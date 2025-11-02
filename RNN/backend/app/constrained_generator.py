"""
Constrained Text Generator
Integrates grammar checking DURING generation to force grammatical outputs
"""

import torch
import numpy as np
import spacy
from typing import List, Set, Dict
from collections import defaultdict


class GrammarConstrainedGenerator:
    """
    Extends text generation with grammar constraints applied during decoding.
    Checks each candidate token for grammaticality before allowing it.
    """
    
    def __init__(self, text_generator):
        """
        Initialize with your existing TextGenerator.
        
        Args:
            text_generator: Your trained TextGenerator instance
        """
        self.generator = text_generator
        
        # Load spaCy for grammar analysis
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("WARNING: spaCy model not found. Grammar constraints disabled.")
            self.nlp = None
        
        # POS tag requirements for valid sentences
        self.required_pos = {'VERB', 'NOUN'}  # Must have verb and noun
        
    def is_grammatical_continuation(self, context_words: List[str], next_word: str) -> bool:
        """
        Check if adding next_word to context creates grammatical text.
        
        Args:
            context_words: List of words generated so far
            next_word: Candidate next word
            
        Returns:
            True if grammatically valid, False otherwise
        """
        if self.nlp is None:
            return True  # No constraints if spaCy unavailable
        
        # Form candidate sentence
        candidate_text = ' '.join(context_words + [next_word])
        doc = self.nlp(candidate_text)
        
        # Rule 1: Don't allow immediate word repetition (unless common words)
        if len(context_words) > 0:
            last_word = context_words[-1].lower()
            if next_word.lower() == last_word:
                # Allow repetition only for very common words
                common_repeatable = {'the', 'a', 'and', 'or', 'of', 'to', 'in'}
                if next_word.lower() not in common_repeatable:
                    return False
        
        # Rule 2: Check for subject-verb agreement in simple cases
        tokens = list(doc)
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                token = tokens[i]
                next_token = tokens[i + 1]
                
                # Check subject-verb agreement
                if token.dep_ == 'nsubj' and next_token.pos_ == 'VERB':
                    # Plural subject should not have singular verb
                    if token.tag_ in ['NNS', 'NNPS']:  # Plural nouns
                        if next_token.tag_ == 'VBZ':  # Singular verb (e.g., "runs")
                            return False
                    
                    # Singular subject should not have plural verb
                    if token.tag_ in ['NN', 'NNP']:  # Singular nouns
                        if next_token.tag_ == 'VBP':  # Plural verb (e.g., "run")
                            return False
        
        # Rule 3: Ensure sentence has proper structure if it's getting long
        if len(context_words) + 1 >= 5:
            pos_tags = {token.pos_ for token in doc}
            
            # Must have at least one verb
            if 'VERB' not in pos_tags:
                # Only allow verbs or words that could become verbs
                next_token_doc = self.nlp(next_word)
                if next_token_doc[0].pos_ != 'VERB':
                    return False
        
        # Rule 4: Punctuation logic
        if next_word in '.!?':
            # Only allow end punctuation after we have complete thought
            if len(context_words) < 3:
                return False
            
            # Must have verb before ending
            pos_tags = {token.pos_ for token in doc}
            if 'VERB' not in pos_tags:
                return False
        
        # Rule 5: Don't start with punctuation (except quotes)
        if len(context_words) == 0 and next_word in ',.!?;:':
            return False
        
        return True
    
    def get_valid_next_tokens(self, context: str, logits: np.ndarray, 
                             top_k: int = 50) -> List[int]:
        """
        Filter logits to only include grammatically valid next tokens.
        
        Args:
            context: Current generated text
            logits: Model output logits for all vocabulary
            top_k: Consider top K candidates from model
            
        Returns:
            List of valid token indices
        """
        if self.nlp is None:
            # No filtering if spaCy unavailable
            top_indices = np.argsort(logits)[-top_k:]
            return top_indices.tolist()
        
        # Get context words
        context_words = context.split()
        
        # Get top-k candidates from model
        top_indices = np.argsort(logits)[-top_k:]
        
        valid_indices = []
        for idx in top_indices:
            # Convert index to word
            word = self.generator.index_to_word.get(idx, '<OOV>')
            if word == '<OOV>':
                continue
            
            # Check if grammatically valid
            if self.is_grammatical_continuation(context_words, word):
                valid_indices.append(idx)
        
        # If no valid tokens found, fall back to top candidates
        # (prevents getting stuck)
        if len(valid_indices) == 0:
            valid_indices = top_indices[-10:].tolist()
        
        return valid_indices
    
    def generate_constrained(self, 
                           seed_text: str,
                           num_words: int = 25,
                           temperature: float = 0.8,
                           top_k: int = 50) -> str:
        """
        Generate text with grammar constraints applied during generation.
        
        Args:
            seed_text: Starting text
            num_words: Number of words to generate
            temperature: Sampling temperature
            top_k: Number of top candidates to consider
            
        Returns:
            Generated text with better grammar
        """
        # Preprocess seed
        seed = self.generator.preprocess_text(seed_text)
        generated = seed
        generated_words = seed.split()
        
        print(f"🔒 Grammar-constrained generation started...")
        print(f"   Seed: '{seed}'")
        
        with torch.no_grad():
            for step in range(num_words):
                # Tokenize context
                token_list = self.generator.tokenizer.texts_to_sequences([generated])[0]
                token_list = token_list[-self.generator.sequence_length:]
                
                # Pad if needed
                if len(token_list) < self.generator.sequence_length:
                    token_list = [0] * (self.generator.sequence_length - len(token_list)) + token_list
                
                # Get model predictions
                input_t = torch.tensor([token_list], dtype=torch.long).to(self.generator.device)
                logits = self.generator.model(input_t)[0].cpu().numpy()
                
                # Apply temperature
                logits = logits / max(temperature, 0.01)
                
                # Get grammatically valid tokens
                valid_indices = self.get_valid_next_tokens(generated, logits, top_k)
                
                # Filter logits to only valid tokens
                filtered_logits = np.full_like(logits, -np.inf)
                filtered_logits[valid_indices] = logits[valid_indices]
                
                # Sample from valid tokens
                exp_logits = np.exp(filtered_logits - np.max(filtered_logits))
                probs = exp_logits / np.sum(exp_logits)
                
                # Handle NaN (shouldn't happen, but safety)
                if np.any(np.isnan(probs)):
                    next_idx = valid_indices[0] if valid_indices else np.argmax(logits)
                else:
                    next_idx = np.random.choice(len(probs), p=probs)
                
                # Get next word
                next_word = self.generator.index_to_word.get(next_idx, '<OOV>')
                
                if next_word == '<OOV>':
                    continue
                
                # Add to generated text
                generated += ' ' + next_word
                generated_words.append(next_word)
                
                # Stop if we hit end punctuation
                if next_word in '.!?' and len(generated_words) >= 5:
                    break
        
        print(f"✓ Generated {len(generated_words) - len(seed.split())} words")
        
        # Apply capitalization rules before returning
        return self._apply_capitalization(generated)
    
    def _apply_capitalization(self, text: str) -> str:
        """Apply capitalization rules to generated text.
        
        Rules:
        1. Capitalize standalone 'i' to 'I'
        2. Capitalize first letter after periods (. ! ?)
        3. Capitalize the first letter of the text
        
        Args:
            text: Generated text to capitalize
            
        Returns:
            Text with proper capitalization
        """
        if not text:
            return text
        
        # Split into words while preserving spaces
        words = text.split()
        
        if not words:
            return text
        
        # Capitalize first word
        if words[0] and words[0][0].isalpha():
            words[0] = words[0][0].upper() + words[0][1:]
        
        # Track if we need to capitalize next word (after sentence-ending punctuation)
        capitalize_next = False
        
        for i in range(len(words)):
            word = words[i]
            
            # Rule 1: Capitalize standalone 'i' to 'I'
            if word == 'i':
                words[i] = 'I'
            
            # Rule 2: Capitalize after sentence-ending punctuation
            if capitalize_next and word and word[0].isalpha():
                words[i] = word[0].upper() + word[1:]
                capitalize_next = False
            
            # Check if this word ends with sentence-ending punctuation
            if word and len(word) > 0:
                # Check if word ends with . ! or ?
                if word[-1] in '.!?':
                    capitalize_next = True
                # Also handle case where punctuation is a separate token
                elif word in '.!?':
                    capitalize_next = True
        
        return ' '.join(words)


def compare_generation_methods(text_generator, seed: str = "the", num_words: int = 25):
    """
    Compare regular generation vs grammar-constrained generation.
    
    Args:
        text_generator: Your trained TextGenerator
        seed: Seed text
        num_words: Words to generate
    """
    print("=" * 90)
    print("COMPARISON: Regular vs Grammar-Constrained Generation")
    print("=" * 90)
    
    # Regular generation
    print("\n1. REGULAR GENERATION (Your current method)")
    print("-" * 90)
    regular_output = text_generator.generate_text(
        seed_text=seed,
        num_words=num_words,
        temperature=0.8,
        top_k=50,
        use_beam_search=False
    )
    print(f"Output: {regular_output}")
    
    # Grammar-constrained generation
    print("\n2. GRAMMAR-CONSTRAINED GENERATION (New method)")
    print("-" * 90)
    constrained_gen = GrammarConstrainedGenerator(text_generator)
    constrained_output = constrained_gen.generate_constrained(
        seed_text=seed,
        num_words=num_words,
        temperature=0.8,
        top_k=50
    )
    print(f"Output: {constrained_output}")
    
    # Analyze quality
    print("\n3. QUALITY ANALYSIS")
    print("-" * 90)
    
    if constrained_gen.nlp:
        # Analyze regular output
        doc_regular = constrained_gen.nlp(regular_output)
        regular_pos = [token.pos_ for token in doc_regular]
        
        # Analyze constrained output
        doc_constrained = constrained_gen.nlp(constrained_output)
        constrained_pos = [token.pos_ for token in doc_constrained]
        
        print(f"Regular output:")
        print(f"  - Has verb: {'VERB' in regular_pos}")
        print(f"  - Has noun: {'NOUN' in regular_pos}")
        print(f"  - Proper punctuation: {regular_output.strip()[-1] in '.!?'}")
        
        print(f"\nConstrained output:")
        print(f"  - Has verb: {'VERB' in constrained_pos}")
        print(f"  - Has noun: {'NOUN' in constrained_pos}")
        print(f"  - Proper punctuation: {constrained_output.strip()[-1] in '.!?'}")
    
    print("\n" + "=" * 90)


# Example usage
if __name__ == "__main__":
    print("This module provides grammar-constrained generation.")
    print("Import GrammarConstrainedGenerator and use with your TextGenerator.")
    print("\nExample:")
    print("  from constrained_generator import GrammarConstrainedGenerator")
    print("  from text_generator import TextGenerator")
    print("  ")
    print("  generator = TextGenerator()")
    print("  generator.load_model('saved_models')")
    print("  ")
    print("  constrained = GrammarConstrainedGenerator(generator)")
    print("  text = constrained.generate_constrained('the', num_words=20)")
