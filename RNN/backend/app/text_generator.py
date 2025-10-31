"""
TextGenerator: RNN-based text generation with LSTM architecture using PyTorch
Handles text preprocessing, model building, training, and generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import re
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    import seaborn as sns
except ImportError:
    sns = None
import os
from pathlib import Path

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")


class LSTMModel(nn.Module):
    """PyTorch LSTM model for text generation with optional pre-trained embeddings."""
    
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_lstm_layers, dropout_rate, 
                 activation_fn='relu', embedding_weights=None, trainable_embeddings=True,
                 recurrent_dropout=0.0):
        super(LSTMModel, self).__init__()
        
        # Create embedding layer
        if embedding_weights is not None:
            # Use pre-trained embeddings (e.g., GloVe)
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_weights),
                freeze=not trainable_embeddings,  # freeze=False means trainable
                padding_idx=0
            )
        else:
            # Learn embeddings from scratch
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Add embedding dropout for regularization
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.3)
        
        # Unidirectional LSTM - only looks at past context (required for causal generation)
        # Note: PyTorch doesn't have direct recurrent_dropout, we simulate with dropout between layers
        self.lstm = nn.LSTM(embedding_dim, lstm_units, num_lstm_layers, 
                           dropout=max(dropout_rate, recurrent_dropout) if num_lstm_layers > 1 else 0,
                           batch_first=True, bidirectional=False)
        
        # Output dimension is lstm_units (unidirectional)
        # Add hidden layer with stronger regularization
        self.dense = nn.Linear(lstm_units, 512)
        self.layer_norm = nn.LayerNorm(512)  # Add layer normalization for stability
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(512, vocab_size)
        
        # Set activation function
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation = nn.GELU()
        elif activation_fn == 'elu':
            self.activation = nn.ELU()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()  # Default
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)  # Apply dropout to embeddings
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last output (unidirectional, past context only)
        x = self.dense(x)
        x = self.layer_norm(x)  # Normalize before activation
        x = self.activation(x)  # Use configurable activation
        x = self.dropout(x)
        x = self.output(x)
        return x


class TextGenerator:
    """Complete text generation system using PyTorch LSTM networks."""
    
    def __init__(self, 
                 sequence_length: int = 50,
                 embedding_dim: int = 100,
                 lstm_units: int = 150,
                 num_lstm_layers: int = 2,
                 dropout_rate: float = 0.2,
                 recurrent_dropout: float = 0.0,
                 vocab_size: Optional[int] = None,
                 activation_fn: str = 'relu',
                 use_glove_embeddings: bool = False,
                 trainable_embeddings: bool = True):
        """Initialize TextGenerator.
        
        Args:
            use_glove_embeddings: Whether to use pre-trained GloVe embeddings
            trainable_embeddings: Whether to fine-tune embeddings during training
            recurrent_dropout: Dropout rate for recurrent connections
        """
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn
        self.use_glove_embeddings = use_glove_embeddings
        self.trainable_embeddings = trainable_embeddings
        self.vocab_size = vocab_size
        
        self.tokenizer = None
        self.model = None
        self.history = None
        self.config = None
        self.embedding_weights = None  # For GloVe embeddings
        self.device = DEVICE
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess raw text - PRESERVE PUNCTUATION for sentence structure."""
        text = text.lower()
        
        # Remove Project Gutenberg headers/footers
        text = re.sub(r'\*\*\*.*?(START|END).*?\*\*\*', '', text, flags=re.DOTALL)
        
        # KEEP punctuation that marks sentence boundaries and dialogue
        # Allow: letters, numbers, apostrophes (for contractions), hyphens, periods, commas, 
        # exclamation marks, question marks, quotes, semicolons, colons, parentheses
        text = re.sub(r"[^a-z0-9\s\.\,\!\?\'\-\"\:\;\(\)—]", '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Filter empty tokens
        words = [w for w in text.split() if w.strip()]
        
        return ' '.join(words)
    
    def prepare_sequences(self, text: str, min_word_freq: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Convert text to sequences for training."""
        words = text.split()
        
        # Count word frequencies to filter rare words
        from collections import Counter
        word_counts = Counter(words)
        
        # Filter rare words BEFORE building vocabulary
        frequent_words = {word for word, count in word_counts.items() if count >= min_word_freq}
        filtered_words = [word if word in frequent_words else '<OOV>' for word in words]
        
        # Build vocabulary ONLY from frequent words (plus OOV)
        unique_frequent_words = set(word for word in filtered_words if word != '<OOV>')
        word_to_idx = {'<OOV>': 0}  # Start with OOV at 0
        word_to_idx.update({word: idx + 1 for idx, word in enumerate(sorted(unique_frequent_words))})
        
        print(f"  Original vocabulary: {len(set(words)):,} words")
        print(f"  After filtering (min_freq={min_word_freq}): {len(word_to_idx)-1:,} words")
        
        # Create a proper Tokenizer class that can be pickled
        class SimpleTokenizer:
            def __init__(self, word_index):
                self.word_index = word_index
            
            def texts_to_sequences(self, texts):
                result = []
                for text in texts:
                    seq = [self.word_index.get(w, 0) for w in text.split()]
                    result.append(seq)
                return result
        
        # Store tokenizer as instance of SimpleTokenizer
        self.tokenizer = SimpleTokenizer(word_to_idx)
        # Cache reverse mapping for fast lookup (index -> word)
        self.index_to_word = {idx: w for w, idx in word_to_idx.items()}
        
        sequences = [word_to_idx.get(w, 0) for w in filtered_words]
        
        # Create training data using sliding window
        X, y = [], []
        for i in range(len(sequences) - self.sequence_length):
            X.append(sequences[i:i + self.sequence_length])
            y.append(sequences[i + self.sequence_length])
        
        return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)
    
    def load_glove_embeddings(self, glove_file: str = 'glove_embeddings/glove.6B.100d.txt'):
        """Load GloVe pre-trained embeddings.
        
        Args:
            glove_file: Path to GloVe embeddings file
            
        Reference: Pennington, J., Socher, R., & Manning, C. D. (2014).
                   "GloVe: Global Vectors for Word Representation"
        """
        import os
        
        if not os.path.exists(glove_file):
            print(f"⚠ GloVe file not found at {glove_file}")
            print(f"  Download GloVe embeddings using: python download_glove.py")
            return False
        
        print(f"📖 Loading GloVe embeddings from {glove_file}...")
        embeddings_index = {}
        
        with open(glove_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % 100000 == 0 and i > 0:
                    print(f"  Loaded {i:,} embeddings...")
                
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                
                if len(coefs) == self.embedding_dim:
                    embeddings_index[word] = coefs
        
        print(f"✓ Loaded {len(embeddings_index):,} GloVe embeddings")
        self.glove_index = embeddings_index
        return True
    
    def create_embedding_matrix(self, word_index: Dict) -> np.ndarray:
        """Create embedding matrix using GloVe weights.
        
        Args:
            word_index: Dictionary mapping words to indices
            
        Returns:
            Embedding matrix (vocab_size, embedding_dim)
        """
        vocab_size = len(word_index)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        glove_index = getattr(self, 'glove_index', {})
        oov_count = 0
        
        for word, idx in word_index.items():
            if word in glove_index:
                embedding_matrix[idx] = glove_index[word]
            elif word.lower() in glove_index:
                embedding_matrix[idx] = glove_index[word.lower()]
            else:
                # Random initialization for OOV words
                embedding_matrix[idx] = np.random.normal(-0.1, 0.1, self.embedding_dim)
                oov_count += 1
        
        coverage = (vocab_size - oov_count) / vocab_size * 100
        print(f"✓ Created embedding matrix ({vocab_size}, {self.embedding_dim})")
        print(f"  GloVe coverage: {coverage:.1f}% ({vocab_size - oov_count:,}/{vocab_size:,} words)")
        print(f"  OOV words (random init): {oov_count:,}")
        
        return embedding_matrix.astype('float32')
    
    def build_model(self, vocab_size: int):
        """Build the PyTorch LSTM model architecture."""
        embedding_weights = None
        
        if self.use_glove_embeddings and hasattr(self, 'glove_index'):
            # Create embedding matrix from GloVe
            embedding_matrix = self.create_embedding_matrix(self.tokenizer.word_index)
            embedding_weights = embedding_matrix
        
        self.model = LSTMModel(vocab_size, self.embedding_dim, self.lstm_units, 
                              self.num_lstm_layers, self.dropout_rate, self.activation_fn,
                              embedding_weights=embedding_weights,
                              trainable_embeddings=self.trainable_embeddings,
                              recurrent_dropout=self.recurrent_dropout).to(self.device)
        self.vocab_size = vocab_size
        return self.model
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              epochs: int = 100,
              batch_size: int = 128,
              validation_split: float = 0.2,
              early_stopping: bool = True,
              learning_rate: float = 0.001,
              gradient_clip: float = 1.0,
              label_smoothing: float = 0.0,
              weight_decay: float = 0.0):
        """Train the LSTM model using PyTorch with improved optimization."""
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.long).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.long).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation DataLoader (to avoid OOM)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss (using AdamW for better generalization with weight decay)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Cosine annealing with warm restarts for better exploration
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=1,
            eta_min=learning_rate * 0.01  # Minimum LR is 1% of initial
        )
        
        # Loss function with label smoothing
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience = 15  # Patience for early stopping
        patience_counter = 0
        
        print(f"✓ Training on {self.device}...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_acc = 0
            batch_count = 0
            
            # Progress bar for batches
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", colour="green")
            
            for X_batch, y_batch in pbar:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += (outputs.argmax(1) == y_batch).float().mean().item()
                batch_count += 1
                
                # Update progress bar description with current loss
                avg_loss = train_loss / batch_count
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Step scheduler after each epoch
            scheduler.step()
            
            train_loss /= batch_count
            train_acc /= batch_count
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_acc = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for X_batch_val, y_batch_val in val_loader:
                    val_outputs = self.model(X_batch_val)
                    val_loss += loss_fn(val_outputs, y_batch_val).item()
                    val_acc += (val_outputs.argmax(1) == y_batch_val).float().mean().item()
                    val_batch_count += 1
            
            val_loss /= max(val_batch_count, 1)
            val_acc /= max(val_batch_count, 1)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['learning_rate'].append(current_lr)
            
            print(f"✓ Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, lr: {current_lr:.6f}")
            
            # Early stopping - check if validation loss improved
            if early_stopping:
                if val_loss < best_val_loss * 0.995:  # Allow small fluctuations
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"⚠ Early stopping at epoch {epoch+1}")
                        # Restore best model
                        if hasattr(self, 'best_model_state'):
                            self.model.load_state_dict(self.best_model_state)
                            print("✓ Restored best model weights")
                        break
        
        self.history = type('History', (), {'history': history})()
        return self.history
    
    def save_model(self, model_dir: str = 'saved_models') -> None:
        """Save model and tokenizer to disk."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save tokenizer word_index as JSON (more portable than pickle)
        tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
        if self.tokenizer is not None:
            with open(tokenizer_path, 'w') as f:
                json.dump(self.tokenizer.word_index, f, indent=2)
            print(f"✓ Tokenizer word_index saved to {tokenizer_path}")
        
        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_lstm_layers': self.num_lstm_layers,
            'dropout_rate': self.dropout_rate,
            'vocab_size': len(self.tokenizer.word_index) if self.tokenizer else 0
        }
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Config saved to {config_path}")
    
    def load_model(self, model_dir: str = 'saved_models') -> None:
        """Load model and tokenizer from disk."""
        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print(f"✓ Config loaded from {config_path}")
        
        # Update from config
        self.sequence_length = self.config.get('sequence_length', 30)
        self.embedding_dim = self.config.get('embedding_dim', 50)
        self.lstm_units = self.config.get('lstm_units', 75)
        self.num_lstm_layers = self.config.get('num_lstm_layers', 1)
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        vocab_size = self.config.get('vocab_size', 2000)
        
        # Load tokenizer
        tokenizer_json_path = os.path.join(model_dir, 'tokenizer.json')
        tokenizer_pkl_path = os.path.join(model_dir, 'tokenizer.pkl')  # Fallback for old format
        
        try:
            # Try new JSON format first
            if os.path.exists(tokenizer_json_path):
                with open(tokenizer_json_path, 'r') as f:
                    word_index = json.load(f)
                
                # Create SimpleTokenizer from word_index
                class SimpleTokenizer:
                    def __init__(self, word_index):
                        self.word_index = word_index
                    
                    def texts_to_sequences(self, texts):
                        result = []
                        for text in texts:
                            seq = [self.word_index.get(w, 0) for w in text.split()]
                            result.append(seq)
                        return result
                
                self.tokenizer = SimpleTokenizer(word_index)
                # Cache reverse mapping for fast lookup (index -> word)
                self.index_to_word = {int(idx): w for w, idx in word_index.items()}
                print(f"✓ Tokenizer loaded from {tokenizer_json_path}")
            else:
                # Try old pickle format (backward compatibility)
                with open(tokenizer_pkl_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print(f"✓ Tokenizer loaded from {tokenizer_pkl_path}")
        except Exception as e:
            print(f"! Warning: tokenizer load failed: {e}")
            print("! Rebuilding tokenizer from training data...")
            
            # Try to rebuild from training text
            try:
                training_data_path = os.path.join(model_dir, '..', 'data', 'training_text.txt')
                if os.path.exists(training_data_path):
                    with open(training_data_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    # Preprocess and rebuild
                    text = self.preprocess_text(text)
                    words = text.split()
                    word_to_idx = {word: idx + 1 for idx, word in enumerate(set(words))}
                    word_to_idx['<OOV>'] = 0
                    
                    # Use SimpleTokenizer class (same as in prepare_sequences)
                    class SimpleTokenizer:
                        def __init__(self, word_index):
                            self.word_index = word_index
                        
                        def texts_to_sequences(self, texts):
                            result = []
                            for text in texts:
                                seq = [self.word_index.get(w, 0) for w in text.split()]
                                result.append(seq)
                            return result
                    
                    self.tokenizer = SimpleTokenizer(word_to_idx)
                    # Cache reverse mapping for fast lookup (index -> word)
                    self.index_to_word = {int(idx): w for w, idx in word_to_idx.items()}
                    print(f"✓ Tokenizer rebuilt with {len(word_to_idx)} words")
                else:
                    print(f"⚠ Training data not found at {training_data_path}")
                    self.tokenizer = None
            except Exception as rebuild_err:
                print(f"✗ Failed to rebuild tokenizer: {rebuild_err}")
                self.tokenizer = None
        
        # Load model
        self.build_model(vocab_size)
        model_path = os.path.join(model_dir, 'model.pt')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✓ Model loaded from {model_path}")

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

    def _safe_sample(self, probs: np.ndarray) -> int:
        """Sample an index from a probability distribution safely.

        Falls back to argmax when the distribution is degenerate or sums to zero.
        """
        if not isinstance(probs, np.ndarray):
            probs = np.array(probs)

        # Replace NaNs with zeros
        probs = np.nan_to_num(probs, nan=0.0)

        total = probs.sum()
        if total <= 0 or np.isclose(total, 0.0):
            # Degenerate distribution: use argmax
            return int(np.argmax(probs))

        # Normalize and sample
        probs = probs / total
        try:
            return int(np.random.choice(len(probs), p=probs))
        except Exception:
            return int(np.argmax(probs))
    
    def _apply_top_k_filtering(self, logits: np.ndarray, top_k: int = 50) -> np.ndarray:
        """Apply top-k filtering to logits."""
        if top_k <= 0 or top_k >= len(logits):
            return logits
        
        indices_to_remove = np.argsort(logits)[:-top_k]
        logits[indices_to_remove] = -np.inf
        return logits
    
    def _apply_top_p_filtering(self, logits: np.ndarray, top_p: float = 0.9) -> np.ndarray:
        """Apply nucleus (top-p) filtering to logits."""
        if top_p >= 1.0:
            return logits
        
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        # Convert to probabilities
        exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
        cumsum_probs = np.cumsum(exp_logits / np.sum(exp_logits))
        
        # Find cutoff
        sorted_indices_to_remove = cumsum_probs > top_p
        if np.any(sorted_indices_to_remove):
            sorted_indices_to_remove[0] = False  # Keep at least one token
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -np.inf
        
        return logits
    
    def _apply_repetition_penalty(self, logits: np.ndarray, recent_words: List[int], 
                                  penalty: float = 1.2, window_size: int = 15) -> np.ndarray:
        """Apply smart repetition penalty to discourage recent words (but allow common ones).
        
        Args:
            logits: Current logits from model
            recent_words: List of recently generated word indices
            penalty: Penalty multiplier (>1.0 reduces probability)
            window_size: How many recent words to consider
        """
        if not recent_words or penalty <= 1.0:
            return logits
        
        # Common words that should be less penalized (they're needed for fluent English)
        common_words = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'was', 'be', 
                       'been', 'are', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                       'would', 'could', 'should', 'may', 'might', 'can', 'at', 'by', 'for',
                       'with', 'on', 'from', 'up', 'as', 'it', 'that', 'this', 'which', 'who',
                       'what', 'when', 'where', 'why', 'how', 'i', 'you', 'he', 'she', 'we', 'they'}
        
        # Convert common words to indices for faster lookup
        common_indices = set()
        for word, idx in self.tokenizer.word_index.items():
            if word.lower() in common_words:
                common_indices.add(idx)
        
        # Look only at the most recent words
        recent_window = recent_words[-window_size:] if len(recent_words) > window_size else recent_words
        
        # Count how many times each word appears in recent window
        from collections import Counter
        word_counts = Counter(recent_window)
        
        for word_idx, count in word_counts.items():
            if 0 <= word_idx < len(logits):
                # Stronger penalty for words that appear multiple times
                multiplier = count if count > 1 else 1
                
                # Reduce penalty for common words
                if word_idx in common_indices:
                    actual_penalty = 1.0 + (penalty - 1.0) * 0.3  # Only 30% penalty for common words
                else:
                    actual_penalty = penalty * multiplier  # Full penalty, scaled by repetition
                
                if actual_penalty > 1.0:
                    logits[word_idx] /= actual_penalty
        
        return logits
    
    def _block_repeated_ngrams(self, logits: np.ndarray, generated_words: List[str], 
                               candidate_idx: int, ngram_size: int = 3) -> bool:
        """Check if adding this word would create a repeated n-gram.
        
        Args:
            logits: Current logits
            generated_words: List of already generated words (strings)
            candidate_idx: Index of candidate word to check
            ngram_size: Size of n-grams to check for repetition
            
        Returns:
            True if this would create a repeated n-gram, False otherwise
        """
        if len(generated_words) < ngram_size:
            return False
        
        # Get the candidate word
        candidate_word = self.index_to_word.get(candidate_idx)
        if not candidate_word or candidate_word == '<OOV>':
            return False
        
        # OPTIMIZATION: Only check recent context (last 20 words) to avoid slowdown
        recent_words = generated_words[-min(20, len(generated_words)):]
        
        # Check if adding this word would complete a repeated n-gram
        for n in range(3, ngram_size + 1):  # Start at 3-grams
            if len(recent_words) >= n - 1:
                # Build the potential n-gram
                potential_ngram = recent_words[-(n-1):] + [candidate_word]
                
                # Check if this exact n-gram appears in recent context
                for i in range(len(recent_words) - n + 1):
                    if recent_words[i:i+n] == potential_ngram:
                        return True
        
        return False
    
    def generate_text(self, 
                     seed_text: str,
                     num_words: int = 50,
                     temperature: float = 0.8,
                     top_k: int = 40,
                     top_p: float = 0.85,
                     use_beam_search: bool = True,
                     beam_width: int = 3,
                     repetition_penalty: float = 2.5,
                     diversity_boost: float = 1.0) -> str:
        """Generate text using the trained model with STRONG repetition prevention.
        
        Args:
            seed_text: Starting text for generation
            num_words: Number of words to generate
            temperature: Temperature for softmax (lower=more deterministic)
            top_k: Keep only top-k most likely tokens (0 to disable)
            top_p: Nucleus sampling threshold (0 to 1, 0 to disable)
            use_beam_search: Whether to use beam search
            beam_width: Number of beams for beam search
            repetition_penalty: STRONGER penalty for repeated words (>1.0 to reduce repetition)
            diversity_boost: Boost for encouraging diverse word choices (>1.0 = more diversity)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Please retrain the model.")

        # Preprocess seed so it matches training preprocessing/tokenization
        seed = self.preprocess_text(seed_text)

        if use_beam_search:
            return self._generate_beam_search(seed, num_words, beam_width, temperature)

        generated = seed
        generated_indices = []  # Track word indices for repetition penalty
        generated_words = seed.split()  # Track actual words

        # OOV index (ensure int)
        oov_idx = int(self.tokenizer.word_index.get('<OOV>', 0))

        with torch.no_grad():
            for step in range(num_words):
                # Tokenize last context
                token_list = self.tokenizer.texts_to_sequences([generated])[0]
                token_list = token_list[-self.sequence_length:]

                # Pad if needed
                if len(token_list) < self.sequence_length:
                    token_list = [0] * (self.sequence_length - len(token_list)) + token_list

                # Predict
                input_t = torch.tensor([token_list], dtype=torch.long).to(self.device)
                logits = self.model(input_t)[0].cpu().numpy()

                # Temperature scaling
                logits = logits / max(temperature, 0.01)

                # ==================== PUNCTUATION BOOST ====================
                # Make periods, commas, and other punctuation more likely
                # This addresses run-on sentences by encouraging punctuation placement
                PUNCTUATION_MARKS = {'.', ',', '!', '?', ';', ':'}
                PUNCTUATION_BOOST = 2.5  # Tune this value (2.0-4.0): higher = more punctuation
                
                for punct in PUNCTUATION_MARKS:
                    punct_idx = self.tokenizer.word_index.get(punct)
                    if punct_idx is not None and 0 <= punct_idx < len(logits):
                        logits[punct_idx] *= PUNCTUATION_BOOST
                # ==================== END PUNCTUATION BOOST ====================

                # Mask OOV token to avoid generating <OOV>
                if 0 <= oov_idx < logits.shape[0]:
                    logits[oov_idx] = -np.inf

                # ==================== BALANCED REPETITION PENALTY ====================
                # Balance: prevent unnatural repetition WITHOUT breaking coherence
                if len(generated_words) > 0:
                    from collections import Counter
                    
                    # Track ALL generated words with their recency
                    word_last_seen = {}
                    for i, word in enumerate(reversed(generated_words)):
                        if word not in word_last_seen:
                            word_last_seen[word] = len(generated_words) - 1 - i
                    
                    # Apply BALANCED penalties (not too aggressive)
                    for word, last_pos in word_last_seen.items():
                        word_idx = self.tokenizer.word_index.get(word)
                        if word_idx and 0 <= word_idx < len(logits):
                            steps_ago = len(generated_words) - last_pos
                            
                            # Define categories of words
                            very_common = {'the', 'a', 'and', 'of', 'to', 'in', 'is', 'was', 
                                          'for', 'with', 'on', 'as', 'by', 'at', 'be', 'are'}
                            common = {'or', 'not', 'it', 'that', 'this', 'i', 'you', 'he', 'his',
                                     'she', 'her', 'them', 'which', 'who', 'had', 'have', 'been'}
                            
                            if word.lower() in very_common:
                                # Very common words: minimal penalty (allow natural flow)
                                if steps_ago == 1:  # Immediately previous
                                    logits[word_idx] -= 8  # Gentle penalty
                                elif steps_ago < 4:  # 1-3 words ago
                                    logits[word_idx] -= 4
                                # Beyond 4 words: no penalty (natural repetition)
                            
                            elif word.lower() in common:
                                # Common words: moderate penalty
                                if steps_ago == 1:  
                                    logits[word_idx] -= 15
                                elif steps_ago < 4:
                                    logits[word_idx] -= 8
                                elif steps_ago < 8:
                                    logits[word_idx] -= 3
                            
                            else:
                                # Rare words: stronger penalty to avoid loops
                                if steps_ago == 1:
                                    logits[word_idx] -= 25  # Block immediate repeat
                                elif steps_ago < 5:
                                    logits[word_idx] -= 15
                                elif steps_ago < 10:
                                    logits[word_idx] -= 8
                                # Beyond 10 words: natural repetition allowed

                # ==================== BLOCK EXACT REPEATED SEQUENCES ====================
                # Prevent exact phrases from repeating (e.g., "the man" appearing twice)
                if len(generated_words) >= 3:
                    # Check last 3 words
                    last_trigram = generated_words[-3:]
                    
                    # Search for this trigram earlier in text
                    for i in range(len(generated_words) - 4):
                        if generated_words[i:i+3] == last_trigram:
                            # Found exact repetition! Block next word to break the cycle
                            if i + 3 < len(generated_words):
                                word_to_block = generated_words[i + 3]
                                block_idx = self.tokenizer.word_index.get(word_to_block)
                                if block_idx and 0 <= block_idx < len(logits):
                                    logits[block_idx] = -np.inf
                
                # Also block repeated bigrams
                if len(generated_words) >= 2:
                    last_bigram = generated_words[-2:]
                    bigram_count = 0
                    for i in range(len(generated_words) - 2):
                        if generated_words[i:i+2] == last_bigram:
                            bigram_count += 1
                    
                    # If bigram appeared 2+ times already, block it from appearing again
                    if bigram_count > 1:
                        for word, idx in self.tokenizer.word_index.items():
                            if idx > 0 and 0 <= idx < len(logits):
                                # Check if adding this word creates the repeated bigram
                                potential_bigram = [generated_words[-1], word]
                                if potential_bigram == last_bigram:
                                    logits[idx] = -np.inf

                # Apply top-k filtering
                if top_k > 0 and top_k < len(logits):
                    indices_to_remove = np.argsort(logits)[:-top_k]
                    logits[indices_to_remove] = -np.inf

                # Apply top-p (nucleus) filtering
                if 0 < top_p < 1.0:
                    sorted_indices = np.argsort(logits)[::-1]
                    sorted_logits = logits[sorted_indices]
                    exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
                    cumsum_probs = np.cumsum(exp_logits / np.sum(exp_logits))
                    sorted_indices_to_remove = cumsum_probs > top_p
                    if np.any(sorted_indices_to_remove):
                        sorted_indices_to_remove[0] = False
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[indices_to_remove] = -np.inf

                # Convert to probabilities
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / (np.sum(exp_logits) + 1e-10)

                # Sample
                predicted_id = self._safe_sample(probs)
                predicted_word = self.index_to_word.get(int(predicted_id))

                # Fallback if word not found
                if (predicted_word is None) or (predicted_word == '<OOV>'):
                    # Pick best non-OOV word
                    probs[oov_idx] = -np.inf
                    predicted_id = np.argmax(probs)
                    predicted_word = self.index_to_word.get(int(predicted_id))

                if predicted_word and predicted_word != '<OOV>':
                    generated += ' ' + predicted_word
                    generated_indices.append(int(predicted_id))
                    generated_words.append(predicted_word)

        return self._apply_capitalization(generated)
    
    def _generate_beam_search(self, 
                             seed_text: str, 
                             num_words: int, 
                             beam_width: int = 3,
                             temperature: float = 0.8,
                             repetition_penalty: float = 1.3,
                             no_repeat_ngram_size: int = 4) -> str:
        """Generate text using beam search for better quality with repetition prevention."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        
        # Use cached reverse mapping
        reverse_idx = getattr(self, 'index_to_word', {v: k for k, v in self.tokenizer.word_index.items()})

        # Initialize beams: (text, log_probability, word_indices)
        beams = [(seed_text, 0.0, [])]

        oov_idx = int(self.tokenizer.word_index.get('<OOV>', 0))

        with torch.no_grad():
            for step in range(num_words):
                new_beams = []

                for text, log_prob, word_indices in beams:
                    # Tokenize
                    token_list = self.tokenizer.texts_to_sequences([text])[0]
                    token_list = token_list[-self.sequence_length:]

                    if len(token_list) < self.sequence_length:
                        token_list = [0] * (self.sequence_length - len(token_list)) + token_list

                    # Predict
                    input_t = torch.tensor([token_list], dtype=torch.long).to(self.device)
                    logits = self.model(input_t)[0].cpu().numpy()
                    logits = logits / max(temperature, 0.01)

                    # Mask OOV
                    logits_masked = logits.copy()
                    if 0 <= oov_idx < logits_masked.shape[0]:
                        logits_masked[oov_idx] = -np.inf
                    
                    # Apply repetition penalty
                    if repetition_penalty > 1.0 and word_indices:
                        logits_masked = self._apply_repetition_penalty(
                            logits_masked, word_indices, 
                            penalty=repetition_penalty, 
                            window_size=15
                        )
                    
                    # Block repeated n-grams (only check top candidates)
                    generated_words = text.split()
                    if no_repeat_ngram_size > 0 and len(generated_words) >= no_repeat_ngram_size - 1:
                        top_k_indices = np.argsort(logits_masked)[-100:]  # Check only top 100
                        for idx in top_k_indices:
                            if self._block_repeated_ngrams(logits_masked, generated_words, idx, no_repeat_ngram_size):
                                logits_masked[idx] = -np.inf

                    # Convert to probs
                    exp_logits = np.exp(logits_masked - np.max(logits_masked))
                    probs = exp_logits / (np.sum(exp_logits) + 1e-10)

                    # Get top beam_width candidates
                    top_indices = np.argsort(probs)[-beam_width * 2:]  # Get more candidates

                    added = 0
                    for idx in reversed(top_indices):
                        if added >= beam_width:
                            break
                        if probs[idx] > 1e-10:
                            word = reverse_idx.get(int(idx))
                            if word and word != '<OOV>':
                                new_text = text + ' ' + word
                                new_log_prob = log_prob + np.log(max(probs[idx], 1e-12))
                                new_indices = word_indices + [int(idx)]
                                new_beams.append((new_text, new_log_prob, new_indices))
                                added += 1

                # Keep only top beam_width sequences
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                beams = new_beams if new_beams else beams

        # Return the best sequence
        result = beams[0][0] if beams else seed_text
        return self._apply_capitalization(result)
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        hist = self.history.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(hist['loss'], label='Training Loss', linewidth=2)
        ax1.plot(hist['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(hist['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(hist['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        return str(self.model)
    
    def count_params(self) -> int:
        """Count total trainable parameters in model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
