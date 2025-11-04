"""
Train the RNN text generation model
Main training orchestrator script
"""

import os
import sys
import time
import json
from pathlib import Path

# GPU CONFIGURATION - Enable GPU if available (PyTorch)
import torch

# Check and configure GPU
if torch.cuda.is_available():
    print(f"✓ Found {torch.cuda.device_count()} GPU(s)")
    print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✓ PyTorch using CUDA for training (fast)")
else:
    print("⚠ No GPU detected - using CPU for training (slower)")

from text_generator import TextGenerator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class TimingMetrics:
    """Utility class to track timing metrics throughout training."""
    
    def __init__(self):
        self.metrics = {}
        self.section_start_times = {}
    
    def start_section(self, section_name: str):
        """Mark the start of a timing section."""
        self.section_start_times[section_name] = time.time()
        print(f"\n[TIMING] Starting: {section_name}")
    
    def end_section(self, section_name: str):
        """Mark the end of a timing section and record duration."""
        if section_name not in self.section_start_times:
            print(f"⚠ Warning: {section_name} was not started")
            return
        
        elapsed = time.time() - self.section_start_times[section_name]
        self.metrics[section_name] = elapsed
        print(f"[TIMING] Completed: {section_name} ({self._format_time(elapsed)})")
        return elapsed
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m ({seconds:.0f}s)"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h ({seconds:.0f}s)"
    
    def get_total_time(self) -> float:
        """Get total time across all sections."""
        return sum(self.metrics.values())
    
    def print_summary(self):
        """Print a comprehensive timing summary."""
        print("\n" + "=" * 70)
        print("TRAINING TIME METRICS SUMMARY")
        print("=" * 70)
        
        total_time = self.get_total_time()
        
        for section, elapsed in self.metrics.items():
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            formatted_time = self._format_time(elapsed)
            print(f"  {section:<30} {formatted_time:>15}  ({percentage:>5.1f}%)")
        
        print("-" * 70)
        print(f"  {'TOTAL TIME':<30} {self._format_time(total_time):>15}  ({100.0:>5.1f}%)")
        print("=" * 70)
    
    def save_metrics(self, output_path: str):
        """Save metrics to a JSON file for documentation."""
        metrics_data = {
            "sections": self.metrics,
            "total_seconds": self.get_total_time(),
            "total_formatted": self._format_time(self.get_total_time()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"\n✓ Metrics saved to: {output_path}")


def load_training_data(data_path: str = 'data/training_text.txt') -> str:
    """Load training data from file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at {data_path}\n"
            "Please download training text first using scripts/download_data.py"
        )
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("RNN TEXT GENERATION - TRAINING PIPELINE")
    print("=" * 70)
    
    # Initialize timing metrics
    timing = TimingMetrics()
    overall_start = time.time()
    
    # Configuration - SIMPLIFIED FOR SMALL DATASET (Anti-overfitting)
    # Smaller model = less memorization, better generalization
    SEQUENCE_LENGTH = 50       # Shorter sequences = less memorization
    EMBEDDING_DIM = 128        # Much smaller embeddings
    LSTM_UNITS = 128           # Reduced capacity (half of before)
    NUM_LSTM_LAYERS = 1        # Single layer = simpler model
    DROPOUT_RATE = 0.4        # High dropout for regularization
    EPOCHS = 25                # Let early stopping decide
    BATCH_SIZE = 32            # Smaller batches = more regularization
    MIN_WORD_FREQ = 1          # Filter rare words
    LEARNING_RATE = 0.001      # Standard learning rate
    GRADIENT_CLIP = 0.5        # Tighter clipping
    LABEL_SMOOTHING = 0.25      # Strong label smoothing to prevent overfitting

    # Paths
    DATA_PATH = r"data\training_text.txt"
    MODEL_DIR = '../saved_models'
    VIZ_DIR = '../visualizations'
    METRICS_PATH = '../visualizations/training_time_metrics.json'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    print("\n[1/5] Loading training data...")
    timing.start_section("Load Training Data")
    try:
        raw_text = load_training_data(DATA_PATH)
        print(f"✓ Loaded {len(raw_text):,} characters")
        print(f"  ({len(raw_text.split()):,} words)")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    timing.end_section("Load Training Data")
    
    print("\n[2/5] Preprocessing text...")
    timing.start_section("Preprocess Text")
    generator = TextGenerator(
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout_rate=DROPOUT_RATE
    )
    
    cleaned_text = generator.preprocess_text(raw_text)
    print(f"✓ Cleaned text: {len(cleaned_text):,} characters")
    print(f"  ({len(cleaned_text.split()):,} words)")
    timing.end_section("Preprocess Text")
    
    print("\n[3/5] Preparing sequences...")
    timing.start_section("Prepare Sequences")
    X, y = generator.prepare_sequences(cleaned_text, min_word_freq=MIN_WORD_FREQ)
    vocab_size = len(generator.tokenizer.word_index)
    print(f"✓ Created {len(X):,} training sequences")
    print(f"  Vocabulary size: {vocab_size:,} words")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {y.shape}")
    timing.end_section("Prepare Sequences")
    
    print("\n[4/5] Building model...")
    timing.start_section("Build Model")
    generator.build_model(vocab_size)
    print(f"✓ Model built successfully")
    print("\nModel Architecture:")
    print(generator.get_model_summary())
    timing.end_section("Build Model")
    
    print("\n[5/5] Training model...")
    print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    timing.start_section("Model Training")
    history = generator.train(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        early_stopping=True,
        learning_rate=LEARNING_RATE,
        gradient_clip=GRADIENT_CLIP,
        label_smoothing=LABEL_SMOOTHING
    )
    timing.end_section("Model Training")
    
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    print("\nSaving model and tokenizer...")
    timing.start_section("Save Model and Tokenizer")
    generator.save_model(MODEL_DIR)
    timing.end_section("Save Model and Tokenizer")
    
    print("\nGenerating visualizations...")
    timing.start_section("Generate Visualizations")
    arch_path = os.path.join(VIZ_DIR, 'model_architecture.png')
    history_path = os.path.join(VIZ_DIR, 'training_history.png')
    
    try:
        generator.visualize_architecture(arch_path)
    except (ImportError, AttributeError) as e:
        print(f"⚠ Skipped architecture diagram: {e}")
    
    try:
        generator.plot_training_history(history_path)
    except (Exception, AttributeError) as e:
        print(f"⚠ Skipped training history plot: {e}")
    timing.end_section("Generate Visualizations")
    
    print("\nTesting text generation...")
    timing.start_section("Text Generation Testing")
    test_prompts = [
        "the",
        "in the morning",
        "it was a dark"
    ]
    
    print("\n--- Generated Text Samples ---")
    for prompt in test_prompts:
        try:
            generated = generator.generate_text(
                seed_text=prompt,
                num_words=25,
                temperature=0.8,           # Lower temp for coherence
                top_k=50,                  # Limit choices
                top_p=0.92,                # Nucleus sampling
                repetition_penalty=1.5,    # Strong anti-repetition
                use_beam_search=False      # Use sampling
            )
            print(f"\nSeed: '{prompt}'")
            print(f"Generated: '{generated}'")
        except Exception as e:
            print(f"Could not generate from '{prompt}': {e}")
    timing.end_section("Text Generation Testing")
    
    # Calculate total time
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    
    # Print timing summary
    timing.print_summary()
    
    # Save metrics to file
    timing.save_metrics(METRICS_PATH)
    
    print(f"\nModel saved to: {MODEL_DIR}")
    print(f"Visualizations saved to: {VIZ_DIR}")
    print(f"Time metrics saved to: {METRICS_PATH}")
    print("\nYou can now run the FastAPI server with:")
    print("  python main.py")


if __name__ == "__main__":
    main()
