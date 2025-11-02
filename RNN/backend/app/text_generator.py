"""
TextGenerator: RNN-based text generation with LSTM architecture using PyTorch
Handles text preprocessing, model building, training, and generation
"""

from __future__ import annotations

import os
import re
import json
import pickle  # legacy fallback only
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    import seaborn as sns  # noqa: F401
except ImportError:
    sns = None  # not required

# ----------------------------- Device ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")


# --------------------------- Model ---------------------------------
class LSTMModel(nn.Module):
    """PyTorch LSTM model for text generation with optional pre-trained embeddings."""
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_units: int,
        num_lstm_layers: int,
        dropout_rate: float,
        activation_fn: str = "relu",
        embedding_weights: Optional[np.ndarray] = None,
        trainable_embeddings: bool = True,
        recurrent_dropout: float = 0.0,
    ):
        super().__init__()

        # Embedding
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_weights),
                freeze=not trainable_embeddings,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.embedding_dropout = nn.Dropout(dropout_rate * 0.3)

        # Unidirectional LSTM (causal)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            dropout=max(dropout_rate, recurrent_dropout) if num_lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.dense = nn.Linear(lstm_units, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(512, vocab_size)

        if activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "elu":
            self.activation = nn.ELU()
        elif activation_fn == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]                # last time step
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


# ------------------------- Text Generator --------------------------
class TextGenerator:
    """Complete text generation system using PyTorch LSTM networks."""
    def __init__(
        self,
        sequence_length: int = 50,
        embedding_dim: int = 100,
        lstm_units: int = 150,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.2,
        recurrent_dropout: float = 0.0,
        vocab_size: Optional[int] = None,
        activation_fn: str = "relu",
        use_glove_embeddings: bool = False,
        trainable_embeddings: bool = True,
    ):
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
        self.index_to_word: Dict[int, str] = {}
        self.model: Optional[LSTMModel] = None
        self.history = None
        self.config = None
        self.embedding_weights = None
        self.device = DEVICE

    # ------------------------- Preprocess --------------------------
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess raw text - PRESERVE PUNCTUATION for sentence structure."""
        text = text.lower()

        # Remove Project Gutenberg headers/footers
        text = re.sub(r"\*\*\*.*?(START|END).*?\*\*\*", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Allow a broad set of punctuation incl. em dash
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\'\-\:\;\"()\—]', "", text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Filter empties
        words = [w for w in text.split() if w.strip()]
        return " ".join(words)

    def prepare_sequences(self, text: str, min_word_freq: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Convert text to sequences for training."""
        words = text.split()

        # Frequency filter
        from collections import Counter
        word_counts = Counter(words)
        frequent_words = {w for w, c in word_counts.items() if c >= min_word_freq}
        filtered_words = [w if w in frequent_words else "<OOV>" for w in words]

        # Vocabulary
        unique_frequent = set(w for w in filtered_words if w != "<OOV>")
        word_to_idx = {"<OOV>": 0}
        word_to_idx.update({w: i + 1 for i, w in enumerate(sorted(unique_frequent))})

        print(f"  Original vocabulary: {len(set(words)):,} words")
        print(f"  After filtering (min_freq={min_word_freq}): {len(word_to_idx) - 1:,} words")

        # Simple tokenizer
        class SimpleTokenizer:
            def __init__(self, word_index: Dict[str, int]):
                self.word_index = word_index

            def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
                out = []
                for t in texts:
                    out.append([self.word_index.get(w, 0) for w in t.split()])
                return out

        self.tokenizer = SimpleTokenizer(word_to_idx)
        self.index_to_word = {idx: w for w, idx in word_to_idx.items()}

        seqs = [word_to_idx.get(w, 0) for w in filtered_words]

        X, y = [], []
        for i in range(len(seqs) - self.sequence_length):
            X.append(seqs[i : i + self.sequence_length])
            y.append(seqs[i + self.sequence_length])

        return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)

    # -------------------- Optional GloVe ---------------------------
    def load_glove_embeddings(self, glove_file: str = "glove_embeddings/glove.6B.100d.txt") -> bool:
        if not Path(glove_file).exists():
            print(f"⚠ GloVe file not found at {glove_file}")
            print("  Download GloVe with: python download_glove.py")
            return False

        print(f"📖 Loading GloVe embeddings from {glove_file}...")
        embeddings_index: Dict[str, np.ndarray] = {}
        with open(glove_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i and i % 100000 == 0:
                    print(f"  Loaded {i:,} embeddings...")
                parts = line.rstrip().split(" ")
                word = parts[0]
                vec = np.asarray(parts[1:], dtype="float32")
                if vec.shape[0] == self.embedding_dim:
                    embeddings_index[word] = vec

        print(f"✓ Loaded {len(embeddings_index):,} GloVe embeddings")
        self.glove_index = embeddings_index
        return True

    def create_embedding_matrix(self, word_index: Dict[str, int]) -> np.ndarray:
        vocab_size = len(word_index)
        mat = np.zeros((vocab_size, self.embedding_dim), dtype="float32")

        glove_index = getattr(self, "glove_index", {})
        oov = 0
        for w, idx in word_index.items():
            v = glove_index.get(w) or glove_index.get(w.lower())
            if v is not None:
                mat[idx] = v
            else:
                mat[idx] = np.random.normal(-0.1, 0.1, self.embedding_dim)
                oov += 1

        coverage = (vocab_size - oov) / max(vocab_size, 1) * 100.0
        print(f"✓ Created embedding matrix ({vocab_size}, {self.embedding_dim})")
        print(f"  GloVe coverage: {coverage:.1f}%  OOV: {oov:,}")
        return mat

    # ------------------- Build / Train -----------------------------
    def build_model(self, vocab_size: Optional[int] = None) -> LSTMModel:
        if vocab_size is None:
            if self.vocab_size is not None:
                vocab_size = self.vocab_size
            elif self.tokenizer is not None:
                vocab_size = (max(self.tokenizer.word_index.values()) + 1) if self.tokenizer.word_index else 1
            else:
                raise ValueError("vocab_size is not known; tokenizer is missing and no explicit value provided.")

        embedding_weights = None
        if self.use_glove_embeddings and hasattr(self, "glove_index") and self.tokenizer is not None:
            embedding_weights = self.create_embedding_matrix(self.tokenizer.word_index)

        self.model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            num_lstm_layers=self.num_lstm_layers,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
            embedding_weights=embedding_weights,
            trainable_embeddings=self.trainable_embeddings,
            recurrent_dropout=self.recurrent_dropout,
        ).to(self.device)
        self.vocab_size = vocab_size
        return self.model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        learning_rate: float = 0.001,
        gradient_clip: float = 1.0,
        label_smoothing: float = 0.0,
        weight_decay: float = 0.0,
    ):
        # Split
        split = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.long, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.long, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=self.device)

        # Loaders
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

        # Optimizer / scheduler / loss
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=learning_rate * 0.01
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "learning_rate": []}
        best_val = float("inf")
        patience, patience_cnt = 15, 0

        print(f"✓ Training on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_acc, batches = 0.0, 0.0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", colour="green")
            for xb, yb in pbar:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()

                train_loss += loss.item()
                train_acc += (out.argmax(1) == yb).float().mean().item()
                batches += 1
                pbar.set_postfix({"loss": f"{train_loss / batches:.4f}"})

            scheduler.step()
            train_loss /= max(batches, 1)
            train_acc /= max(batches, 1)

            # Validation
            self.model.eval()
            val_loss, val_acc, vbatches = 0.0, 0.0, 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    vo = self.model(xv)
                    val_loss += loss_fn(vo, yv).item()
                    val_acc += (vo.argmax(1) == yv).float().mean().item()
                    vbatches += 1
            val_loss /= max(vbatches, 1)
            val_acc /= max(vbatches, 1)

            lr_now = optimizer.param_groups[0]["lr"]
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["learning_rate"].append(lr_now)

            print(
                f"✓ Epoch {epoch+1}/{epochs} "
                f"- loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, lr: {lr_now:.6f}"
            )

            # Early stopping
            if early_stopping:
                if val_loss < best_val * 0.995:
                    best_val = val_loss
                    patience_cnt = 0
                    self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        print(f"⚠ Early stopping at epoch {epoch+1}")
                        if hasattr(self, "best_model_state"):
                            self.model.load_state_dict(self.best_model_state)
                            self.model.to(self.device).eval()
                            print("✓ Restored best model weights")
                        break

        self.history = type("History", (), {"history": history})()
        return self.history

    # -------------------- Save / Load -------------------------------
    def save_model(self, model_dir: str = "saved_models") -> None:
        """Save model (state_dict), tokenizer.json, and config.json."""
        mdir = Path(model_dir)
        mdir.mkdir(parents=True, exist_ok=True)

        model_path = mdir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path}")

        if self.tokenizer is not None:
            tok_path = mdir / "tokenizer.json"
            with tok_path.open("w", encoding="utf-8") as f:
                json.dump(self.tokenizer.word_index, f, indent=2)
            print(f"✓ Tokenizer word_index saved to {tok_path}")

        cfg = {
            "sequence_length": self.sequence_length,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout_rate": self.dropout_rate,
            "vocab_size": (len(self.tokenizer.word_index) if self.tokenizer else (self.vocab_size or 0)),
        }
        cfg_path = mdir / "config.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        print(f"✓ Config saved to {cfg_path}")

    @staticmethod
    def _is_lfs_pointer(path: Path) -> bool:
        try:
            with path.open("rb") as f:
                head = f.read(64)
            return head.startswith(b"version https://git-lfs.github.com/spec/v1")
        except Exception:
            return False

    def load_model(self, model_dir: str = "saved_models") -> None:
        """Load model and tokenizer from disk (JSON tokenizer; pickle as legacy fallback)."""
        mdir = Path(model_dir).resolve()

        # ---- Config
        cfg_path = mdir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json at {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)
        print(f"✓ Config loaded from {cfg_path}")

        # Refresh params from config
        self.sequence_length = self.config.get("sequence_length", self.sequence_length)
        self.embedding_dim = self.config.get("embedding_dim", self.embedding_dim)
        self.lstm_units = self.config.get("lstm_units", self.lstm_units)
        self.num_lstm_layers = self.config.get("num_lstm_layers", self.num_lstm_layers)
        self.dropout_rate = self.config.get("dropout_rate", self.dropout_rate)
        cfg_vocab = self.config.get("vocab_size")

        # ---- Tokenizer (JSON preferred)
        tok_json = mdir / "tokenizer.json"
        tok_pkl = mdir / "tokenizer.pkl"  # legacy fallback
        try:
            if tok_json.exists():
                with tok_json.open("r", encoding="utf-8") as f:
                    word_index: Dict[str, int] = json.load(f)

                class SimpleTokenizer:
                    def __init__(self, word_index: Dict[str, int]):
                        self.word_index = word_index

                    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
                        out = []
                        for t in texts:
                            out.append([self.word_index.get(w, 0) for w in t.split()])
                        return out

                self.tokenizer = SimpleTokenizer(word_index)
                self.index_to_word = {idx: w for w, idx in word_index.items()}
                print(f"✓ Tokenizer loaded from {tok_json}")
            elif tok_pkl.exists():
                with tok_pkl.open("rb") as f:
                    self.tokenizer = pickle.load(f)
                # Build reverse mapping if possible
                wi = getattr(self.tokenizer, "word_index", {})
                self.index_to_word = {idx: w for w, idx in wi.items()} if wi else {}
                print(f"✓ Tokenizer loaded from {tok_pkl}")
            else:
                print("⚠ No tokenizer file found; attempting best-effort rebuild from training data (if present)")
                self.tokenizer = None
                self.index_to_word = {}
        except Exception as e:
            print(f"! Warning: tokenizer load failed: {e}")
            self.tokenizer = None
            self.index_to_word = {}

        # ---- Build model (determine vocab)
        vocab_for_build: Optional[int] = None
        if cfg_vocab and isinstance(cfg_vocab, int) and cfg_vocab > 0:
            vocab_for_build = cfg_vocab
        elif self.tokenizer is not None and getattr(self.tokenizer, "word_index", None):
            vocab_for_build = max(self.tokenizer.word_index.values()) + 1

        self.build_model(vocab_for_build)

        # ---- Weights
        model_path = mdir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing weights at {model_path}")

        if self._is_lfs_pointer(model_path):
            raise RuntimeError(
                f"{model_path} looks like a Git LFS pointer, not real weights. "
                "Ensure your deploy fetches LFS blobs (git lfs fetch && git lfs checkout) "
                "or commit the real binary file."
            )

        try:
            state = torch.load(str(model_path), map_location=self.device)
            if not isinstance(state, dict):
                raise TypeError("Expected a state_dict (dict) in model.pt")
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PyTorch weights from {model_path}. "
                f"Expected a state_dict saved with torch.save(model.state_dict(), ...). "
                f"Underlying error: {e}"
            )

    # ------------------ Post-processing / Sampling ------------------
    def _apply_capitalization(self, text: str) -> str:
        if not text:
            return text
        words = text.split()
        if not words:
            return text

        # Capitalize first token
        if words[0] and words[0][0].isalpha():
            words[0] = words[0][0].upper() + words[0][1:]

        capitalize_next = False
        for i, word in enumerate(words):
            if word == "i":
                words[i] = "I"
            if capitalize_next and word and word[0].isalpha():
                words[i] = word[0].upper() + word[1:]
                capitalize_next = False
            if word and word[-1] in ".!?":
                capitalize_next = True
            elif word in ".!?":
                capitalize_next = True

        return " ".join(words)

    def _safe_sample(self, probs: np.ndarray) -> int:
        if not isinstance(probs, np.ndarray):
            probs = np.array(probs)
        probs = np.nan_to_num(probs, nan=0.0)
        total = probs.sum()
        if total <= 0 or np.isclose(total, 0.0):
            return int(np.argmax(probs))
        probs = probs / total
        try:
            return int(np.random.choice(len(probs), p=probs))
        except Exception:
            return int(np.argmax(probs))

    def _apply_top_k_filtering(self, logits: np.ndarray, top_k: int = 50) -> np.ndarray:
        if top_k <= 0 or top_k >= len(logits):
            return logits
        indices_to_remove = np.argsort(logits)[:-top_k]
        logits[indices_to_remove] = -np.inf
        return logits

    def _apply_top_p_filtering(self, logits: np.ndarray, top_p: float = 0.9) -> np.ndarray:
        if top_p >= 1.0:
            return logits
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
        cumsum_probs = np.cumsum(exp_logits / np.sum(exp_logits))
        to_remove = cumsum_probs > top_p
        if np.any(to_remove):
            to_remove[0] = False
            logits[sorted_indices[to_remove]] = -np.inf
        return logits

    def _apply_repetition_penalty(
        self,
        logits: np.ndarray,
        recent_words: List[int],
        penalty: float = 1.2,
        window_size: int = 15,
    ) -> np.ndarray:
        if not recent_words or penalty <= 1.0:
            return logits

        common_words = {
            "the", "a", "an", "and", "or", "of", "to", "in", "is", "was", "be",
            "been", "are", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "at", "by", "for",
            "with", "on", "from", "up", "as", "it", "that", "this", "which", "who",
            "what", "when", "where", "why", "how", "i", "you", "he", "she", "we", "they",
        }
        common_idx = {idx for w, idx in self.tokenizer.word_index.items() if w.lower() in common_words}

        recent_window = recent_words[-window_size:] if len(recent_words) > window_size else recent_words
        from collections import Counter
        counts = Counter(recent_window)

        for idx, cnt in counts.items():
            if 0 <= idx < len(logits):
                multiplier = cnt if cnt > 1 else 1
                if idx in common_idx:
                    actual = 1.0 + (penalty - 1.0) * 0.3
                else:
                    actual = penalty * multiplier
                if actual > 1.0:
                    logits[idx] /= actual
        return logits

    def _block_repeated_ngrams(
        self,
        logits: np.ndarray,
        generated_words: List[str],
        candidate_idx: int,
        ngram_size: int = 3,
    ) -> bool:
        if len(generated_words) < ngram_size:
            return False
        candidate_word = self.index_to_word.get(candidate_idx)
        if not candidate_word or candidate_word == "<OOV>":
            return False

        recent = generated_words[-min(20, len(generated_words)):]
        for n in range(3, ngram_size + 1):
            if len(recent) >= n - 1:
                potential = recent[-(n - 1):] + [candidate_word]
                for i in range(len(recent) - n + 1):
                    if recent[i : i + n] == potential:
                        return True
        return False

    # ---------------------- Generation ------------------------------
    def generate_text(
        self,
        seed_text: str,
        num_words: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.85,
        use_beam_search: bool = True,
        beam_width: int = 3,
        repetition_penalty: float = 2.5,
        diversity_boost: float = 1.0,
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Please retrain the model.")

        seed = self.preprocess_text(seed_text)

        if use_beam_search:
            return self._generate_beam_search(seed, num_words, beam_width, temperature)

        generated = seed
        generated_indices: List[int] = []
        generated_words = seed.split()
        oov_idx = int(self.tokenizer.word_index.get("<OOV>", 0))

        with torch.no_grad():
            for _ in range(num_words):
                token_list = self.tokenizer.texts_to_sequences([generated])[0]
                token_list = token_list[-self.sequence_length:]
                if len(token_list) < self.sequence_length:
                    token_list = [0] * (self.sequence_length - len(token_list)) + token_list

                inp = torch.tensor([token_list], dtype=torch.long, device=self.device)
                logits = self.model(inp)[0].detach().cpu().numpy()
                logits = logits / max(temperature, 0.01)

                # punctuation boost
                for punct in {".", ",", "!", "?", ";", ":"}:
                    idx = self.tokenizer.word_index.get(punct)
                    if idx is not None and 0 <= idx < logits.shape[0]:
                        logits[idx] *= 2.5

                # mask OOV
                if 0 <= oov_idx < logits.shape[0]:
                    logits[oov_idx] = -np.inf

                # balanced repetition control
                if generated_words:
                    word_last_seen = {}
                    for i, w in enumerate(reversed(generated_words)):
                        if w not in word_last_seen:
                            word_last_seen[w] = len(generated_words) - 1 - i

                    very_common = {
                        "the", "a", "and", "of", "to", "in", "is", "was", "for", "with", "on",
                        "as", "by", "at", "be", "are",
                    }
                    common = {
                        "or", "not", "it", "that", "this", "i", "you", "he", "his", "she", "her",
                        "them", "which", "who", "had", "have", "been",
                    }

                    for w, last_pos in word_last_seen.items():
                        idx = self.tokenizer.word_index.get(w)
                        if idx is None or not (0 <= idx < logits.shape[0]):
                            continue
                        steps_ago = len(generated_words) - last_pos
                        if w.lower() in very_common:
                            if steps_ago == 1:
                                logits[idx] -= 8
                            elif steps_ago < 4:
                                logits[idx] -= 4
                        elif w.lower() in common:
                            if steps_ago == 1:
                                logits[idx] -= 15
                            elif steps_ago < 4:
                                logits[idx] -= 8
                            elif steps_ago < 8:
                                logits[idx] -= 3
                        else:
                            if steps_ago == 1:
                                logits[idx] -= 25
                            elif steps_ago < 5:
                                logits[idx] -= 15
                            elif steps_ago < 10:
                                logits[idx] -= 8

                # block exact repeated sequences (tri/bigrams)
                if len(generated_words) >= 3:
                    last3 = generated_words[-3:]
                    for i in range(len(generated_words) - 4):
                        if generated_words[i : i + 3] == last3 and i + 3 < len(generated_words):
                            wblock = generated_words[i + 3]
                            bidx = self.tokenizer.word_index.get(wblock)
                            if bidx and 0 <= bidx < logits.shape[0]:
                                logits[bidx] = -np.inf
                if len(generated_words) >= 2:
                    last2 = generated_words[-2:]
                    count = 0
                    for i in range(len(generated_words) - 2):
                        if generated_words[i : i + 2] == last2:
                            count += 1
                    if count > 1:
                        for w, idx in self.tokenizer.word_index.items():
                            if 0 <= idx < logits.shape[0]:
                                if [generated_words[-1], w] == last2:
                                    logits[idx] = -np.inf

                # top-k / top-p
                if top_k > 0 and top_k < logits.shape[0]:
                    remove = np.argsort(logits)[:-top_k]
                    logits[remove] = -np.inf
                if 0.0 < top_p < 1.0:
                    sorted_idx = np.argsort(logits)[::-1]
                    sorted_logits = logits[sorted_idx]
                    exp = np.exp(sorted_logits - np.max(sorted_logits))
                    csum = np.cumsum(exp / (exp.sum() + 1e-10))
                    to_remove = csum > top_p
                    if np.any(to_remove):
                        to_remove[0] = False
                        logits[sorted_idx[to_remove]] = -np.inf

                # probs & sample
                exp = np.exp(logits - np.max(logits))
                probs = exp / (exp.sum() + 1e-10)
                next_id = self._safe_sample(probs)
                next_word = self.index_to_word.get(int(next_id))

                if (next_word is None) or (next_word == "<OOV>"):
                    probs[oov_idx] = -np.inf if 0 <= oov_idx < probs.shape[0] else probs[oov_idx]
                    next_id = int(np.argmax(probs))
                    next_word = self.index_to_word.get(next_id)

                if next_word and next_word != "<OOV>":
                    generated += " " + next_word
                    generated_indices.append(next_id)
                    generated_words.append(next_word)

        return self._apply_capitalization(generated)

    def _generate_beam_search(
        self,
        seed_text: str,
        num_words: int,
        beam_width: int = 3,
        temperature: float = 0.8,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 4,
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        reverse_idx = self.index_to_word if self.index_to_word else {v: k for k, v in self.tokenizer.word_index.items()}
        beams = [(seed_text, 0.0, [])]
        oov_idx = int(self.tokenizer.word_index.get("<OOV>", 0))

        with torch.no_grad():
            for _ in range(num_words):
                new_beams = []
                for text, logp, word_ids in beams:
                    token_list = self.tokenizer.texts_to_sequences([text])[0]
                    token_list = token_list[-self.sequence_length:]
                    if len(token_list) < self.sequence_length:
                        token_list = [0] * (self.sequence_length - len(token_list)) + token_list

                    inp = torch.tensor([token_list], dtype=torch.long, device=self.device)
                    logits = self.model(inp)[0].detach().cpu().numpy()
                    logits = logits / max(temperature, 0.01)

                    logits_masked = logits.copy()
                    if 0 <= oov_idx < logits_masked.shape[0]:
                        logits_masked[oov_idx] = -np.inf

                    if repetition_penalty > 1.0 and word_ids:
                        logits_masked = self._apply_repetition_penalty(
                            logits_masked, word_ids, penalty=repetition_penalty, window_size=15
                        )

                    gen_words = text.split()
                    if no_repeat_ngram_size > 0 and len(gen_words) >= no_repeat_ngram_size - 1:
                        topc = np.argsort(logits_masked)[-100:]
                        for idx in topc:
                            if self._block_repeated_ngrams(logits_masked, gen_words, idx, no_repeat_ngram_size):
                                logits_masked[idx] = -np.inf

                    exp = np.exp(logits_masked - np.max(logits_masked))
                    probs = exp / (exp.sum() + 1e-10)

                    top_idx = np.argsort(probs)[-beam_width * 2:]
                    added = 0
                    for idx in reversed(top_idx):
                        if added >= beam_width:
                            break
                        p = float(probs[idx])
                        if p <= 1e-12:
                            continue
                        w = reverse_idx.get(int(idx))
                        if w and w != "<OOV>":
                            new_text = text + " " + w
                            new_beams.append((new_text, logp + np.log(max(p, 1e-12)), word_ids + [int(idx)]))
                            added += 1

                new_beams = sorted(new_beams, key=lambda t: t[1], reverse=True)[:beam_width]
                beams = new_beams if new_beams else beams

        return self._apply_capitalization(beams[0][0] if beams else seed_text)

    # ------------------------ Utilities ----------------------------
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        if self.history is None:
            print("No training history available.")
            return
        hist = self.history.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(hist["loss"], label="Training Loss", linewidth=2)
        ax1.plot(hist["val_loss"], label="Validation Loss", linewidth=2)
        ax1.set_title("Model Loss", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(hist["accuracy"], label="Training Accuracy", linewidth=2)
        ax2.plot(hist["val_accuracy"], label="Validation Accuracy", linewidth=2)
        ax2.set_title("Model Accuracy", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()

    def get_model_summary(self) -> str:
        return str(self.model)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
