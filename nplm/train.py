from __future__ import annotations
import argparse
import json
import math
import time
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown

nltk.download("punkt", quiet=True)
nltk.download("brown", quiet=True)

from tqdm import tqdm


def write_jsonl_splits(
    shard_dir: str,
    document_unit: str = "sentence",
    seed: int = 92,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1) ) -> Tuple[str, str, str]:
    """
    Create JSONL shards (train/valid/test) from the Brown corpus..

    Args:
        shard_dir (str): Directory to save the JSONL files.
        document_unit (str): Unit of text to consider as a document ("sentence" or "paragraph").
        seed (int): Random seed for shuffling.
        splits (Tuple[float, float, float]): Proportions for train, validation, and test splits.
    Returns:
        Tuple[str, str, str]: Paths to the train, validation, and test JSONL
    """

    random.seed(seed)
    out_base = Path(shard_dir)
    train_dir = out_base / "train"
    valid_dir = out_base / "valid"
    test_dir = out_base / "test"

    for d in [train_dir, valid_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Prepare text units
    texts: List[str] = []
    if document_unit == "sentence":
        texts = brown.sents()
        texts = [" ".join(sent) for sent in texts]
    else:
        for fileid in brown.fileids():
            sents = brown.sents(fileids=[fileid])
            texts.append(" ".join([" ".join(s) for s in sents]))

    # shuffle and split
    random.shuffle(texts)
    n = len(texts)
    n_train = int(splits[0] * n)
    n_valid = int(splits[1] * n)

    train_texts = texts[:n_train]
    valid_texts = texts[n_train : n_train + n_valid]
    test_texts = texts[n_train + n_valid :]

    def dump(list_texts, out_path: Path):
        fp= out_path/"data.jsonl"
        with fp.open("w", encoding="utf-8") as f:
            for t in list_texts:
                obj = {"text": t}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return str(fp)

    train_path = dump(train_texts, train_dir)
    valid_path = dump(valid_texts, valid_dir)
    test_path = dump(test_texts, test_dir)

    print(f"Wrote {len(train_texts)} train, {len(valid_texts)} valid, {len(test_texts)} test lines to {shard_dir}.")
    return train_path, valid_path, test_path

class Vocabulary:
    """
    A simple word-level vocabulary class.
    """
    def __init__(self, min_freq: int = 2, max_vocab: int = None):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.freqs: Dict[str, int] = {}
        self.min_freq = min_freq # this is used to filter rare words
        self.max_vocab = max_vocab

        # Special tokens
        self.pad = "<pad>"
        self.unk = "<unk>" # unknown word
        self.bos = "<bos>"
        self.eos = "<eos>"
        self.specials = [self.pad, self.unk, self.bos, self.eos]

    def build(self, jsonl_dir: str) -> None:
        """
        Build vocabulary from JSONL files in the specified directory.

        Args:
            jsonl_dir (str): Directory containing .jsonl or .jsonl.gz files.
        """
        p = Path(jsonl_dir)/"data.jsonl"

        # count token frequencies
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                tokens = word_tokenize(text)
                for tok in tokens:
                    self.freqs[tok] = self.freqs.get(tok, 0) + 1

        # sort by frequency and apply min_freq and max_vocab constraints
        sorted_tokens = sorted(
                [t for t, c in self.freqs.items() if c >= self.min_freq],
                key=lambda x: -self.freqs[x]
        )
        sorted_tokens = sorted_tokens[: self.max_vocab - len(self.specials)]

        self.id_to_token = self.specials + sorted_tokens
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert a tokens to IDs.

        Args:
            tokens (List[str]): List of tokens.
        Returns:
            List[int]: List of token IDs.
        """
        return [self.token_to_id.get(t, self.token_to_id[self.unk]) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """
        Convert IDs back to tokens.

        Args:
            ids (List[int]): List of token IDs.
        Returns:
            List[str]: List of tokens.
        """
        return [self.id_to_token[i] for i in ids]

    def save(self, path: str) -> None:
        """
        Save the vocabulary to a JSON file.

        Args:
            path (str): Path to save the vocabulary JSON.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.id_to_token},f)

    @classmethod
    def load(cls, path: str) -> Vocabulary:
        """
        Load a vocabulary from a JSON file.

        Args:
            path (str): Path to the vocabulary JSON.
        Returns:
            Vocabulary: Loaded Vocabulary object.
        """
        with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        vocab = cls()
        vocab.id_to_token = data["itos"]
        vocab.token_to_id = {t: i for i, t in enumerate(vocab.id_to_token)}
        return vocab

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad]
    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk]
    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos]
    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos]

class NextWordDataset(Dataset):
    """Dataset that generates (n-1)-gram contexts to predict the next word."""

    def __init__(self, jsonl_path: str, vocab: Vocabulary, n: int = 5):
        self.examples: List[Tuple[List[int], int]] = []
        self.vocab = vocab
        self.context_size = n - 1

        p = Path(jsonl_path)
        if p.is_dir():
            p = p/"data.jsonl"
        assert p.exists(), f"JSONL file not found: {p}"

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                tokens = [vocab.bos] + word_tokenize(text) + [vocab.eos]
                ids = vocab.encode(tokens)

                # Generate (context, target) pairs
                for i in range(self.context_size, len(ids)):
                    context = ids[i - self.context_size : i]
                    target = ids[i]
                    self.examples.append((context, target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.examples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class NPLM(torch.nn.Module):
    """A simple Neural Probabilistic Language Model (NPLM) given (n-1) gram context."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128, context_size: int = 4, hidden_dim: int = 256):
        """
        Initialize the NPLM model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            context_size (int): Size of the context (n-1 for n-grams).
            hidden_dim (int): Dimension of the hidden layer.
        """
        super().__init__()
        self.context_size = context_size
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        input_dim= embedding_dim * context_size
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, context_size).
        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, vocab_size).
        """
        embed = self.embeddings(inputs)
        embed = embed.view(embed.size(0), -1)
        return self.ff(embed)

def train_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for contexts, targets in tqdm(dataloader, desc="Training", leave=False):
        contexts, targets = contexts.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(contexts)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * contexts.size(0)
        total_tokens += contexts.size(0)

    return total_loss / total_tokens

def eval_loss(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module, device: torch.device) -> float:
    """Evaluate average negative log-likelihood on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ctx, tgt in tqdm(dataloader, desc="eval batches", leave=False):
            ctx, tgt = ctx.to(device), tgt.to(device)
            logits = model(ctx)
            loss = criterion(logits, tgt)
            total_loss += loss.item() * ctx.size(0)
            total_tokens += ctx.size(0)
    return total_loss / total_tokens

def perplexity_from_nll(nll: float) -> float:
    """Convert negative log-likelihood to perplexity."""
    try:
        return math.exp(nll)
    except OverflowError:
        return float("inf")

def save_training_plot(train_losses: List[float], valid_losses: List[float],
                       filename: str = "artifacts/training_plot.png") -> None:

    """Save a plot of training and validation losses."""
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train NLL", linestyle='--')
    plt.plot(epochs, valid_losses, label="Validation NLL", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved training plot -> {filename}")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train a Neural Probabilistic Language Model (NPLM)")
    p.add_argument("--shard_dir", type=str, default="data/brown_jsonl", help="Where to write/read JSONL shards")
    p.add_argument("--vocab_path", type=str, default="artifacts/brown_word_vocab.json", help="Path to save/load vocab")
    p.add_argument("--min_freq", type=int, default=2, help="Minimum frequency for a token to be included")
    p.add_argument("--max_vocab", type=int, default=20000, help="Maximum vocabulary size")
    p.add_argument("--n", type=int, default=5, help="N-gram size: model uses n-1 words to predict nth")
    p.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    p.add_argument("--hid_dim", type=int, default=256, help="Hidden layer dimension")
    p.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    p.add_argument("--rebuild_shards", action="store_true", help="Rebuild JSONL shards from Brown corpus")
    return p.parse_args()

def main() -> None:
    """Main training and evaluation routine."""
    args = parse_args()
    os.makedirs(os.path.dirname(args.vocab_path) or ".", exist_ok=True)


    # Prepare JSONL shards
    if args.rebuild_shards or not (Path(args.shard_dir) / "train" / "data.jsonl").exists():
        print("Building JSONL shards from Brown corpus...")
        write_jsonl_splits(args.shard_dir, document_unit="sentence")
    else:
        print("Using existing shards in", args.shard_dir)


    # Build/load vocabulary
    if not Path(args.vocab_path).exists():
        print("Building vocabulary...")
        vocab = Vocabulary(min_freq=args.min_freq, max_vocab=args.max_vocab)
        vocab.build(jsonl_dir=str(Path(args.shard_dir) / "train"))
        vocab.save(args.vocab_path)
        print(f"Saved vocab with size {len(vocab.id_to_token)} -> {args.vocab_path}")
    else:
        print("Loading vocabulary...")
        vocab = Vocabulary.load(args.vocab_path)
        print(f"Loaded vocab with size {len(vocab.id_to_token)}")


    vocab_size = len(vocab.id_to_token)
    context_size = args.n - 1


    # Datasets
    train_ds = NextWordDataset(str(Path(args.shard_dir) / "train"), vocab=vocab, n=args.n)
    valid_ds = NextWordDataset(str(Path(args.shard_dir) / "valid"), vocab=vocab, n=args.n)
    test_ds = NextWordDataset(str(Path(args.shard_dir) / "test"), vocab=vocab, n=args.n)


    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)


    # Device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("Using device:", device)


    # Model
    model = NPLM(vocab_size=vocab_size, embedding_dim=args.emb_dim, context_size=context_size, hidden_dim=args.hid_dim)
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)


    # Training loop
    best_valid_nll = float("inf")
    train_losses, valid_losses = [], []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_nll = train_epoch(model, train_loader, optim, criterion, device)
        valid_nll = eval_loss(model, valid_loader, criterion, device)
        print(f" train NLL: {train_nll:.4f}, ppl: {perplexity_from_nll(train_nll):.2f}")
        print(f" valid NLL: {valid_nll:.4f}, ppl: {perplexity_from_nll(valid_nll):.2f}")
        if valid_nll < best_valid_nll:
            best_valid_nll = valid_nll
            best_valid_ppl = perplexity_from_nll(best_valid_nll)
            ckpt_path = Path("artifacts") / "best_model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "vocab_path": args.vocab_path}, ckpt_path)
            print("Saved best model ->", ckpt_path)
        train_losses.append(train_nll)
        valid_losses.append(valid_nll)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.2f} minutes.")
    save_training_plot(train_losses, valid_losses, filename="artifacts/training_plot.png")

    # Final evaluation
    test_nll = eval_loss(model, test_loader, criterion, device)
    test_ppl = perplexity_from_nll(test_nll)
    print(f"Test NLL: {test_nll:.4f}, ppl: {test_ppl:.2f}")

    # Save final report
    report = {"vocab_size": vocab_size,
            "emb_dim": args.emb_dim,
            "context_size": context_size,
            "hid_dim": args.hid_dim,
            "test_nll": round(test_nll,2),
            "test_ppl": round(test_ppl,2),
            "best_valid_ppl": round(best_valid_ppl,2),
            "valid_nll": round(best_valid_nll,2),
            "train_time_mins": round(elapsed / 60,),
            "epochs": args.epochs}
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        print("Saved report -> artifacts/metrics.json")

if __name__ == "__main__":
    main()
