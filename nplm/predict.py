"""
predict.py

Use a trained NPLM model to predict the next word given a context.
"""

import torch
import warnings
import json
from pathlib import Path
from nltk.tokenize import word_tokenize
from nplm.train import NPLM, Vocabulary

warnings.simplefilter("ignore", category=FutureWarning)

def load_model(model_path: str, vocab_path: str, device: str = "cpu") -> tuple[NPLM, Vocabulary]:
    """
    Load the trained NPLM model and vocabulary.

    Args:
        model_path (str): Path to saved model weights (.pt or .pth file).
        vocab_path (str): Path to vocabulary JSON file.
        device (str): Device to map model to ("cpu" or "cuda").

    Returns:
        tuple[NPLM, Vocabulary]: Loaded model and vocabulary.
    """
    # Load vocab
    vocab = Vocabulary.load(vocab_path)

    # Load model config from report
    report_path = Path(model_path).parent / "report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    model = NPLM(
        vocab_size=report["vocab_size"],
        embedding_dim=report["emb_dim"],
        context_size=report["context_size"],
        hidden_dim=report["hid_dim"],
    ).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model, vocab


def predict_next_word(model: NPLM, vocab: Vocabulary, context: str, device: str = "cpu") -> str:
    """
    Predict the most likely next word for a given context.

    Args:
        model (NPLM): Trained NPLM model.
        vocab (Vocabulary): Vocabulary object.
        context (str): Input context string (n-1 words).
        device (str): Device to run prediction on.

    Returns:
        str: Predicted next word.
    """
    tokens = word_tokenize(context.lower())
    ids = [vocab.token_to_id.get(tok, vocab.token_to_id["<unk>"]) for tok in tokens]

    # Ensure correct length
    if len(ids) != model.context_size:
        raise ValueError(f"Expected {model.context_size} tokens, got {len(ids)}")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)  # shape: [1, vocab_size]
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()

    return vocab.id_to_token[pred_id]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/best_model.pt", help="Path to trained model")
    parser.add_argument("--vocab", type=str, default="artifacts/brown_word_vocab.json", help="Path to vocabulary JSON")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--context", type=str, required=True, help="Input context of n-1 words")

    args = parser.parse_args()

    model, vocab = load_model(args.model, args.vocab, args.device)
    next_word = predict_next_word(model, vocab, args.context, args.device)

    print(f"Context: {args.context}")
    print(f"Predicted next word: {next_word}")

