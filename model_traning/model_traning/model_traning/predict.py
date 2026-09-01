#!/usr/bin/env python3
"""
Sinhala Fake News Detection — CLI Predictor

Usage:
    python predict.py "අද කොළඹදී විශාල රැස්වීමක් පැවැත්විණි..."
    echo "පුවත් අන්තර්ගතය..." | python predict.py

Uses the best model saved by allmode.ipynb. Prefers a fine-tuned transformer
(mBERT/XLM-R) when one has been saved; otherwise falls back to the frozen
mBERT + GRU head (mbert_gru_best.pt).
"""

import argparse, sys, re, json, os, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Architecture (same as allmode.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
class GRUClassifierPreEmbed(nn.Module):
    """GRU accepting pre-computed embeddings (batch, seq, embed_dim)."""
    def __init__(self, embed_dim, hidden_dim=128, num_layers=2,
                 num_classes=2, dropout=0.5, emb_dropout=0.25,
                 bidirectional=False):
        super().__init__()
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        n_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * n_directions, num_classes)

    def forward(self, x):
        x = self.emb_dropout(x)
        out, hn = self.gru(x)
        if self.gru.bidirectional:
            last = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last = hn[-1]
        out = self.fc(self.dropout(last))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 2. Text Cleaning (same as allmode.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

def remove_special_chars(text):
    cleaned = re.sub(r'[^\u0D80-\u0DFF\u200C\u200D\s\w]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def clean_text(text):
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_special_chars(text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model Loading (fine-tuned transformer preferred, mBERT+GRU fallback)
# ──────────────────────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 150

def load_best_model(device):
    """Return (kind, tokenizer, model) for the best saved model.

    kind == 'transformer' → fine-tuned AutoModelForSequenceClassification
    kind == 'preembed'    → frozen mBERT (AutoModel) for embedding extraction
    """
    from transformers import AutoTokenizer, AutoModel

    # 1) Fine-tuned transformer best (saved by allmode.ipynb if it won)
    for meta_name, pt_name, base in [
        ("mbert_finetuned_best_meta.json", "mbert_finetuned_best.pt", "bert-base-multilingual-cased"),
        ("xlmr_finetuned_best_meta.json", "xlmr_finetuned_best.pt", "xlm-roberta-base"),
    ]:
        if os.path.exists(meta_name) and os.path.exists(pt_name):
            with open(meta_name, encoding="utf-8") as f:
                meta = json.load(f)
            tokenizer = AutoTokenizer.from_pretrained(meta.get("tokenizer", base))
            model = torch.load(pt_name, map_location=device, weights_only=False)
            model.eval()
            model.to(device)
            return "transformer", tokenizer, model

    # 2) Fallback: frozen mBERT embeddings + GRU head
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    model.eval()
    model.to(device)
    return "preembed", tokenizer, model


def extract_embeddings(text, tokenizer, model, device, max_len=MAX_SEQ_LEN):
    """Convert a single text string to mBERT token-level embeddings."""
    encoded = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    # Shape: (1, seq_len, 768)
    return outputs.last_hidden_state


# ──────────────────────────────────────────────────────────────────────────────
# 4. Prediction
# ──────────────────────────────────────────────────────────────────────────────
def predict(text, kind, tokenizer, model, classifier, device):
    """Return (label: str, confidence: float)."""
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "UNKNOWN", 0.0

    with torch.no_grad():
        if kind == "transformer":
            enc = tokenizer(
                cleaned, max_length=MAX_SEQ_LEN, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
        else:
            emb = extract_embeddings(cleaned, tokenizer, model, device)
            classifier.eval()
            logits = classifier(emb)          # (1, 2)

        probs = torch.softmax(logits, dim=1).squeeze(0)  # (2,)
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    label = "REAL" if pred_idx == 1 else "FAKE"
    return label, confidence


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sinhala Fake News Detection — Predict REAL or FAKE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py \"අද කොළඹදී විශාල රැස්වීමක්...\"\n"
            "  echo \"පුවත් අන්තර්ගතය...\" | python predict.py\n"
            "  python predict.py --json file.txt\n"
        ),
    )
    parser.add_argument("text", nargs="?", help="News article text to classify")
    parser.add_argument("--file", "-f", help="Read text from a file")
    parser.add_argument("--json", "-j", help="Read JSON file with 'content' key(s)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show cleaning steps & raw probabilities")

    args = parser.parse_args()

    # ── Read input text ────────────────────────────────────────────────────
    input_text = None

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            input_text = f.read().strip()
    elif args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                content = item.get("content", item.get("text", ""))
                if not content:
                    continue
                result = run_single(content, args.verbose)
                print(f"[{i+1}] {result}")
            return
        elif isinstance(data, dict):
            input_text = data.get("content", data.get("text", ""))
    elif args.text:
        input_text = args.text
    else:
        # Read from stdin (pipe)
        if not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(1)

    if not input_text:
        print("Error: No input text provided.", file=sys.stderr)
        sys.exit(1)

    result = run_single(input_text, args.verbose)
    print(result)


def run_single(text: str, verbose: bool = False) -> str:
    """Predict a single text and return a formatted result string."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"[Device] {device}", file=sys.stderr)
        print(f"[Original] {text[:120]}...", file=sys.stderr)
        cleaned = clean_text(text)
        print(f"[Cleaned]  {cleaned[:120]}...", file=sys.stderr)

    # ── Load best model ────────────────────────────────────────────────────
    if verbose:
        print("[Loading best model...]", file=sys.stderr)
    kind, tokenizer, model = load_best_model(device)

    classifier = None
    if kind == "preembed":
        classifier = GRUClassifierPreEmbed(embed_dim=768)
        state = torch.load("mbert_gru_best.pt", map_location=device, weights_only=True)
        classifier.load_state_dict(state)
        classifier.to(device)
        if verbose:
            print("[Loaded: frozen mBERT + GRU head]", file=sys.stderr)
    else:
        if verbose:
            print("[Loaded: fine-tuned transformer]", file=sys.stderr)

    # ── Predict ────────────────────────────────────────────────────────────
    label, confidence = predict(text, kind, tokenizer, model, classifier, device)

    if verbose:
        print(file=sys.stderr)

    pct = confidence * 100
    if label == "REAL":
        return f"✅ REAL  ({pct:.1f}% confidence)"
    else:
        return f"❌ FAKE  ({pct:.1f}% confidence)"


if __name__ == "__main__":
    main()
