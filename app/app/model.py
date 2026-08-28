"""
XLM-R (frozen embeddings) + GRU head — prediction pipeline.

Loads the artifacts copied into this folder:
  - xlm-r_gru_best.pt        (GRU classifier-head state_dict)
  - xlm-r_gru_best_meta.json (test-set metrics)

Requires: torch, transformers (see requirements.txt). The XLM-R backbone is
downloaded from HuggingFace on first use (~1.1 GB) unless already cached.

The text cleaning and GRU architecture mirror allmode.ipynb exactly.
"""

import json
import re
import threading
from pathlib import Path

import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xlm-r_gru_best.pt"
META_PATH = BASE_DIR / "xlm-r_gru_best_meta.json"
XLMR_NAME = "xlm-roberta-base"
LOCAL_MODEL_DIR = BASE_DIR / "xlmr_model"   # offline copy of the XLM-R backbone
MAX_SEQ_LEN = 150


def _model_source():
    """Use the local xlmr_model/ folder once fully downloaded, else the Hub.

    The downloader writes model.safetensors.ok when model.safetensors is
    complete, so a partial file is never loaded.
    """
    marker = LOCAL_MODEL_DIR / "model.safetensors.ok"
    if (LOCAL_MODEL_DIR / "config.json").exists():
        if marker.exists():
            return str(LOCAL_MODEL_DIR)
        if (LOCAL_MODEL_DIR / "model.safetensors").exists():
            raise RuntimeError(
                "XLM-R model is still downloading into app/xlmr_model/ "
                "(model.safetensors is incomplete). Wait for it to finish "
                "(~1.12 GB) before analyzing.")
    return XLMR_NAME


# ── Text cleaning (same as allmode.ipynb / predict.py) ────────────────────────
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


# ── GRU head (same architecture as allmode.ipynb) ─────────────────────────────
class GRUClassifierPreEmbed(nn.Module):
    """GRU accepting pre-computed token-level embeddings (batch, seq, embed)."""

    def __init__(self, embed_dim=768, hidden_dim=128, num_layers=2,
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
        return self.fc(self.dropout(last))


# ── Singleton model handle (loaded lazily on first prediction) ────────────────
_lock = threading.Lock()
_handle = None  # (tokenizer, xlmr, classifier, device)


def is_ready():
    return _handle is not None


def _load():
    """Load (tokenizer, XLM-R model, GRU head, device) once, thread-safely."""
    global _handle
    if _handle is not None:
        return _handle
    with _lock:
        if _handle is None:
            try:
                from transformers import AutoTokenizer, AutoModel
            except ImportError:
                raise RuntimeError(
                    "transformers is not installed. Run: pip install -r requirements.txt")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            source = _model_source()
            tokenizer = AutoTokenizer.from_pretrained(source)
            xlmr = AutoModel.from_pretrained(source)
            xlmr.eval().to(device)

            classifier = GRUClassifierPreEmbed(embed_dim=768)
            state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            classifier.load_state_dict(state)
            classifier.eval().to(device)

            _handle = (tokenizer, xlmr, classifier, device)
    return _handle


def meta():
    """Test-set metrics from the meta JSON (empty dict if unavailable)."""
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def predict(text):
    """Classify a Sinhala article. Returns a dict with label and probabilities.

    Raises ValueError for empty input; RuntimeError if the model fails to load.
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        raise ValueError("Text is empty after cleaning")

    tokenizer, xlmr, classifier, device = _load()

    encoded = tokenizer(
        cleaned,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        emb = xlmr(**encoded).last_hidden_state   # (1, seq, 768)
        logits = classifier(emb)                  # (1, 2)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred = int(probs.argmax())
    return {
        "label": "REAL" if pred == 1 else "FAKE",
        "confidence": round(float(probs[pred]), 6),
        "probability_fake": round(float(probs[0]), 6),
        "probability_real": round(float(probs[1]), 6),
        "cleaned_text": cleaned,
    }
