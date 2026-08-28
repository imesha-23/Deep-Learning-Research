#!/usr/bin/env python3
"""
cli2.py — FastText-based Sinhala Fake News Predictor (no transformers)

No mBERT, no XLM-R, no GPU needed. ~200MB for FastText model.

Usage:
    First run (trains FastText + classifier from CSV data):
        python cli2.py "අද කොළඹදී විශාල රැස්වීමක්..."

    Subsequent runs (loads cached artifacts):
        python cli2.py "පුවත් අන්තර්ගතය..."
        echo "text..." | python cli2.py
        python cli2.py -f article.txt
"""

import argparse, sys, os, re, json, pickle, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path(__file__).parent / ".cli2_cache"
ARTIFACT_DIR.mkdir(exist_ok=True)

FT_MODEL_PATH   = ARTIFACT_DIR / "fasttext.model"
VOCAB_PATH      = ARTIFACT_DIR / "vocab.pkl"
VECTORIZER_PATH = ARTIFACT_DIR / "vectorizer.pkl"
CLASSIFIER_PATH = ARTIFACT_DIR / "classifier.pkl"
META_PATH       = ARTIFACT_DIR / "meta.json"

EMBED_DIM = 100
MAX_SEQ_LEN = 150
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Text Cleaning
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
# 2. Sinhala Tokenization + Stopword Removal
# ──────────────────────────────────────────────────────────────────────────────
SINHALA_STOPWORDS = {
    'සහ', 'හා', 'හෝ', 'නමුත්', 'එහෙත්', 'නිසා', 'බැවින්', 'නම්',
    'මෙම', 'ඒ', 'එම', 'එය', 'මෙය', 'ඔහු', 'ඇය', 'ඔවුන්', 'ඔවුහු',
    'ඔහුගේ', 'ඇගේ', 'ඔවුන්ගේ', 'මෙහි', 'එහි', 'මෙතැන', 'එතැන',
    'ද', 'ම', 'ත්', 'වෙත', 'විසින්', 'සඳහා', 'අතර', 'පිළිබඳ', 'ලෙස',
    'ලෙසම', 'ලෙසින්', 'කෙසේ', 'යනු', 'යන', 'බව', 'යයි', 'කිව',
    'වේ', 'වෙයි', 'ඇත', 'නැත', 'ඇති', 'නැති', 'විය', 'වී', 'කර', 'කළ',
    'කරයි', 'කරනු', 'කරන', 'ලබා', 'ලැබී', 'ලැබෙන', 'දී', 'ගෙන', 'ගොස්',
    'තුළ', 'සිට', 'දක්වා', 'පිට', 'අනුව', 'යටතේ', 'හරහා',
    'එකක්', 'දෙකක්', 'කිහිපයක්', 'සියලු', 'සෑම', 'ඇතැම්', 'බොහෝ',
    'කවුද', 'කුමක්ද', 'කොතෙකද', 'කෙසේද', 'ඇයි', 'කවදා',
    'මෙලෙස', 'ඒ්', 'වන', 'වූ', 'යැයි', 'ඉදිරිපත්', 'ප්‍රකාශ',
    'ප්', 'ක්', 'රී', 'ශ්', 'මේ',
}

def tokenize(text):
    tokens = text.split()
    return [t for t in tokens
            if re.fullmatch(r'[\u0D80-\u0DFF\u200C\u200D]+', t) and len(t) >= 2]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in SINHALA_STOPWORDS]

def tokenize_and_filter(text):
    return remove_stopwords(tokenize(text))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Document Vectorizer (average FastText word vectors)
# ──────────────────────────────────────────────────────────────────────────────
def doc_vector(tokens, ft_model, embed_dim=EMBED_DIM):
    """Average word vectors for a document. Returns zero vector if no known tokens."""
    vectors = []
    for t in tokens:
        if t in ft_model.wv:
            vectors.append(ft_model.wv[t])
    if not vectors:
        return np.zeros(embed_dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def train(data_csv="TRUE.csv", data_csv_false="FALSE.csv"):
    """Train FastText + classifier from CSV files. Returns (accuracy on test set)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from gensim.models import FastText

    print("[cli2] Loading data...", file=sys.stderr)

    # ── Load (CSV columns: id,domain,source_type,datestamp,content) ────────
    data_dir = Path(__file__).parent
    true_df  = pd.read_csv(data_dir / data_csv,  usecols=["domain", "content"])
    false_df = pd.read_csv(data_dir / data_csv_false, usecols=["domain", "content"])
    true_df.rename(columns={"domain": "source"}, inplace=True)
    false_df.rename(columns={"domain": "source"}, inplace=True)
    true_df["label"]  = 1
    false_df["label"] = 0
    df = pd.concat([true_df, false_df], ignore_index=True)
    df.dropna(subset=["content"], inplace=True)
    df["content"] = df["content"].astype(str)

    # ── Clean + Tokenize ──────────────────────────────────────────────────
    print("[cli2] Cleaning & tokenizing...", file=sys.stderr)
    df["cleaned"] = df["content"].apply(clean_text)
    df["tokens"]  = df["cleaned"].apply(tokenize_and_filter)

    all_corpus = df["tokens"].tolist()
    texts      = df["tokens"].tolist()
    labels     = df["label"].values

    # ── Train FastText ─────────────────────────────────────────────────────
    print("[cli2] Training FastText (this may take a minute)...", file=sys.stderr)
    ft_model = FastText(
        sentences=all_corpus,
        vector_size=EMBED_DIM,
        window=5,
        min_count=1,
        workers=4,
        epochs=30,
        sg=1,
        seed=42,
    )
    ft_model.save(str(FT_MODEL_PATH))
    print(f"[cli2] FastText saved  ({len(ft_model.wv):,} words)", file=sys.stderr)

    # ── Vectorize ──────────────────────────────────────────────────────────
    print("[cli2] Vectorizing documents...", file=sys.stderr)
    X = np.array([doc_vector(t, ft_model) for t in texts], dtype=np.float32)
    y = labels

    # ── Train Classifier ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42,
    )

    print("[cli2] Training Logistic Regression classifier...", file=sys.stderr)
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")

    print(f"[cli2] Test accuracy  : {acc*100:.2f}%", file=sys.stderr)
    print(f"[cli2] Macro F1       : {f1_m:.4f}", file=sys.stderr)
    print(f"[cli2] Weighted F1    : {f1_w:.4f}", file=sys.stderr)
    print(file=sys.stderr)

    # ── Save artifacts ─────────────────────────────────────────────────────
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(doc_vector, f)

    meta = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_m), 4),
        "weighted_f1": round(float(f1_w), 4),
        "embed_dim": EMBED_DIM,
        "num_samples": len(df),
        "architecture": "FastText + LogisticRegression",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[cli2] Artifacts saved to {ARTIFACT_DIR}/", file=sys.stderr)
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# 5. Prediction
# ──────────────────────────────────────────────────────────────────────────────
def load_pipeline():
    """Load FastText model + classifier. Returns (ft_model, classifier)."""
    from gensim.models import FastText as GensimFastText
    ft_model = GensimFastText.load(str(FT_MODEL_PATH))
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    return ft_model, clf


def predict(text, ft_model, clf):
    """Return (label_str, confidence)."""
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "UNKNOWN", 0.0
    tokens = tokenize_and_filter(cleaned)
    vec = doc_vector(tokens, ft_model).reshape(1, -1)
    prob = clf.predict_proba(vec)[0]         # [prob_false, prob_true]
    pred = clf.predict(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    confidence = float(prob[pred])
    return label, confidence


# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="cli2 — FastText Sinhala Fake News Predictor (no GPU/transformers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli2.py \"අද කොළඹදී විශාල රැස්වීමක්...\"\n"
            "  echo \"පුවත් අන්තර්ගතය...\" | python cli2.py\n"
            "  python cli2.py -f article.txt\n"
            "  python cli2.py --train   (force retrain)\n"
        ),
    )
    parser.add_argument("text", nargs="?", help="News article text")
    parser.add_argument("--file", "-f", help="Read from file")
    parser.add_argument("--json", "-j", help="Read JSON file with 'content' key(s)")
    parser.add_argument("--train", action="store_true", help="Force retrain model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show debug info")

    args = parser.parse_args()

    # ── (Re)train if requested or artifacts missing ────────────────────────
    artifacts_exist = all(p.exists() for p in [
        FT_MODEL_PATH, CLASSIFIER_PATH, VECTORIZER_PATH, META_PATH,
    ])

    if args.train or not artifacts_exist:
        if not artifacts_exist:
            print("[cli2] First run — training FastText model...", file=sys.stderr)
        train()
        print(file=sys.stderr)

    # ── Show model info ────────────────────────────────────────────────────
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        if args.verbose:
            print(f"[Model] FastText (100d) + LogisticRegression", file=sys.stderr)
            print(f"[Test]  Acc: {meta['accuracy']*100:.2f}%  Macro F1: {meta['macro_f1']:.4f}", file=sys.stderr)

    # ── Read input ─────────────────────────────────────────────────────────
    input_text = None
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            input_text = f.read().strip()
    elif args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            ft_model, clf = load_pipeline()
            for i, item in enumerate(data):
                content = item.get("content", item.get("text", ""))
                if not content:
                    continue
                label, conf = predict(content, ft_model, clf)
                print(f"[{i+1}] {'✅ REAL' if label=='REAL' else '❌ FAKE'}  ({conf*100:.1f}%)")
            return
        elif isinstance(data, dict):
            input_text = data.get("content", data.get("text", ""))
    elif args.text:
        input_text = args.text
    else:
        if not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(1)

    if not input_text:
        print("Error: No input text.", file=sys.stderr)
        sys.exit(1)

    # ── Predict ────────────────────────────────────────────────────────────
    ft_model, clf = load_pipeline()
    label, confidence = predict(input_text, ft_model, clf)

    pct = confidence * 100
    if label == "REAL":
        print(f"✅ REAL  ({pct:.1f}%)")
    else:
        print(f"❌ FAKE  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
