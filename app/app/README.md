# Sinhala Fake News Detection — Web App (plan.md Phase 6)

Flask web interface that classifies Sinhala news articles as **REAL** or **FAKE**
and, when FAKE, displays the applicable **Online Safety Act, No. 9 of 2024**
penalties (sections 12, 14, 17, 19) with a downloadable PDF penalty notice.

## Model

`XLM-R (frozen embeddings) + GRU` — the copied artifacts in this folder:

| File | Purpose |
|---|---|
| `xlm-r_gru_best.pt` | GRU classifier-head weights (embed 768 → GRU 2×128 → 2) |
| `xlm-r_gru_best_meta.json` | Test metrics (acc 85.37%, Macro F1 0.8496) |
| `allmode_comparison_results.csv` | Comparison of all 12 embedding×architecture combinations |

The XLM-R backbone (`xlm-roberta-base`) is downloaded from HuggingFace on first
prediction (~1.1 GB, cached afterwards).

## Setup & Run

```bash
cd app
pip install -r requirements.txt
python3.12 app.py
```

Then open <http://127.0.0.1:5000>.

## Features

- Paste Sinhala text → REAL/FAKE prediction with confidence + class probabilities
- FAKE → table of applicable Online Safety Act sections and maximum penalties
- "Download PDF Report" → penalty notice (`reportlab`/`fpdf2`-style PDF via fpdf2)
- Health endpoint: `GET /health`
- Model comparison table (from `allmode_comparison_results.csv`)

## API

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/` | GET | — | web page |
| `/health` | GET | — | model status + metrics |
| `/predict` | POST | `{"text": "…"}` | `{label, confidence, probability_fake, probability_real, penalties, article_id, timestamp}` |
| `/report` | POST | `{"text": "…", "article_id": "…"}` | PDF file download |
