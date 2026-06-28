# Deep Learning Based Detection of Fake News on Sri Lanka in the Sinhala Language with Automated Legal Penalty Notification
Deep learning based detection of fake news in Sinhala-language content related to Sri Lanka, paired with an automated tool that informs publishers of false content about the legal consequences stipulated under the **Online Safety Act, No. 9 of 2024**.

**Final Year Individual Project — COM4901**
Faculty of Computer Science & Engineering, KIU Campus Pvt (Ltd)
Student: H.M.I.L. Jayathilake (11350) · BSc (Hons) Software Engineering, 8th Batch


---

## Project Status

🟡 **In Progress — Interim Stage **

| Component | Status |
|---|---|
| Dataset collection | In progress (1046 / 1,500 target samples) |
| Data cleaning & pre-processing | ✅ Completed |
| Exploratory Data Analysis (EDA) | ✅ Completed |
| FastText embedding training | ✅ Completed |
| TextCNN + FastText model | ✅ Completed |
| LSTM / BiLSTM / GRU models | 🔄 Implemented, training in progress |
| mBERT / XLM-R embeddings | 🔄 Implemented, extraction/training in progress |
| Comparative model evaluation (12 combinations) | 🔄 In progress |
| Automated Legal Penalty Notification Tool | ⬜ Not yet started |
| Prototype system (UI/UX) | ⬜ Not yet started |

See the [Interim Report](#) for full details on progress, challenges, and the revised work plan.

---

## Overview

Sri Lanka has seen a sharp rise in Sinhala-language misinformation across Facebook, Twitter/X, WhatsApp, and Telegram — particularly around elections, the economic crisis, and public health events. Existing fake news detection research and tooling overwhelmingly target English and other high-resource languages, leaving Sinhala largely unaddressed, partly due to the lack of any standard public dataset.

This project addresses that gap in two parts:

1. **Detection** — a manually annotated Sinhala dataset and a set of deep learning classifiers (CNN, LSTM, BiLSTM, GRU) trained over both static (FastText) and contextual multilingual transformer (mBERT, XLM-R) embeddings, to classify Sinhala news/social-media text as **Real** or **Fake**.
2. **Legal accountability** — an automated notification tool that, when content is classified as fake, surfaces the relevant penalty provisions under Sections 12, 14, 17, and 19 of the Online Safety Act, No. 9 of 2024. This tool is informational only and does not constitute legal advice or a formal legal proceeding.

**Scope note:** this project is text-only. Image- and video-based fake news detection is explicitly out of scope.

---

## Repository Structure

```
.
├── data/
│   ├── TRUE.csv                  # Real news samples (source, content, label=1)
│   └── FALSE.csv                 # Fake news samples (source, content, label=0)
├── notebooks/
│   ├── analys.ipynb              # EDA notebook (see below)
│   └── model_training.ipynb      # Model training & comparison notebook (see below)
├── models/                       # Saved model weights (.pt) + metadata (.json)
├── docs/
│   ├── Final_Year_Research_Proposal.pdf
│   └── Interim_Progress_Report.docx
└── README.md
```

> Folder names above reflect the intended layout. Adjust to match your actual repo if it differs.

---

## Dataset

| Field | Description |
|---|---|
| `id` | Unique identifier |
| `title` | News headline |
| `content` | Full text content |
| `date` | Publication date |
| `source` | News source / publisher |
| `source_type` | Platform type (e.g. news site, Facebook, WhatsApp) |
| `label` | `1` = Real, `0` = Fake |

- **Language:** Sinhala
- **Classification:** Binary (Real / Fake)
- **Target size:** 1,500 samples · **Current size:** 437 samples (327 Real / 110 Fake)
- **Timeframe:** last 5 years, with a `year_range` field for temporal stratification
- **Sources (real news):** BBC Sinhala, Ada Derana, News First, Lankadeepa, Government Information Department, official press releases
- **Sources (fake news):** Facebook public groups/pages, viral WhatsApp/Telegram content (collected with consent), known fake news sites, FactCheck.lk archives

Data collection for WhatsApp/Telegram content follows a dedicated ethics protocol (no personal identifiers, consent-based collection, compliance with the Personal Data Protection Act No. 9 of 2022). See the proposal document in `docs/` for full details.

**Note on annotation:** inter-annotator agreement (Cohen's Kappa) has so far only been piloted using proxy heuristic "annotators" (digit ratio, Sinhala-script ratio, article length, source trust) to validate the IAA computation pipeline. Annotation by the trained 3–5 person human annotator team (target κ ≥ 0.61) is still pending.

---

## Notebooks

### `analys.ipynb` — Exploratory Data Analysis

Loads `TRUE.csv` / `FALSE.csv` and walks through:

1. Class distribution
2. Article length analysis
3. Source / publisher analysis (including real–fake source overlap)
4. Character-level analysis (Sinhala Unicode ratio)
5. Feature correlation heatmap
6. Sinhala stop word definition
7. Most common Sinhala words (top unigrams, per class)
8. Exclusive words: True-only vs. False-only
9. Summary statistics table
10. Cohen's Kappa — pilot inter-annotator agreement test

Run top-to-bottom; no GPU required.

### `model_training.ipynb` — Model Training & Comparison

Implements and (where training has completed) evaluates **12 combinations** of embedding method × architecture:

| Embedding | Dim | Architectures |
|---|---|---|
| FastText (trained on corpus) | 100 | TextCNN, LSTM, BiLSTM, GRU |
| mBERT (frozen, token-level) | 768 | TextCNN, LSTM, BiLSTM, GRU |
| XLM-R (frozen, token-level) | 768 | TextCNN, LSTM, BiLSTM, GRU |

Pipeline stages:

1. Text cleaning (URL/emoji/special-character removal)
2. Tokenisation + Sinhala stop word removal
3. Unicode (NFC) normalisation
4. FastText embedding training
5. Stratified 70/15/15 train/val/test split
6. Shared training loop (weighted sampling, AdamW, label smoothing, gradient clipping, ReduceLROnPlateau, early stopping)
7. Evaluation (Accuracy, Macro F1, Weighted F1, confusion matrix, classification report)
8. Aggregated comparison table, bar charts, and Macro-F1/Accuracy heatmaps across all 12 combinations
9. Best model selection (by Macro F1) and export (`.pt` weights + `.json` metadata)

GPU strongly recommended for the mBERT/XLM-R embedding extraction steps.

---

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

**`requirements.txt`** (core dependencies used across both notebooks):

```
torch
transformers
gensim
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Then place `TRUE.csv` and `FALSE.csv` in the working directory (or update the paths at the top of each notebook) and run via Jupyter Notebook or Google Colab.

---

## Models

Trained model weights are saved as `{embedding}_{architecture}_best.pt`, with an accompanying `{...}_meta.json` containing test accuracy, Macro F1, Weighted F1, best epoch, and embedding dimension. The best-performing combination overall (by Macro F1 across all 12 runs) is saved separately once full comparative training is complete.

---

## Legal Penalty Notification Tool (Planned)

Not yet implemented — planned for the next project phase. When complete, it will:

1. Accept a Sinhala-language news item via the prototype interface.
2. Classify it as REAL or FAKE using the best-performing trained model.
3. If REAL → display a "Verified" result with a confidence score.
4. If FAKE → display a "False News Detected" result with a structured penalty report referencing the applicable section(s) of the Online Safety Act, No. 9 of 2024:
   - **Section 12** — false statement threatening national security/public health/public order
   - **Section 14** — false statement provoking riot
   - **Section 17** — online cheating via false statement
   - **Section 19** — false statement inducing an offence against the State

This tool is informational only and does not constitute legal advice.

---

## Evaluation Metrics

Accuracy · Precision · Recall · F1-score (Macro & Weighted) · Confusion Matrix · ROC-AUC

---

## Tools & Technologies

Python · PyTorch · Hugging Face Transformers (mBERT, XLM-R) · Gensim (FastText) · scikit-learn · Pandas / NumPy · Matplotlib / Seaborn · BeautifulSoup / Scrapy (data collection) · Jupyter Notebook / Google Colab

---

## References

1. K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake news detection on social media: A data mining perspective," *ACM SIGKDD Explor. Newsl.*, vol. 19, no. 1, pp. 22–36, 2017.
2. X. Zhou and R. Zafarani, "A survey of fake news: Fundamental theories, detection methods, and opportunities," *ACM Comput. Surv.*, vol. 53, no. 5, 2020.
3. H. Allcott and M. Gentzkow, "Social media and fake news in the 2016 election," *J. Econ. Perspect.*, vol. 31, no. 2, pp. 211–236, 2017.
4. C. Buntain and J. Golbeck, "Automatically identifying fake news in popular Twitter threads," in *Proc. IEEE Int. Conf. Smart Cloud*, 2017, pp. 208–215.
5. FactCheck.lk — [https://www.factcheck.lk](https://www.factcheck.lk)
6. Parliament of Sri Lanka, *Online Safety Act, No. 9 of 2024*. [https://www.documents.gov.lk](https://www.documents.gov.lk)

---

## License & Ethics

Data collection involving WhatsApp/Telegram content follows a formal ethics protocol: only publicly forwarded or consented messages are collected, personal identifiers are anonymised, and collection complies with the Personal Data Protection Act No. 9 of 2022 of Sri Lanka. This repository is submitted as part of an academic Final Year Individual Project and is not intended for production deployment without further legal and ethical review.
