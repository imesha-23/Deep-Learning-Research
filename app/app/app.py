"""
Sinhala Fake News Detection — Flask web app (plan.md Phase 6).

Features:
  - Text input → REAL / FAKE prediction with confidence score
  - If FAKE → applicable Online Safety Act No. 9 of 2024 penalties (sections 12/14/17/19)
  - PDF penalty-notice report download

Run:
    pip install -r requirements.txt
    python3.12 app.py
    # → http://127.0.0.1:5000
"""

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_file

import model
import penalties

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


# ── Model comparison table (from allmode_comparison_results.csv) ──────────────
def load_comparison():
    rows = []
    path = BASE_DIR / "allmode_comparison_results.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


# ── PDF penalty notice (fpdf2) ────────────────────────────────────────────────
def _sinhala_font_path():
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansSinhala-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansSinhala-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSinhala.ttf",
        "/System/Library/Fonts/Supplemental/NotoSansSinhala-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def build_report_pdf(result):
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 is not installed. Run: pip install fpdf2")

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, "Sinhala Fake News Detection - Penalty Notice", ln=1, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "Automated legal notification tool - Online Safety Act, No. 9 of 2024 (Sri Lanka)",
              ln=1, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"Analysis ID : {result['article_id']}", ln=1)
    pdf.cell(0, 6, f"Timestamp   : {result['timestamp']}", ln=1)
    pdf.ln(3)

    # Article excerpt (Sinhala font when available)
    sinhala_font = _sinhala_font_path()
    if sinhala_font:
        pdf.add_font("Sinhala", "", sinhala_font)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Article text (excerpt)", ln=1)
    pdf.set_font("Sinhala" if sinhala_font else "Helvetica", "", 10)
    excerpt = result["cleaned_text"][:500]
    excerpt += "..." if len(result["cleaned_text"]) > 500 else ""
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, excerpt)
    pdf.ln(3)

    # Verdict
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Prediction", ln=1)
    pdf.set_font("Helvetica", "B", 13)
    verdict = f"{result['label']}  (confidence: {result['confidence'] * 100:.1f}%)"
    pdf.cell(0, 8, verdict, ln=1)
    pdf.ln(3)

    if result["label"] == "FAKE":
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Applicable Online Safety Act sections (ranked)", ln=1)
        pdf.ln(1)
        for p in result["penalties"]:
            heading = f"Section {p['section']}"
            if p.get("primary"):
                heading += "  (PRIMARY)"
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, heading, ln=1)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"Offence         : {p['title']}")
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"Maximum penalty : {p['penalty']}")
            if p.get("matched_keywords"):
                kw_text = "Matched keywords: " + ", ".join(p["matched_keywords"])
                pdf.set_font("Sinhala" if sinhala_font else "Helvetica", "", 8)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 4, kw_text)
            pdf.ln(1)
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, "No penalty sections apply - the article is classified as REAL.")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 4,
                   "Disclaimer: This notice is generated automatically for research/demonstration "
                   "purposes only and does not constitute legal advice.")

    return bytes(pdf.output())


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    meta = model.meta()
    return render_template(
        "index.html",
        meta=meta,
        comparison=load_comparison(),
        max_penalty=penalties.PENALTIES,
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if model.is_ready() else "loading",
        "model": "XLM-R (frozen embeddings) + GRU",
        "metrics": model.meta(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        result = model.predict(text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    result["article_id"] = str(uuid4())[:8].upper()
    result["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result["penalties"] = penalties.score_penalties(result["cleaned_text"]) \
        if result["label"] == "FAKE" else []
    return jsonify(result)


@app.route("/report", methods=["POST"])
def report():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        result = model.predict(text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    result["article_id"] = data.get("article_id") or str(uuid4())[:8].upper()
    result["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result["penalties"] = penalties.score_penalties(result["cleaned_text"]) \
        if result["label"] == "FAKE" else []

    try:
        pdf_bytes = build_report_pdf(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"penalty_notice_{result['article_id']}.pdf",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
