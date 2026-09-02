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
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, render_template_string, request, send_file

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
def _sinhala_font_candidates():
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansSinhala-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansSinhala-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSinhala.ttf",
        "/System/Library/Fonts/Supplemental/NotoSansSinhala-Regular.ttf",
    ]
    # Windows: Nirmala UI ships with Windows 8+ and includes Sinhala glyphs.
    windir = os.environ.get("WINDIR")
    if windir:
        fonts_dir = Path(windir) / "Fonts"
        candidates += [
            str(fonts_dir / "NirmalaUI.ttf"),
            str(fonts_dir / "Nirmala.ttf"),
        ]
        candidates += sorted(str(p) for p in fonts_dir.glob("NotoSansSinhala*.ttf"))
    # Per-user fonts (e.g. Noto Sans Sinhala installed from the Microsoft Store).
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        user_fonts = Path(localappdata) / "Microsoft" / "Windows" / "Fonts"
        candidates += sorted(str(p) for p in user_fonts.glob("NotoSansSinhala*.ttf"))
    return candidates


def _sinhala_font_path():
    for path in _sinhala_font_candidates():
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
    except Exception as e:
        # Any other failure (e.g. no Sinhala-capable font installed, which
        # makes fpdf2 raise UnicodeEncodeError on the Sinhala text). Return
        # JSON so the UI can show a readable message instead of an HTML page.
        return jsonify({"error": f"PDF generation failed: {e}"}), 500

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"penalty_notice_{result['article_id']}.pdf",
    )


# ── OpenAPI 3.0 spec + Swagger UI (zero-dependency; UI assets from CDN) ───────
OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Sinhala Fake News Detection API",
        "version": "1.0.0",
        "description": (
            "Flask REST API for the Sinhala fake-news detection prototype. "
            "Classifies articles as REAL/FAKE (XLM-R frozen backbone + GRU head), "
            "scores applicable Online Safety Act No. 9 of 2024 penalties "
            "(\u00a712/14/17/19) when the label is FAKE, and generates a PDF "
            "penalty notice."
        ),
    },
    "paths": {
        "/": {
            "get": {
                "summary": "Web UI",
                "description": "Serves the browser front end (index.html).",
                "responses": {
                    "200": {
                        "description": "HTML page",
                        "content": {"text/html": {"schema": {"type": "string"}}},
                    }
                },
            }
        },
        "/health": {
            "get": {
                "summary": "Service and model status",
                "responses": {
                    "200": {
                        "description": "OK — model ready or still loading",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthResponse"}
                            }
                        },
                    }
                },
            }
        },
        "/predict": {
            "post": {
                "summary": "Classify a Sinhala article",
                "description": (
                    "Returns REAL/FAKE with confidence scores. The penalties "
                    "array is populated only when the label is FAKE."
                ),
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PredictionRequest"}
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Prediction result",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PredictionResponse"}
                            }
                        },
                    },
                    "400": {
                        "description": "Missing/empty text, or text empty after cleaning",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                    "500": {
                        "description": "Model failed to load",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                },
            }
        },
        "/report": {
            "post": {
                "summary": "Generate PDF penalty notice",
                "description": (
                    "Runs the same prediction as /predict but returns a "
                    "downloadable PDF penalty notice (fpdf2)."
                ),
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PredictionRequest"}
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "PDF penalty notice",
                        "content": {
                            "application/pdf": {
                                "schema": {"type": "string", "format": "binary"}
                            }
                        },
                    },
                    "400": {
                        "description": "Missing/empty text, or text empty after cleaning",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                    "500": {
                        "description": "Model failed to load or fpdf2 not installed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                },
            }
        },
    },
    "components": {
        "schemas": {
            "ErrorResponse": {
                "type": "object",
                "properties": {"error": {"type": "string"}},
                "required": ["error"],
            },
            "PredictionRequest": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Sinhala article text",
                        "example": "\u0dc1\u0dca\u200d\u0dbb\u0dd3 \u0dbd\u0d82\u0d9a\u0dcf\u0dc0\u0dda \u0dc3\u0dd2\u0dba\u0dbd\u0dd4\u0db8 \u0dbb\u0ddd\u0dc4\u0dbd\u0dca\u0dc0\u0dbd \u0d96\u0dc2\u0db0 \u0dad\u0ddc\u0d9c \u0db4\u0dca\u200d\u0dbb\u0db8\u0dcf\u0dab\u0dc0\u0dad\u0dca \u0db6\u0dc0 \u0dc3\u0dd6\u0d9b\u0dca\u200d\u0dba \u0d85\u0db8\u0dcf\u0dad\u0dca\u200d\u0dba\u0dcf\u0d82\u0dc1\u0dba \u0d85\u0daf \u0db1\u0dd2\u0dc0\u0dda\u0daf\u0db1\u0dba \u0d9a\u0dc5\u0dda\u0dba.",
                    },
                },
                "required": ["text"],
            },
            "Penalty": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "example": "12"},
                    "title": {"type": "string"},
                    "penalty": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "score": {"type": "integer"},
                    "matched_keywords": {"type": "array", "items": {"type": "string"}},
                    "coverage": {
                        "type": "number",
                        "format": "float",
                        "description": "Fraction of the section's keyword list that matched (0-1)",
                    },
                    "primary": {"type": "boolean"},
                },
            },
            "PredictionResponse": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["REAL", "FAKE"]},
                    "confidence": {"type": "number", "format": "float"},
                    "probability_fake": {"type": "number", "format": "float"},
                    "probability_real": {"type": "number", "format": "float"},
                    "cleaned_text": {"type": "string"},
                    "article_id": {"type": "string", "example": "1A2B3C4D"},
                    "timestamp": {"type": "string", "example": "2026-08-27 10:00:00 UTC"},
                    "penalties": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Penalty"},
                    },
                },
            },
            "HealthResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok", "loading"]},
                    "model": {"type": "string"},
                    "metrics": {"type": "object"},
                },
            },
        }
    },
}


@app.route("/openapi.json")
def openapi_spec():
    return jsonify(OPENAPI_SPEC)


SWAGGER_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Swagger UI — Sinhala Fake News Detection API</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"/>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function () {
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
        docExpansion: "list",
      });
    };
  </script>
</body>
</html>"""


@app.route("/docs")
def docs():
    return render_template_string(SWAGGER_UI_HTML)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
