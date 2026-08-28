"""
Online Safety Act, No. 9 of 2024 (Sri Lanka) — penalty sections displayed
when an article is classified as FAKE. Values match plan.md Phase 6.

Each section carries a heuristic Sinhala keyword list used to rank which
section is most likely applicable to a given article (see score_penalties).
"""

PENALTIES = [
    {
        "section": "12",
        "title": "False statement threatening national security, public health, or public order",
        "penalty": "5 years imprisonment + Rs. 500,000 fine",
        "keywords": [
            "ජාතික ආරක්ෂාව", "ජාතික ආරක්ෂක", "මහජන සෞඛ්‍ය", "පොදු පිළිවෙළ",
            "පොදු ආරක්ෂාව", "කැරැල්ල", "කුමන්ත්‍රණ", "යුද්ධ", "වසංගත",
            "බෝවන රෝග", "ත්‍රස්තවාදය", "ආරක්ෂක තර්ජන",
        ],
    },
    {
        "section": "14",
        "title": "Wantonly provoking a riot by false statement",
        "penalty": "5 years imprisonment + Rs. 500,000 fine (if rioting occurs)",
        "keywords": [
            "කෝලාහල", "ප්‍රචණ්ඩත්වය", "රැලිය", "ප්‍රකෝප", "ගැටුම්", "කැරලි",
            "පහරදීම්", "විරෝධතා", "අවුල් සහගත",
        ],
    },
    {
        "section": "17",
        "title": "Online cheating via false statement",
        "penalty": "7 years imprisonment + Rs. 700,000 fine",
        "keywords": [
            "වංචා", "මුදල් වංචා", "සයිබර්", "අන්තර්ජාල", "ව්‍යාජ", "බැංකු",
            "ගිණුම්", "රැවටීම", "මුදල් අයකර", "ඔන්ලයින්", "වංචනික",
        ],
    },
    {
        "section": "19",
        "title": "False statement to induce an offence against the State",
        "penalty": "7 years imprisonment + Rs. 700,000 fine",
        "keywords": [
            "රාජ්‍යයට එරෙහි", "රාජ්‍ය විරෝධී", "බලහත්කාර", "අපරාධයක්",
            "කුමන්ත්‍රණය", "පාලනය පෙරලා", "ව්‍යවස්ථාව උල්ලංඝන", "රාජ්‍ය බලය",
        ],
    },
]


def score_penalties(text):
    """Rank sections by how many of their keywords appear in the (cleaned) text.

    Returns the penalty dicts sorted by score (descending, stable — ties keep
    the original 12/14/17/19 order), each augmented with:
      'score'            (int)          — number of distinct matched keywords
      'matched_keywords' (list[str])    — which keywords matched
      'primary'          (bool)         — True only for the single top section
                                          when its score > 0
    """
    scored = []
    for p in PENALTIES:
        matched = [kw for kw in p["keywords"] if kw in text]
        scored.append({**p, "score": len(matched),
                       "matched_keywords": matched, "primary": False})
    scored.sort(key=lambda d: d["score"], reverse=True)
    if scored and scored[0]["score"] > 0:
        scored[0]["primary"] = True
    return scored
