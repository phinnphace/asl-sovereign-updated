import requests
import json
import pandas as pd

_STUDIO_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemma-2-9b-it:generateContent"
)

_PROMPT = """You are an expert AI diagnostic routing agent for American Sign Language (ASL) recognition models.
Read the user complaint or error description and determine what training data regime the failing model was trained on.

TRAINING REGIMES:
1. LANDMARK-ONLY — 21-hand-landmark keypoints only. Fails on body-anchored signs (chest/shoulder/face touch), two-hand crossing signs. Signature confusions: K→V, D→I, G→D total collapse.
2. FULL LANDMARKS — hands + face + body pose. Better spatial coverage, limited on continuous flow.
3. HYBRID — RGB frames + pose. Errors are contextual, not geometric.

OUTPUT: Return ONLY valid JSON. No markdown. No preamble.

JSON STRUCTURE:
{
    "failure_mode": "one-line classification",
    "training_regime": "LANDMARK-ONLY | FULL LANDMARKS | HYBRID",
    "deployment_risk": "HIGH | MODERATE | LOW — one sentence",
    "firing_pairs": ["K→V", "D→I"],
    "provenance_diagnosis": "paragraph explaining training data fingerprint",
    "recommendation": "specific actionable advice"
}

EXAMPLE:
Input: "K always comes back as V — never gets it right"
Output: {"failure_mode": "Geometric Collapse — handshape confusion", "training_regime": "LANDMARK-ONLY", "deployment_risk": "HIGH — systematic failure will not resolve with fine-tuning", "firing_pairs": ["K→V"], "provenance_diagnosis": "K→V total collapse is a diagnostic fingerprint of landmark-only training. The model cannot distinguish these geometrically similar handshapes because it was trained exclusively on 21-point hand keypoints without visual texture.", "recommendation": "DO NOT fine-tune. Switch to a full-body or hybrid model. Fine-tuning on more K/V examples will not fix a geometric prior in the training data."}

Input: "It sees the hand fine but misses signs where I touch my shoulder"
Output: {"failure_mode": "Body-Anchor Occlusion — spatial sign failure", "training_regime": "LANDMARK-ONLY", "deployment_risk": "HIGH — structural gap, not a fine-tuning problem", "firing_pairs": [], "provenance_diagnosis": "Body-anchored signs require full-body topology. A landmark-only model has no body coordinate system and fails systematically on all body-referenced signs.", "recommendation": "Route to a full-body pose estimation model or hybrid architecture. Fine-tuning is wasted compute."}
"""


def analyze_failure_mode(user_input: str, api_key: str) -> dict:
    payload = {
        "contents": [{
            "parts": [{"text": f"{_PROMPT}\n\nINPUT:\n{user_input}\n\nOUTPUT:"}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 512,
        },
    }

    try:
        resp = requests.post(
            f"{_STUDIO_URL}?key={api_key}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "failure_mode": "Parse Error",
                "training_regime": "Unknown",
                "deployment_risk": "Unknown",
                "firing_pairs": [],
                "provenance_diagnosis": "Model returned unstructured text.",
                "recommendation": "Check API response format.",
                "raw_output": text,
            }
    except requests.exceptions.RequestException as e:
        return {
            "failure_mode": "Connection Error",
            "training_regime": "Unknown",
            "deployment_risk": "Unknown",
            "firing_pairs": [],
            "provenance_diagnosis": f"Could not reach Google AI Studio: {e}",
            "recommendation": "Verify GOOGLE_API_KEY in .streamlit/secrets.toml.",
        }


def decode_csv(df: pd.DataFrame) -> dict:
    cols = [c.lower().strip() for c in df.columns]

    # Inference log: actual_letter + model_output
    if "actual_letter" in cols and "model_output" in cols:
        df = df.copy()
        df.columns = cols
        total = len(df)
        correct = (df["actual_letter"].str.upper() == df["model_output"].str.upper()).sum()
        overall = round(correct / total * 100, 1) if total else 0.0

        per_letter = {}
        top_errors = {}
        for letter, grp in df.groupby(df["actual_letter"].str.upper()):
            n = len(grp)
            hits = (grp["model_output"].str.upper() == letter).sum()
            per_letter[letter] = {"accuracy": hits / n if n else 0.0}
            wrong = grp[grp["model_output"].str.upper() != letter]["model_output"].str.upper()
            if not wrong.empty:
                top_errors[letter] = wrong.value_counts().index.tolist()[:3]

        return {
            "type": "inference_log",
            "overall_accuracy": overall,
            "total_images": total,
            "per_letter": per_letter,
            "top_errors": top_errors,
        }

    # Per-letter accuracy: letter + accuracy
    if "letter" in cols and "accuracy" in cols:
        df = df.copy()
        df.columns = cols
        per_letter = {
            str(row["letter"]).upper(): {"accuracy": float(row["accuracy"])}
            for _, row in df.iterrows()
        }
        overall = (
            round(sum(v["accuracy"] for v in per_letter.values()) / len(per_letter) * 100, 1)
            if per_letter else 0.0
        )
        return {
            "type": "inference_log",
            "overall_accuracy": overall,
            "total_images": len(df),
            "per_letter": per_letter,
            "top_errors": {},
        }

    # Confusion matrix: square NxN with letter labels
    if len(df.columns) > 1 and len(df) == len(df.columns):
        labels = [str(c).upper() for c in df.columns]
        return {
            "type": "confusion_matrix",
            "shape": (len(labels), len(labels)),
            "labels": labels,
        }

    return {
        "type": "unknown",
        "message": (
            "Could not detect format. Expected columns: "
            "actual_letter/model_output, letter/accuracy, or NxN confusion matrix."
        ),
    }
