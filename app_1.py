"""
app.py — The Provenance Decoder Ring
=====================================
ASL model training data diagnostics via Gemma (Google AI Studio).
Bidirectional data flywheel: diagnoses user complaints AND collects
real-world failure language that doesn't exist anywhere else.

Certified by Ted — a tabby cat whose visibility to Gemma 4eb via Ollama
confirmed correct vision encoder function after a week of bitsandbytes
quantization hell. Ted visits my stoop. Ted is the mascot. Ted is serene. 
"""

import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import base64
from pathlib import Path

from provenance_decoder import analyze_failure_mode, decode_csv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Provenance Decoder Ring",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6c757d;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 600;
        color: #212529;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 11px;
        color: #adb5bd;
        margin-top: 3px;
    }
    .lookup-landmark {
        background: #FAECE7;
        border: 1px solid #F0997B;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    .lookup-full {
        background: #E1F5EE;
        border: 1px solid #5DCAA5;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    .lookup-hybrid {
        background: #E6F1FB;
        border: 1px solid #85B7EB;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    .lookup-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .lookup-mono {
        font-family: monospace;
        font-size: 11px;
        line-height: 1.6;
        opacity: 0.85;
    }
    .pair-badge-landmark {
        display: inline-block;
        background: #F0997B44;
        color: #4A1B0C;
        font-size: 11px;
        padding: 1px 7px;
        border-radius: 4px;
        margin: 2px;
        font-family: monospace;
    }
    .pair-badge-full {
        display: inline-block;
        background: #5DCAA544;
        color: #04342C;
        font-size: 11px;
        padding: 1px 7px;
        border-radius: 4px;
        margin: 2px;
        font-family: monospace;
    }
    .pair-badge-hybrid {
        display: inline-block;
        background: #85B7EB44;
        color: #042C53;
        font-size: 11px;
        padding: 1px 7px;
        border-radius: 4px;
        margin: 2px;
        font-family: monospace;
    }
    .risk-high { color: #993C1D; font-weight: 600; }
    .risk-moderate { color: #854F0B; font-weight: 600; }
    .risk-low { color: #3B6D11; font-weight: 600; }
    .accuracy-blurb {
        border-left: 3px solid #adb5bd;
        padding-left: 1rem;
        color: #6c757d;
        font-style: italic;
        font-size: 14px;
        line-height: 1.7;
        margin-top: 0.75rem;
    }
    .ted-caption {
        font-size: 11px;
        color: #adb5bd;
        text-align: center;
        margin-top: 4px;
        font-style: italic;
    }
    .result-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    .firing-pair {
        display: inline-block;
        background: #FAECE7;
        color: #4A1B0C;
        font-family: monospace;
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
        border: 1px solid #F0997B;
    }
    div[data-testid="stHorizontalBlock"] { gap: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Google Sheets persistence ─────────────────────────────────────────────────

@st.cache_resource
def get_sheets_client():
    """Get authenticated Google Sheets client from Streamlit secrets."""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        return None


def log_diagnosis(user_input: str, result: dict):
    """Append a diagnosis to Google Sheets flywheel log."""
    try:
        client = get_sheets_client()
        if client is None:
            return
        sheet_id = st.secrets.get("SHEETS_ID", "")
        if not sheet_id:
            return
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet("diagnoses")
        ws.append_row([
            datetime.utcnow().isoformat(),
            user_input[:500],
            result.get("failure_mode", ""),
            result.get("training_regime", ""),
            result.get("deployment_risk", ""),
            json.dumps(result.get("firing_pairs", [])),
            result.get("recommendation", "")[:300],
        ])
    except Exception:
        pass  # Flywheel logging is best-effort, never block the user


def get_diagnosis_count() -> int:
    """Get total number of diagnoses run (flywheel counter)."""
    try:
        client = get_sheets_client()
        if client is None:
            return 0
        sheet_id = st.secrets.get("SHEETS_ID", "")
        if not sheet_id:
            return 0
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet("diagnoses")
        return max(0, len(ws.get_all_values()) - 1)  # subtract header row
    except Exception:
        return 0


# ── Image helpers ─────────────────────────────────────────────────────────────

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── API key ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():

    api_key = get_api_key()
    diagnosis_count = get_diagnosis_count()

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_ted = st.columns([5, 1])
    with col_title:
        st.markdown("## 🔍 The Provenance Decoder Ring")
        st.markdown(
            "ASL model training data diagnostics · "
            "Gemma 4 E4B · "
            "Mantel r=0.945 cross-dataset stability · "
            "[asl-sovereign on GitHub](https://github.com/phinnphace/asl-sovereign)"
        )

    with col_ted:
        ted_path = Path(__file__).parent / "ted_mascot.jpg"
        if ted_path.exists():
            st.markdown(
                f'<img src="data:image/jpeg;base64,{img_to_base64(str(ted_path))}" '
                f'style="width:100%;border-radius:50%;border:2px solid #e9ecef;" alt="Ted">',
                unsafe_allow_html=True
            )
            st.markdown('<div class="ted-caption">Ted · certified this pipeline</div>', unsafe_allow_html=True)

    st.divider()

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Diagnostic accuracy</div>
          <div class="metric-value">67%</div>
          <div class="metric-sub">grows with user data</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Mantel correlation</div>
          <div class="metric-value">r=0.945</div>
          <div class="metric-sub">Roboflow vs ISL, p&lt;0.001</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Images audited</div>
          <div class="metric-value">2,370</div>
          <div class="metric-sub">ASL + ISL · 24 static letters</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Diagnoses run</div>
          <div class="metric-value">{diagnosis_count if diagnosis_count > 0 else "—"}</div>
          <div class="metric-sub">flywheel · grows with use</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Diagnostic chat ───────────────────────────────────────────────────────
    st.markdown("### Diagnostic chat")
    st.markdown(
        "Describe what your model gets wrong — in plain language or as a confusion matrix description. "
        "Gemma reads your complaint and maps it to the training data topology that caused it."
    )

    user_input = st.text_area(
        "Your complaint or matrix description:",
        placeholder=(
            'e.g. "it misses my face and I have to perform for the camera just to get it to see me"\n'
            'or "the confusion matrix has massive blocky clusters — M, N, and T are all bleeding together"\n'
            'or "K is basically useless, it always says V"'
        ),
        height=110,
        label_visibility="collapsed",
    )

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        diagnose_clicked = st.button("🔍 Diagnose", type="primary", use_container_width=True)
    with col_note:
        if not api_key:
            st.warning("⚠️ No API key found. Add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`.")

    # ── Diagnosis result ──────────────────────────────────────────────────────
    if diagnose_clicked:
        if not user_input.strip():
            st.warning("Please describe the failure first.")
        elif not api_key:
            st.error("No API key configured. Cannot run diagnosis.")
        else:
            with st.spinner("Gemma is reading the failure geometry..."):
                result = analyze_failure_mode(user_input, api_key)
                log_diagnosis(user_input, result)

            st.success("Diagnosis complete.")

            r1, r2 = st.columns(2)

            with r1:
                st.markdown("**Failure mode**")
                st.markdown(f"#### {result.get('failure_mode', '—')}")

                st.markdown("**Training regime detected**")
                st.info(result.get("training_regime", "—"))

                st.markdown("**Matrix signature**")
                st.markdown(f"`{result.get('matrix_signature', '—')}`")

                firing = result.get("firing_pairs", [])
                if firing:
                    st.markdown("**Firing decoder ring pairs**")
                    pairs_html = " ".join(
                        f'<span class="firing-pair">{p}</span>' for p in firing
                    )
                    st.markdown(pairs_html, unsafe_allow_html=True)

            with r2:
                risk = result.get("deployment_risk", "—")
                risk_class = (
                    "risk-high" if "HIGH" in risk.upper()
                    else "risk-moderate" if "MODERATE" in risk.upper()
                    else "risk-low"
                )
                st.markdown("**Deployment risk**")
                st.markdown(
                    f'<span class="{risk_class}">{risk}</span>',
                    unsafe_allow_html=True
                )

                st.markdown("**Provenance diagnosis**")
                st.markdown(result.get("provenance_diagnosis", "—"))

                st.markdown("**Recommendation**")
                st.error(result.get("recommendation", "—"))

                st.markdown("**Compute optimization**")
                st.markdown(f"*{result.get('compute_saved', '—')}*")

            if "raw_output" in result:
                with st.expander("Raw model output"):
                    st.code(result["raw_output"])

    st.divider()

    # ── Two-column: Lookup table + Dataset upload ─────────────────────────────
    col_lookup, col_upload = st.columns(2)

    # ── Lookup table ──────────────────────────────────────────────────────────
    with col_lookup:
        st.markdown("### Confusion fingerprint lookup")
        st.markdown(
            "Reference cards for the three training regimes. "
            "Print these out — the confusion pairs are stable across datasets."
        )

        st.markdown("""
        <div class="lookup-landmark">
          <div class="lookup-title">🟠 Landmark-only (21-point / hand-only)</div>
          <div class="lookup-mono">
            Matrix shape: blocky off-diagonal clusters<br>
            M / N / T → high mutual confusion<br>
            A / S / E → cluster collapse<br>
            K → V (total, ≥85%)<br>
            D → I dominant · G ↔ D · P ↔ D
          </div>
          <div style="margin-top:8px;">
            <span class="pair-badge-landmark">K→V</span>
            <span class="pair-badge-landmark">D→I</span>
            <span class="pair-badge-landmark">M→N</span>
            <span class="pair-badge-landmark">M→T</span>
            <span class="pair-badge-landmark">A→S</span>
            <span class="pair-badge-landmark">G→D</span>
            <span class="pair-badge-landmark">P→D</span>
          </div>
          <div style="margin-top:8px;font-size:12px;color:#4A1B0C;">
            Deployment risk: <strong>HIGH</strong> · silent systematic failure · no refusal signal
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="lookup-full">
          <div class="lookup-title">🟢 Full-body landmark (hands + face + body + temporal)</div>
          <div class="lookup-mono">
            Matrix shape: softer diagonals<br>
            Classic pairs reduced but not gone<br>
            Failures are distributed<br>
            Co-articulation / continuous flow fails<br>
            Static dictionary signing OK · motion fails
          </div>
          <div style="margin-top:8px;">
            <span class="pair-badge-full">M vs N (reduced)</span>
            <span class="pair-badge-full">co-artic fail</span>
            <span class="pair-badge-full">flow break</span>
          </div>
          <div style="margin-top:8px;font-size:12px;color:#04342C;">
            Deployment risk: <strong>MODERATE-HIGH</strong> · signer-context dependent
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="lookup-hybrid">
          <div class="lookup-title">🔵 Hybrid RGB + pose (multimodal)</div>
          <div class="lookup-mono">
            Matrix shape: thin diagonal · diffuse noise<br>
            Classic geometric pairs largely resolved<br>
            Remaining errors: sequence-driven<br>
            Handshapes correct · context/motion fails<br>
            Long signing strings break · static OK
          </div>
          <div style="margin-top:8px;">
            <span class="pair-badge-hybrid">context errors</span>
            <span class="pair-badge-hybrid">temporal</span>
            <span class="pair-badge-hybrid">sequence</span>
          </div>
          <div style="margin-top:8px;font-size:12px;color:#042C53;">
            Deployment risk: <strong>MODERATE</strong> · errors contextual not geometric
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>")
        st.markdown(
            "<div style='font-size:11px;color:#adb5bd;'>Based on: FSboard (Georg et al. 2024, CC BY 4.0) · "
            "Mantel test r=0.945 p<0.001 (Roboflow vs ISL, n=9,999 permutations) · "
            "Gemma 4 E4B via Ollama</div>",
            unsafe_allow_html=True
        )

    # ── Dataset upload ─────────────────────────────────────────────────────────
    with col_upload:
        st.markdown("### Dataset upload")
        st.markdown(
            "Upload a confusion matrix CSV, per-letter accuracy CSV, or raw inference log. "
            "The decoder ring fires automatically."
        )

        uploaded = st.file_uploader(
            "Drop your file here",
            type=["csv"],
            help="Accepted: confusion matrix (NxN), per-letter accuracy (letter/accuracy columns), "
                 "or inference log (actual_letter/model_output columns)",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

                with st.expander("Preview"):
                    st.dataframe(df.head(10), use_container_width=True)

                summary = decode_csv(df)
                dtype = summary.get("type", "unknown")

                if dtype == "inference_log":
                    acc = summary["overall_accuracy"]
                    total = summary["total_images"]
                    st.markdown(f"**Overall accuracy:** {acc}% across {total} images")

                    per_letter = summary.get("per_letter", {})
                    if per_letter:
                        rows = []
                        for letter, data in sorted(per_letter.items()):
                            acc_l = data.get("accuracy", 0)
                            rows.append({
                                "Letter": letter,
                                "Accuracy": f"{acc_l:.1%}",
                                "Status": "⚠️ Collapse" if acc_l < 0.05 else ("🔴 Low" if acc_l < 0.20 else "✅"),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    top_errors = summary.get("top_errors", {})
                    if top_errors and api_key:
                        st.markdown("**Auto-diagnosing from upload...**")
                        worst = sorted(top_errors.items(), key=lambda x: len(x[1]), reverse=True)[:3]
                        error_desc = "; ".join(
                            f"{l} confused with {', '.join(v)}" for l, v in worst
                        )
                        auto_input = f"Per-letter accuracy data: overall {acc}%. Top error pairs: {error_desc}."
                        with st.spinner("Running decoder ring on uploaded data..."):
                            result = analyze_failure_mode(auto_input, api_key)
                            log_diagnosis(auto_input, result)

                        st.markdown("**Decoder ring result:**")
                        st.info(f"**Regime:** {result.get('training_regime', '—')}")
                        st.markdown(result.get("provenance_diagnosis", "—"))
                        st.error(f"**Recommendation:** {result.get('recommendation', '—')}")

                elif dtype == "confusion_matrix":
                    st.markdown(f"**Confusion matrix detected:** {summary['shape'][0]}×{summary['shape'][1]}")
                    st.markdown("Labels: " + ", ".join(summary.get("labels", [])))
                    if api_key:
                        st.markdown("**Auto-diagnosing from matrix...**")
                        matrix_desc = (
                            f"Confusion matrix with {summary['shape'][0]} classes: "
                            f"{', '.join(summary.get('labels', [])[:10])}. "
                            "Please analyze the matrix structure."
                        )
                        with st.spinner("Running decoder ring..."):
                            result = analyze_failure_mode(matrix_desc, api_key)
                            log_diagnosis(matrix_desc, result)
                        st.info(f"**Regime:** {result.get('training_regime', '—')}")
                        st.markdown(result.get("provenance_diagnosis", "—"))

                elif dtype == "unknown":
                    st.warning(summary.get("message", "Unknown format."))

            except Exception as e:
                st.error(f"Could not parse file: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "**Accepted formats:**\n"
            "- Inference log: `actual_letter`, `model_output` columns\n"
            "- Per-letter accuracy: `letter`, `accuracy` columns\n"
            "- Confusion matrix: NxN CSV with letter labels"
        )

    st.divider()

    # ── Accuracy blurb ────────────────────────────────────────────────────────
    st.markdown("### About this accuracy score")

    acc_col, ted_col = st.columns([4, 1])
    with acc_col:
        st.markdown(
            "<div style='font-size:32px;font-weight:600;margin-bottom:4px;'>67%</div>"
            "<div style='font-size:14px;color:#6c757d;margin-bottom:0.75rem;'>"
            "baseline diagnostic accuracy · increases with user participation</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="accuracy-blurb">'
            "Our baseline diagnostic accuracy is 67% due to ecosystem constraints, not poor diagnostics. "
            "We refuse to manipulate data or take other standard machine learning practices to amplify this accuracy "
            "when simply waiting and letting the tool be used honestly and transparently provides all of the information necessary. "
            "This tool is designed as a bidirectional data flywheel — your complaints and corrections are the training signal."
            "</div>",
            unsafe_allow_html=True
        )

    with ted_col:
        ted_path = Path(__file__).parent / "ted_mascot.jpg"
        if ted_path.exists():
            st.markdown(
                f'<img src="data:image/jpeg;base64,{img_to_base64(str(ted_path))}" '
                f'style="width:100%;border-radius:10px;border:1px solid #e9ecef;" alt="Ted">',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="ted-caption">'
                'Ted confirmed the pipeline works.<br>'
                'The model saw him clearly via Ollama.<br>'
                'HuggingFace could not see Ted.'
                '</div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.caption(
        "Provenance Decoder Ring · "
        "Gemma 4 E4B (gemma4:e4b-it-q4_K_M via Ollama) for research · "
        "Gemma Instruct via Google AI Studio for diagnostics · "
        "Framework: *Seeing through the Map* · "
        "FSboard reference: Georg et al. 2024 (CC BY 4.0) · "
        "ISL dataset: Biswas 2024 · "
        "Mantel r=0.945 p<0.001"
    )


if __name__ == "__main__":
    main()
