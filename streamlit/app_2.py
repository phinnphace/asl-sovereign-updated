"""
app.py — The Provenance Decoder Ring
=====================================
ASL model training data diagnostics via confusion matrix structure.
Bidirectional data flywheel: diagnoses models AND collects real-world
failure data that doesn't exist anywhere else.

Certified by Ted — a tabby cat whose visibility to Gemma 4 E4B via Ollama
confirmed correct vision encoder function after a week of bitsandbytes hell.
Ted visits the stoop. Ted is the mascot. Ted is serene.
"""

import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import base64
from pathlib import Path
import requests
from io import BytesIO

from provenance_decoder import analyze_failure_mode, decode_csv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Provenance Decoder Ring",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TED_URL = "https://raw.githubusercontent.com/phinnphace/asl-sovereign/main/ted.jpg"

# ── Custom CSS — Comic Vintage Superhero / Cryptograph ────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Special+Elite&family=Share+Tech+Mono&display=swap');

  /* ── Root palette ── */
  :root {
    --ink:       #1A1209;
    --paper:     #F5EDD6;
    --paper-dark:#E8D9B0;
    --red:       #C0281A;
    --amber:     #D4820A;
    --blue:      #1B4F8A;
    --green:     #1A6B3A;
    --rule:      #1A1209;
    --mono:      'Share Tech Mono', monospace;
    --display:   'Bangers', cursive;
    --body:      'Special Elite', cursive;
  }

  /* ── Global paper texture ── */
  .stApp {
    background-color: var(--paper) !important;
    background-image:
      repeating-linear-gradient(
        0deg,
        transparent,
        transparent 27px,
        rgba(26,18,9,0.06) 27px,
        rgba(26,18,9,0.06) 28px
      );
  }

  /* ── Streamlit chrome cleanup ── */
  .block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; max-width: 1200px !important; }
  h1, h2, h3, h4 { font-family: var(--display) !important; letter-spacing: 0.04em !important; color: var(--ink) !important; }
  p, li, div { font-family: var(--body) !important; color: var(--ink) !important; }
  .stTextArea textarea { font-family: var(--mono) !important; background: #FEFBF0 !important; border: 2px solid var(--ink) !important; border-radius: 0 !important; }
  .stButton > button {
    font-family: var(--display) !important;
    letter-spacing: 0.08em !important;
    font-size: 18px !important;
    background: var(--red) !important;
    color: #FEFBF0 !important;
    border: 3px solid var(--ink) !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 var(--ink) !important;
    transition: all 0.1s !important;
  }
  .stButton > button:hover {
    transform: translate(2px, 2px) !important;
    box-shadow: 2px 2px 0 var(--ink) !important;
  }
  .stFileUploader {
    border: 3px dashed var(--ink) !important;
    background: #FEFBF0 !important;
    border-radius: 0 !important;
    padding: 0.5rem !important;
  }
  .stAlert { border-radius: 0 !important; border: 2px solid var(--ink) !important; }
  hr { border: 2px solid var(--ink) !important; margin: 1.5rem 0 !important; }

  /* ── Masthead ── */
  .masthead {
    border: 4px solid var(--ink);
    background: var(--ink);
    color: var(--paper) !important;
    padding: 0.6rem 1.2rem;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .masthead-title {
    font-family: var(--display) !important;
    font-size: 42px;
    color: var(--paper) !important;
    letter-spacing: 0.05em;
    line-height: 1;
    text-shadow: 3px 3px 0 var(--red);
  }
  .masthead-sub {
    font-family: var(--mono) !important;
    font-size: 11px;
    color: #C8B98A !important;
    line-height: 1.6;
  }
  .masthead-badge {
    background: var(--red);
    color: var(--paper) !important;
    font-family: var(--display) !important;
    font-size: 13px;
    letter-spacing: 0.1em;
    padding: 2px 10px;
    border: 2px solid var(--paper);
    display: inline-block;
    transform: rotate(-1.5deg);
    margin-top: 4px;
  }

  /* ── Metric cards ── */
  .metric-card {
    border: 3px solid var(--ink);
    background: #FEFBF0;
    padding: 0.75rem 1rem;
    box-shadow: 4px 4px 0 var(--ink);
    position: relative;
  }
  .metric-label {
    font-family: var(--mono) !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6B5B3A !important;
    margin-bottom: 2px;
  }
  .metric-value {
    font-family: var(--display) !important;
    font-size: 32px;
    color: var(--ink) !important;
    line-height: 1;
  }
  .metric-sub {
    font-family: var(--mono) !important;
    font-size: 10px;
    color: #8B7355 !important;
    margin-top: 3px;
  }

  /* ── Codex / lookup cards ── */
  .codex-card {
    border: 3px solid var(--ink);
    background: #FEFBF0;
    box-shadow: 5px 5px 0 var(--ink);
    margin-bottom: 1.25rem;
    overflow: hidden;
  }
  .codex-header {
    padding: 0.5rem 0.75rem;
    border-bottom: 3px solid var(--ink);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .codex-header-landmark { background: var(--red); }
  .codex-header-full     { background: var(--green); }
  .codex-header-hybrid   { background: var(--blue); }
  .codex-header-title {
    font-family: var(--display) !important;
    font-size: 18px;
    color: #FEFBF0 !important;
    letter-spacing: 0.05em;
  }
  .codex-header-cer {
    font-family: var(--mono) !important;
    font-size: 12px;
    color: #FEFBF0 !important;
    opacity: 0.85;
  }
  .codex-body {
    padding: 0.75rem;
    display: flex;
    gap: 1rem;
  }
  .codex-matrix {
    font-family: var(--mono) !important;
    font-size: 10.5px;
    line-height: 1.55;
    color: var(--ink) !important;
    background: var(--paper);
    border: 2px solid var(--ink);
    padding: 0.5rem 0.6rem;
    flex: 1;
    white-space: pre;
  }
  .codex-pairs {
    flex: 1;
  }
  .codex-pairs-label {
    font-family: var(--mono) !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6B5B3A !important;
    margin-bottom: 5px;
  }
  .codex-badge {
    display: inline-block;
    font-family: var(--mono) !important;
    font-size: 11px;
    padding: 1px 7px;
    border: 2px solid var(--ink);
    margin: 2px;
  }
  .codex-badge-landmark { background: #FAECE7; color: var(--ink) !important; }
  .codex-badge-full     { background: #E1F5EE; color: var(--ink) !important; }
  .codex-badge-hybrid   { background: #E6F1FB; color: var(--ink) !important; }
  .codex-risk {
    font-family: var(--mono) !important;
    font-size: 11px;
    margin-top: 8px;
    padding: 4px 8px;
    border: 2px solid var(--ink);
    display: inline-block;
  }
  .risk-high     { background: #FAECE7; color: var(--red) !important; font-weight: bold; }
  .risk-moderate { background: #FFF8E1; color: var(--amber) !important; font-weight: bold; }
  .risk-low      { background: #E1F5EE; color: var(--green) !important; font-weight: bold; }

  /* ── Result box ── */
  .result-box {
    border: 3px solid var(--ink);
    background: #FEFBF0;
    padding: 1rem 1.25rem;
    box-shadow: 5px 5px 0 var(--ink);
    margin-top: 1rem;
  }
  .result-label {
    font-family: var(--mono) !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6B5B3A !important;
    margin-bottom: 3px;
  }
  .firing-pair {
    display: inline-block;
    background: #FAECE7;
    font-family: var(--mono) !important;
    font-size: 12px;
    padding: 2px 8px;
    border: 2px solid var(--red);
    margin: 2px;
    color: var(--ink) !important;
  }

  /* ── Ted ── */
  .ted-frame {
    border: 4px solid var(--ink);
    box-shadow: 5px 5px 0 var(--ink);
    overflow: hidden;
    background: var(--paper-dark);
  }
  .ted-caption {
    font-family: var(--mono) !important;
    font-size: 9.5px;
    color: #6B5B3A !important;
    text-align: center;
    padding: 4px;
    background: var(--paper-dark);
    border-top: 2px solid var(--ink);
  }

  /* ── Accuracy blurb ── */
  .accuracy-blurb {
    border-left: 5px solid var(--ink);
    padding-left: 1rem;
    font-family: var(--body) !important;
    font-size: 14px;
    line-height: 1.8;
    color: #3A2E1A !important;
    margin-top: 0.5rem;
  }
  .accuracy-number {
    font-family: var(--display) !important;
    font-size: 56px;
    color: var(--ink) !important;
    line-height: 1;
  }
  .accuracy-denom {
    font-family: var(--mono) !important;
    font-size: 12px;
    color: #8B7355 !important;
  }

  /* ── Stamp ── */
  .stamp {
    display: inline-block;
    border: 4px solid var(--red);
    color: var(--red) !important;
    font-family: var(--display) !important;
    font-size: 22px;
    letter-spacing: 0.15em;
    padding: 4px 14px;
    transform: rotate(-8deg);
    opacity: 0.85;
    margin-left: 12px;
    vertical-align: middle;
  }

  /* ── Section headers ── */
  .section-rule {
    border-top: 4px double var(--ink);
    border-bottom: 1px solid var(--ink);
    padding: 4px 0;
    font-family: var(--display) !important;
    font-size: 22px;
    letter-spacing: 0.06em;
    color: var(--ink) !important;
    margin: 1.25rem 0 0.75rem 0;
  }

  /* ── Footer ── */
  .footer-rule {
    border-top: 4px solid var(--ink);
    font-family: var(--mono) !important;
    font-size: 10px;
    color: #6B5B3A !important;
    padding-top: 6px;
    margin-top: 2rem;
    line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)


# ── Google Sheets persistence ─────────────────────────────────────────────────

@st.cache_resource
def get_sheets_client():
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception:
        return None


def log_diagnosis(user_input: str, result: dict):
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
        pass


def get_diagnosis_count() -> int:
    try:
        client = get_sheets_client()
        if client is None:
            return 0
        sheet_id = st.secrets.get("SHEETS_ID", "")
        if not sheet_id:
            return 0
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet("diagnoses")
        return max(0, len(ws.get_all_values()) - 1)
    except Exception:
        return 0


# ── Ted loader ────────────────────────────────────────────────────────────────

@st.cache_data
def load_ted_b64() -> str:
    """Load Ted from GitHub. Returns base64 string or empty string on failure."""
    try:
        r = requests.get(TED_URL, timeout=8)
        r.raise_for_status()
        return base64.b64encode(r.content).decode()
    except Exception:
        return ""


def get_api_key() -> str:
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return ""


# ── Codex card renderer ───────────────────────────────────────────────────────

def codex_card(regime: str, cer: str, matrix_ascii: str, pairs: list, risk: str, risk_class: str, desc_lines: list):
    header_class = f"codex-header-{regime}"
    badge_class  = f"codex-badge-{regime}"

    labels = {
        "landmark": "① LANDMARK-ONLY  ·  Hand Keypoints (21-pt)",
        "full":     "② FULL LANDMARKS  ·  Hands + Face + Body",
        "hybrid":   "③ HYBRID  ·  RGB Frames + Pose Keypoints",
    }

    pairs_html = " ".join(
        f'<span class="codex-badge {badge_class}">{p}</span>' for p in pairs
    )
    desc_html = "<br>".join(desc_lines)

    st.markdown(f"""
    <div class="codex-card">
      <div class="codex-header {header_class}">
        <span class="codex-header-title">{labels[regime]}</span>
        <span class="codex-header-cer">CER {cer}</span>
      </div>
      <div class="codex-body">
        <div class="codex-matrix">{matrix_ascii}</div>
        <div class="codex-pairs">
          <div class="codex-pairs-label">Confusion signature</div>
          {pairs_html}
          <br><br>
          <div style="font-family:'Share Tech Mono',monospace;font-size:10.5px;line-height:1.6;color:#3A2E1A;">
            {desc_html}
          </div>
          <div class="codex-risk {risk_class}">{risk}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    api_key        = get_api_key()
    diagnosis_count = get_diagnosis_count()
    ted_b64        = load_ted_b64()

    # ── Masthead ──────────────────────────────────────────────────────────────
    col_title, col_ted = st.columns([5, 1])

    with col_title:
        st.markdown("""
        <div class="masthead">
          <div>
            <div class="masthead-title">💍 THE PROVENANCE DECODER RING</div>
            <div class="masthead-sub">
              ASL MODEL TRAINING DATA DIAGNOSTICS  ·  CONFUSION MATRIX STRUCTURE ANALYSIS<br>
              Gemma 4 E4B  ·  Mantel r=0.945 cross-dataset stability  ·
              <a href="https://github.com/phinnphace/asl-sovereign"
                 style="color:#C8B98A;">asl-sovereign on GitHub</a>
            </div>
            <span class="masthead-badge">CLASSIFIED INSTRUMENT</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ted:
        if ted_b64:
            st.markdown(f"""
            <div class="ted-frame">
              <img src="data:image/jpeg;base64,{ted_b64}"
                   style="width:100%;display:block;" alt="Ted">
              <div class="ted-caption">
                TED · STOOP TABBY<br>
                CERTIFIED THIS PIPELINE
                <br>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ted-frame" style="padding:1rem;text-align:center;">
              <div style="font-family:'Bangers',cursive;font-size:28px;">TED</div>
              <div class="ted-caption">Unable to load Ted.<br>The pipeline is still certified.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Metrics strip ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Diagnostic confidence", "Accumulating", "grows with user submissions"),
        ("Mantel correlation",    "r=0.945",      "Roboflow vs ISL · p<0.001"),
        ("Images audited",        "2,370",         "ASL + ISL · 24 static letters"),
        ("Diagnoses run",
         str(diagnosis_count) if diagnosis_count > 0 else "—",
         "flywheel · grows with use"),
    ]
    for col, (label, value, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Diagnostic input ──────────────────────────────────────────────────────
    st.markdown('<div class="section-rule">① DIAGNOSTIC TRANSMISSION</div>', unsafe_allow_html=True)
    st.markdown(
        "Describe what your're experiencing in language that makes sense to you about your current use, "
        "Let the ring decode your signal."
    )

    user_input = st.text_area(
        "Input:",
        placeholder=(
            '"K always comes back as V — it never gets it right"\n'
            '"Massive blocky clusters — M, N, and T are all bleeding together"\n'
            '"The model sees hands clearly and is confidently wrong"'
        ),
        height=100,
        label_visibility="collapsed",
    )

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        diagnose_clicked = st.button("💍 DECODE", type="primary", use_container_width=True)
    with col_note:
        if not api_key:
            st.warning("⚠️ No API key found. Add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`.")

    # ── Diagnosis result ──────────────────────────────────────────────────────
    if diagnose_clicked:
        if not user_input.strip():
            st.warning("Transmit a signal first.")
        elif not api_key:
            st.error("No API key configured. Cannot run diagnosis.")
        else:
            with st.spinner("Reading failure geometry..."):
                result = analyze_failure_mode(user_input, api_key)
                log_diagnosis(user_input, result)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown('<div class="result-label">Failure mode</div>', unsafe_allow_html=True)
                st.markdown(f"#### {result.get('failure_mode', '—')}")

                st.markdown('<div class="result-label">Training regime detected</div>', unsafe_allow_html=True)
                st.info(result.get("training_regime", "—"))

                st.markdown('<div class="result-label">Matrix signature</div>', unsafe_allow_html=True)
                st.markdown(f"`{result.get('matrix_signature', '—')}`")

                firing = result.get("firing_pairs", [])
                if firing:
                    st.markdown('<div class="result-label">Firing decoder ring pairs</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="result-label">Deployment risk</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="codex-risk {risk_class}">{risk}</span>', unsafe_allow_html=True)

                st.markdown('<div class="result-label">Provenance diagnosis</div>', unsafe_allow_html=True)
                st.markdown(result.get("provenance_diagnosis", "—"))

                st.markdown('<div class="result-label">Recommendation</div>', unsafe_allow_html=True)
                st.error(result.get("recommendation", "—"))

                st.markdown('<div class="result-label">Compute note</div>', unsafe_allow_html=True)
                st.markdown(f"*{result.get('compute_saved', '—')}*")

            st.markdown('</div>', unsafe_allow_html=True)

            if "raw_output" in result:
                with st.expander("Raw transmission"):
                    st.code(result["raw_output"])

    # ── Two column: Codex + Upload ─────────────────────────────────────────────
    st.markdown('<div class="section-rule">② CONFUSION CODEX  ·  REFERENCE FINGERPRINTS</div>', unsafe_allow_html=True)

    col_lookup, col_upload = st.columns([3, 2])

    with col_lookup:
        st.markdown(
            "Three training regimes. Three confusion fingerprints. "
            "Locate your model's pattern. The pairs are stable across datasets — "
            "Mantel r=0.945, p<0.001.",
            unsafe_allow_html=True
        )

        # ── LANDMARK-ONLY ──
        codex_card(
            regime="landmark",
            cer="16.7%",
            matrix_ascii=(
                "  Pred →  M   N   T   A   S\n"
                "True ↓  ┌───────────────────┐\n"
                "  M     │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n"
                "  N     │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n"
                "  T     │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n"
                "  A     │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n"
                "  S     │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n"
                "        └───────────────────┘\n"
                " ▓=high confusion  ░=low"
            ),
            pairs=["K→V (total)", "D→I", "M↔N↔T", "A↔S", "G→D", "P→D"],
            risk="⚠ DEPLOYMENT RISK: HIGH",
            risk_class="risk-high",
            desc_lines=[
                "Blocky off-diagonal clusters",
                "Geometric pairs dominate errors",
                "Failures are predictable + repeatable",
                "Silent systematic misidentification",
                "No refusal signal",
            ]
        )

        # ── FULL LANDMARKS ──
        codex_card(
            regime="full",
            cer="11.1%",
            matrix_ascii=(
                "  Pred →  M   N   T   A   S\n"
                "True ↓  ┌───────────────────┐\n"
                "  M     │▓▓▓ ▒▒▒ ▒▒▒ ░░░ ░░░│\n"
                "  N     │▒▒▒ ▓▓▓ ▒▒▒ ░░░ ░░░│\n"
                "  T     │▒▒▒ ▒▒▒ ▓▓▓ ░░░ ░░░│\n"
                "  A     │░░░ ░░░ ░░░ ▓▓▓ ▒▒▒│\n"
                "  S     │░░░ ░░░ ░░░ ▒▒▒ ▓▓▓│\n"
                "        └───────────────────┘\n"
                " ▓=correct  ▒=some confusion  ░=low"
            ),
            pairs=["M vs N (reduced)", "co-artic fail", "flow break"],
            risk="⚠ DEPLOYMENT RISK: MODERATE-HIGH",
            risk_class="risk-moderate",
            desc_lines=[
                "Softer diagonals — clusters reduced",
                "Classic pairs weakened not gone",
                "Failures become distributed",
                "Static signing OK · motion fails",
                "Signer-context dependent",
            ]
        )

        # ── HYBRID ──
        codex_card(
            regime="hybrid",
            cer="7.2%",
            matrix_ascii=(
                "  Pred →  M   N   T   A   S\n"
                "True ↓  ┌───────────────────┐\n"
                "  M     │▓▓▓ ░░░ ░░░ ░░░ ░░░│\n"
                "  N     │░░░ ▓▓▓ ░░░ ░░░ ░░░│\n"
                "  T     │░░░ ░░░ ▓▓▓ ░░░ ░░░│\n"
                "  A     │░░░ ░░░ ░░░ ▓▓▓ ░░░│\n"
                "  S     │░░░ ░░░ ░░░ ░░░ ▓▓▓│\n"
                "        └───────────────────┘\n"
                " ▓=correct  ░=low confusion"
            ),
            pairs=["context errors", "temporal drift", "sequence breaks"],
            risk="✓ DEPLOYMENT RISK: MODERATE",
            risk_class="risk-low",
            desc_lines=[
                "Thin diagonal · diffuse noise",
                "Classic geometric pairs resolved",
                "Remaining errors: sequence-driven",
                "Handshapes correct · context fails",
                "Long signing strings break",
            ]
        )

        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;color:#8B7355;">'
            'Reference: FSboard (Georg et al. 2024, CC BY 4.0) · arxiv.org/abs/2407.15806 · '
            'Mantel r=0.945 p&lt;0.001 (n=9,999 permutations) · Gemma 4 E4B via Ollama'
            '</div>',
            unsafe_allow_html=True
        )

    # ── Dataset upload ─────────────────────────────────────────────────────────
    with col_upload:
        st.markdown("##### Upload your data")
        st.markdown(
            "Upload a confusion matrix CSV, per-letter accuracy CSV, "
            "or raw inference log. The decoder ring fires automatically. "
            "No confusion matrix on hand? Describe your model's failures in "
            "the diagnostic box above — plain language works."
        )

        uploaded = st.file_uploader(
            "Drop file",
            type=["csv"],
            help=(
                "Accepted formats:\n"
                "· Inference log: actual_letter / model_output columns\n"
                "· Per-letter accuracy: letter / accuracy columns\n"
                "· Confusion matrix: NxN CSV with letter labels"
            ),
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

                with st.expander("Preview"):
                    st.dataframe(df.head(10), use_container_width=True)

                summary = decode_csv(df)
                dtype   = summary.get("type", "unknown")

                if dtype == "inference_log":
                    acc   = summary["overall_accuracy"]
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
                                "Status": (
                                    "⚠ Collapse" if acc_l < 0.05
                                    else "🔴 Low" if acc_l < 0.20
                                    else "✓"
                                ),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    top_errors = summary.get("top_errors", {})
                    if top_errors and api_key:
                        st.markdown("**Auto-diagnosing from upload...**")
                        worst = sorted(top_errors.items(), key=lambda x: len(x[1]), reverse=True)[:3]
                        error_desc = "; ".join(
                            f"{l} confused with {', '.join(v)}" for l, v in worst
                        )
                        auto_input = f"Per-letter accuracy: overall {acc}%. Top error pairs: {error_desc}."
                        with st.spinner("Running decoder ring on uploaded data..."):
                            result = analyze_failure_mode(auto_input, api_key)
                            log_diagnosis(auto_input, result)
                        st.info(f"**Regime:** {result.get('training_regime', '—')}")
                        st.markdown(result.get("provenance_diagnosis", "—"))
                        st.error(f"**Recommendation:** {result.get('recommendation', '—')}")

                elif dtype == "confusion_matrix":
                    st.markdown(f"**Confusion matrix:** {summary['shape'][0]}×{summary['shape'][1]}")
                    st.markdown("Labels: " + ", ".join(summary.get("labels", [])))
                    if api_key:
                        matrix_desc = (
                            f"Confusion matrix {summary['shape'][0]} classes: "
                            f"{', '.join(summary.get('labels', [])[:10])}. Analyze structure."
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
            "- Confusion matrix: NxN CSV with letter labels\n\n"
            "**No file?** Use the diagnostic box above."
        )

    # ── Confidence / accuracy section ─────────────────────────────────────────
    st.markdown('<div class="section-rule">③ ON CONFIDENCE</div>', unsafe_allow_html=True)

    acc_col, ted_col = st.columns([4, 1])
    with acc_col:
        st.markdown("""
        <div class="accuracy-number">Accumulating
          <span style="font-family:'Share Tech Mono',monospace;font-size:18px;vertical-align:middle;">
          <span class="stamp">LIVE</span></span>
        </div>
        <div class="accuracy-denom">construct validity confirmed · absolute accuracy accumulates with use</div>
        <div class="accuracy-blurb">
          The validation suite establishes that this instrument measures something real and stable —
          bootstrap persistence at 1,000 iterations, Gaussian noise probe confirming weight-encoded
          confusion structure, Mantel r=0.945 cross-dataset. What it does not yet have is an
          absolute accuracy percentage against known-provenance models, because a labeled set of
          models with known training regimes does not exist in the literature.<br><br>
          We refuse to manufacture ground truth to produce a confidence number.
          Every diagnosis submitted becomes part of the ground truth as provenance becomes known.
          The confidence number you want is the thing this instrument is designed to produce —
          honestly, from real use.
        </div>
        """, unsafe_allow_html=True)

    with ted_col:
        if ted_b64:
            st.markdown(f"""
            <div class="ted-frame">
              <img src="data:image/jpeg;base64,{ted_b64}"
                   style="width:100%;display:block;" alt="Ted">
              <div class="ted-caption">
                Ted confirmed the pipeline.<br>
                Gemma saw Ted via Ollama.<br>
                HuggingFace could not see Ted.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer-rule">
      PROVENANCE DECODER RING  ·
      Gemma 4 E4B (gemma4:e4b-it-q4_K_M via Ollama) for research  ·
      Gemma Instruct via Google AI Studio for diagnostics  ·
      FSboard reference: Georg et al. 2024 (CC BY 4.0) · arxiv.org/abs/2407.15806  ·
      ISL dataset: Biswas 2024 · doi:10.17632/n34wm8sb3x.1  ·
      Mantel r=0.945 p&lt;0.001 (n=9,999 permutations)  ·
      Bootstrap persistence: 100% across 1,000 iterations
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
