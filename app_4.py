"""
app.py — The Provenance Decoder Ring
=====================================
ASL model training data diagnostics via confusion matrix structure.
Bidirectional data flywheel: diagnoses models AND collects real-world
failure data that doesn't exist anywhere else.

Certified by Ted — a tabby cat whose visibility to Gemma 4 E4B via Ollama
confirmed correct vision encoder function after a week of bitsandbytes hell.
Ted visits the stoop. Ted is the mascot. Ted is serene.

Diagnostic engine: Gemma 2 Instruct (Google AI Studio)
Audited subject:   Gemma 4 E4B (gemma4:e4b-it-q4_K_M via Ollama)
"""

import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import base64
import requests

from provenance_decoder import analyze_failure_mode, decode_csv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Provenance Decoder Ring",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TED_URL = "https://raw.githubusercontent.com/phinnphace/asl-sovereign/main/ted.jpg"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Special+Elite&family=Share+Tech+Mono&display=swap');

  :root {
    --ink:    #1A1209;
    --paper:  #F5EDD6;
    --cream:  #FEFBF0;
    --tan:    #E8D9B0;
    --red:    #C0281A;
    --amber:  #C47A0A;
    --blue:   #1B4F8A;
    --green:  #1A6B3A;
    --display: 'Bangers', cursive;
    --body:    'Special Elite', cursive;
    --mono:    'Share Tech Mono', monospace;
  }

  /* Paper texture */
  .stApp {
    background-color: var(--paper) !important;
    background-image: repeating-linear-gradient(
      0deg, transparent, transparent 27px,
      rgba(26,18,9,0.05) 27px, rgba(26,18,9,0.05) 28px
    );
  }

  /* Remove Streamlit top chrome padding */
  .stApp > div:first-child { padding-top: 0 !important; }
  .block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1240px !important;
  }
  header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
  }

  /* Global type */
  h1,h2,h3,h4,h5 { font-family: var(--display) !important; letter-spacing: 0.04em !important; color: var(--ink) !important; }
  p, li, label, div { font-family: var(--body) !important; color: var(--ink) !important; }
  code, pre { font-family: var(--mono) !important; }

  /* Inputs */
  .stTextArea textarea {
    font-family: var(--mono) !important;
    font-size: 15px !important;
    background: var(--cream) !important;
    border: 3px solid var(--ink) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
  }
  .stFileUploader {
    border: 3px dashed var(--ink) !important;
    background: var(--cream) !important;
    border-radius: 0 !important;
  }

  /* Primary button */
  .stButton > button {
    font-family: var(--display) !important;
    letter-spacing: 0.1em !important;
    font-size: 22px !important;
    background: var(--red) !important;
    color: var(--cream) !important;
    border: 3px solid var(--ink) !important;
    border-radius: 0 !important;
    box-shadow: 5px 5px 0 var(--ink) !important;
    padding: 0.4rem 1.2rem !important;
    transition: all 0.08s !important;
  }
  .stButton > button:hover {
    transform: translate(3px,3px) !important;
    box-shadow: 2px 2px 0 var(--ink) !important;
  }

  /* ── MASTHEAD — text side only, Ted handled by st.columns ── */
  .masthead-block {
    background: var(--ink);
    padding: 1.1rem 1.4rem 1rem 1.4rem;
    border-bottom: 6px solid var(--red);
    margin-bottom: 0;
  }
  .masthead-title {
    font-family: var(--display) !important;
    font-size: 44px;
    color: var(--paper) !important;
    letter-spacing: 0.05em;
    line-height: 1;
    text-shadow: 3px 3px 0 var(--red);
    white-space: nowrap;
    margin: 0 0 6px 0;
  }
  .masthead-sub {
    font-family: var(--mono) !important;
    font-size: 11px;
    color: #C8B98A !important;
    line-height: 1.75;
  }
  .masthead-sub a { color: #E8C96A !important; text-decoration: none; }
  .masthead-sub a:hover { text-decoration: underline; }
  .masthead-badge {
    font-family: var(--display) !important;
    font-size: 14px;
    letter-spacing: 0.14em;
    background: var(--red);
    color: var(--cream) !important;
    border: 2px solid var(--paper);
    padding: 3px 14px;
    display: inline-block;
    transform: rotate(-2deg);
    margin-top: 10px;
  }

  /* Ted column — sits inside masthead bg via CSS hack */
  .ted-col-bg {
    background: var(--ink);
    padding: 0.8rem 1rem 0.8rem 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
  }
  .speech-bubble {
    background: var(--cream);
    border: 3px solid var(--ink);
    border-radius: 12px 12px 12px 2px;
    padding: 5px 14px;
    font-family: var(--display) !important;
    font-size: 19px;
    color: var(--ink) !important;
    letter-spacing: 0.04em;
    position: relative;
    margin-bottom: 12px;
    white-space: nowrap;
    display: inline-block;
  }
  /* tail points down toward Ted */
  .speech-bubble::after {
    content: '';
    position: absolute;
    bottom: -13px;
    left: 20px;
    border: 7px solid transparent;
    border-top-color: var(--ink);
  }
  .speech-bubble::before {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 22px;
    border: 5px solid transparent;
    border-top-color: var(--cream);
    z-index: 1;
  }
  .ted-img-frame {
    border: 4px solid var(--paper);
    box-shadow: 4px 4px 0 var(--red);
    overflow: hidden;
    width: 108px;
    height: 108px;
  }
  .ted-img-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .ted-caption {
    font-family: var(--mono) !important;
    font-size: 9px;
    color: #C8B98A !important;
    text-align: center;
    margin-top: 6px;
    line-height: 1.5;
  }

  /* ── METRICS ── */
  .metric-card {
    border: 3px solid var(--ink);
    background: var(--cream);
    padding: 0.8rem 1rem;
    box-shadow: 4px 4px 0 var(--ink);
  }
  .metric-label {
    font-family: var(--mono) !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7A6540 !important;
    margin-bottom: 2px;
  }
  .metric-value {
    font-family: var(--display) !important;
    font-size: 34px;
    color: var(--ink) !important;
    line-height: 1;
  }
  .metric-sub {
    font-family: var(--mono) !important;
    font-size: 10px;
    color: #9A7F52 !important;
    margin-top: 3px;
  }

  /* ── SECTION RULE ── */
  .section-rule {
    font-family: var(--display) !important;
    font-size: 26px;
    letter-spacing: 0.06em;
    color: var(--ink) !important;
    border-top: 4px double var(--ink);
    border-bottom: 1px solid var(--ink);
    padding: 5px 0;
    margin: 1.8rem 0 1rem 0;
  }

  /* ── CODEX CARDS ── */
  .codex-card {
    border: 3px solid var(--ink);
    background: var(--cream);
    box-shadow: 5px 5px 0 var(--ink);
    margin-bottom: 1.5rem;
    overflow: hidden;
  }
  .codex-header {
    padding: 0.55rem 0.9rem;
    border-bottom: 3px solid var(--ink);
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .codex-header-landmark { background: var(--red); }
  .codex-header-full     { background: var(--green); }
  .codex-header-hybrid   { background: var(--blue); }
  .codex-header-title {
    font-family: var(--display) !important;
    font-size: 20px;
    color: var(--cream) !important;
    letter-spacing: 0.05em;
  }
  .codex-header-cer {
    font-family: var(--mono) !important;
    font-size: 12px;
    color: rgba(254,251,240,0.8) !important;
  }
  .codex-body {
    padding: 0.8rem 0.9rem;
    display: flex;
    gap: 0.9rem;
  }
  .codex-matrix {
    font-family: var(--mono) !important;
    font-size: 11px;
    line-height: 1.5;
    color: var(--ink) !important;
    background: var(--paper);
    border: 2px solid var(--ink);
    padding: 0.5rem 0.7rem;
    white-space: pre;
    flex-shrink: 0;
  }
  .codex-right { flex: 1; }
  .codex-usecase {
    font-family: var(--body) !important;
    font-size: 13px;
    color: var(--ink) !important;
    line-height: 1.7;
    margin-bottom: 0.5rem;
    border-left: 4px solid var(--ink);
    padding-left: 0.6rem;
  }
  .codex-pairs-label {
    font-family: var(--mono) !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7A6540 !important;
    margin: 0.5rem 0 4px 0;
  }
  .codex-badge {
    display: inline-block;
    font-family: var(--mono) !important;
    font-size: 11px;
    padding: 1px 8px;
    border: 2px solid var(--ink);
    margin: 2px;
    color: var(--ink) !important;
  }
  .codex-badge-landmark { background: #FAECE7; }
  .codex-badge-full     { background: #E1F5EE; }
  .codex-badge-hybrid   { background: #E6F1FB; }
  .codex-risk {
    font-family: var(--mono) !important;
    font-size: 11px;
    margin-top: 0.6rem;
    padding: 3px 10px;
    border: 2px solid var(--ink);
    display: inline-block;
    color: var(--ink) !important;
  }
  .risk-high     { background: #FAECE7; }
  .risk-moderate { background: #FFF8E1; }
  .risk-low      { background: #E1F5EE; }

  /* ── RESULT BOX ── */
  .result-box {
    border: 3px solid var(--ink);
    background: var(--cream);
    padding: 1.2rem 1.4rem;
    box-shadow: 5px 5px 0 var(--ink);
    margin-top: 1rem;
  }
  .result-label {
    font-family: var(--mono) !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7A6540 !important;
    margin-bottom: 3px;
  }
  .firing-pair {
    display: inline-block;
    background: #FAECE7;
    font-family: var(--mono) !important;
    font-size: 13px;
    padding: 2px 9px;
    border: 2px solid var(--red);
    margin: 2px;
    color: var(--ink) !important;
  }

  /* ── CONFIDENCE ── */
  .confidence-number {
    font-family: var(--display) !important;
    font-size: 64px;
    color: var(--ink) !important;
    line-height: 1;
  }
  .confidence-blurb {
    font-family: var(--body) !important;
    font-size: 15px;
    line-height: 1.9;
    color: #3A2E1A !important;
    border-left: 5px solid var(--ink);
    padding-left: 1rem;
    margin-top: 0.6rem;
  }
  .stamp {
    display: inline-block;
    border: 4px solid var(--red);
    color: var(--red) !important;
    font-family: var(--display) !important;
    font-size: 24px;
    letter-spacing: 0.18em;
    padding: 3px 16px;
    transform: rotate(-7deg);
    opacity: 0.9;
    margin-left: 14px;
    vertical-align: middle;
  }

  /* ── FOOTER ── */
  .footer-strip {
    border-top: 4px solid var(--ink);
    padding-top: 8px;
    margin-top: 2.5rem;
    font-family: var(--mono) !important;
    font-size: 10px;
    color: #7A6540 !important;
    line-height: 1.8;
  }

  div[data-testid="stHorizontalBlock"] { gap: 1.2rem; }
  .stAlert { border-radius: 0 !important; border: 2px solid var(--ink) !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_sheets_client():
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = ["https://spreadsheets.google.com/feeds",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception:
        return None

def log_diagnosis(user_input: str, result: dict):
    try:
        client = get_sheets_client()
        if not client:
            return
        sheet_id = st.secrets.get("SHEETS_ID", "")
        if not sheet_id:
            return
        ws = client.open_by_key(sheet_id).worksheet("diagnoses")
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
        if not client:
            return 0
        sheet_id = st.secrets.get("SHEETS_ID", "")
        if not sheet_id:
            return 0
        ws = client.open_by_key(sheet_id).worksheet("diagnoses")
        return max(0, len(ws.get_all_values()) - 1)
    except Exception:
        return 0

@st.cache_data
def load_ted_b64() -> str:
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

def codex_card(regime, cer, matrix_ascii, use_cases, pairs, risk, risk_class):
    labels = {
        "landmark": "① LANDMARK-ONLY  ·  21-Point Hand Keypoints",
        "full":     "② FULL LANDMARKS  ·  Hands + Face + Body",
        "hybrid":   "③ HYBRID  ·  RGB Frames + Pose",
    }
    badge_class  = f"codex-badge-{regime}"
    header_class = f"codex-header-{regime}"
    pairs_html   = " ".join(
        f'<span class="codex-badge {badge_class}">{p}</span>' for p in pairs
    )
    uc_html = "<br>".join(f"· {u}" for u in use_cases)
    st.markdown(f"""
    <div class="codex-card">
      <div class="codex-header {header_class}">
        <span class="codex-header-title">{labels[regime]}</span>
        <span class="codex-header-cer">CER {cer}</span>
      </div>
      <div class="codex-body">
        <div class="codex-matrix">{matrix_ascii}</div>
        <div class="codex-right">
          <div class="codex-usecase">{uc_html}</div>
          <div class="codex-pairs-label">Confusion signature</div>
          {pairs_html}
          <br>
          <span class="codex-risk {risk_class}">{risk}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    api_key         = get_api_key()
    diagnosis_count = get_diagnosis_count()
    ted_b64         = load_ted_b64()

    # ── MASTHEAD ──────────────────────────────────────────────────────────────
    # Text block rendered as HTML; Ted rendered via st.columns to avoid
    # f-string HTML injection issues with base64 image data.

    col_text, col_ted = st.columns([5, 1])

    with col_text:
        st.markdown("""
        <div class="masthead-block">
          <div class="masthead-title">💍 THE PROVENANCE DECODER RING</div>
          <div class="masthead-sub">
            ASL MODEL TRAINING DATA DIAGNOSTICS &nbsp;·&nbsp;
            READ THE CONFUSION MATRIX &nbsp;·&nbsp;
            KNOW WHAT YOUR MODEL LEARNED<br>
            Audited subject: <strong style="color:#E8C96A;">Gemma 4 E4B</strong>
            &nbsp;·&nbsp; Diagnostic engine: Gemma 2 Instruct
            &nbsp;·&nbsp; Mantel r=0.945
            &nbsp;·&nbsp; <a href="https://github.com/phinnphace/asl-sovereign">asl-sovereign ↗</a>
            &nbsp;·&nbsp; <a href="https://arxiv.org/abs/2407.15806">FSboard paper ↗</a>
          </div>
          <span class="masthead-badge">CLASSIFIED INSTRUMENT</span>
        </div>
        """, unsafe_allow_html=True)

    with col_ted:
        # Dark background to match masthead
        st.markdown(
            '<div style="background:#1A1209;padding:0.8rem 0.5rem 0.8rem 0;'
            'height:100%;display:flex;flex-direction:column;'
            'align-items:center;justify-content:center;">',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="speech-bubble">Ted said so.</div>',
            unsafe_allow_html=True
        )
        if ted_b64:
            st.markdown(
                f'<div class="ted-img-frame">'
                f'<img src="data:image/jpeg;base64,{ted_b64}" alt="Ted">'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="ted-img-frame" style="display:flex;align-items:center;'
                'justify-content:center;background:#2A2010;">'
                '<span style="font-family:\'Bangers\',cursive;font-size:36px;'
                'color:#F5EDD6;">TED</span></div>',
                unsafe_allow_html=True
            )
        st.markdown(
            '<div class="ted-caption">TED · STOOP TABBY<br>'
            'Certified this pipeline.<br>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── METRICS ───────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, value, sub) in zip([m1, m2, m3, m4], [
        ("Cross-dataset stability", "r=0.945", "Mantel · Roboflow vs ISL · p<0.001"),
        ("Bootstrap persistence",  "100%",     "5 shared pairs · 1,000 iterations"),
        ("Images audited",         "2,370",    "ASL + ISL · 24 static letters"),
        ("Diagnoses run",
         str(diagnosis_count) if diagnosis_count > 0 else "—",
         "flywheel · grows with use"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DIAGNOSTIC INPUT ──────────────────────────────────────────────────────
    st.markdown('<div class="section-rule">① SEND YOUR SIGNAL</div>', unsafe_allow_html=True)
    st.markdown(
        "Share what your experiencing in your own way, plain language, letter patterns, "
        "You can also upload your data below. The ring decodes it."
    )

    user_input = st.text_area(
        "Input:",
        placeholder=(
            '"K always comes back as V — never gets it right"\n'
            '"M, N, and T are bleeding together"\n'
            '"Sees the hand clearly. Confidently wrong."'
        ),
        height=95,
        label_visibility="collapsed",
    )

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        diagnose_clicked = st.button("💍 DECODE", type="primary", use_container_width=True)
    with col_note:
        if not api_key:
            st.warning("⚠️ No API key — add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`")

    if diagnose_clicked:
        if not user_input.strip():
            st.warning("Transmit a signal first.")
        elif not api_key:
            st.error("No API key configured.")
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
                firing = result.get("firing_pairs", [])
                if firing:
                    st.markdown('<div class="result-label">Firing pairs</div>', unsafe_allow_html=True)
                    st.markdown(
                        " ".join(f'<span class="firing-pair">{p}</span>' for p in firing),
                        unsafe_allow_html=True
                    )
            with r2:
                risk = result.get("deployment_risk", "—")
                risk_class = ("risk-high" if "HIGH" in risk.upper()
                              else "risk-moderate" if "MODERATE" in risk.upper()
                              else "risk-low")
                st.markdown('<div class="result-label">Deployment risk</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="codex-risk {risk_class}">{risk}</span>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="result-label">Diagnosis</div>', unsafe_allow_html=True)
                st.markdown(result.get("provenance_diagnosis", "—"))
                st.markdown('<div class="result-label">Recommendation</div>', unsafe_allow_html=True)
                st.error(result.get("recommendation", "—"))
            st.markdown('</div>', unsafe_allow_html=True)

            if "raw_output" in result:
                with st.expander("Raw transmission"):
                    st.code(result["raw_output"])

    # ── CODEX + UPLOAD ────────────────────────────────────────────────────────
    st.markdown('<div class="section-rule">② THE CONFUSION CODEX</div>', unsafe_allow_html=True)
    st.markdown(
        "For the analogue folks — here are the confusion matrices. "
        "Find your model's pattern and look it up."
    )

    col_lookup, col_upload = st.columns([3, 2])

    with col_lookup:
        codex_card(
            regime="landmark",
            cer="16.7%",
            matrix_ascii=(
                "  Pred→  M   N   T   A   S\n"
                "True↓  ┌─────────────────┐\n"
                "  M    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n"
                "  N    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n"
                "  T    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n"
                "  A    │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n"
                "  S    │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n"
                "       └─────────────────┘\n"
                "  ▓ high confusion  ░ low"
            ),
            use_cases=[
                "Beginner / demo systems · controlled setting only",
                "Expect consistent confusion pairs — they will not resolve",
                "NOT suitable for real ASL, continuous signing, or sentences",
            ],
            pairs=["K→V (total)", "D→I", "M↔N↔T", "A↔S", "G→D", "P→D"],
            risk="⚠ DEPLOYMENT RISK: HIGH  ·  silent systematic failure",
            risk_class="risk-high",
        )

        codex_card(
            regime="full",
            cer="11.1%",
            matrix_ascii=(
                "  Pred→  M   N   T   A   S\n"
                "True↓  ┌─────────────────┐\n"
                "  M    │▓▓▓ ▒▒▒ ▒▒▒ ░░░ ░░░│\n"
                "  N    │▒▒▒ ▓▓▓ ▒▒▒ ░░░ ░░░│\n"
                "  T    │▒▒▒ ▒▒▒ ▓▓▓ ░░░ ░░░│\n"
                "  A    │░░░ ░░░ ░░░ ▓▓▓ ▒▒▒│\n"
                "  S    │░░░ ░░░ ░░░ ▒▒▒ ▓▓▓│\n"
                "       └─────────────────┘\n"
                "  ▓ correct  ▒ some confusion  ░ low"
            ),
            use_cases=[
                "Research-grade ASL recognition",
                "Static dictionary signing — works well",
                "Continuous flow and motion signing — still limited",
            ],
            pairs=["M vs N (reduced)", "co-artic drift", "flow breaks"],
            risk="⚠ DEPLOYMENT RISK: MODERATE-HIGH  ·  signer-context dependent",
            risk_class="risk-moderate",
        )

        codex_card(
            regime="hybrid",
            cer="7.2%",
            matrix_ascii=(
                "  Pred→  M   N   T   A   S\n"
                "True↓  ┌─────────────────┐\n"
                "  M    │▓▓▓ ░░░ ░░░ ░░░ ░░░│\n"
                "  N    │░░░ ▓▓▓ ░░░ ░░░ ░░░│\n"
                "  T    │░░░ ░░░ ▓▓▓ ░░░ ░░░│\n"
                "  A    │░░░ ░░░ ░░░ ▓▓▓ ░░░│\n"
                "  S    │░░░ ░░░ ░░░ ░░░ ▓▓▓│\n"
                "       └─────────────────┘\n"
                "  ▓ correct  ░ low confusion"
            ),
            use_cases=[
                "ASL → English translation pipelines",
                "Full visual + temporal models — the right tool here",
                "Errors are contextual, not geometric — manageable in production",
            ],
            pairs=["context errors", "temporal drift", "sequence breaks"],
            risk="✓ DEPLOYMENT RISK: MODERATE  ·  errors contextual not geometric",
            risk_class="risk-low",
        )

        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
            'color:#9A7F52;margin-top:0.25rem;">'
            'FSboard · Georg et al. 2024 · CC BY 4.0 · arxiv.org/abs/2407.15806'
            '</div>',
            unsafe_allow_html=True
        )

    with col_upload:
        st.markdown("##### Upload your data")
        st.markdown(
            "Drop a confusion matrix, per-letter accuracy CSV, or raw inference log. "
            "No file? Describe the failure in plain language above — that works too."
        )

        uploaded = st.file_uploader(
            "Drop file",
            type=["csv"],
            help=(
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
                        rows = [{"Letter": l,
                                 "Accuracy": f"{d.get('accuracy',0):.1%}",
                                 "Status": ("⚠ Collapse" if d.get('accuracy',0) < 0.05
                                            else "🔴 Low" if d.get('accuracy',0) < 0.20
                                            else "✓")}
                                for l, d in sorted(per_letter.items())]
                        st.dataframe(
                            pd.DataFrame(rows), use_container_width=True, hide_index=True
                        )
                    top_errors = summary.get("top_errors", {})
                    if top_errors and api_key:
                        worst = sorted(
                            top_errors.items(), key=lambda x: len(x[1]), reverse=True
                        )[:3]
                        error_desc = "; ".join(
                            f"{l} confused with {', '.join(v)}" for l, v in worst
                        )
                        auto_input = (
                            f"Per-letter accuracy: overall {acc}%. "
                            f"Top errors: {error_desc}."
                        )
                        with st.spinner("Running decoder ring..."):
                            result = analyze_failure_mode(auto_input, api_key)
                            log_diagnosis(auto_input, result)
                        st.info(f"**Regime:** {result.get('training_regime', '—')}")
                        st.markdown(result.get("provenance_diagnosis", "—"))
                        st.error(f"**Recommendation:** {result.get('recommendation', '—')}")

                elif dtype == "confusion_matrix":
                    st.markdown(
                        f"**Matrix:** {summary['shape'][0]}×{summary['shape'][1]}"
                    )
                    if api_key:
                        matrix_desc = (
                            f"Confusion matrix {summary['shape'][0]} classes: "
                            f"{', '.join(summary.get('labels', [])[:10])}. "
                            "Analyze structure."
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

        st.markdown("<br>")
        st.markdown(
            "**Formats accepted:**\n"
            "- Inference log: `actual_letter`, `model_output`\n"
            "- Per-letter accuracy: `letter`, `accuracy`\n"
            "- Confusion matrix: NxN CSV with letter labels"
        )

    # ── CONFIDENCE ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-rule">③ ON CONFIDENCE</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown("""
        <div class="confidence-number">Accumulating
          <span class="stamp">LIVE</span>
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                    color:#9A7F52;margin-top:4px;">
          construct validity confirmed · absolute accuracy accumulates with use
        </div>
        <div class="confidence-blurb">
          The validation suite confirms this instrument measures something real and stable.
          What it does not yet have is a classification accuracy percentage against
          known-provenance models — because that labeled ground truth doesn't exist yet.<br><br>
          We won't manufacture it. Every diagnosis you submit becomes part of that ground
          truth as provenance becomes known. The confidence number you want is what
          this instrument is designed to produce — honestly, from real use.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if ted_b64:
            st.markdown(
                f'<div style="border:4px solid #1A1209;box-shadow:4px 4px 0 #1A1209;'
                f'overflow:hidden;">'
                f'<img src="data:image/jpeg;base64,{ted_b64}" '
                f'style="width:100%;display:block;" alt="Ted">'
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:9px;'
                f'color:#7A6540;text-align:center;padding:5px;'
                f'background:#E8D9B0;border-top:2px solid #1A1209;">'
                f'Gemma 4 saw Ted via Ollama.<br>'
                f'HuggingFace could not see Ted.'
                f'</div></div>',
                unsafe_allow_html=True
            )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer-strip">
      THE PROVENANCE DECODER RING &nbsp;·&nbsp;
      Audited subject: Gemma 4 E4B (gemma4:e4b-it-q4_K_M via Ollama) &nbsp;·&nbsp;
      Diagnostic engine: Gemma 2 Instruct (Google AI Studio) &nbsp;·&nbsp;
      FSboard: Georg et al. 2024 (CC BY 4.0) · arxiv.org/abs/2407.15806 &nbsp;·&nbsp;
      ISL dataset: Biswas 2024 · doi:10.17632/n34wm8sb3x.1 &nbsp;·&nbsp;
      Mantel r=0.945 p&lt;0.001 &nbsp;·&nbsp;
      Bootstrap 100% · 1,000 iterations
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
