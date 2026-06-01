import React from 'react'
import { Link } from 'react-router-dom'

const F = {
  display: "Bangers, cursive",
  body:    "'Special Elite', cursive",
  mono:    "'Share Tech Mono', monospace",
}
const C = {
  ink:    "#1A1209",
  paper:  "#F5EDD6",
  cream:  "#FEFBF0",
  tan:    "#E8D9B0",
  red:    "#C0281A",
  amber:  "#C47A0A",
  blue:   "#1B4F8A",
  green:  "#1A6B3A",
  dark:   "#0F0A04",
  rust:   "#8B3A1A",
  wood:   "#6B4C2A",
}

// ── Crate / shelf card ────────────────────────────────────────────────────────
function Crate({ label, tag, children, accent = C.wood, open = false }) {
  const [expanded, setExpanded] = React.useState(open)
  return (
    <div style={{
      border: `3px solid ${C.wood}`,
      background: '#1C1208',
      boxShadow: `6px 6px 0 ${C.dark}`,
      marginBottom: '1.25rem',
      overflow: 'hidden',
    }}>
      <div
        onClick={() => setExpanded(e => !e)}
        style={{
          background: accent,
          padding: '0.55rem 1rem',
          borderBottom: `3px solid ${C.wood}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem' }}>
          <span style={{ fontFamily: F.display, fontSize: 20, color: C.tan, letterSpacing: '0.05em' }}>{label}</span>
          {tag && <span style={{ fontFamily: F.mono, fontSize: 10, color: 'rgba(232,217,176,0.7)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{tag}</span>}
        </div>
        <span style={{ fontFamily: F.mono, fontSize: 14, color: C.tan, opacity: 0.8 }}>
          {expanded ? '▼ OPEN' : '▶ OPEN'}
        </span>
      </div>
      {expanded && (
        <div style={{ padding: '0.85rem 1rem', borderTop: `1px solid ${C.wood}` }}>
          {children}
        </div>
      )}
    </div>
  )
}

// ── Placeholder label ─────────────────────────────────────────────────────────
function Placeholder({ text }) {
  return (
    <div style={{
      border: `2px dashed ${C.wood}`,
      padding: '1.5rem',
      textAlign: 'center',
      fontFamily: F.mono,
      fontSize: 12,
      color: '#6B5A3A',
      letterSpacing: '0.08em',
      background: 'rgba(107,76,42,0.08)',
    }}>
      [ {text} ]
    </div>
  )
}

// ── Stat row ──────────────────────────────────────────────────────────────────
function StatRow({ label, value, note }) {
  return (
    <div style={{ display: 'flex', gap: '1rem', alignItems: 'baseline', padding: '0.35rem 0', borderBottom: `1px solid rgba(107,76,42,0.3)` }}>
      <span style={{ fontFamily: F.mono, fontSize: 11, color: '#8B7355', minWidth: 200 }}>{label}</span>
      <span style={{ fontFamily: F.display, fontSize: 18, color: C.tan, letterSpacing: '0.04em' }}>{value}</span>
      {note && <span style={{ fontFamily: F.mono, fontSize: 10, color: '#6B5A3A' }}>{note}</span>}
    </div>
  )
}

// ── Body text ─────────────────────────────────────────────────────────────────
function Body({ children }) {
  return (
    <div style={{ fontFamily: F.body, fontSize: 14, color: '#C8B090', lineHeight: 1.85 }}>
      {children}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
export default function Library() {
  React.useEffect(() => {
    document.body.style.backgroundColor = '#0F0A04'
    document.body.style.backgroundImage = 'repeating-linear-gradient(0deg,transparent,transparent 27px,rgba(107,76,42,0.08) 27px,rgba(107,76,42,0.08) 28px)'
  }, [])

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 1rem 4rem 1rem' }}>

      {/* ── HEADER ── */}
      <div style={{
        background: C.dark,
        borderBottom: `6px solid ${C.wood}`,
        padding: '1rem 1.5rem',
        marginLeft: '-1rem', marginRight: '-1rem',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div>
          <div style={{ fontFamily: F.display, fontSize: 'clamp(24px,3.5vw,42px)', color: C.tan, letterSpacing: '0.05em', lineHeight: 1, textShadow: `3px 3px 0 ${C.rust}` }}>
            📚 TED'S LIBRARY
          </div>
          <div style={{ fontFamily: F.mono, fontSize: 11, color: '#8B7355', marginTop: 6, lineHeight: 1.7 }}>
            THE BACK ROOM · WHERE THE REAL WORK LIVES · METHODOLOGY · DATA · CITATIONS
          </div>
        </div>
        <Link to="/" style={{
          fontFamily: F.mono, fontSize: 11, color: '#8B7355',
          border: `1px solid ${C.wood}`, padding: '4px 12px',
          textDecoration: 'none', letterSpacing: '0.08em',
          background: 'rgba(107,76,42,0.15)',
        }}>
          ← Back to the Ring
        </Link>
      </div>

      {/* ── INTRO ── */}
      <div style={{ padding: '1rem 0 0.5rem 0' }}>
        <Body>
          This is the back room. Everything on the front page is true —
          here's where we show our work. Crates are clickable. Open what you need.
        </Body>
      </div>

      {/* ── CRATES ── */}

      <Crate label="DATASETS" tag="three sources" accent={C.wood} open>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem' }}>
          {[
            { name: 'Roboflow ASL', detail: '~835 valid predictions · 25 letters · real-world diverse · wildtype dataset', link: 'https://universe.roboflow.com/asl-q517z-sd2no' },
            { name: 'ISL Mendeley', detail: '14,300 images · 26 letters · 3 signers · 2 backgrounds · CC BY 4.0 · controlled comparison', link: 'https://doi.org/10.17632/n34wm8sb3x.1' },
            { name: 'Sign MNIST', detail: '720 images · 28×28 grayscale · canonical benchmark · ~45% refusal rate — resolution failure', link: 'https://www.kaggle.com/datamunge/sign-language-mnist' },
          ].map(d => (
            <div key={d.name} style={{ border: `2px solid ${C.wood}`, padding: '0.7rem', background: 'rgba(107,76,42,0.1)' }}>
              <div style={{ fontFamily: F.display, fontSize: 16, color: C.tan, marginBottom: 4 }}>{d.name}</div>
              <div style={{ fontFamily: F.body, fontSize: 12, color: '#9A7F52', lineHeight: 1.6, marginBottom: 6 }}>{d.detail}</div>
              <a href={d.link} target="_blank" rel="noreferrer" style={{ fontFamily: F.mono, fontSize: 10, color: C.amber, letterSpacing: '0.06em' }}>SOURCE ↗</a>
            </div>
          ))}
        </div>
      </Crate>

      <Crate label="PAPERS + CITATIONS" tag="standing on the shoulders of" accent="#3A2810">
        <Body>
          <div style={{ marginBottom: '0.75rem' }}>
            <span style={{ fontFamily: F.display, fontSize: 16, color: C.tan, display: 'block', marginBottom: 3 }}>FSboard — Georg et al. 2024</span>
            The reference framework. Over 3 million characters of ASL fingerspelling collected via smartphones. Provides the three confusion matrix fingerprints used in this instrument.
            CC BY 4.0 · <a href="https://arxiv.org/abs/2407.15806" style={{ color: C.amber }}>arxiv.org/abs/2407.15806</a>
          </div>
          <div style={{ marginBottom: '0.75rem' }}>
            <span style={{ fontFamily: F.display, fontSize: 16, color: C.tan, display: 'block', marginBottom: 3 }}>WAR Strategy — Abd Al-Latief et al. 2024</span>
            100% accuracy on curated ASL datasets via 8 preprocessing steps + metaheuristic optimization. Illustrates the gap between benchmark accuracy and practical performance.
            <a href="https://doi.org/10.1007/s44227-024-00039-8" style={{ color: C.amber }}> doi ↗</a>
          </div>
          <div style={{ marginBottom: '0.75rem' }}>
            <span style={{ fontFamily: F.display, fontSize: 16, color: C.tan, display: 'block', marginBottom: 3 }}>Fingerspelling PoseNet — Fayyazsanavi et al. 2024</span>
            Improved continuous fingerspelling translation using pose-based transformer models. WACV 2024.
            <a href="https://openaccess.thecvf.com/content/WACV2024W/WVLL/papers/Fayyazsanavi_Fingerspelling_PoseNet_Enhancing_Fingerspelling_Translation_With_Pose-Based_Transformer_Models_WACVW_2024_paper.pdf" style={{ color: C.amber }}> PDF ↗</a>
          </div>
          <div style={{ marginBottom: '0.75rem' }}>
            <span style={{ fontFamily: F.display, fontSize: 16, color: C.tan, display: 'block', marginBottom: 3 }}>Real-Time ASL via Deep Learning — Alsharif et al. 2025</span>
            Sensors 25(7). Documents M/N/T and A/T confusion pairs in landmark-only systems.
            <a href="https://doi.org/10.3390/s25072138" style={{ color: C.amber }}> doi ↗</a>
          </div>
          <div>
            <span style={{ fontFamily: F.display, fontSize: 16, color: C.tan, display: 'block', marginBottom: 3 }}>3D Camera ISL — Ulrich et al. 2026</span>
            Sensors 26(3). Depth improves separability of geometrically similar hand configurations.
            <a href="https://doi.org/10.3390/s26031059" style={{ color: C.amber }}> doi ↗</a>
          </div>
        </Body>
      </Crate>

      <Crate label="INITIAL TESTS" tag="confusion matrices · first results" accent="#2A1F0E">
        <div style={{ marginBottom: '1rem' }}>
          <StatRow label="Roboflow · simple prompt · temp 0.1" value="19.6%" note="835 images · 25 letters" />
          <StatRow label="ISL · simple prompt · temp 0.1"      value="19.4%" note="1,187 images · 25 letters · 0 refusals" />
          <StatRow label="Sign MNIST · simple prompt"          value="6.9%"  note="720 images · ~42% refusal · resolution failure" />
          <StatRow label="Roboflow · elaborate prompt · temp 1.0" value="14.9%" note="original run · 17.3% refusal" />
        </div>
        <Placeholder text="CONFUSION MATRIX IMAGES — cm_roboflow_normalized.png · decoder_ring_isl.png" />
      </Crate>

      <Crate label="PUTTING IT TOGETHER WITH FSBOARD" tag="the decoder ring hypothesis" accent="#1A2A1A">
        <Body>
          The systematic misidentification patterns — K always becoming V, D/T/X always becoming I —
          suggested a structural explanation. The Google/Kaggle ASL fingerspelling competition used
          MediaPipe landmark points rather than raw images for training. If models trained on that
          competition data learned ASL handshapes as geometric point configurations rather than visual
          patterns, the failure modes would be predictable.<br /><br />
          FSboard's ablation study is the key: when you systematically remove non-hand information
          from a full multimodal system, the blocky confusion structure re-emerges. M/N/T cluster
          back together. A/S ambiguity comes back. The confusion clusters are not inherent to ASL —
          they are artifacts of limited representation.<br /><br />
          The FSboard confusion matrices are reference fingerprints. Any model's confusion matrix
          can be compared against these fingerprints to read backwards toward what information
          the model was actually operating on.
        </Body>
        <div style={{ marginTop: '1rem' }}>
          <Placeholder text="FSboard spectrum chart · spectrum scores: Roboflow 0.196 · ISL 0.199" />
        </div>
      </Crate>

      <Crate label="GEMMA 4 E4B + OLLAMA" tag="the specimen · the pipeline" accent="#1A1A2A">
        <Body>
          Gemma 4 E4B is the audited subject — not the diagnostic engine. It is a full
          vision-language model that produces blocky confusion clusters statistically
          indistinguishable from landmark-only models. This should not happen.
          The confusion matrix is the model card we were not given.<br /><br />
          HuggingFace + bitsandbytes NF4 quantization silently corrupts Gemma 4's vision encoder.
          The llm_int8_skip_modules parameter does not correctly map to Gemma 4's internal layer names.
          Every image returns "gray static noise." Confirmed via Ted — a neighbor's tabby cat whose
          visibility to the model established correct vision encoder function.<br /><br />
          Ollama uses GGUF format with llama.cpp. Pre-baked quantization preserves the vision encoder.
          Pipeline validation: Ted was described in precise, accurate detail. The pipeline worked.
        </Body>
        <div style={{ marginTop: '1rem' }}>
          <StatRow label="Model" value="gemma4:e4b-it-q4_K_M" note="via Ollama · local Windows machine" />
          <StatRow label="Prompt" value="Simple" note="What letter of the ASL alphabet does this hand show? Answer with just the letter." />
          <StatRow label="Temperature" value="0.1" note="lower = more consistent · elaborate prompt = 0% on letter A" />
          <StatRow label="conda env" value="asl_gemma4" note="Python 3.11 · torch 2.5.1+cu121 · never use asl_sovereign" />
        </div>
        <div style={{ marginTop: '1rem' }}>
          <a href="https://github.com/phinnphace/asl-sovereign/blob/main/comeedyofgemmaerrors.md"
            target="_blank" rel="noreferrer"
            style={{ fontFamily: F.mono, fontSize: 11, color: C.amber, letterSpacing: '0.06em' }}>
            THE FULL COMEDY OF ERRORS → comeedyofgemmaerrors.md ↗
          </a>
        </div>
      </Crate>

      <Crate label="VALIDATION SUITE" tag="six tests · all passed" accent="#2A1A0E">
        <div style={{ marginBottom: '0.5rem' }}>
          <StatRow label="Bootstrap jiggle · 1,000 iterations"   value="100%"   note="5 shared pairs · K→V mean=1.0 CV=0.0" />
          <StatRow label="Gaussian noise probe · σ 0.0–0.5"      value="PASSED" note="sharp rho drop at σ=0.1 · stable weight core confirmed" />
          <StatRow label="Gray image baseline"                    value="PASSED" note="zero valid predictions · output layer priors ruled out" />
          <StatRow label="Mantel test · 9,999 permutations"       value="r=0.945" note="p&lt;0.001 · Roboflow vs ISL vs FSboard landmark reference" />
          <StatRow label="Thinking mode ablation"                 value="CLOSED" note="COT hurts classification · silent token unresolvable via Ollama" />
          <StatRow label="Token budget ablation"                  value="DOCUMENTED" note="num_ctx does not move prompt_eval_count off 295" />
        </div>
        <div style={{ marginTop: '1rem' }}>
          <Placeholder text="GAUSSIAN NOISE PROBE CHART · rho vs sigma · both datasets" />
        </div>
        <div style={{ marginTop: '0.75rem' }}>
          <Placeholder text="BOOTSTRAP PERSISTENCE TABLE · 5 shared pairs · 9 ISL pairs · 6 Roboflow pairs" />
        </div>
      </Crate>

      <Crate label="GEMMA 2 FINE-TUNING" tag="the diagnostic engine · in progress" accent="#1A2A20">
        <Body>
          Gemma 2 Instruct (Google AI Studio) powers the diagnostic chatbot on the front page.
          It was chosen specifically because it is <em>not</em> the audited subject — the instrument
          and the specimen are different models by design.<br /><br />
          Fine-tuning is underway using real-world failure descriptions collected via the Google Sheet
          flywheel. Approximately 12 validated phrases at time of writing. The model trains as it is used —
          each diagnosis submitted and confirmed becomes a training example.<br /><br />
          The rationale: manufactured ground truth produces manufactured confidence.
          Real-world input produces real confidence. We accumulate, we don't simulate.
        </Body>
        <div style={{ marginTop: '1rem' }}>
          <StatRow label="Base model"          value="Gemma 2 Instruct" note="Google AI Studio · gemma-2-9b-it" />
          <StatRow label="Training phrases"    value="20"               note="v2 dataset · grows with flywheel" />
          <StatRow label="Training approach"   value="Flywheel"         note="user submissions → confirmed → training data" />
          <StatRow label="Fine-tuning notebook" value="gemma2training.ipynb" note="Google Colab · ASL_Project/Drive" />
        </div>
        <div style={{ marginTop: '1rem' }}>
          <Placeholder text="GOOGLE SHEET · live diagnosis log · link pending public share" />
        </div>
      </Crate>

      {/* ── FOOTER ── */}
      <div style={{ borderTop: `3px solid ${C.wood}`, paddingTop: 8, marginTop: '2rem', fontFamily: F.mono, fontSize: 10, color: '#6B5A3A', lineHeight: 1.8 }}>
        TED'S LIBRARY &nbsp;·&nbsp;
        THE PROVENANCE DECODER RING &nbsp;·&nbsp;
        <a href="https://github.com/phinnphace/asl-sovereign" style={{ color: '#6B5A3A' }}>asl-sovereign on GitHub ↗</a>
        &nbsp;·&nbsp;
        <Link to="/" style={{ color: '#6B5A3A' }}>← Back to the Ring</Link>
      </div>
    </div>
  )
}
