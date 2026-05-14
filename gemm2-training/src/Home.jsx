import React, { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'

// Just drop your actual Modal URL right here
const API_BASE = "https://phinnphace--asl-decoder-cloud-fastapi-app.modal.run/api"
const TED_URL  = 'https://raw.githubusercontent.com/phinnphace/asl-sovereign/main/ted.jpg'

// ── Shared styles ─────────────────────────────────────────────────────────────
const F = {
  display: "Bangers, cursive",
  body:    "'Special Elite', cursive",
  mono:    "'Share Tech Mono', monospace",
}
const C = {
  ink:   "#1A1209",
  paper: "#F5EDD6",
  cream: "#FEFBF0",
  tan:   "#E8D9B0",
  red:   "#C0281A",
  amber: "#C47A0A",
  blue:  "#1B4F8A",
  green: "#1A6B3A",
}

// ── Spikey burst ──────────────────────────────────────────────────────────────
function SpikeBurst({ width = 180, color = "#FFE033", text, textColor = C.red }) {
  const pts = Array.from({ length: 20 }, (_, i) => {
    const a = (i / 20) * Math.PI * 2 - Math.PI / 2
    const r = i % 2 === 0 ? 48 : 34
    return `${50 + Math.cos(a) * r},${50 + Math.sin(a) * r}`
  }).join(' ')
  return (
    <svg viewBox="0 0 100 100" width={width} style={{ overflow: 'visible', display: 'block' }}>
      <polygon points={pts} fill={color} stroke={C.ink} strokeWidth="2" />
      {text && (
        <text x="50" y="57" textAnchor="middle"
          fontFamily="Bangers, cursive" fontSize="16"
          fill={textColor} stroke={C.ink} strokeWidth="0.8" paintOrder="stroke"
          letterSpacing="1">{text}</text>
      )}
    </svg>
  )
}

// ── Speech bubble ─────────────────────────────────────────────────────────────
function SpeechBubble({ children }) {
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <div style={{
        background: C.cream,
        border: `3px solid ${C.ink}`,
        borderRadius: '12px 12px 12px 2px',
        padding: '5px 14px',
        fontFamily: F.display,
        fontSize: 18,
        letterSpacing: '0.05em',
        color: C.ink,
        whiteSpace: 'nowrap',
        boxShadow: `3px 3px 0 ${C.ink}`,
        position: 'relative',
      }}>
        {children}
        <span style={{
          position: 'absolute', bottom: -13, left: 16,
          borderLeft: '7px solid transparent',
          borderRight: '7px solid transparent',
          borderTop: `13px solid ${C.ink}`,
        }} />
        <span style={{
          position: 'absolute', bottom: -8, left: 18,
          borderLeft: '5px solid transparent',
          borderRight: '5px solid transparent',
          borderTop: `9px solid ${C.cream}`,
          zIndex: 1,
        }} />
      </div>
    </div>
  )
}

// ── Codex card ────────────────────────────────────────────────────────────────
function CodexCard({ regime, cer, matrix, plain, useCases, pairs, risk, riskLevel }) {
  const hdr = { landmark: C.red, full: C.green, hybrid: C.blue }[regime]
  const bdg = { landmark: '#FAECE7', full: '#E1F5EE', hybrid: '#E6F1FB' }[regime]
  const rc  = { high: { bg: '#FAECE7', fg: C.red }, moderate: { bg: '#FFF8E1', fg: C.amber }, low: { bg: '#E1F5EE', fg: C.green } }[riskLevel]
  const title = {
    landmark: '① LANDMARK-ONLY',
    full:     '② FULL LANDMARKS',
    hybrid:   '③ HYBRID',
  }[regime]

  return (
    <div style={{ border: `3px solid ${C.ink}`, background: C.cream, boxShadow: `5px 5px 0 ${C.ink}`, marginBottom: '1.25rem', overflow: 'hidden' }}>
      <div style={{ background: hdr, padding: '0.5rem 0.9rem', borderBottom: `3px solid ${C.ink}`, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontFamily: F.display, fontSize: 20, color: C.cream, letterSpacing: '0.04em' }}>{title} · CER {cer}</span>
      </div>

      {/* Plain language summary */}
      <div style={{ padding: '0.6rem 0.9rem 0 0.9rem', fontFamily: F.body, fontSize: 15, color: C.ink, lineHeight: 1.7, borderBottom: `2px dashed ${C.tan}` }}>
        {plain}
      </div>

      <div style={{ padding: '0.7rem 0.9rem', display: 'flex', gap: '0.85rem' }}>
        <pre style={{
          fontFamily: F.mono, fontSize: 10.5, lineHeight: 1.5, color: C.ink,
          background: C.paper, border: `2px solid ${C.ink}`,
          padding: '0.45rem 0.6rem', margin: 0, flexShrink: 0, whiteSpace: 'pre',
        }}>{matrix}</pre>
        <div style={{ flex: 1 }}>
          <div style={{ fontFamily: F.mono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', color: '#7A6540', marginBottom: 6 }}>Use when</div>
          <div style={{ fontFamily: F.body, fontSize: 13, lineHeight: 1.7, borderLeft: `4px solid ${C.ink}`, paddingLeft: '0.6rem', marginBottom: 8 }}>
            {useCases.map((u, i) => <div key={i}>· {u}</div>)}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 8 }}>
            {pairs.map((p, i) => (
              <span key={i} style={{ fontFamily: F.mono, fontSize: 11, padding: '1px 8px', border: `2px solid ${C.ink}`, background: bdg }}>{p}</span>
            ))}
          </div>
          <span style={{ fontFamily: F.mono, fontSize: 11, padding: '3px 10px', border: `2px solid ${C.ink}`, background: rc.bg, color: rc.fg, fontWeight: 'bold' }}>{risk}</span>
        </div>
      </div>
    </div>
  )
}

// ── Metric card ───────────────────────────────────────────────────────────────
function MetricCard({ label, value, sub }) {
  return (
    <div style={{ border: `3px solid ${C.ink}`, background: C.cream, padding: '0.75rem 1rem', boxShadow: `4px 4px 0 ${C.ink}` }}>
      <div style={{ fontFamily: F.mono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#7A6540' }}>{label}</div>
      <div style={{ fontFamily: F.display, fontSize: 34, color: C.ink, lineHeight: 1 }}>{value}</div>
      <div style={{ fontFamily: F.mono, fontSize: 10, color: '#9A7F52', marginTop: 3 }}>{sub}</div>
    </div>
  )
}

// ── Chat message ──────────────────────────────────────────────────────────────
function ChatMessage({ role, content, result }) {
  const isUser = role === 'user'
  const regimeColor = r => r === 'LANDMARK-ONLY' ? C.red : r === 'FULL LANDMARKS' ? C.green : C.blue
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: '0.75rem' }}>
      <div style={{
        maxWidth: '78%',
        border: `2px solid ${C.ink}`,
        padding: '0.6rem 0.9rem',
        background: isUser ? C.ink : C.cream,
        boxShadow: isUser ? `3px 3px 0 ${C.red}` : `3px 3px 0 ${C.ink}`,
        fontFamily: F.body,
        fontSize: 14,
        color: isUser ? C.paper : C.ink,
        lineHeight: 1.6,
      }}>
        {content}
        {result && result.training_regime && (
          <div style={{ marginTop: '0.5rem', borderTop: `2px solid ${C.tan}`, paddingTop: '0.5rem' }}>
            <div style={{ fontFamily: F.mono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', color: '#7A6540', marginBottom: 4 }}>Fingerprint detected</div>
            <div style={{ fontFamily: F.display, fontSize: 20, letterSpacing: '0.05em', color: regimeColor(result.training_regime) }}>{result.training_regime}</div>
            {result.firing_pairs?.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginTop: 5 }}>
                {result.firing_pairs.map((p, i) => (
                  <span key={i} style={{ fontFamily: F.mono, fontSize: 11, padding: '1px 7px', border: `2px solid ${C.red}`, background: '#FAECE7' }}>{p}</span>
                ))}
              </div>
            )}
            <div style={{ marginTop: 6, fontFamily: F.body, fontSize: 13, color: '#3A2E1A', lineHeight: 1.6 }}>{result.provenance_diagnosis}</div>
            <div style={{ marginTop: 6, padding: '4px 8px', border: `2px solid ${C.red}`, background: '#FAECE7', fontFamily: F.mono, fontSize: 11, color: C.red }}>
              ⚠ {result.recommendation}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ══════════════════════════════════════════════════════════════════════════════
// HOME PAGE
// ══════════════════════════════════════════════════════════════════════════════
export default function Home() {
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: "Describe what your model gets wrong — plain language works fine. Or upload a CSV of your results. I'll read the fingerprint.",
    result: null,
  }])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [diagCount, setDiagCount] = useState(null)
  const fileRef   = useRef(null)
  const chatEnd   = useRef(null)

  useEffect(() => {
    document.body.style.backgroundColor = C.paper
    document.body.style.backgroundImage = 'repeating-linear-gradient(0deg,transparent,transparent 27px,rgba(26,18,9,0.05) 27px,rgba(26,18,9,0.05) 28px)'
    fetch(`${API_BASE}/count`).then(r => r.json()).then(d => setDiagCount(d.count)).catch(() => {})
  }, [])

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const send = async (text, file = null) => {
    if (!text.trim() && !file) return
    setMessages(p => [...p, { role: 'user', content: file ? `📎 ${file.name}` : text, result: null }])
    setInput('')
    setLoading(true)
    try {
      let result
      if (file) {
        const fd = new FormData(); fd.append('file', file)
        const r = await fetch(`${API_BASE}/decode-csv`, { method: 'POST', body: fd })
        result = await r.json()
      } else {
        const r = await fetch(`${API_BASE}/diagnose`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          // Using user_text to correctly map to our Python Pydantic model
          body: JSON.stringify({ user_text: text }),
        })
        result = await r.json()
      }
      // Using result.diagnosis to pull the exact dictionary key returned by Modal
      setMessages(p => [...p, { role: 'assistant', content: result.diagnosis || 'Decoded.', result }])
      setDiagCount(p => (p || 0) + 1)
    } catch {
      setMessages(p => [...p, { role: 'assistant', content: 'Could not reach the decoder backend. Is it running?', result: null }])
    }
    setLoading(false)
  }

  return (
    <div style={{ maxWidth: 1240, margin: '0 auto', padding: '0 1rem 4rem 1rem' }}>

      {/* ── HEADER ── */}
      <div style={{
        background: C.ink, borderBottom: `6px solid ${C.red}`,
        padding: '1rem 1.5rem 0.9rem 1.5rem',
        marginLeft: '-1rem', marginRight: '-1rem',
        display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
        position: 'relative',
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontFamily: F.display, fontSize: 'clamp(26px,4vw,48px)', color: C.paper, letterSpacing: '0.05em', lineHeight: 1, textShadow: `4px 4px 0 ${C.red}`, whiteSpace: 'nowrap' }}>
            💍 THE PROVENANCE DECODER RING
          </div>
          <div style={{ fontFamily: F.mono, fontSize: 11, color: '#C8B98A', marginTop: 7, lineHeight: 1.75 }}>
            ASL MODEL TRAINING DATA DIAGNOSTICS &nbsp;·&nbsp; READ THE FINGERPRINT &nbsp;·&nbsp; KNOW WHAT YOUR MODEL LEARNED<br />
            Audited subject: <strong style={{ color: '#E8C96A' }}>Gemma 4 E4B</strong>
            &nbsp;·&nbsp; Diagnostic engine: <strong style={{ color: '#E8C96A' }}>Gemma 2 Instruct</strong>
            &nbsp;·&nbsp; <a href="https://github.com/phinnphace/asl-sovereign" style={{ color: '#E8C96A' }}>asl-sovereign ↗</a>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: 10 }}>
            <span style={{ fontFamily: F.display, fontSize: 13, letterSpacing: '0.14em', background: C.red, color: C.cream, border: `2px solid ${C.paper}`, padding: '2px 14px', display: 'inline-block', transform: 'rotate(-2deg)' }}>
              CLASSIFIED INSTRUMENT
            </span>
            <Link to="/library" style={{
              fontFamily: F.mono, fontSize: 11, color: '#C8B98A',
              border: '1px solid #5A4A2A', padding: '2px 10px',
              textDecoration: 'none', letterSpacing: '0.08em',
              transition: 'color 0.15s',
            }}>
              🚪 Ted's Library →
            </Link>
          </div>
        </div>

        {/* Ted */}
        <div style={{ position: 'relative', flexShrink: 0, marginLeft: '1.5rem', marginTop: '-0.25rem' }}>
          <div style={{ position: 'absolute', top: -28, right: -16, width: 70, zIndex: 0, transform: 'rotate(18deg)' }}>
            <SpikeBurst width={70} color="#FFE033" text="POW!" textColor={C.red} />
          </div>
          <div style={{ position: 'relative', zIndex: 2, marginBottom: 8, marginLeft: 4 }}>
            <SpeechBubble>Ted said so.</SpeechBubble>
          </div>
          <div style={{ width: 118, height: 118, border: `4px solid ${C.paper}`, boxShadow: `5px 5px 0 ${C.red}`, overflow: 'hidden', transform: 'rotate(2deg)', position: 'relative', zIndex: 1 }}>
            <img src={TED_URL} alt="Ted" style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'center 30%', display: 'block' }} />
          </div>
          <div style={{ fontFamily: F.mono, fontSize: 9, color: '#C8B98A', textAlign: 'center', marginTop: 5, lineHeight: 1.5 }}>
            TED · STOOP TABBY<br />Certified this pipeline.
          </div>
        </div>
      </div>

      {/* ── METRICS ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '1rem', margin: '1.25rem 0' }}>
        <MetricCard label="Fingerprints matched" value="2,370" sub="images · ASL + ISL · 24 letters" />
        <MetricCard label="Cross-dataset match"  value="r=0.945" sub="same result, different sign languages" />
        <MetricCard label="Stability check"      value="100%"   sub="same fingerprints across 1,000 tests" />
        <MetricCard label="Diagnoses run"         value={diagCount !== null ? diagCount.toLocaleString() : '—'} sub="grows with use" />
      </div>

      {/* ── INTRO ── */}
      <div style={{ border: `3px solid ${C.ink}`, background: C.cream, padding: '1rem 1.25rem', boxShadow: `5px 5px 0 ${C.ink}`, marginBottom: '1.5rem' }}>
        <div style={{ fontFamily: F.display, fontSize: 28, color: C.ink, letterSpacing: '0.04em', marginBottom: 8 }}>
          Training leaves fingerprints.
        </div>
        <div style={{ fontFamily: F.body, fontSize: 15, lineHeight: 1.85, color: '#3A2E1A', maxWidth: 780 }}>
          When a computer vision model is taught to recognize sign language, <em>how</em> it was taught gets baked in.
          A model trained on hand skeleton data makes different mistakes than one trained on real video.
          Those mistakes are predictable. They form a pattern — a fingerprint.<br /><br />
          This tool reads that fingerprint. You describe what's going wrong, or upload your results,
          and we tell you what your model was probably trained on — and what that means for how you use it.
          You don't need to know anything about machine learning to use it.
        </div>
      </div>

      {/* ── SEND YOUR SIGNAL ── */}
      <div style={{ fontFamily: F.display, fontSize: 26, letterSpacing: '0.06em', color: C.ink, borderTop: `4px double ${C.ink}`, borderBottom: `1px solid ${C.ink}`, padding: '5px 0', margin: '0 0 1rem 0' }}>
        ① SEND YOUR SIGNAL
      </div>
      <p style={{ fontFamily: F.body, fontSize: 15, color: '#3A2E1A', marginBottom: '1rem', lineHeight: 1.7 }}>
        Tell us what's going wrong. Plain language is fine. Upload a file if you have one.
        The ring does the rest.
      </p>

      {/* Chat */}
      <div style={{ border: `3px solid ${C.ink}`, background: C.cream, boxShadow: `5px 5px 0 ${C.ink}`, height: 340, overflowY: 'auto', padding: '1rem', marginBottom: '0.75rem' }}>
        {messages.map((m, i) => <ChatMessage key={i} {...m} />)}
        {loading && <div style={{ fontFamily: F.mono, fontSize: 13, color: '#7A6540', padding: '0.5rem' }}>Reading the fingerprint...</div>}
        <div ref={chatEnd} />
      </div>

      <div style={{ display: 'flex', border: `3px solid ${C.ink}`, background: C.cream, boxShadow: `4px 4px 0 ${C.ink}` }}>
        <button onClick={() => fileRef.current?.click()} title="Upload CSV or image"
          style={{ background: C.tan, border: 'none', borderRight: `3px solid ${C.ink}`, padding: '0 0.85rem', cursor: 'pointer', fontSize: 20, flexShrink: 0 }}>
          📎
        </button>
        <input ref={fileRef} type="file" accept=".csv,image/*" style={{ display: 'none' }} onChange={e => { const f = e.target.files?.[0]; if (f) send('', f); e.target.value = '' }} />
        <textarea value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }}
          placeholder='"K always comes back as V" · "M, N, T are bleeding together" · upload a CSV ↑'
          rows={2}
          style={{ flex: 1, fontFamily: F.mono, fontSize: 14, background: 'transparent', border: 'none', outline: 'none', padding: '0.6rem 0.75rem', color: C.ink, resize: 'none', lineHeight: 1.5 }}
        />
        <button onClick={() => send(input)} disabled={loading}
          style={{ fontFamily: F.display, fontSize: 18, letterSpacing: '0.1em', background: loading ? '#9A7F52' : C.red, color: C.cream, border: 'none', borderLeft: `3px solid ${C.ink}`, padding: '0 1.25rem', cursor: loading ? 'not-allowed' : 'pointer', flexShrink: 0 }}>
          💍 DECODE
        </button>
      </div>
      <div style={{ fontFamily: F.mono, fontSize: 10, color: '#9A7F52', marginTop: 5 }}>
        Enter to send · Shift+Enter for new line · 📎 to attach CSV or image
      </div>

      {/* ── CONFUSION CODEX ── */}
      <div style={{ fontFamily: F.display, fontSize: 26, letterSpacing: '0.06em', color: C.ink, borderTop: `4px double ${C.ink}`, borderBottom: `1px solid ${C.ink}`, padding: '5px 0', margin: '2rem 0 1rem 0' }}>
        ② THE CONFUSION CODEX
      </div>
      <p style={{ fontFamily: F.body, fontSize: 15, color: '#3A2E1A', marginBottom: '1.25rem', lineHeight: 1.7 }}>
        Three ways a model gets trained. Three fingerprints. Find yours and look it up —
        analogue style.
      </p>

      <CodexCard
        regime="landmark" cer="16.7%"
        plain="Trained on a 21-point skeleton of the hand. Fast and simple — thinks about hand shape only, nothing else."
        matrix={
          "  Pred→  M   N   T   A   S\n" +
          "True↓  ┌─────────────────┐\n" +
          "  M    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
          "  N    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
          "  T    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
          "  A    │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n" +
          "  S    │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n" +
          "       └─────────────────┘\n" +
          "  ▓ confused  ░ fine"
        }
        useCases={[
          "Demo or classroom projects in controlled settings",
          "Expect the same letters to fail every single time",
          "Not for real-world ASL, sentences, or continuous signing",
        ]}
        pairs={["K→V (always)", "D→I", "M/N/T cluster", "A/S mix-up"]}
        risk="⚠ HIGH RISK · fails silently, no warning"
        riskLevel="high"
      />

      <CodexCard
        regime="full" cer="11.1%"
        plain="Trained on hands, face, and body together. Sees more context — better at understanding how the whole body moves when signing."
        matrix={
          "  Pred→  M   N   T   A   S\n" +
          "True↓  ┌─────────────────┐\n" +
          "  M    │▓▓▓ ▒▒▒ ▒▒▒ ░░░ ░░░│\n" +
          "  N    │▒▒▒ ▓▓▓ ▒▒▒ ░░░ ░░░│\n" +
          "  T    │▒▒▒ ▒▒▒ ▓▓▓ ░░░ ░░░│\n" +
          "  A    │░░░ ░░░ ░░░ ▓▓▓ ▒▒▒│\n" +
          "  S    │░░░ ░░░ ░░░ ▒▒▒ ▓▓▓│\n" +
          "       └─────────────────┘\n" +
          "  ▓ fine  ▒ sometimes wrong  ░ fine"
        }
        useCases={[
          "Research-level ASL recognition",
          "Single signs and static spelling — works well",
          "Continuous flowing signing — still has trouble",
        ]}
        pairs={["M/N sometimes", "flow breaks", "context gaps"]}
        risk="⚠ MODERATE-HIGH RISK · context-dependent"
        riskLevel="moderate"
      />

      <CodexCard
        regime="hybrid" cer="7.2%"
        plain="Trained on actual video frames plus body position data. The most complete picture of how someone signs — mistakes are situational, not structural."
        matrix={
          "  Pred→  M   N   T   A   S\n" +
          "True↓  ┌─────────────────┐\n" +
          "  M    │▓▓▓ ░░░ ░░░ ░░░ ░░░│\n" +
          "  N    │░░░ ▓▓▓ ░░░ ░░░ ░░░│\n" +
          "  T    │░░░ ░░░ ▓▓▓ ░░░ ░░░│\n" +
          "  A    │░░░ ░░░ ░░░ ▓▓▓ ░░░│\n" +
          "  S    │░░░ ░░░ ░░░ ░░░ ▓▓▓│\n" +
          "       └─────────────────┘\n" +
          "  ▓ fine  ░ fine"
        }
        useCases={[
          "ASL to English translation pipelines",
          "Production systems with real signers",
          "When mistakes need to be manageable, not structural",
        ]}
        pairs={["sequence drift", "context errors"]}
        risk="✓ MODERATE RISK · errors are situational"
        riskLevel="low"
      />

      <div style={{ fontFamily: F.mono, fontSize: 10, color: '#9A7F52', marginBottom: '2rem' }}>
        Reference fingerprints from FSboard · Georg et al. 2024 · CC BY 4.0 ·{' '}
        <a href="https://arxiv.org/abs/2407.15806" style={{ color: '#9A7F52' }}>arxiv.org/abs/2407.15806</a>
        {' '}· <Link to="/library" style={{ color: '#9A7F52' }}>Full methodology in Ted's Library →</Link>
      </div>

      {/* ── ON CONFIDENCE — Ted on the shelf ── */}
      <div style={{ position: 'relative', margin: '2.5rem 0 0 0' }}>

        {/* Section rule — Ted sits on this */}
        <div style={{ borderTop: `4px double ${C.ink}`, borderBottom: `1px solid ${C.ink}`, padding: '5px 0', position: 'relative' }}>
          <span style={{ fontFamily: F.display, fontSize: 26, letterSpacing: '0.06em', color: C.ink }}>
            ③ ON CONFIDENCE
          </span>

          {/* Ted on the shelf — absolute positioned to sit ON the rule */}
          <div style={{
            position: 'absolute',
            right: 60,
            top: -130,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}>
            {/* Spikey burst behind bubble */}
            <div style={{ position: 'absolute', top: -18, right: -22, width: 65, zIndex: 0, transform: 'rotate(-12deg)' }}>
              <SpikeBurst width={65} color="#FFE033" />
            </div>
            {/* Bubble */}
            <div style={{ position: 'relative', zIndex: 2, marginBottom: 6 }}>
              <SpeechBubble>Ted says so.</SpeechBubble>
            </div>
            {/* Ted photo — sitting on the rule line */}
            <div style={{
              width: 100, height: 100,
              border: `3px solid ${C.ink}`,
              boxShadow: `4px 4px 0 ${C.red}`,
              overflow: 'hidden',
              transform: 'rotate(-1.5deg)',
              zIndex: 1,
              position: 'relative',
            }}>
              <img src={TED_URL} alt="Ted" style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'center 20%', display: 'block' }} />
            </div>
          </div>
        </div>

        {/* Confidence content */}
        <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: '1.5rem', marginTop: '1rem', alignItems: 'start' }}>
          <div>
            <div style={{ fontFamily: F.display, fontSize: 58, color: C.ink, lineHeight: 1 }}>
              Accumulating
              <span style={{ display: 'inline-block', border: `4px solid ${C.red}`, color: C.red, fontFamily: F.display, fontSize: 22, letterSpacing: '0.18em', padding: '3px 14px', transform: 'rotate(-7deg)', opacity: 0.9, marginLeft: 14, verticalAlign: 'middle' }}>LIVE</span>
            </div>
            <div style={{ fontFamily: F.mono, fontSize: 11, color: '#9A7F52', marginTop: 4 }}>
              validity confirmed · absolute accuracy grows with use
            </div>
            <div style={{ fontFamily: F.body, fontSize: 15, lineHeight: 1.9, color: '#3A2E1A', borderLeft: `5px solid ${C.ink}`, paddingLeft: '1rem', marginTop: '0.75rem' }}>
              We can tell you that this instrument is measuring something real and stable —
              the same fingerprints show up across completely different sign language datasets,
              different signers, different conditions.<br /><br />
              What we can't yet tell you is <em>how often we get it right</em> — because to score that,
              you'd need a library of models with known training histories, and that library doesn't exist yet.
              We refuse to make up a number.<br /><br />
              Every diagnosis you submit helps build that library. The confidence number you want
              is the thing this instrument is designed to produce — honestly, from real use.
            </div>
          </div>
          <div style={{ border: `4px solid ${C.ink}`, boxShadow: `4px 4px 0 ${C.ink}`, overflow: 'hidden' }}>
            <img src={TED_URL} alt="Ted" style={{ width: '100%', display: 'block', objectFit: 'cover', objectPosition: 'center 30%', maxHeight: 200 }} />
            <div style={{ fontFamily: F.mono, fontSize: 9, color: '#7A6540', textAlign: 'center', padding: 5, background: C.tan, borderTop: `2px solid ${C.ink}` }}>
              Gemma 4 saw Ted via Ollama.<br />HuggingFace could not see Ted.
            </div>
          </div>
        </div>
      </div>

      {/* ── FOOTER ── */}
      <div style={{ borderTop: `4px solid ${C.ink}`, paddingTop: 8, marginTop: '2.5rem', fontFamily: F.mono, fontSize: 10, color: '#7A6540', lineHeight: 1.8 }}>
        THE PROVENANCE DECODER RING &nbsp;·&nbsp;
        Audited subject: Gemma 4 E4B (gemma4:e4b-it-q4_K_M via Ollama) &nbsp;·&nbsp;
        Diagnostic engine: Gemma 2 Instruct (Google AI Studio) &nbsp;·&nbsp;
        FSboard: Georg et al. 2024 (CC BY 4.0) &nbsp;·&nbsp;
        ISL dataset: Biswas 2024 · doi:10.17632/n34wm8sb3x.1 &nbsp;·&nbsp;
        <Link to="/library" style={{ color: '#7A6540' }}>Ted's Library →</Link>
      </div>
    </div>
  )
}