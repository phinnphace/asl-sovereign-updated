import React, { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'

const API_BASE = "https://phinnphace--asl-decoder-cloud-fastapi-app.modal.run/api"
const TED_URL  = '/Tedcrop.png'

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

// ── Strip [EVIDENCE_GRID]...[/EVIDENCE_GRID] from bot reply before display ────
function stripEvidenceGrid(text) {
  const idx = text.indexOf('[EVIDENCE_GRID]')
  return idx !== -1 ? text.slice(0, idx).trim() : text
}

// ── File upload helpers ───────────────────────────────────────────────────────
// All uploads are converted to a message string and enter the conversation
// through /api/chat. Qwen2.5-7B is text-only, so images get a verbal prompt
// asking the user to describe what they see — consistent with the listening posture.

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload  = e => resolve(e.target.result)
    reader.onerror = reject
    reader.readAsText(file)
  })
}

function summarizeCSV(content, filename) {
  const lines = content.trim().split('\n').filter(l => l.trim())
  if (lines.length === 0) return `I uploaded ${filename} but it appears to be empty.`
  const headers = lines[0].split(',').map(h => h.trim())
  // Small enough to include in full — let the model read it directly
  if (content.length <= 1800) {
    return `I'm uploading my results as a CSV (${filename}):\n\n${content}`
  }
  // Larger file — send headers + sample rows
  const rowCount   = lines.length - 1
  const sampleRows = lines.slice(1, 5).join('\n')
  return (
    `I'm uploading a CSV (${filename}) — ${rowCount} rows, ${headers.length} columns.\n` +
    `Headers: ${headers.join(', ')}\n` +
    `First rows:\n${sampleRows}`
  )
}

async function processUploadedFile(file) {
  const isImage = file.type.startsWith('image/')
  const isCSV   = file.name.toLowerCase().endsWith('.csv') || file.type === 'text/csv'

  if (isImage) {
    // Intentionally unsupported — see the note in the UI.
    return null
  }

  if (isCSV) {
    try {
      const content = await readFileAsText(file)
      return summarizeCSV(content, file.name)
    } catch {
      return `I tried to upload ${file.name} but couldn't read it.`
    }
  }

  // Any other text-readable format — try it
  try {
    const content = await readFileAsText(file)
    const preview = content.slice(0, 1200)
    return `I'm uploading ${file.name}:\n\n${preview}${content.length > 1200 ? '\n[truncated]' : ''}`
  } catch {
    return `I uploaded ${file.name} — not sure how to read it. Want me to describe what I'm seeing?`
  }
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
  const hdr   = { landmark: C.red, full: C.green, hybrid: C.blue }[regime]
  const bdg   = { landmark: '#FAECE7', full: '#E1F5EE', hybrid: '#E6F1FB' }[regime]
  const rc    = { high: { bg: '#FAECE7', fg: C.red }, moderate: { bg: '#FFF8E1', fg: C.amber }, low: { bg: '#E1F5EE', fg: C.green } }[riskLevel]
  const title = { landmark: '⑀ LANDMARK-ONLY', full: '⑁ FULL LANDMARKS', hybrid: '⑂ HYBRID' }[regime]

  return (
    <div style={{ border: `3px solid ${C.ink}`, background: C.cream, boxShadow: `5px 5px 0 ${C.ink}`, marginBottom: '1.25rem', overflow: 'hidden' }}>
      <div style={{ background: hdr, padding: '0.5rem 0.9rem', borderBottom: `3px solid ${C.ink}`, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontFamily: F.display, fontSize: 20, color: C.cream, letterSpacing: '0.04em' }}>{title} · CER {cer}</span>
      </div>
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
function ChatMessage({ role, content, complete }) {
  const isUser = role === 'user'
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
        whiteSpace: 'pre-wrap',
      }}>
        {content}
        {complete && (
          <div style={{
            marginTop: '0.5rem',
            paddingTop: '0.5rem',
            borderTop: `2px dashed ${C.tan}`,
            fontFamily: F.mono,
            fontSize: 10,
            color: '#7A6540',
            letterSpacing: '0.08em',
          }}>
            ✓ DECODED · LOGGED TO DECODER RING
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// HOME PAGE
// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: "Hey — what's on your mind? Just so you know, I'm a listener bot built to gather real-world experiences with vision models, sort what you tell me, and run it through the decoder ring (check out Ted the cat for the nerdy details on what that means). You get back an answer on what the model or system you've been using was likely trained on — which tells you its limitations and what it's actually built for. And your words make the decoder ring stronger for everyone. I'm here to figure out where the tech is going wrong. You sign how you sign. That's the whole point. So... what's been going on?",
    complete: false,
  }])

  // Full message history sent to Modal on each turn
  const [conversationHistory, setConversationHistory] = useState([])

  const [input,   setInput]   = useState('')
  const [loading, setLoading] = useState(false)
  const chatEnd = useRef(null)
  const fileRef = useRef(null)

  useEffect(() => {
    document.body.style.backgroundColor = C.paper
    document.body.style.backgroundImage = 'repeating-linear-gradient(0deg,transparent,transparent 27px,rgba(26,18,9,0.05) 27px,rgba(26,18,9,0.05) 28px)'
  }, [])

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (text, file = null) => {
    if (!text.trim() && !file) return

    // Convert file to message string if provided
    let messageText = text.trim()
    if (file) messageText = await processUploadedFile(file)
    if (!messageText) return

    // Show filename label for uploads, raw text otherwise
    const displayContent = file ? `📎 ${file.name}\n\n${messageText}` : messageText

    setMessages(p => [...p, { role: 'user', content: displayContent, complete: false }])
    setInput('')
    setLoading(true)

    const updatedHistory = [
      ...conversationHistory,
      { role: 'user', content: messageText },
    ]

    try {
      const r = await fetch(`${API_BASE}/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ messages: updatedHistory }),
      })

      if (!r.ok) throw new Error(`HTTP ${r.status}`)

      const data = await r.json()

      // Strip [EVIDENCE_GRID] block from display — backend data only
      const displayText = stripEvidenceGrid(data.reply)

      setMessages(p => [...p, {
        role:     'assistant',
        content:  displayText,
        complete: data.conversation_complete,
      }])

      // Keep raw reply (with grid markup) in API history for model context
      setConversationHistory([
        ...updatedHistory,
        { role: 'assistant', content: data.reply },
      ])

    } catch {
      setMessages(p => [...p, {
        role:     'assistant',
        content:  'Could not reach the decoder backend. Is it running?',
        complete: false,
      }])
    }

    setLoading(false)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) }
  }

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (file) { await send('', file); e.target.value = '' }
  }

  const resetConversation = () => {
    setConversationHistory([])
    setMessages([{
      role: 'assistant',
      content: "Hey — what's on your mind? Just so you know, I'm a listener bot built to gather real-world experiences with vision models, sort what you tell me, and run it through the decoder ring (check out Ted the cat for the nerdy details on what that means). You get back an answer on what the model or system you've been using was likely trained on — which tells you its limitations and what it's actually built for. And your words make the decoder ring stronger for everyone. I'm here to figure out where the tech is going wrong. You sign how you sign. That's the whole point. So... what's been going on?",
      complete: false,
    }])
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
            📍 THE PROVENANCE DECODER RING
          </div>
          <div style={{ fontFamily: F.mono, fontSize: 11, color: '#C8B98A', marginTop: 7, lineHeight: 1.75 }}>
            YOU DON'T ADJUST TO THIS. &nbsp;·&nbsp; IT COMES TO YOU. &nbsp;·&nbsp; TALK HOWEVER YOU ACTUALLY TALK.<br />
            Diagnostic engine: <strong style={{ color: '#E8C96A' }}>Qwen2.5-7B-Instruct</strong>
            &nbsp;·&nbsp; Audited subject: <strong style={{ color: '#E8C96A' }}>Gemma 4 E4B</strong>
            &nbsp;·&nbsp; <a href="https://github.com/phinnphace/asl-sovereign" style={{ color: '#E8C96A' }}>asl-sovereign ↗</a>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: 10 }}>
            <span style={{ fontFamily: F.display, fontSize: 13, letterSpacing: '0.14em', background: C.red, color: C.cream, border: `2px solid ${C.paper}`, padding: '2px 14px', display: 'inline-block', transform: 'rotate(-2deg)' }}>
              CLASSIFIED INSTRUMENT
            </span>
            <Link to="/library" style={{ fontFamily: F.mono, fontSize: 11, color: '#C8B98A', border: '1px solid #5A4A2A', padding: '2px 10px', textDecoration: 'none', letterSpacing: '0.08em' }}>
              🪶 Ted's Library →
            </Link>
          </div>
        </div>

        {/* Ted in header */}
        <div style={{ position: 'relative', flexShrink: 0, marginLeft: '1.5rem', marginTop: '-0.25rem' }}>
          <div style={{ position: 'absolute', top: -28, right: -16, width: 70, zIndex: 0, transform: 'rotate(18deg)' }}>
            <SpikeBurst width={70} color="#FFE033" text="POW!" textColor={C.red} />
          </div>
          <div style={{ position: 'relative', zIndex: 2, marginBottom: 8, marginLeft: 4 }}>
            <SpeechBubble>Ted said so.</SpeechBubble>
          </div>
          <div style={{ width: 118, overflow: 'visible', transform: 'rotate(2deg)', position: 'relative', zIndex: 1 }}>
            <img src={TED_URL} alt="Ted" style={{ width: '100%', height: 'auto', display: 'block' }} />
          </div>
          <div style={{ fontFamily: F.mono, fontSize: 9, color: '#C8B98A', textAlign: 'center', marginTop: 5, lineHeight: 1.5 }}>
            TED · STOOP TABBY<br />Certified this pipeline.
          </div>
        </div>
      </div>

      {/* ── METRICS ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '1rem', margin: '1.25rem 0' }}>
        <MetricCard label="Diagnostic Confidence" value="Accumulating" sub="grows with user submissions" />
        <MetricCard label="Mantel Correlation"     value="r=0.945"      sub="Roboflow vs ISL · p<0.001" />
        <MetricCard label="Images Audited"         value="2,370"        sub="ASL + ISL · 24 static letters" />
        <MetricCard label="Training Data"          value="Live"         sub="every real submission counts" />
      </div>

      {/* ── INTRO ── */}
      <div style={{ border: `3px solid ${C.ink}`, background: C.cream, padding: '1rem 1.25rem', boxShadow: `5px 5px 0 ${C.ink}`, marginBottom: '1.5rem' }}>
        <div style={{ fontFamily: F.display, fontSize: 28, color: C.ink, letterSpacing: '0.04em', marginBottom: 8 }}>
          Be you so we can get to you.
        </div>
        <div style={{ fontFamily: F.body, fontSize: 15, lineHeight: 1.85, color: '#3A2E1A', maxWidth: 780 }}>
          ASL signers have been doing the adapting for a long time. Tools get built with great benchmark numbers
          — measured on lab conditions, on curated datasets, on hands that happened to be in the room.
          Real signers in real conditions are a different story.<br /><br />
          This is the inverse. You don't adjust to this interface. You talk however you actually talk.
          The model on the other end was trained to meet you there — to read the fingerprint in what you describe
          and tell you what was probably wrong with the training data that got you here.
          No machine learning knowledge required. No formatting required. Just say what's going wrong.
        </div>
      </div>

      {/* ── JUST TALK ── */}
      <div style={{ fontFamily: F.display, fontSize: 26, letterSpacing: '0.06em', color: C.ink, borderTop: `4px double ${C.ink}`, borderBottom: `1px solid ${C.ink}`, padding: '5px 0', margin: '0 0 1rem 0' }}>
        ⑀ JUST TALK
      </div>
      <p style={{ fontFamily: F.body, fontSize: 15, color: '#3A2E1A', marginBottom: '1rem', lineHeight: 1.7 }}>
        Say what's going wrong — however you'd say it to a friend. Upload a CSV or paste your results — or just talk.
        Every real complaint from a real signer makes this better for the next one.
      </p>

      {/* ── CHAT WINDOW ── */}
      <div style={{ border: `3px solid ${C.ink}`, background: C.cream, boxShadow: `5px 5px 0 ${C.ink}`, height: 340, overflowY: 'auto', padding: '1rem', marginBottom: '0.75rem' }}>
        {messages.map((m, i) => (
          <ChatMessage key={i} role={m.role} content={m.content} complete={m.complete} />
        ))}
        {loading && (
          <div style={{ fontFamily: F.mono, fontSize: 13, color: '#7A6540', padding: '0.5rem' }}>
            Reading the fingerprint…
          </div>
        )}
        <div ref={chatEnd} />
      </div>

      {/* ── INPUT ROW ── */}
      <div style={{ display: 'flex', border: `3px solid ${C.ink}`, background: C.cream, boxShadow: `4px 4px 0 ${C.ink}` }}>
        {/* File upload button */}
        <button
          onClick={() => fileRef.current?.click()}
          title="Upload CSV, confusion matrix image, or any results file"
          style={{ background: C.tan, border: 'none', borderRight: `3px solid ${C.ink}`, padding: '0 0.85rem', cursor: 'pointer', fontSize: 20, flexShrink: 0 }}
        >
          📎
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.txt,.json"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={'"K always comes back as V" · "it can\'t tell M from N" · "the whole middle cluster is wrong" · or upload something ↑'}
          rows={2}
          style={{ flex: 1, fontFamily: F.mono, fontSize: 14, background: 'transparent', border: 'none', outline: 'none', padding: '0.6rem 0.75rem', color: C.ink, resize: 'none', lineHeight: 1.5 }}
        />
        <button
          onClick={() => send(input)}
          disabled={loading}
          style={{ fontFamily: F.display, fontSize: 18, letterSpacing: '0.1em', background: loading ? '#9A7F52' : C.red, color: C.cream, border: 'none', borderLeft: `3px solid ${C.ink}`, padding: '0 1.25rem', cursor: loading ? 'not-allowed' : 'pointer', flexShrink: 0 }}
        >
          📍 DECODE
        </button>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 5 }}>
        <div style={{ fontFamily: F.mono, fontSize: 10, color: '#9A7F52' }}>
          Enter to send · Shift+Enter for new line · 📎 to attach CSV or text file
        </div>
        {conversationHistory.length > 0 && (
          <button
            onClick={resetConversation}
            style={{ fontFamily: F.mono, fontSize: 10, color: '#9A7F52', background: 'none', border: `1px solid #C8B98A`, padding: '2px 10px', cursor: 'pointer', letterSpacing: '0.08em' }}
          >
            ↺ NEW CONVERSATION
          </button>
        )}
      </div>

      {/* ── NO VISION MODEL NOTE ── */}
      <div style={{
        display: 'block',
        fontFamily: F.mono,
        fontSize: 10,
        color: C.red,
        border: `1.5px solid ${C.red}`,
        background: '#FAECE7',
        padding: '4px 12px',
        marginTop: 6,
        letterSpacing: '0.06em',
      }}>
        ⚠ Since the entire point of this project is addressing failures of vision models,
        this tool does not use one. JPEG / PNG not supported.
        You can use the Codex here, the chatbot, or upload a dataset.
        If this is still wrong, please say so. Ted knows what he is doing, but I do not.
      </div>

      {/* ── CONFUSION CODEX ── */}
      <div style={{ fontFamily: F.display, fontSize: 26, letterSpacing: '0.06em', color: C.ink, borderTop: `4px double ${C.ink}`, borderBottom: `1px solid ${C.ink}`, padding: '5px 0', margin: '2rem 0 1rem 0' }}>
        ⑁ THE CONFUSION CODEX
      </div>
      <p style={{ fontFamily: F.body, fontSize: 15, color: '#3A2E1A', marginBottom: '1.25rem', lineHeight: 1.7 }}>
        Three ways a model gets trained. Three fingerprints. Find yours and look it up — analogue style.
      </p>

      <CodexCard
        regime="landmark" cer="16.7%"
        plain="Trained on a 21-point skeleton of the hand. Fast and simple — thinks about hand shape only, nothing else."
        matrix={
          "  Pred→  M   N   T   A   S\n" +
          "True↓  ┌──────────────────┐\n" +
          "  M    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
          "  N    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
          "  T    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
          "  A    │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n" +
          "  S    │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n" +
          "       └──────────────────┘\n" +
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
          "True↓  ┌──────────────────┐\n" +
          "  M    │▓▓▓ ▒▒▒ ▒▒▒ ░░░ ░░░│\n" +
          "  N    │▒▒▒ ▓▓▓ ▒▒▒ ░░░ ░░░│\n" +
          "  T    │▒▒▒ ▒▒▒ ▓▓▓ ░░░ ░░░│\n" +
          "  A    │░░░ ░░░ ░░░ ▓▓▓ ▒▒▒│\n" +
          "  S    │░░░ ░░░ ░░░ ▒▒▒ ▓▓▓│\n" +
          "       └──────────────────┘\n" +
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
          "True↓  ┌──────────────────┐\n" +
          "  M    │▓▓▓ ░░░ ░░░ ░░░ ░░░│\n" +
          "  N    │░░░ ▓▓▓ ░░░ ░░░ ░░░│\n" +
          "  T    │░░░ ░░░ ▓▓▓ ░░░ ░░░│\n" +
          "  A    │░░░ ░░░ ░░░ ▓▓▓ ░░░│\n" +
          "  S    │░░░ ░░░ ░░░ ░░░ ▓▓▓│\n" +
          "       └──────────────────┘\n" +
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

      <div style={{ fontFamily: F.mono, fontSize: 10, color: '#9A7F52', marginBottom: '1.5rem' }}>
        Reference fingerprints from FSboard · Georg et al. 2024 · CC BY 4.0 ·{' '}
        <a href="https://arxiv.org/abs/2407.15806" style={{ color: '#9A7F52' }}>arxiv.org/abs/2407.15806</a>
      </div>

      {/* ── LIBRARY DOOR ── */}
      <Link to="/library" style={{ textDecoration: 'none' }}>
        <div style={{
          border: `4px solid ${C.ink}`, background: '#1C1208',
          padding: '1.25rem 1.5rem', boxShadow: `8px 8px 0 ${C.ink}`, marginBottom: '2.5rem',
          display: 'flex', alignItems: 'center', gap: '1.5rem', cursor: 'pointer',
        }}>
          <div style={{ fontSize: 48, flexShrink: 0 }}>🪶</div>
          <div>
            <div style={{ fontFamily: F.display, fontSize: 26, color: '#E8D9B0', letterSpacing: '0.05em', lineHeight: 1.1 }}>TED'S LIBRARY</div>
            <div style={{ fontFamily: F.body, fontSize: 14, color: '#8B7355', marginTop: 4, lineHeight: 1.6 }}>
              Nerdy, technical documentation this way. Datasets, validation tests, citations, the full comedy of errors. Watch your step.
            </div>
            <div style={{ fontFamily: F.mono, fontSize: 10, color: '#6B5A3A', marginTop: 6, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              → Methodology · Data · Gemma 4 Audit · Validation Suite
            </div>
          </div>
        </div>
      </Link>

      {/* ── ON CONFIDENCE ── */}
      <div style={{ position: 'relative', margin: '2.5rem 0 0 0' }}>
        <div style={{ borderTop: `4px double ${C.ink}`, borderBottom: `1px solid ${C.ink}`, padding: '5px 0' }}>
          <span style={{ fontFamily: F.display, fontSize: 26, letterSpacing: '0.06em', color: C.ink }}>⑂ ON CONFIDENCE</span>
        </div>
        <div style={{ marginTop: '1rem' }}>
          <div style={{ fontFamily: F.display, fontSize: 58, color: C.ink, lineHeight: 1 }}>
            Accumulating
            <span style={{ display: 'inline-block', border: `4px solid ${C.red}`, color: C.red, fontFamily: F.display, fontSize: 22, letterSpacing: '0.18em', padding: '3px 14px', transform: 'rotate(-7deg)', opacity: 0.9, marginLeft: 14, verticalAlign: 'middle' }}>LIVE</span>
          </div>
          <div style={{ fontFamily: F.mono, fontSize: 11, color: '#9A7F52', marginTop: 4 }}>
            construct validity confirmed · absolute accuracy accumulates with use
          </div>
          <div style={{ fontFamily: F.body, fontSize: 15, lineHeight: 1.9, color: '#3A2E1A', borderLeft: `5px solid ${C.ink}`, paddingLeft: '1rem', marginTop: '0.75rem', maxWidth: 780 }}>
            The validation suite establishes that this instrument measures something real and stable —
            bootstrap persistence at 1,000 iterations, Gaussian noise probe confirming weight-encoded
            confusion structure, Mantel r=0.945 cross-dataset. What it does not yet have is an absolute
            accuracy percentage against known-provenance models, because a labeled set of models with
            known training regimes does not exist in the literature.<br /><br />
            We can tell you that this instrument is measuring something real. We refuse to make up a number
            for how often we get it right. Every diagnosis submitted helps build that number honestly.
          </div>
        </div>
      </div>

      {/* ── TED ABOVE FOOTER ── */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'flex-end', marginBottom: '-10px', paddingRight: '2rem', position: 'relative', zIndex: 2 }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ position: 'relative', marginBottom: 6 }}>
            <div style={{ position: 'absolute', top: -20, right: -18, width: 60, zIndex: 0, transform: 'rotate(15deg)' }}>
              <SpikeBurst width={60} color="#FFE033" />
            </div>
            <div style={{ position: 'relative', zIndex: 2 }}>
              <SpeechBubble>I said so.</SpeechBubble>
            </div>
          </div>
          <div style={{ width: 110, overflow: 'visible', transform: 'rotate(2deg)' }}>
            <img src={TED_URL} alt="Ted" style={{ width: '100%', height: 'auto', display: 'block' }} />
          </div>
          <div style={{ fontFamily: F.mono, fontSize: 9, color: '#7A6540', textAlign: 'center', marginTop: 4, lineHeight: 1.5 }}>
            TED · STOOP TABBY<br />Certified this pipeline.
          </div>
        </div>
      </div>

      {/* ── FOOTER ── */}
      <div style={{ borderTop: `4px solid ${C.ink}`, paddingTop: 8, marginTop: '0.5rem', fontFamily: F.mono, fontSize: 10, color: '#7A6540', lineHeight: 1.8 }}>
        THE PROVENANCE DECODER RING &nbsp;·&nbsp;
        Audited subject: Gemma 4 E4B (gemma4:e4b-it-q4_K_M via Ollama) &nbsp;·&nbsp;
        Diagnostic engine: Qwen2.5-7B-Instruct &nbsp;·&nbsp;
        FSboard: Georg et al. 2024 (CC BY 4.0) &nbsp;·&nbsp;
        ISL dataset: Biswas 2024 · doi:10.17632/n34w8sb3x.1 &nbsp;·&nbsp;
        <Link to="/library" style={{ color: '#7A6540' }}>Ted's Library →</Link>
      </div>
    </div>
  )
}
