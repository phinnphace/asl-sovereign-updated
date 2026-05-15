import { useState, useEffect, useRef } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TED_URL = "https://raw.githubusercontent.com/phinnphace/asl-sovereign/main/ted.jpg";

// ── Comic burst SVG shapes ────────────────────────────────────────────────────
function Burst({ className, color = "#FFE033", size = 120, label, labelColor = "#C0281A" }) {
  const pts = Array.from({ length: 16 }, (_, i) => {
    const angle = (i / 16) * Math.PI * 2;
    const r = i % 2 === 0 ? size / 2 : size / 2 * 0.72;
    return `${50 + Math.cos(angle) * r},${50 + Math.sin(angle) * r}`;
  }).join(" ");
  return (
    <svg viewBox="0 0 100 100" className={className} style={{ overflow: "visible" }}>
      <polygon points={pts} fill={color} stroke="#1A1209" strokeWidth="2.5" />
      {label && (
        <text x="50" y="56" textAnchor="middle" fontSize="22"
          fontFamily="Bangers, cursive" fill={labelColor}
          stroke="#1A1209" strokeWidth="1" paintOrder="stroke"
          letterSpacing="2">
          {label}
        </text>
      )}
    </svg>
  );
}

// ── Speech bubble ─────────────────────────────────────────────────────────────
function SpeechBubble({ children, className = "" }) {
  return (
    <div className={`relative inline-block ${className}`}>
      <div style={{
        background: "#FEFBF0",
        border: "3px solid #1A1209",
        borderRadius: "14px 14px 14px 2px",
        padding: "6px 16px",
        fontFamily: "Bangers, cursive",
        fontSize: "20px",
        letterSpacing: "0.05em",
        color: "#1A1209",
        whiteSpace: "nowrap",
        position: "relative",
        boxShadow: "3px 3px 0 #1A1209",
      }}>
        {children}
        {/* tail */}
        <span style={{
          position: "absolute", bottom: -14, left: 18,
          width: 0, height: 0,
          borderLeft: "8px solid transparent",
          borderRight: "8px solid transparent",
          borderTop: "14px solid #1A1209",
        }} />
        <span style={{
          position: "absolute", bottom: -9, left: 20,
          width: 0, height: 0,
          borderLeft: "6px solid transparent",
          borderRight: "6px solid transparent",
          borderTop: "10px solid #FEFBF0",
          zIndex: 1,
        }} />
      </div>
    </div>
  );
}

// ── Codex card ────────────────────────────────────────────────────────────────
function CodexCard({ regime, cer, matrix, useCases, pairs, risk, riskLevel }) {
  const colors = {
    landmark: { header: "#C0281A", badge: "#FAECE7" },
    full:     { header: "#1A6B3A", badge: "#E1F5EE" },
    hybrid:   { header: "#1B4F8A", badge: "#E6F1FB" },
  };
  const labels = {
    landmark: "① LANDMARK-ONLY · 21-Point Hand Keypoints",
    full:     "② FULL LANDMARKS · Hands + Face + Body",
    hybrid:   "③ HYBRID · RGB Frames + Pose",
  };
  const riskColors = {
    high:     { bg: "#FAECE7", color: "#C0281A" },
    moderate: { bg: "#FFF8E1", color: "#C47A0A" },
    low:      { bg: "#E1F5EE", color: "#1A6B3A" },
  };
  const rc = riskColors[riskLevel] || riskColors.moderate;
  const c  = colors[regime];

  return (
    <div style={{
      border: "3px solid #1A1209",
      background: "#FEFBF0",
      boxShadow: "5px 5px 0 #1A1209",
      marginBottom: "1.25rem",
      overflow: "hidden",
    }}>
      <div style={{
        background: c.header,
        padding: "0.5rem 0.85rem",
        borderBottom: "3px solid #1A1209",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
      }}>
        <span style={{ fontFamily: "Bangers, cursive", fontSize: 19, color: "#FEFBF0", letterSpacing: "0.04em" }}>
          {labels[regime]}
        </span>
        <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 11, color: "rgba(254,251,240,0.8)" }}>
          CER {cer}
        </span>
      </div>
      <div style={{ padding: "0.75rem 0.85rem", display: "flex", gap: "0.85rem" }}>
        <pre style={{
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: 10.5,
          lineHeight: 1.55,
          color: "#1A1209",
          background: "#F5EDD6",
          border: "2px solid #1A1209",
          padding: "0.45rem 0.6rem",
          margin: 0,
          flexShrink: 0,
          whiteSpace: "pre",
        }}>{matrix}</pre>
        <div style={{ flex: 1 }}>
          <div style={{
            borderLeft: "4px solid #1A1209",
            paddingLeft: "0.6rem",
            fontFamily: "'Special Elite', cursive",
            fontSize: 13,
            lineHeight: 1.7,
            marginBottom: "0.5rem",
          }}>
            {useCases.map((u, i) => <div key={i}>· {u}</div>)}
          </div>
          <div style={{
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: 10,
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            color: "#7A6540",
            marginBottom: 4,
          }}>Confusion signature</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginBottom: 8 }}>
            {pairs.map((p, i) => (
              <span key={i} style={{
                fontFamily: "'Share Tech Mono', monospace",
                fontSize: 11,
                padding: "1px 8px",
                border: "2px solid #1A1209",
                background: c.badge,
              }}>{p}</span>
            ))}
          </div>
          <span style={{
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: 11,
            padding: "3px 10px",
            border: "2px solid #1A1209",
            background: rc.bg,
            color: rc.color,
            fontWeight: "bold",
          }}>{risk}</span>
        </div>
      </div>
    </div>
  );
}

// ── Metric card ───────────────────────────────────────────────────────────────
function MetricCard({ label, value, sub }) {
  return (
    <div style={{
      border: "3px solid #1A1209",
      background: "#FEFBF0",
      padding: "0.75rem 1rem",
      boxShadow: "4px 4px 0 #1A1209",
    }}>
      <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.12em", color: "#7A6540" }}>{label}</div>
      <div style={{ fontFamily: "Bangers, cursive", fontSize: 34, color: "#1A1209", lineHeight: 1 }}>{value}</div>
      <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 10, color: "#9A7F52", marginTop: 3 }}>{sub}</div>
    </div>
  );
}

// ── Chat message ──────────────────────────────────────────────────────────────
function ChatMessage({ role, content, result }) {
  const isUser = role === "user";
  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: "0.75rem",
    }}>
      <div style={{
        maxWidth: "75%",
        border: "2px solid #1A1209",
        padding: "0.6rem 0.85rem",
        background: isUser ? "#1A1209" : "#FEFBF0",
        boxShadow: isUser ? "3px 3px 0 #C0281A" : "3px 3px 0 #1A1209",
        fontFamily: "'Special Elite', cursive",
        fontSize: 14,
        color: isUser ? "#F5EDD6" : "#1A1209",
        lineHeight: 1.6,
      }}>
        {content}
        {result && (
          <div style={{ marginTop: "0.5rem", borderTop: "2px solid #1A1209", paddingTop: "0.5rem" }}>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em", color: "#7A6540", marginBottom: 4 }}>
              Regime Detected
            </div>
            <div style={{
              fontFamily: "Bangers, cursive", fontSize: 18, letterSpacing: "0.05em",
              color: result.training_regime === "LANDMARK-ONLY" ? "#C0281A" :
                     result.training_regime === "FULL LANDMARKS" ? "#1A6B3A" : "#1B4F8A",
            }}>{result.training_regime}</div>
            {result.firing_pairs?.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginTop: 6 }}>
                {result.firing_pairs.map((p, i) => (
                  <span key={i} style={{
                    fontFamily: "'Share Tech Mono', monospace", fontSize: 11,
                    padding: "1px 7px", border: "2px solid #C0281A",
                    background: "#FAECE7", color: "#1A1209",
                  }}>{p}</span>
                ))}
              </div>
            )}
            <div style={{ marginTop: 6, fontFamily: "'Special Elite', cursive", fontSize: 12, color: "#3A2E1A" }}>
              {result.provenance_diagnosis}
            </div>
            <div style={{
              marginTop: 6, padding: "4px 8px",
              border: "2px solid #C0281A", background: "#FAECE7",
              fontFamily: "'Share Tech Mono', monospace", fontSize: 11, color: "#C0281A",
            }}>
              ⚠ {result.recommendation}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN APP
// ══════════════════════════════════════════════════════════════════════════════

export default function App() {
  const [messages, setMessages]     = useState([{
    role: "assistant",
    content: "Describe your model's failure — plain language, letter patterns, or upload a CSV. I'll decode it.",
    result: null,
  }]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [diagCount, setDiagCount]   = useState(null);
  const [activeTab, setActiveTab]   = useState("chat"); // chat | codex | confidence
  const fileRef                     = useRef(null);
  const chatEndRef                  = useRef(null);

  useEffect(() => {
    // Load fonts
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Bangers&family=Special+Elite&family=Share+Tech+Mono&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);

    // Paper texture background
    document.body.style.backgroundColor = "#F5EDD6";
    document.body.style.backgroundImage = "repeating-linear-gradient(0deg,transparent,transparent 27px,rgba(26,18,9,0.05) 27px,rgba(26,18,9,0.05) 28px)";
    document.body.style.margin = 0;

    // Fetch diagnosis count
    fetch(`${API_BASE}/count`).then(r => r.json()).then(d => setDiagCount(d.count)).catch(() => {});
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text, file = null) => {
    if (!text.trim() && !file) return;
    const userMsg = { role: "user", content: file ? `📎 ${file.name}` : text, result: null };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      let result;
      if (file) {
        const fd = new FormData();
        fd.append("file", file);
        const r = await fetch(`${API_BASE}/decode-csv`, { method: "POST", body: fd });
        result = await r.json();
      } else {
        const r = await fetch(`${API_BASE}/diagnose`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        result = await r.json();
      }
      setMessages(prev => [...prev, { role: "assistant", content: result.summary || "Decoded.", result }]);
      setDiagCount(prev => (prev || 0) + 1);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Could not reach the decoder. Check that the backend is running.",
        result: null,
      }]);
    }
    setLoading(false);
  };

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) sendMessage("", file);
    e.target.value = "";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div style={{ maxWidth: 1240, margin: "0 auto", padding: "0 1rem 3rem 1rem" }}>

      {/* ── HEADER ── */}
      <div style={{
        background: "#1A1209",
        borderBottom: "6px solid #C0281A",
        padding: "1rem 1.4rem 0.8rem 1.4rem",
        marginLeft: "-1rem",
        marginRight: "-1rem",
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "space-between",
        position: "relative",
        overflow: "visible",
      }}>
        {/* Title block */}
        <div style={{ flex: 1 }}>
          <div style={{
            fontFamily: "Bangers, cursive",
            fontSize: "clamp(28px, 4vw, 48px)",
            color: "#F5EDD6",
            letterSpacing: "0.05em",
            lineHeight: 1,
            textShadow: "4px 4px 0 #C0281A",
            whiteSpace: "nowrap",
          }}>
            💍 THE PROVENANCE DECODER RING
          </div>
          <div style={{
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: 11,
            color: "#C8B98A",
            marginTop: 7,
            lineHeight: 1.75,
          }}>
            ASL MODEL TRAINING DATA DIAGNOSTICS &nbsp;·&nbsp;
            READ THE CONFUSION MATRIX &nbsp;·&nbsp;
            KNOW WHAT YOUR MODEL LEARNED<br />
            Audited subject: <strong style={{ color: "#E8C96A" }}>Gemma 4 E4B</strong>
            &nbsp;·&nbsp; Diagnostic engine: <strong style={{ color: "#E8C96A" }}>Gemma 2 Instruct</strong>
            &nbsp;·&nbsp; Mantel r=0.945
            &nbsp;·&nbsp; <a href="https://github.com/phinnphace/asl-sovereign" style={{ color: "#E8C96A" }}>asl-sovereign ↗</a>
            &nbsp;·&nbsp; <a href="https://arxiv.org/abs/2407.15806" style={{ color: "#E8C96A" }}>FSboard ↗</a>
          </div>
          <span style={{
            fontFamily: "Bangers, cursive",
            fontSize: 13,
            letterSpacing: "0.14em",
            background: "#C0281A",
            color: "#FEFBF0",
            border: "2px solid #F5EDD6",
            padding: "2px 14px",
            display: "inline-block",
            transform: "rotate(-2deg)",
            marginTop: 10,
          }}>CLASSIFIED INSTRUMENT</span>
        </div>

        {/* Ted — breaks out of the header */}
        <div style={{
          position: "relative",
          flexShrink: 0,
          marginLeft: "1.5rem",
          marginTop: "-0.5rem",
        }}>
          {/* POW burst behind Ted */}
          <div style={{
            position: "absolute",
            top: -30, right: -20,
            width: 80, height: 80,
            zIndex: 0,
            transform: "rotate(15deg)",
          }}>
            <Burst size={80} color="#FFE033" label="POW!" labelColor="#C0281A" />
          </div>

          {/* Speech bubble */}
          <div style={{ position: "relative", zIndex: 3, marginBottom: 10, marginLeft: 8 }}>
            <SpeechBubble>Ted said so.</SpeechBubble>
          </div>

          {/* Ted photo — slightly rotated, breaking the border */}
          <div style={{
            position: "relative",
            zIndex: 2,
            width: 120,
            height: 120,
            border: "4px solid #F5EDD6",
            boxShadow: "5px 5px 0 #C0281A",
            overflow: "hidden",
            transform: "rotate(2deg)",
            marginBottom: 4,
          }}>
            <img
              src={TED_URL}
              alt="Ted"
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                objectPosition: "center 30%",
                display: "block",
              }}
            />
          </div>
          <div style={{
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: 9,
            color: "#C8B98A",
            textAlign: "center",
            lineHeight: 1.5,
          }}>
            TED · STOOP TABBY<br />
            Certified this pipeline.<br />
            HuggingFace could not see Ted.
          </div>
        </div>
      </div>

      {/* ── METRICS ── */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: "1rem",
        margin: "1.25rem 0",
      }}>
        <MetricCard label="Cross-dataset stability" value="r=0.945" sub="Mantel · Roboflow vs ISL · p<0.001" />
        <MetricCard label="Bootstrap persistence"   value="100%"    sub="5 shared pairs · 1,000 iterations" />
        <MetricCard label="Images audited"          value="2,370"   sub="ASL + ISL · 24 static letters" />
        <MetricCard
          label="Diagnoses run"
          value={diagCount !== null ? diagCount.toLocaleString() : "—"}
          sub="flywheel · grows with use"
        />
      </div>

      {/* ── TABS ── */}
      <div style={{ display: "flex", gap: 0, marginBottom: "1.25rem", borderBottom: "3px solid #1A1209" }}>
        {[
          { id: "chat",       label: "① SEND YOUR SIGNAL" },
          { id: "codex",      label: "② THE CONFUSION CODEX" },
          { id: "confidence", label: "③ ON CONFIDENCE" },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              fontFamily: "Bangers, cursive",
              fontSize: 16,
              letterSpacing: "0.06em",
              padding: "0.5rem 1.1rem",
              border: "3px solid #1A1209",
              borderBottom: activeTab === tab.id ? "3px solid #FEFBF0" : "3px solid #1A1209",
              background: activeTab === tab.id ? "#FEFBF0" : "#E8D9B0",
              color: "#1A1209",
              cursor: "pointer",
              marginBottom: -3,
              boxShadow: activeTab === tab.id ? "none" : "2px 2px 0 #1A1209",
              transition: "all 0.08s",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── TAB: CHAT ── */}
      {activeTab === "chat" && (
        <div>
          <p style={{ fontFamily: "'Special Elite', cursive", fontSize: 15, color: "#3A2E1A", marginBottom: "1rem" }}>
            Tell us what's going wrong — plain language, letter patterns, upload a CSV, or drop an image.
            The ring decodes it.
          </p>

          {/* Chat window */}
          <div style={{
            border: "3px solid #1A1209",
            background: "#FEFBF0",
            boxShadow: "5px 5px 0 #1A1209",
            height: 420,
            overflowY: "auto",
            padding: "1rem",
            marginBottom: "0.75rem",
          }}>
            {messages.map((m, i) => (
              <ChatMessage key={i} {...m} />
            ))}
            {loading && (
              <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 13, color: "#7A6540", padding: "0.5rem" }}>
                Reading failure geometry...
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input bar */}
          <div style={{
            display: "flex",
            gap: 0,
            border: "3px solid #1A1209",
            background: "#FEFBF0",
            boxShadow: "4px 4px 0 #1A1209",
          }}>
            {/* File/image upload */}
            <button
              onClick={() => fileRef.current?.click()}
              title="Upload CSV or image"
              style={{
                background: "#E8D9B0",
                border: "none",
                borderRight: "3px solid #1A1209",
                padding: "0 0.85rem",
                cursor: "pointer",
                fontFamily: "Bangers, cursive",
                fontSize: 20,
                color: "#1A1209",
                flexShrink: 0,
              }}
            >📎</button>
            <input ref={fileRef} type="file" accept=".csv,image/*" style={{ display: "none" }} onChange={handleFile} />

            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder='"K always comes back as V" · "M, N, T are bleeding together" · upload a CSV or image ↑'
              rows={2}
              style={{
                flex: 1,
                fontFamily: "'Share Tech Mono', monospace",
                fontSize: 14,
                background: "transparent",
                border: "none",
                outline: "none",
                padding: "0.6rem 0.75rem",
                color: "#1A1209",
                resize: "none",
                lineHeight: 1.5,
              }}
            />
            <button
              onClick={() => sendMessage(input)}
              disabled={loading}
              style={{
                fontFamily: "Bangers, cursive",
                fontSize: 18,
                letterSpacing: "0.1em",
                background: loading ? "#9A7F52" : "#C0281A",
                color: "#FEFBF0",
                border: "none",
                borderLeft: "3px solid #1A1209",
                padding: "0 1.25rem",
                cursor: loading ? "not-allowed" : "pointer",
                flexShrink: 0,
                transition: "background 0.1s",
              }}
            >
              💍 DECODE
            </button>
          </div>
          <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 10, color: "#9A7F52", marginTop: 5 }}>
            Enter to send · Shift+Enter for new line · attach CSV or image with 📎
          </div>
        </div>
      )}

      {/* ── TAB: CODEX ── */}
      {activeTab === "codex" && (
        <div>
          <p style={{ fontFamily: "'Special Elite', cursive", fontSize: 15, color: "#3A2E1A", marginBottom: "1rem" }}>
            For the analogue folks — here are the confusion matrices. Find your model's pattern and look it up.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 0 }}>
            <CodexCard
              regime="landmark" cer="16.7%"
              matrix={
                "  Pred→  M   N   T   A   S\n" +
                "True↓  ┌─────────────────┐\n" +
                "  M    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
                "  N    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
                "  T    │▓▓▓ ▓▓▓ ▓▓▓ ░░░ ░░░│\n" +
                "  A    │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n" +
                "  S    │░░░ ░░░ ░░░ ▓▓▓ ▓▓▓│\n" +
                "       └─────────────────┘\n" +
                "  ▓ high confusion  ░ low"
              }
              useCases={[
                "Beginner / demo systems · controlled setting only",
                "Expect consistent confusion pairs — they will not resolve",
                "NOT suitable for real ASL, continuous signing, or sentences",
              ]}
              pairs={["K→V (total)", "D→I", "M↔N↔T", "A↔S", "G→D", "P→D"]}
              risk="⚠ DEPLOYMENT RISK: HIGH · silent systematic failure"
              riskLevel="high"
            />
            <CodexCard
              regime="full" cer="11.1%"
              matrix={
                "  Pred→  M   N   T   A   S\n" +
                "True↓  ┌─────────────────┐\n" +
                "  M    │▓▓▓ ▒▒▒ ▒▒▒ ░░░ ░░░│\n" +
                "  N    │▒▒▒ ▓▓▓ ▒▒▒ ░░░ ░░░│\n" +
                "  T    │▒▒▒ ▒▒▒ ▓▓▓ ░░░ ░░░│\n" +
                "  A    │░░░ ░░░ ░░░ ▓▓▓ ▒▒▒│\n" +
                "  S    │░░░ ░░░ ░░░ ▒▒▒ ▓▓▓│\n" +
                "       └─────────────────┘\n" +
                "  ▓ correct  ▒ some confusion  ░ low"
              }
              useCases={[
                "Research-grade ASL recognition",
                "Static dictionary signing — works well",
                "Continuous flow and motion signing — still limited",
              ]}
              pairs={["M vs N (reduced)", "co-artic drift", "flow breaks"]}
              risk="⚠ DEPLOYMENT RISK: MODERATE-HIGH · signer-context dependent"
              riskLevel="moderate"
            />
            <CodexCard
              regime="hybrid" cer="7.2%"
              matrix={
                "  Pred→  M   N   T   A   S\n" +
                "True↓  ┌─────────────────┐\n" +
                "  M    │▓▓▓ ░░░ ░░░ ░░░ ░░░│\n" +
                "  N    │░░░ ▓▓▓ ░░░ ░░░ ░░░│\n" +
                "  T    │░░░ ░░░ ▓▓▓ ░░░ ░░░│\n" +
                "  A    │░░░ ░░░ ░░░ ▓▓▓ ░░░│\n" +
                "  S    │░░░ ░░░ ░░░ ░░░ ▓▓▓│\n" +
                "       └─────────────────┘\n" +
                "  ▓ correct  ░ low confusion"
              }
              useCases={[
                "ASL → English translation pipelines",
                "Full visual + temporal models — the right tool here",
                "Errors are contextual, not geometric — manageable in production",
              ]}
              pairs={["context errors", "temporal drift", "sequence breaks"]}
              risk="✓ DEPLOYMENT RISK: MODERATE · errors contextual not geometric"
              riskLevel="low"
            />
          </div>
          <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 10, color: "#9A7F52" }}>
            FSboard · Georg et al. 2024 · CC BY 4.0 · arxiv.org/abs/2407.15806
          </div>
        </div>
      )}

      {/* ── TAB: CONFIDENCE ── */}
      {activeTab === "confidence" && (
        <div style={{ display: "grid", gridTemplateColumns: "3fr 1fr", gap: "1.5rem", alignItems: "start" }}>
          <div>
            <div style={{ fontFamily: "Bangers, cursive", fontSize: 64, color: "#1A1209", lineHeight: 1 }}>
              Accumulating
              <span style={{
                display: "inline-block",
                border: "4px solid #C0281A",
                color: "#C0281A",
                fontFamily: "Bangers, cursive",
                fontSize: 24,
                letterSpacing: "0.18em",
                padding: "3px 16px",
                transform: "rotate(-7deg)",
                opacity: 0.9,
                marginLeft: 14,
                verticalAlign: "middle",
              }}>LIVE</span>
            </div>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 11, color: "#9A7F52", marginTop: 4 }}>
              construct validity confirmed · absolute accuracy accumulates with use
            </div>
            <div style={{
              fontFamily: "'Special Elite', cursive",
              fontSize: 15,
              lineHeight: 1.9,
              color: "#3A2E1A",
              borderLeft: "5px solid #1A1209",
              paddingLeft: "1rem",
              marginTop: "0.75rem",
            }}>
              The validation suite confirms this instrument measures something real and stable.
              What it does not yet have is a classification accuracy percentage against
              known-provenance models — because that labeled ground truth doesn't exist yet.
              <br /><br />
              We won't manufacture it. Every diagnosis you submit becomes part of that ground
              truth as provenance becomes known. The confidence number you want is what
              this instrument is designed to produce — honestly, from real use.
            </div>
          </div>
          <div>
            <div style={{ border: "4px solid #1A1209", boxShadow: "4px 4px 0 #1A1209", overflow: "hidden" }}>
              <img src={TED_URL} alt="Ted" style={{ width: "100%", display: "block", objectFit: "cover", objectPosition: "center 30%" }} />
              <div style={{
                fontFamily: "'Share Tech Mono', monospace", fontSize: 9,
                color: "#7A6540", textAlign: "center", padding: 5,
                background: "#E8D9B0", borderTop: "2px solid #1A1209",
              }}>
                Gemma 4 saw Ted via Ollama.<br />HuggingFace could not see Ted.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── FOOTER ── */}
      <div style={{
        borderTop: "4px solid #1A1209",
        paddingTop: 8,
        marginTop: "2.5rem",
        fontFamily: "'Share Tech Mono', monospace",
        fontSize: 10,
        color: "#7A6540",
        lineHeight: 1.8,
      }}>
        THE PROVENANCE DECODER RING &nbsp;·&nbsp;
        Audited subject: Gemma 4 E4B (gemma4:e4b-it-q4_K_M via Ollama) &nbsp;·&nbsp;
        Diagnostic engine: Gemma 2 Instruct (Google AI Studio) &nbsp;·&nbsp;
        FSboard: Georg et al. 2024 (CC BY 4.0) · arxiv.org/abs/2407.15806 &nbsp;·&nbsp;
        ISL dataset: Biswas 2024 · doi:10.17632/n34wm8sb3x.1 &nbsp;·&nbsp;
        Mantel r=0.945 p&lt;0.001 &nbsp;·&nbsp; Bootstrap 100% · 1,000 iterations
      </div>
    </div>
  );
}
