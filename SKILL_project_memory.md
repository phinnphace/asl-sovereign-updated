---
name: project-memory
description: Set up and maintain a structured project memory system in docs/project_notes/
  that tracks bugs with solutions, architectural decisions, key project facts, and work
  history. Use this skill when asked to "set up project memory", "track our decisions",
  "log a bug fix", "update project memory", or "initialize memory system".
---

## When to Use This Skill

Invoke this skill when:

- Starting a new project that will accumulate knowledge over time
- The project already has recurring bugs or decisions that should be documented
- The user asks to "set up project memory" or "track our decisions"
- Encountering a problem that feels familiar ("didn't we solve this before?")
- Before proposing an architectural change (check existing decisions first)

---

## Core Capabilities

### 1. Initial Setup — Create Memory Infrastructure

When invoked for the first time in a project, create the following structure:

```
docs/
└── project_notes/
    ├── bugs.md         # Bug log with solutions
    ├── decisions.md    # Architectural Decision Records
    ├── key_facts.md    # Project configuration and constants
    └── issues.md       # Work log with ticket references
```

**Directory naming rationale:** Using `docs/project_notes/` instead of `memory/`
makes it look like standard engineering organization, not AI-specific tooling.

---

### 2. Configure CLAUDE.md — Memory-Aware Behavior

Add or update the following section in the project's `CLAUDE.md` file:

```markdown
## Project Memory System

### Memory-Aware Protocols

**Before proposing architectural changes:**
- Check `docs/project_notes/decisions.md` for existing decisions
- Verify the proposed approach doesn't conflict with past choices

**When encountering errors or bugs:**
- Search `docs/project_notes/bugs.md` for similar issues
- Apply known solutions if found
- Document new bugs and solutions when resolved

**When looking up project configuration:**
- Check `docs/project_notes/key_facts.md` for credentials, ports, URLs
- Prefer documented facts over assumptions
```

---

## Memory Entry Formats

### bugs.md — Bug Log Entry

```markdown
## [Short descriptive title]

**Date:** YYYY-MM-DD
**Status:** Open | Resolved | Won't Fix

**Issue:**
What went wrong. Exact error message if available.

**Root Cause:**
Why it happened.

**Solution:**
What fixed it. Exact steps or code if applicable.

**Prevention:**
What to do to avoid this in the future.
```

---

### decisions.md — Architectural Decision Record (ADR)

```markdown
## [Decision title]

**Date:** YYYY-MM-DD
**Status:** Accepted | Superseded | Deprecated

**Context:**
What situation prompted this decision.

**Decision:**
What was decided.

**Alternatives Considered:**
What else was evaluated and why it was rejected.

**Consequences:**
What this decision constrains or enables going forward.
```

---

### key_facts.md — Project Constants

```markdown
## [Fact category or name]

**Value / Detail:**
The fact itself — path, constant, URL, credential reference, constraint.

**Why it matters:**
Context for why this is recorded here.

**Last verified:** YYYY-MM-DD
```

---

### issues.md — Work Log Entry

```markdown
## [Issue or task title]

**Date opened:** YYYY-MM-DD
**Status:** Open | In Progress | Closed
**Reference:** [ticket number, PR, or conversation link if applicable]

**Description:**
What the issue is or what work is needed.

**Notes:**
Running log of progress, blockers, and resolutions.

**Closed:** YYYY-MM-DD [if applicable]
```

---

## Loading Protocol (Session Start)

When `/project memory` is invoked at the start of a session:

1. Check `/mnt/user-data/uploads/` for any uploaded memory files (`CLAUDE.md`,
   `bugs.md`, `decisions.md`, `key_facts.md`, `issues.md`). If found, copy them
   into the working directory at `docs/project_notes/` before doing anything else.
   This makes them the live working copies for this session.
2. Locate `CLAUDE.md` in the project root — read it in full
3. Read all four files in `docs/project_notes/` that are present
4. Note any files that are missing without erroring
5. Output a structured session summary:

```
## Project Memory Loaded

**Project:** [name]
**Last known state:** [from CLAUDE.md]

**Open bugs:** [count + titles]
**Open issues:** [count + titles]
**Recent decisions:** [last 2-3 ADR titles]
**Key facts loaded:** [count]

Ready. (Remind me to run /project-memory close before ending this session.)
```

---

## Update Protocol

When the user asks to log, update, or record something:

- **New bug found** → append entry to `bugs.md`
- **Bug resolved** → update Status field and fill Solution + Prevention
- **Decision made** → append ADR to `decisions.md`
- **Fact established** → append to `key_facts.md`
- **Work item** → append to `issues.md`

Always confirm the write with: `Logged to [filename].`

---

## Close Protocol (Session End)

When the user says `/project-memory close`, or asks to "save memory", "wrap up",
or "close the session":

1. Write the current in-memory state of all five files to `/mnt/user-data/outputs/`:
   - `CLAUDE.md`
   - `docs/project_notes/bugs.md`
   - `docs/project_notes/decisions.md`
   - `docs/project_notes/key_facts.md`
   - `docs/project_notes/issues.md`

2. Call `present_files` on all five so download links appear.

3. Output this closing message:

```
## Project Memory Saved

Files are ready to download above.

To persist across sessions:
  Save these to your project folder in Google Drive (or equivalent),
  maintaining the docs/project_notes/ subfolder structure.
  Upload them at the start of your next session before invoking /project-memory.
```

**Why this is necessary:** The session working directory (`/home/claude/`) resets
between sessions. Files written there are lost. `/mnt/user-data/outputs/` produces
download links. The user's cloud storage (Google Drive, etc.) is the durable store.
This skill cannot write directly to Drive — the user must download and re-upload.

**Proactive close reminder:** If the user appears to be wrapping up a session
(says "thanks", "that's all for now", "we're done") and `/project-memory close`
has not been called this session, remind them:
"You have unsaved project memory from this session — want me to run
`/project-memory close` before you go?"

---

## Linking Convention

`CLAUDE.md` and `AGENTS.md` (if present) should reference the notes files explicitly
so any agent reading the root config knows where memory lives:

```markdown
## Memory
Project notes are in `docs/project_notes/`.
See: bugs.md · decisions.md · key_facts.md · issues.md
```
