# Key Facts — ASL-Sovereign Dashboard

## Project Identity

**Value / Detail:**
- Project name: ASL-Sovereign Dashboard
- Owner: phinn (phinnphace@gmail.com)
- Repo: private GitHub repo (made private 2026-05-15 due to security incident)

**Why it matters:**
Repo is private — coordinate before sharing links or making public.

**Last verified:** 2026-05-15

---

## Stack

**Value / Detail:**
- Frontend: React + Tailwind CSS, hosted on Vercel
- Backend/API: Modal (serverless Python functions) — chatbot endpoint
- User data storage: Google Sheets
- Previous stack (deprecated): Streamlit

**Why it matters:**
Three moving parts (Vercel + Modal + Google Sheets) must stay in sync. Auth/API keys span all three services.

**Last verified:** 2026-05-15

---

## Vercel Deployment

**Value / Detail:**
- Repo is NOT currently linked to Vercel (manual deploys or separate pipeline)
- Vercel has flagged this and is prompting to link the repo

**Why it matters:**
Do not link repo to Vercel until git history is fully clean — Vercel build logs could expose secrets if dirty history is present.

**Last verified:** 2026-05-15

---

## API Key Exposure Incident

**Value / Detail:**
- An API key was exposed in the repo (date: ~2026-05-14 or shortly before)
- Keys have since been rotated
- Old key values must be scrubbed from all git history before repo can be made public or Vercel-linked
- An SSH OSU key of unknown origin also appeared in the repo — suspected accidental commit

**Why it matters:**
Scrubbing must use exact old key strings (even rotated keys must be purged from history). Confirm with phinn which keys to target before running filter-repo or BFG.

**Last verified:** 2026-05-15

---

## Git Hygiene Status

**Value / Detail:**
- Repo history is currently dirty: contains exposed API keys, an SSH OSU key of unknown origin, and general disorganization from rapid iteration
- Recommended cleanup tool: `git-filter-repo` (preferred over BFG for precision)
- After history rewrite, a force push is required: `git push --force --all`

**Why it matters:**
Force push will overwrite remote history — all collaborators must re-clone afterward.

**Last verified:** 2026-05-15
