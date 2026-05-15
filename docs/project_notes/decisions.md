# Architectural Decisions — ASL-Sovereign Dashboard

## Migrate from Streamlit to React + Tailwind on Vercel

**Date:** ~2026-05-13 (approx)
**Status:** Accepted

**Context:**
Initial prototype was built on Streamlit for rapid development. Needed a more production-ready, customizable frontend with proper hosting.

**Decision:**
Move to React + Tailwind CSS hosted on Vercel.

**Alternatives Considered:**
- Streamlit (rejected: limited UI flexibility, not suitable for production dashboard)

**Consequences:**
- Requires managing a proper build/deploy pipeline
- Vercel hosting adds a deployment layer to coordinate
- More flexibility for UI/UX but more moving parts

---

## Use Modal for Serverless Chatbot Backend

**Date:** ~2026-05-13 (approx)
**Status:** Accepted

**Context:**
The dashboard includes a chatbot feature that requires a backend inference endpoint. Needed a serverless option that could scale without managing infrastructure.

**Decision:**
Use Modal as the serverless Python backend for the chatbot API.

**Alternatives Considered:**
- Vercel serverless functions (rejected or not chosen — reasons TBD)
- Self-hosted (rejected: too much overhead)

**Consequences:**
- Modal manages compute; secrets must be stored in Modal's secrets manager
- API key for chatbot service must be kept in sync between Modal and any calling services
- Adds a third-party dependency to the stack

---

## Use Google Sheets for User Data Storage

**Date:** ~2026-05-13 (approx)
**Status:** Accepted

**Context:**
Needed a lightweight, accessible data store for user data that doesn't require standing up a database.

**Decision:**
Use Google Sheets as the user data backend via the Sheets API.

**Alternatives Considered:**
- Supabase, Firebase, or similar (not chosen — likely due to simplicity preference)

**Consequences:**
- Google Sheets API credentials must be managed securely (never committed to repo)
- Scaling limits apply (Sheets not suited for high-volume writes)
- Easy to inspect/edit data manually which is convenient for early development

---

## Keep Repo Private Until Git History Is Cleaned

**Date:** 2026-05-15
**Status:** Accepted

**Context:**
API key exposure and presence of unknown SSH key in git history required immediate containment.

**Decision:**
Made repo private on 2026-05-15. Will remain private until a full history scrub is completed and verified.

**Alternatives Considered:**
- Deleting and recreating the repo (rejected: loses all history and issues)

**Consequences:**
- Vercel linking is blocked until history is clean (risk of exposing secrets via build logs)
- Any collaborators lose access until repo is made public again or explicitly re-invited
