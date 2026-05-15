# Bug Log — ASL-Sovereign Dashboard

## API Key Exposed in Git History

**Date:** 2026-05-15
**Status:** CLOSED ✓ — key rotated + history scrubbed 2026-05-15

**Issue:**
An API key was committed to the repository and exposed publicly. The repo was subsequently made private to contain the damage.

**Root Cause:**
Key was likely hardcoded in source or a config/env file that was not excluded by .gitignore, then committed and pushed.

**Solution:**
- Key has been rotated (new key active)
- History scrub pending using `git-filter-repo` — see issues.md for tracking

**Prevention:**
- Never commit .env files or any file containing raw API keys
- Add .env, .env.local, .env.* to .gitignore before first commit
- Use environment variables in Vercel dashboard and Modal secrets manager instead of hardcoding
- Consider adding `git-secrets` or similar pre-commit hook going forward

---

## SSH OSU Key Found in Repo

**Date:** 2026-05-15
**Status:** CLOSED ✓ — scrubbed from history 2026-05-15

**Issue:**
An SSH OSU (Ohio State University) key appeared in the repo. Origin is unknown — phinn does not use or own this key.

**Root Cause:**
Unknown. Possibilities: accidental copy-paste into a committed file, a tool or script that wrote it, or a misconfigured SSH config file that got committed.

**Solution:**
Pending investigation of git log to identify which commit introduced it and what file it's in. Must be scrubbed from history alongside other secrets.

**Prevention:**
Audit what files are being staged before each commit (`git diff --cached`). Never commit SSH keys, .ssh/ directories, or key material.

---

## App Broken After API Key Rotation

**Date:** 2026-05-15
**Status:** Open

**Issue:**
After rotating the exposed API key, the chatbot and/or other integrations stopped working correctly. The app has not been fully functional since the incident.

**Root Cause:**
New key not yet propagated to all services (Vercel env vars, Modal secrets, possibly hardcoded references in code).

**Solution:**
Pending — need to audit all locations where the old key was referenced and update each:
1. Vercel environment variables (dashboard)
2. Modal secrets
3. Any remaining hardcoded references in source (which should then be removed and moved to env vars)

**Prevention:**
Centralize all secrets in environment variable managers. Document where each secret is used.
