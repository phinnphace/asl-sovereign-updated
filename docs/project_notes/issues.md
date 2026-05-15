# Work Log — ASL-Sovereign Dashboard

## Git History Scrub & Repo Cleanup

**Date opened:** 2026-05-15
**Status:** CLOSED ✓
**Reference:** Cowork session 2026-05-15

**Description:**
The git repo contained exposed secrets (API keys, unknown SSH OSU key) across its commit history. The repo was made private as a temporary measure. Full history rewrite completed 2026-05-15.

**Notes:**
- Repo made private: 2026-05-15 ✓
- Old API key rotated: ✓
- History scrubbed with `git filter-repo` (run from PowerShell — sandbox had Windows filesystem lock file permission issues): ✓
- Files removed from all history: `kaggleaccess_token.txt`, `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAA.txt`, `googleapi.txt` ✓
- Verified clean: `git log --all -p | grep KGAT_` returned no results ✓
- Force pushed: `git push --force --all && git push --force --tags` ✓
- .gitignore updated to block token/key/SSH files going forward ✓
- Remaining: rotate Kaggle token, Google service account key, confirm Gemini key is post-exposure
- Remaining: re-clone on any other machines (force push rewrites history)
- Remaining: link repo to Vercel once all keys rotated

**Closed:** 2026-05-15

---

## Restore App Functionality Post-Key Rotation

**Date opened:** 2026-05-15
**Status:** Open
**Reference:** Cowork session 2026-05-15

**Description:**
After rotating the exposed API key, the chatbot and integrations stopped working. New key needs to be propagated to Vercel env vars, Modal secrets, and any source references cleaned up.

**Notes:**
- Audit all files for hardcoded key references (should be zero after cleanup)
- Update Vercel environment variables dashboard
- Update Modal secrets
- Test chatbot end-to-end after updates

**Closed:** —

---

## Set Up .gitignore Properly

**Date opened:** 2026-05-15
**Status:** Open

**Description:**
Ensure .gitignore covers all sensitive file patterns before repo is reopened or any new commits are made.

**Notes:**
Must include at minimum:
- .env
- .env.local
- .env.*
- *.pem
- *.key
- .ssh/
- Any service account JSON files (Google Sheets credentials)

**Closed:** —
