# Deployment verification checklist

Use this after Render reports both services as **Live**. A created service is not proof that the application is healthy.

## Render backend

- [ ] Service `rag-failure-diagnostics-api-kajal` is Live.
- [ ] `GEMINI_API_KEY` is configured in the backend service environment.
- [ ] `API_AUTH_TOKEN` has a generated value. Keep it private.
- [ ] `CORS_ORIGINS` exactly matches the deployed static-site origin.
- [ ] Open `/health`; expect HTTP 200 and `{"status":"healthy","database":"ready",...}`.
- [ ] Run the smoke test below with the Render-generated API token.

## Render frontend

- [ ] Static site `rag-failure-diagnostics-web-kajal` is Live.
- [ ] `VITE_API_URL` points to the backend URL (no trailing slash required).
- [ ] Open the site and enter the backend's `API_AUTH_TOKEN` in the API token field.
- [ ] Verify document list, upload, query, incident diagnosis, and benchmark screens.

## Smoke test

Install no extra dependencies; Python standard library only:

```powershell
python scripts/smoke_test.py https://YOUR-BACKEND.onrender.com
python scripts/smoke_test.py https://YOUR-BACKEND.onrender.com --token YOUR_API_AUTH_TOKEN
```

Without a token, public health/root endpoints should return 200 and protected API endpoints should return 401. With a valid token, protected read endpoints should return 200.

## Important free-hosting data warning

The current Render Blueprint config points SQLite to `/tmp/rag_diagnostics.db`. On an ephemeral filesystem, uploaded documents, experiments, traces, and incidents may disappear on restart/redeploy. Do not use this free deployment as durable storage. For persistent portfolio demos, attach a persistent disk (if available on the selected plan) or migrate to managed PostgreSQL/object storage and configure backups.

## Data backup check

- [ ] With the API bearer token, download `/api/admin/database-backup` and verify the file begins with the SQLite database header.
- [ ] Store the backup in a private location with restricted access; it contains the complete indexed corpus and operational history.
- [ ] Repeat backups before every Render restart/redeploy while the database remains on ephemeral `/tmp` storage.
- [ ] Test restore procedure on a separate copy and never overwrite the only backup.

## Report actual results

Record the Render deployment ID, commit SHA, timestamp, smoke-test output, CI run URL, and whether a Gemini-backed query succeeded. Do not label the project production-ready until these checks pass.
