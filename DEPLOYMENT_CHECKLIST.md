# Deployment verification checklist

The current deployed demo uses the following services:

- Frontend: https://rag-failure-diagnostics-web-live.onrender.com
- Backend: https://rag-failure-diagnostics-api-kajal.onrender.com
- Repository: https://github.com/kajalnain143-hub/rag-failure-diagnostics

## Render backend

- [x] Service `rag-failure-diagnostics-api-kajal` is deployed and marked Live.
- [ ] `GEMINI_API_KEY` is configured in the backend service environment.
- [ ] `API_AUTH_TOKEN` has a generated value. Keep it private.
- [x] `CORS_ORIGINS` includes the deployed frontend origin and the legacy frontend origin.
- [x] Render health logs show HTTP 200 responses from `/health`.
- [ ] Run the smoke test below with the Render-generated API token.

## Render frontend

- [x] Static site `rag-failure-diagnostics-web-live` is deployed and marked Live.
- [x] Production Vite build completed and Render logged "Your site is live".
- [x] `VITE_API_URL` points to the backend URL.
- [ ] Open the site and enter the backend's `API_AUTH_TOKEN` in the API token field.
- [ ] Verify document list, upload, query, incident diagnosis, and benchmark screens in a browser.

## Smoke test

Install no extra dependencies; Python standard library only:

```powershell
python scripts/smoke_test.py https://rag-failure-diagnostics-api-kajal.onrender.com
python scripts/smoke_test.py https://rag-failure-diagnostics-api-kajal.onrender.com --token YOUR_API_AUTH_TOKEN
```

Without a token, public health/root endpoints should return 200 and protected API endpoints should return 401. With a valid token, protected read endpoints should return 200.

## Important free-hosting data warning

The current backend uses SQLite at `/tmp/rag_diagnostics.db`. On an ephemeral filesystem, uploaded documents, experiments, traces, and incidents may disappear on restart/redeploy. Do not use this free deployment as durable storage. PostgreSQL + pgvector has not been implemented in this repository. For persistent use, implement and test a managed PostgreSQL adapter and backup/restore workflow before migrating live data.

## Data backup check

- [x] A bearer-token-protected SQLite online backup endpoint is implemented at `/api/admin/database-backup`.
- [ ] With the API bearer token, download the endpoint and verify the file begins with the SQLite database header.
- [ ] Store the backup in a private location with restricted access; it contains the complete indexed corpus and operational history.
- [ ] Repeat backups before every Render restart/redeploy while the database remains on ephemeral `/tmp` storage.
- [ ] Test restore procedure on a separate copy and never overwrite the only backup.

## Report actual results

Record the Render deployment ID, commit SHA, timestamp, smoke-test output, CI run URL, and whether a Gemini-backed query succeeded. Do not label the project production-ready until these checks pass.
