# MedHelp v2 - clinical-guideline assistant (separate backend + frontend, no Docker)

Answers questions **only** from open-access clinical guidelines (WHO, CDC, ACC/AHA, KDIGO ...; a curated list of 43 documents), with
citations, the guideline's own evidence grade and a plain-language glossary.

> **Educational use only.** MedHelp summarises published guidelines. It is not medical advice.

```
medhelp/
├── backend/                  FastAPI service  (deploy this to Render / Cloud Run / any Python host)
│   ├── app/                  engine, API, sessions, cache, keyword index, providers ...
│   ├── data/                 manifest.json (shipped), keyword.sqlite.gz (you build it), SOURCES.md, your PDFs
│   ├── tests/                179 offline unit tests
│   ├── requirements.txt      small runtime set (no torch, no LlamaIndex, no embedding model)
│   ├── requirements-ingest.txt   heavy set, ONLY for the machine where you build the index
│   └── .env.example
├── frontend/                 Streamlit UI     (deploy to Streamlit Community Cloud or any Python host)
│   ├── app.py   medhelp_ui/   tests/   requirements.txt   .streamlit/
├── render.yaml               optional Render blueprint for the backend
└── .github/workflows/keepalive.yml   optional cron that keeps the free Qdrant cluster alive
```

## What changed from the first version

| Area | Before | Now |
|---|---|---|
| Chat history | kept in the browser, re-sent every request | **server-side sessions** per browser tab (`session_id`), isolated, size/time-bounded; the frontend keeps a copy and re-sends it if the server restarted |
| Query embeddings | `bge-large` loaded in the API (~2 GB RAM) | **Cloudflare Workers AI** hosts the *same* model, so your existing Qdrant index stays valid (`EMBED_PROVIDER=local` still works) |
| Keyword search | in-memory BM25 rebuilt from Qdrant on every cold start | **SQLite FTS5 file** built at ingestion and shipped with the backend (opens instantly, almost no RAM) |
| Backend imports | LlamaIndex, torch | plain `requests` + `qdrant-client`; LlamaIndex/torch are used **only for ingestion on your machine** |
| Answers | one blocking call | **real token streaming** (SSE), plus `POST /query` for non-streaming clients |
| New | | **compare mode**, **citation verification**, **semantic answer cache**, **suggested follow-ups**, **voice questions**, **Markdown/PDF export**, optional **API reranker**, rate limiting + request queue, API-key protection |

> **Renamed from MedRAG.** Only the app's name changed. The default Qdrant collection is still called `medrag_guidelines` on purpose (it is the name of your existing collection); leave `QDRANT_COLLECTION` alone unless you really create a new collection.

`TOP_K`, `RETRIEVE_CANDIDATES`, `CONTEXT_CHAR_BUDGET` and the other retrieval settings are unchanged.

---

## 1. Accounts you need (all free tiers)

* **Groq** key - <https://console.groq.com/keys> (answers, glossary, follow-ups, speech-to-text)
* **Qdrant Cloud** cluster URL + key - <https://cloud.qdrant.io> (you probably already have one)
* **Cloudflare** account ID + a Workers AI API token - for query-time embeddings
* *(optional)* Voyage AI key for the reranker

## 2. Build the index on your own machine (one time, then occasionally)

The deployed API never embeds documents. You do that locally, writing straight to Qdrant.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ingest.txt
cp .env.example .env                                       # fill QDRANT_URL / QDRANT_API_KEY (and the rest)

# put your guideline PDFs in backend/data/ (or let the tool download them), then:
python -m app.ingestion                # embeds only NEW/changed PDFs, writes data/keyword.sqlite.gz at the end
```

**Already have an index from v1?** It is still valid (same model, same chunking, same version stamp). Do **not**
re-ingest; just write the keyword file:

```bash
python -m app.keyword_db build         # scrolls Qdrant, writes data/keyword.sqlite.gz
```

Then check that Cloudflare's vectors match the local model (needs the local model once, ~1.3 GB download):

```bash
python -m app.embeddings --check       # expect "OK - safe to use the API embedder with your existing index"
```

If it prints MISMATCH, use `EMBED_PROVIDER=local` on a host with enough RAM, or re-ingest with a matching provider.

**Commit `backend/data/manifest.json` and `backend/data/keyword.sqlite.gz`.** The API needs both. I could not measure
the size of the keyword file for your real data; it should be tens of MB (GitHub's per-file limit is 100 MB). If it is
missing or out of date (its recorded Qdrant point count differs from Qdrant's), the API rebuilds a fresh copy from Qdrant
in the background and answers with semantic search only until that finishes.

## 3. Run locally

```bash
# terminal 1 - backend (from backend/, with requirements.txt installed; EMBED_PROVIDER=local needs the ingest set)
uvicorn app.api:app --port 8000 --workers 1

# terminal 2 - frontend (from frontend/)
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml     # API_URL=http://localhost:8000 (API_KEY if you set one)
streamlit run app.py
```

Interactive API docs: <http://localhost:8000/docs>. Keep `--workers 1`: sessions, the cache and the keyword
file handle live in that one process.

## 4. Deploy without Docker

Free-tier limits change often; the figures below are what I found when researching this (October 2026), so re-check them
when you sign up.

### Backend

The API needs roughly **100-150 MB** of RAM with the lean set-up (my estimate from reading the code, **not measured**;
`GET /health` reports the real figure as `memory_mb` after you deploy).

**Option A - Google Cloud Run (best technical fit; needs a card on the account, free allowance covers small use)**

```bash
cd backend
gcloud run deploy medhelp-api --source . --region YOUR_REGION --allow-unauthenticated \
  --cpu 1 --memory 1Gi --min-instances 0 --max-instances 1 --cpu-boost \
  --set-env-vars EMBED_PROVIDER=cloudflare,QDRANT_URL=...,CLOUDFLARE_ACCOUNT_ID=... \
  --set-secrets GROQ_API_KEY=groq-key:latest,QDRANT_API_KEY=qdrant-key:latest,CLOUDFLARE_API_TOKEN=cf-token:latest,API_KEY=medhelp-api-key:latest
```

`--source .` uses Google's buildpacks (no Dockerfile); `backend/Procfile` tells them how to start the app.
**`--max-instances 1` matters**: sessions and the cache live in memory. Set a $1 budget alert. Free instances scale to
zero, so the first request after idle waits for a cold start.

**Option B - Render free web service (no card)**

New Web Service -> connect the repo (or use `render.yaml`) -> *Root directory* `backend`, *Build* `pip install -r requirements.txt`,
*Start* `uvicorn app.api:app --host 0.0.0.0 --port $PORT --workers 1`, health check `/health`, then add the secrets from `.env.example`.
Render's free instance has 512 MB RAM and 0.1 vCPU and sleeps after about 15 idle minutes (expect a ~1 minute wake-up).
It fits the lean backend; it would not fit the old torch-based one.

**Not recommended anymore:** Hugging Face Docker Spaces (new free accounts can no longer create them on free CPU, as far as I found),
Fly.io (no free tier for new users). Koyeb's free tier status was reported inconsistently.

### Frontend (Streamlit)

*Streamlit Community Cloud*: new app -> repo, main file `frontend/app.py` -> Advanced settings -> Secrets:

```toml
API_URL = "https://your-backend-url"
API_KEY = "same value as the backend's API_KEY"
```

(I did not re-verify Community Cloud's current limits.) Alternatively run it as a second Python service on any host:
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0` from `frontend/`. On Render, two always-on free services
would exceed the free instance-hour allowance of one workspace, so prefer Community Cloud for the UI.

### Keeping the free services alive

* **Qdrant free clusters** are suspended after about a week without activity and deleted after about four weeks.
  `GET /keepalive` runs a real Qdrant query; the included GitHub Action calls it twice a week
  (add repository secrets `BACKEND_URL` and `BACKEND_API_KEY`).
* The Action also wakes a sleeping Render instance; you do not need to keep it awake all day.

### Protect your quotas

Set `API_KEY` on the backend and the same value in the frontend's secrets. Without it anyone who finds your backend URL
can spend your Groq and Cloudflare allowance. Browsers never see the key: Streamlit's server calls the API.

---

## 5. Features in detail

* **Sessions** - the frontend puts a random session id in the URL (`?sid=...`), so a refresh resumes the conversation
  (text only; sources are not restored). The server keeps the last 20 turns per session for 2 hours, at most 500 sessions
  (least recently used evicted). Refusals about personal health and small talk are **not** stored. If the server forgot a session
  (restart, scale-to-zero) the frontend re-sends its own copy once; after that the server is the source of truth.
* **Compare mode** - tick *Compare guidelines*, pick 2-3 publishers. Retrieval runs separately inside each publisher
  (Qdrant payload filter), and the **existing** `TOP_K` / context budget is split between them (6 passages -> 3+3 or 2+2+2). The
  answer is structured as *Where they agree / Where they differ / Only one covers / Strength of Recommendation*.
* **Citation check** - local, no LLM call: each cited sentence is compared with the passage it cites (content words and every
  number). A sentence citing several passages is judged against their combined text, and each citation must contribute something.
  It catches wrong citations and invented numbers; it cannot judge meaning. Problems are listed as warnings.
* **Streaming** - `POST /query/stream` sends Server-Sent Events: `status`, `token`..., `final` (or `error`).
* **Answer cache** - a question whose embedding is >= 0.97 similar to a cached one (or identical after normalising) is answered
  without the LLM. Scoped per mode, only successful answers, expires after 12 h. A follow-up whose rewrite failed is never cached.
  Lower `CACHE_SIMILARITY` for more hits, raise it (or set `CACHE_ENABLED=false`) if you ever see a wrong match.
* **Follow-ups** - three suggested questions after each answer (fast model, run in parallel with the glossary call; personal-medical
  suggestions are filtered out).
* **Voice** - record in the sidebar -> Groq Whisper (`whisper-large-v3-turbo`) -> an **editable transcript** -> you press Send. Clips are
  limited to 2 MB; audio is never stored. Needs Streamlit >= 1.40 and an HTTPS page (Community Cloud is).
* **Export** - Markdown (always) and PDF (needs `fpdf2`; built-in Latin-1 font, so symbols such as >= are transliterated).
* **Reranker (optional)** - `RERANK_PROVIDER=voyage` re-orders the best 12 fused passages through Voyage's `rerank-2.5-lite`
  API (no RAM). It never changes how many passages are used. Without a payment method on the Voyage account the rate limit is
  only 3 requests/minute.
* **Fair use** - `RATE_LIMIT_PER_MINUTE` questions per session/IP, at most `MAX_CONCURRENT_QUERIES` answered at once (others queue up to
  `QUEUE_WAIT_SECONDS`, then get a polite 429). Groq's free plan allows roughly 8K tokens/minute for the whole app, so this turns
  simultaneous users into short waits instead of errors.

## 6. API reference

All endpoints except `/`, `/health` need `X-API-Key` when `API_KEY` is set.

| Method & path | Purpose |
|---|---|
| `GET /health` | status, `index_ready`, keyword index state, embed provider, `memory_mb`, session/cache/queue stats |
| `GET /keepalive` | touches Qdrant (for cron) |
| `GET /documents` | `{documents, publishers}` |
| `POST /sessions` / `GET /sessions/{id}` / `DELETE /sessions/{id}` | create / read text turns / delete |
| `POST /query` | `{question, session_id?, history?, compare?}` -> full result |
| `POST /query/stream` | same body; SSE events `status`, `token`, `final`, `error` |
| `POST /transcribe` | multipart `file` (+ optional `session_id`) -> `{text}` |

Result fields added in v2: `mode`, `suggested_followups`, `verification[]`, `cached`, `session_id`, `session_recovered`,
`sources[].publisher`.

## 7. Tests

```bash
cd backend  && python -m unittest discover -s tests -t .      # no network, no API keys needed
cd frontend && python -m unittest discover -s tests -t .
```

`backend/tests/test_api.py` exercises the real FastAPI app and is skipped unless `fastapi` and `httpx` are installed
(`pip install httpx`). The frontend smoke tests run `app.py` against a stub of Streamlit.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` says `misconfigured` | it lists the missing settings |
| Answers say "I do not know" for everything | index empty/unreachable, or `EMBED_PROVIDER=cloudflare` vectors differ from your index: run `python -m app.embeddings --check` |
| `keyword_index: building` for minutes | the shipped `keyword.sqlite.gz` is missing/stale; commit a fresh one (`python -m app.keyword_db build`) |
| `keyword_index: unavailable` | host Python lacks SQLite FTS5 (rare) or Qdrant was unreachable; semantic search still works |
| 429 "free allowance for today is used up" | Cloudflare's daily neuron allowance or Groq's limits; wait or upgrade |
| Voice button missing | Streamlit older than 1.40 |
| First request after idle is slow | free hosts sleep; the UI says so when the backend is unreachable |
