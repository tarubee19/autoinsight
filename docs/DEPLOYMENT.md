# Deployment guide

> **Status of these instructions.** The Docker images, compose files and the single-container start script are written
> and their configuration is validated, and the single-container script was booted and ran a complete analysis during
> development. The author's development machine had **no Docker daemon**, so image builds, `docker compose up` and the
> hosting-platform steps below have **not been executed end to end**. The GitHub Actions workflow builds every image and
> smoke-tests compose — treat a green CI run as your first verification, and expect to adjust platform-specific clicks
> (platform UIs change).

## Choose a deployment

| Goal | Use | LLM-written code? |
|---|---|---|
| Try it on your laptop | [A. Local, no Docker](#a-local-no-docker) | offline templates (or Docker sandbox if you have Docker) |
| Run the full stack on your machine | [B. Docker Compose](#b-docker-compose) | offline; **C** adds a real LLM |
| Real LLM + hardened sandbox on **your own machine** | [C. Compose + LLM override](#c-compose--real-llm--docker-sandbox-your-own-machine-only) | yes, in Docker |
| Free public demo | [D. Hugging Face Space](#d-hugging-face-space-free-public-demo) or [E. Render/Railway/Fly/Cloud Run](#e-any-single-container-host) | **no** — offline only |
| A server you control | [F. VPS + Caddy](#f-vps-with-docker-compose-and-https) | offline; LLM mode only with a separate sandbox host |

**Rule of thumb:** anything reachable from the internet runs in **offline mode**. See [SECURITY.md](SECURITY.md).

## A. Local, no Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[all]"
autoinsight serve                       # API   http://localhost:8000/docs
streamlit run app/streamlit_app.py      # UI    http://localhost:8501   (second terminal)
```

## B. Docker Compose

```bash
docker compose up --build               # UI http://localhost:8501   API http://localhost:8000/docs
docker compose down                     # stop (runs are kept in ./runs)
```

Two services: `api` (FastAPI + the pipeline, offline mode, local sandbox for the vetted templates only) and `ui`
(Streamlit, talks to `http://api:8000`). Runs are persisted in `./runs` on the host. To protect the API set
`AUTOINSIGHT_API_KEY` in `.env` — compose passes it to both services.

## C. Compose + real LLM + Docker sandbox (your own machine only)

```bash
cp .env.example .env                     # set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
docker compose -f docker-compose.yml -f docker-compose.llm.yml up --build
```

What the override does: builds the **sandbox image** (`autoinsight-sandbox:latest`, same pinned numeric stack as the API
image — the model is trained in one and loaded in the other), runs the API as root so it can use the **Docker socket**,
switches to `AUTOINSIGHT_SANDBOX_MODE=docker`, and sets `AUTOINSIGHT_HOST_RUNS_DIR=${PWD}/runs`.

* **Why `AUTOINSIGHT_HOST_RUNS_DIR`?** The API container starts *sibling* containers through the host's Docker daemon.
  Their bind mounts are resolved on the **host**, so the API must translate its in-container `/app/runs/...` paths to the
  host path. Start compose from the repository root so `${PWD}` is right.
* ⚠️ **Mounting the Docker socket is root-equivalent on the host.** Use this on a laptop or a throw-away VM only.

Without compose: `docker build -f docker/Dockerfile.sandbox -t autoinsight-sandbox:latest .` once, then
`autoinsight run ...` on the host with an API key in the environment (`autoinsight doctor` shows whether the sandbox
image is ready).

## D. Hugging Face Space (free public demo)

Result: a public URL such as `https://huggingface.co/spaces/<you>/autoinsight`, running offline mode.

1. Push this repository to GitHub first (see the README) and make sure it is **public** — the Space builds from it.
2. Create a Hugging Face account → **New Space** → name `autoinsight` → **SDK: Docker** → blank template → *CPU basic (free)* → Public.
3. Make sure `deploy/hf-space/Dockerfile` (the `REPO_URL` line) and `deploy/hf-space/README.md` point to this repository.
4. Add those two files to the Space repository — **the file must be named `README.md` and keep its YAML header**
   (`sdk: docker`, `app_port: 7860`):
   ```bash
   git clone https://huggingface.co/spaces/<you>/autoinsight hf-autoinsight
   cp deploy/hf-space/Dockerfile deploy/hf-space/README.md hf-autoinsight/
   cd hf-autoinsight && git add . && git commit -m "Deploy AutoInsight" && git push
   ```
5. The Space builds (a few minutes) and starts. It runs one container: the API on `127.0.0.1:8000` and Streamlit on `$PORT`
   (7860) — `docker/start_demo.sh`. Defaults for a public demo: offline mode, 10 MB uploads, runs deleted after 24 h,
   runs stored in `/tmp` (ephemeral).
6. Add the Space URL to the top of the README.

Notes: free Spaces sleep after inactivity (first load takes ~30 s); to redeploy after you change the code, restart or
"Factory reboot" the Space so the Dockerfile re-clones the repository. Set `AUTOINSIGHT_API_KEY` as a Space *secret* only
if you also change the UI to send it; for the demo the API is bound to localhost inside the container and not exposed.

## E. Any single-container host

`docker/Dockerfile.demo` builds one image with API + UI, honours `$PORT`, and defaults to offline mode, so it fits hosts that
run a single web container.

```bash
docker build -f docker/Dockerfile.demo -t autoinsight-demo .
docker run --rm -p 7860:7860 autoinsight-demo        # http://localhost:7860
```

* **Render**: New → Web Service → connect the GitHub repo → *Runtime: Docker* → *Dockerfile path*: `docker/Dockerfile.demo`.
  Render injects `PORT`; no other variables are needed. Free instances spin down when idle.
* **Railway / Fly.io**: deploy from the repo with the same Dockerfile path; make sure the platform's internal port equals `$PORT`.
* **Google Cloud Run**: `gcloud run deploy autoinsight --source . ` with the Dockerfile set to `docker/Dockerfile.demo`
  (or build and push the image); it sets `PORT`. Cloud Run's filesystem is ephemeral — fine for a demo.

Because the Streamlit app is served with XSRF protection disabled in `start_demo.sh` (needed behind some proxies for file
upload), do not put anything sensitive behind this container.

## F. VPS with Docker Compose and HTTPS

For a machine you control (any small Linux VPS). Offline mode only on a public address.

```bash
git clone https://github.com/<you>/autoinsight.git && cd autoinsight
cat > .env <<'EOF'
AUTOINSIGHT_API_KEY=change-me-to-a-long-random-string
AUTOINSIGHT_RETENTION_HOURS=24
AUTOINSIGHT_MAX_UPLOAD_MB=10
EOF
docker compose up -d --build
```

Put a reverse proxy in front for HTTPS and rate limiting. With Caddy (`/etc/caddy/Caddyfile`):

```caddyfile
autoinsight.example.com {
    reverse_proxy localhost:8501
}
api.autoinsight.example.com {
    reverse_proxy localhost:8000
}
```

Then firewall ports 8000/8501 so only the proxy can reach them (e.g. bind them to `127.0.0.1` in the compose `ports:` entries).
If the API is public, `AUTOINSIGHT_API_KEY` is mandatory (clients send `X-API-Key`).

## Environment variable reference

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | – | Enable LLM mode with that provider |
| `AUTOINSIGHT_LLM_PROVIDER` | `auto` | `auto` / `anthropic` / `openai` / `offline` |
| `AUTOINSIGHT_MODEL` | provider default | Model name override |
| `AUTOINSIGHT_SANDBOX_MODE` | `auto` | `auto` (Docker if the image is ready, else local) / `docker` / `local` |
| `AUTOINSIGHT_SANDBOX_IMAGE` | `autoinsight-sandbox:latest` | Image used for sandboxed execution |
| `AUTOINSIGHT_SANDBOX_TIMEOUT_S` | `120` | Wall-clock limit per program (the trainer gets at least 300 s) |
| `AUTOINSIGHT_SANDBOX_MEMORY_MB` | `2048` | Container memory limit |
| `AUTOINSIGHT_MAX_CODE_ATTEMPTS` | `3` | LLM attempts per code-writing agent before the vetted fallback |
| `AUTOINSIGHT_ALLOW_UNSAFE_LOCAL` | `0` | Allow LLM-written code without Docker (**dev only**) |
| `AUTOINSIGHT_SEND_SAMPLE_VALUES` | `1` | Send 3 example values per column to the LLM (`0` = names, types, stats only) |
| `AUTOINSIGHT_MAX_UPLOAD_MB` | `50` | Upload size limit (API) |
| `AUTOINSIGHT_RUNS_DIR` | `runs` | Where runs are stored |
| `AUTOINSIGHT_HOST_RUNS_DIR` | – | Host path of the runs dir when the API itself runs in a container and uses Docker mode |
| `AUTOINSIGHT_RETENTION_HOURS` | `0` | Delete runs older than this (0 = keep) |
| `AUTOINSIGHT_API_KEY` | – | If set, the API requires an `X-API-Key` header; the UI sends it too |
| `AUTOINSIGHT_SAMPLES_DIR` | auto | Location of the bundled sample CSVs (set in the Docker images) |
| `AUTOINSIGHT_SEED` | `42` | Random seed for the model and sampling |
| `API_URL` | `http://localhost:8000` | (UI) where the API is |
| `PORT` | `7860` | (demo image) port for Streamlit |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Refusing to run: An LLM is configured, but no Docker sandbox is available` | Intended. Build the sandbox image (`docker build -f docker/Dockerfile.sandbox -t autoinsight-sandbox:latest .`), or use `--provider offline` |
| `autoinsight doctor` says *Docker CLI found* but *sandbox image not available* | The image is not built, or the Docker daemon is not running/reachable |
| Docker mode: `permission denied` on `/in` or `/out` (not observed — Docker mode could not be run during development) | The code makes each run's output directory world-writable for the sandbox user (uid 65534); on SELinux hosts the bind mounts may additionally need a relabel (`:z`) or a different runs location |
| Compose LLM mode: sandbox cannot find the input files | `AUTOINSIGHT_HOST_RUNS_DIR` must be the **host** path of `./runs`; start compose from the repo root |
| `401 Missing or invalid X-API-Key` | Send the header, or unset `AUTOINSIGHT_API_KEY` for local use |
| `413 File larger than N MB` | Raise `AUTOINSIGHT_MAX_UPLOAD_MB` |
| Model step takes minutes on a small machine | Gradient boosting uses all cores; on busy or very small CPUs set `OMP_NUM_THREADS=1` or `2`. (Observed during development: ~4 s uncontended vs. minutes when two heavy jobs shared a 2-core machine.) |
| UI shows "Cannot reach the API" | Start the API first; check `API_URL` |
| Model loads fail after changing library versions | The model is pickled: train and serve with the same `constraints.txt` versions (the images do) |
| Streamlit upload fails behind a proxy | Allow WebSocket upgrades; `--server.enableXsrfProtection=false` is already set in the demo script |
