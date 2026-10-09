# Testing the agentic workspace

Manual end-to-end test of the agentic workspace, once against Anthropic (Claude) directly and once against other providers (e.g. OpenAI) through the LiteLLM proxy. Background on the architecture is in the registry's [agentic_workspace.md](https://github.com/ssciwr/biocypher-components-registry/blob/main/docs/agentic_workspace.md), the route contract in [API.md](https://github.com/ssciwr/biocypher-components-registry/blob/main/docs/API.md).

**All commands below run in a checkout of [biocypher-components-registry](https://github.com/ssciwr/biocypher-components-registry)** (the backend, its compose files and `deploy/litellm/config.yaml` live there), not in this repository.

There are three ways to drive the workspace; each section below covers all three:

| Mode | What runs | Good for |
|---|---|---|
| CLI | `src/core/workspace/client_loop.py` in your terminal | quickest check of LLM + MCP connectivity |
| Local dev | `uvicorn` backend + Vite frontend (`pnpm run dev`) | UI testing, fast iteration |
| Docker Compose | backend container (+ LiteLLM container) | testing the containerized setup (sandbox user, permissions) |

## Prerequisites

- `uv sync --group dev` in the repository root.
- A separate Python environment for adapter development with BioCypher and cookiecutter installed. The CLI asks for it on start; the agent runs `run_command` in it.
- For the UI: `.env` created from `.envsample` with a GitHub OAuth app (`GITHUB_OAUTH_CLIENT_ID`, `GITHUB_OAUTH_CLIENT_SECRET`, callback `http://127.0.0.1:8000/api/v1/auth/github/callback`) and `AUTH_SESSION_SECRET`. Workspace sessions require GitHub sign-in.
- For the frontend: Node.js 24+ and pnpm, then `cd frontend && pnpm install`.
- Reachable MCP server: `https://mcp.biocypher.org/mcp` by default, or set `BIOCYPHER_MCP_URL`.

Quick MCP check (no LLM key needed):

    uv run python src/core/workspace/client_loop.py --list-tools

## A. Anthropic (Claude)

Needs an Anthropic API key. Make sure `ANTHROPIC_BASE_URL` is **unset**: thinking and prompt caching are only on by default when talking to Anthropic directly.

### CLI

    unset ANTHROPIC_BASE_URL
    export ANTHROPIC_API_KEY="<your Anthropic key>"
    export CLAUDE_MODEL="claude-opus-4-8"   # optional, this is the default
    uv run python src/core/workspace/client_loop.py

### Local dev (UI)

    # terminal 1
    unset ANTHROPIC_BASE_URL
    uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000

    # terminal 2
    cd frontend && pnpm run dev

Open `http://127.0.0.1:5173/workspace`, sign in with GitHub, and paste your Anthropic key when the UI asks for it. The key is stored for that session only.

### Docker Compose

    docker compose -f docker-compose-sqlite.yml up --build

The compose frontend service is only a placeholder, so test the UI with `pnpm run dev` as above (the Vite proxy forwards to the backend on port 8000).

### What to expect

- `[thinking...]` markers (CLI) or a thinking indicator (UI) on harder prompts.
- In the CLI `[usage]` lines: `cache_read` > 0 from the second request of a turn on.

## B. Other providers via LiteLLM (e.g. OpenAI)

The backend only speaks the Anthropic Messages API; LiteLLM translates it to the provider. Model aliases live in `deploy/litellm/config.yaml`; adjust them to the models you want to compare.

Add to `.env`:

    OPENAI_API_KEY="sk-..."                         # provider key, stays inside the proxy
    LITELLM_MASTER_KEY="sk-a-random-long-string"   # what the backend/UI uses as "API key"
    CLAUDE_MODEL="gpt-4.1"                          # alias from deploy/litellm/config.yaml

`LITELLM_MASTER_KEY` is the proxy's admin key. Do not hand it to other people; for multiple testers use LiteLLM virtual keys (needs LiteLLM's database, not part of this setup).

### Start only the proxy (for CLI and local dev)

    docker compose -f docker-compose-sqlite.yml -f docker-compose.litellm.yml up litellm

The proxy listens on `127.0.0.1:4000`. Sanity check:

    curl -s http://127.0.0.1:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"

### CLI

    export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
    export ANTHROPIC_API_KEY="$LITELLM_MASTER_KEY"
    export CLAUDE_MODEL="gpt-4.1"
    uv run python src/core/workspace/client_loop.py

### Local dev (UI)

    # terminal 1
    export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
    export CLAUDE_MODEL="gpt-4.1"
    uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000

    # terminal 2
    cd frontend && pnpm run dev

In the UI, paste the `LITELLM_MASTER_KEY` (not the OpenAI key) as API key.

### Docker Compose (backend + proxy)

    docker compose -f docker-compose-sqlite.yml -f docker-compose.litellm.yml up --build

The overlay points the backend at `http://litellm:4000` and passes `CLAUDE_MODEL` / `CLAUDE_MAX_TOKENS`. Again use `pnpm run dev` for the UI.

### Switching models

`CLAUDE_MODEL` is read once at startup:

- CLI / local dev: change the variable and restart the process.
- Compose: change `CLAUDE_MODEL` in `.env`, then `docker compose -f docker-compose-sqlite.yml -f docker-compose.litellm.yml up -d backend`.

New models: add an entry to `deploy/litellm/config.yaml` and restart the `litellm` service.

### What to expect

- No thinking markers and `cache_read=0 cache_write=0` in usage: thinking and prompt caching are off by default behind a proxy (`CLAUDE_THINKING`, `CLAUDE_PROMPT_CACHE` to override; untested whether LiteLLM accepts them for OpenAI models).
- A 400 about `max_tokens`: the model has a smaller output limit; lower `CLAUDE_MAX_TOKENS`.
- Weaker tool use from small models (wrong tool arguments, skipping the mandatory cookiecutter step). That is a model capability result, not a backend bug; note it in the comparison.

## Test script (both setups)

Run the same prompts in each setup so results are comparable.

1. **Tool listing:** "Which BioCypher tools do you have available?" — expect an MCP tool call and a list.
2. **MCP lookup:** "What does the BioCypher schema configuration look like for a protein–protein interaction adapter?" — expect MCP calls, answer grounded in tool results.
3. **Adapter creation:** "Create a new BioCypher adapter for a dataset of protein interactions." — expect `check_project_exists` and `get_cookiecutter_instructions`, then `cookiecutter` via `run_command`, files appearing in the workspace pane, and a `pytest` run at the end.
4. **Interrupt:** start a long prompt and interrupt it — the turn ends with "interrupted", the next prompt works.
5. **Wrong key:** start a new session with an invalid key and send a message (the key is only checked on the first turn) — expect "Your API key was rejected. Check it or use another key."
6. **Unknown model** (LiteLLM only): set `CLAUDE_MODEL` to an alias not in the config — expect a generic "Error from AI model stream" in the UI and the LiteLLM error in the backend log.

For a quantitative comparison across models (task ladder, grading, metrics) see [evaluating_models_workspace.md](./evaluating_models_workspace.md) and the harness in this repository ([README](../README.md)). For a quick comparison, record per model: did steps 1–3 succeed, number of turns/tool calls, whether tests passed, input/output tokens from the `[usage]` lines, and notable failures.

## Troubleshooting

- **"could not connect to MCP server"**: check `BIOCYPHER_MCP_URL` and `--list-tools`.
- **401 on workspace routes**: not signed in with GitHub, or a stale session token; start a new session.
- **Errors in the UI are generic by design**: the details are in the backend log (terminal or `docker compose logs backend`) and, for LiteLLM, in `docker compose logs litellm`.
