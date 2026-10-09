# biocypher-workspace-eval

Testing and quantitative model evaluation of the BioCypher **agentic workspace** (the LLM + BioCypher MCP backend of [biocypher-components-registry](https://github.com/ssciwr/biocypher-components-registry)).

Models (Claude directly, others such as open-weight models via a LiteLLM proxy) build a BioCypher adapter for the [Synthetic Protein Interaction Dataset](https://doi.org/10.5281/zenodo.21455347) on a task ladder L0–L6; every run is graded automatically against a ground truth.

| Document | Content |
|---|---|
| [docs/testing_agentic_workspace.md](docs/testing_agentic_workspace.md) | manual end-to-end test of the workspace, with Claude and via LiteLLM |
| [docs/evaluating_models_workspace.md](docs/evaluating_models_workspace.md) | design of the quantitative evaluation: dataset, task ladder, prompts, grading, metrics, verified toolchain facts |

## Related repositories

The evaluation expects two checkouts next to this one (paths in `evaluation/config.toml`, overridable by environment variables):

| Repository | Role | Override |
|---|---|---|
| `../biocypher-components-registry` | the backend under test: the runner imports its workspace modules (`src/core/workspace/`) from the checkout; check out the backend version you want to evaluate | `EVAL_REGISTRY_REPO` |
| `../synthetic-ppi-reference` | reference solution: its root commit is the pristine cookiecutter scaffold seeded into L3–L5 workspaces, its working tree the reference adapter from which the grader's mutants are derived | `EVAL_REFERENCE_REPO` |

The git commits of both (and of this repository) are recorded in every run's `meta.json`.

## Layout

| Path | Purpose |
|---|---|
| `evaluation/config.toml` | pins (MCP URL, repositories, Python), run settings, levels (dataset, seeding, budgets), models |
| `evaluation/prompts/L0.md` … `L6.md` | the exact user prompts (hashed into every run's metadata) |
| `evaluation/data/` | pinned dataset variants and `ground_truth.json` (see [evaluation/data/README.md](evaluation/data/README.md)) |
| `evaluation/prepare_data.py` | regenerates `evaluation/data/` from Zenodo with checksum verification |
| `evaluation/backend.py` | imports the workspace backend from the registry checkout |
| `evaluation/runner.py` | runs models × levels × repetitions through real workspace sessions |
| `evaluation/grader.py` | grades a run; `validate` checks the grader against the reference and mutants |
| `evaluation/probe.py` | runs inside the graded project's environment and dumps adapter tuples / Parquet output |
| `evaluation/mutants.py` | broken variants of the reference adapter (grader validation, test strength) |
| `evaluation/metrics.py` | per-run metrics (`runs.csv`) and the comparison table (`report.md`) |
| `evaluation/events.py` | event-log helpers shared by grader and metrics |
| `tests/` | tests of the harness |

## Usage

Prerequisites: [uv](https://docs.astral.sh/uv/), `git`, network access (MCP server, Biolink ontology, package index), the two checkouts above, and the API keys named in `evaluation/config.toml` (`api_key_env`). For proxied models, start LiteLLM from the registry checkout first (see [docs/testing_agentic_workspace.md](docs/testing_agentic_workspace.md), section B).

```bash
uv sync

# once, and after any change to the grader, mutants or reference
uv run python -m evaluation.grader validate

# run (resumable: finished runs are skipped)
uv run python -m evaluation.runner --models gpt-4.1 --levels L0 L1 --repetitions 3
uv run python -m evaluation.runner                       # everything in config.toml

# regrade one run, then aggregate
uv run python -m evaluation.grader grade evaluation/runs/gpt-4.1/L3/rep00
uv run python -m evaluation.metrics                      # -> evaluation/runs/runs.csv, report.md
```

Each run directory `evaluation/runs/<model>/<level>/repNN/` contains `meta.json` (versions, hashes, budgets, termination), `events.jsonl` (every workspace event with timestamps), `workspace/` (copy of the final workspace without virtual environments) and `grade.json` (named checks, pass/fail, partial score, claims). Use `EVAL_CONFIG=<file>` to run a separate configuration round.

## How a run works

1. A `SessionManager` session of the registry backend is created in-process (no HTTP, no GitHub auth) with a real MCP connection.
2. The workspace gets its own Python 3.13 venv with the `workspace_env` packages; `run_command` uses it (the user-selected environment in real use).
3. Seeding: L3–L5 get the scaffold from the reference repo's root commit in `synthetic-ppi/` with the dataset in `synthetic-ppi/data/`; L1, L2, L6 get the dataset in `data/`.
4. The level prompt is sent. The runner stops the turn when the call budget (LLM calls) or time budget is used up. If a turn ends without the expected final line, the scripted reply from `config.toml` is sent, at most `max_replies` times; each counts as an intervention.
5. The workspace is copied into the run directory and graded.

The runner configures the backend through its module-level settings (`MODEL`, `MAX_TOKENS`, `RESULT_MAX_CHARS`) and replaces its private `_exec_bin`. A refactoring of the registry's workspace modules can break this, so rerun the tests after updating the registry checkout.

## Grading

The grader works on a copy of the workspace and runs the model's project with `uv run --project`, i.e. with the dependencies the model declared. Data-handling checks run on the adapter's tuples, because BioCypher deduplicates by ID and does not enforce declared types, so the graph output alone hides errors (section 3.4 of the design doc). Level pass criteria are the `REQUIRED` checks in `evaluation/grader.py`.

**Security:** grading executes model-written code (adapter, tests, build script). Subprocesses get an allowlisted environment without credentials, but they run with your user's file and network access. Run evaluations of untrusted models in a container or VM.

## Tests

```bash
uv run pytest                  # fast, no network; runner tests need the registry checkout
EVAL_SLOW=1 uv run pytest      # plus end-to-end grader validation
```

## Known limitations

- Runs are sequential; one evaluation of all levels × 5 repetitions × several models takes hours.
- Recovery and truncation metrics pair tool results with calls by order; an interrupted turn can shift that pairing for its last calls.
- Test strength replaces the model's adapter module with mutants of the reference; tests importing other names from that module count the mutant as caught.
- Context-window overflows are reported by the provider as generic errors (`provider_error`); check the backend log to distinguish them.

## License and citation

MIT, see [LICENSE](LICENSE); the dataset has its own MIT license ([evaluation/data/README.md](evaluation/data/README.md)). Citation metadata: [CITATION.cff](CITATION.cff).

---

*AI involvement in this project is declared in [aidecl.yaml](./aidecl.yaml) following the [AI Declaration Format](https://ai-declaration.org).*
