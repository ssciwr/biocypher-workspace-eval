# Evaluating open-weight models in the agentic workspace

How to quantitatively compare models (open-weight via LiteLLM, Claude as reference) on building a simple BioCypher adapter for the [Synthetic Protein Interaction Dataset](https://doi.org/10.5281/zenodo.21455347) (Zenodo, version 1.0.5). Setup of the proxy and the workspace is described in [testing_agentic_workspace.md](./testing_agentic_workspace.md).

**Implementation:** the harness (runner, grader, metrics, prompts, pinned data) is in [`evaluation/`](../evaluation/) of this repository (usage in the [README](../README.md)); the backend under test is a checkout of [biocypher-components-registry](https://github.com/ssciwr/biocypher-components-registry); the reference solution is the separate repository [synthetic-ppi-reference](https://github.com/ssciwr/synthetic-ppi-reference). This document is the design rationale; where numbers or texts differ, the files there are canonical.

The goal is not one "score" per model but a profile that shows **where** in the orchestration a model breaks: emitting tool calls, following the mandatory protocol, writing correct code, recovering from errors, or reporting its own result honestly.

## 1. Design principles

1. **Fixed, versioned inputs.** Dataset, prompts, system prompt, tool set, MCP server version, `MCP_RESULT_MAX_CHARS` and `CLAUDE_MAX_TOKENS` are identical for every model. Changing any of them starts a new evaluation round. No per-model prompt tuning.
2. **Graded by machine, not by reading transcripts.** Every task has acceptance criteria that a hidden grader checks against a ground truth. Transcripts are only read to classify failures.
3. **A task ladder, not one big task.** Each level isolates one capability. A model failing to scaffold should not automatically score zero on "can it write the adapter code", so higher levels start from a **seeded reference state** (see 3.2).
4. **No human in the loop.** One user prompt per run. If the model stops and asks a question, the harness answers with one fixed reply (`Proceed with your best judgement and complete the task.`), at most twice; each is counted as an intervention.
5. **Budgets instead of open-ended runs.** Each run has a hard cap on LLM calls, wall time and tokens. Hitting a cap is a recorded outcome (`budget_exhausted`), not a crash.
6. **Repetitions.** Agent runs are nondeterministic even at temperature 0 (batching, sampling in serving stacks). Run every (model, level) pair **n ≥ 5** times, better 10, and report rates with confidence intervals.

## 2. The dataset

Foundation: **Synthetic Protein Interaction Dataset**, version 1.0.5, [doi:10.5281/zenodo.21455347](https://doi.org/10.5281/zenodo.21455347), MIT license (Carreño 2026). It contains synthetic protein–protein interactions in the OmniPath column layout (the same columns as a subset of the registry's `data/in/sample_networks_omnipath.tsv`), plus a `croissant.jsonld` metadata file.

| Property | Value |
|---|---|
| File | `synthetic_protein_interactions.tsv`, 23 data rows, 15 columns |
| Columns | `source`, `target`, `source_genesymbol`, `target_genesymbol`, `is_directed`, `is_stimulation`, `is_inhibition`, `consensus_direction`, `consensus_stimulation`, `consensus_inhibition`, `type`, `ncbi_tax_id_source`, `entity_type_source`, `ncbi_tax_id_target`, `entity_type_target` |
| Nodes | 15 proteins (UniProt accessions), three organisms (9606, 10090, 10116) |
| Interaction types | activation, binding, inhibition, phosphorylation, ubiquitination |
| md5 | `155577b25e2e8460a59e8a096875edfa` (TSV), `0ad5e18be753bc06b7993fed50f6ccb1` (croissant) |

Why it suits the evaluation: it is small (fits every context window), synthetic (the interactions cannot be recalled from real databases), has no separate node file (nodes must be derived from both edge endpoints), and ships **built-in data-quality issues**, partly documented in the record description and partly found by inspection:

| Issue | In the data | Specified handling |
|---|---|---|
| exact duplicate row | `TP53 → CREB1` ubiquitination, twice (documented) | keep once |
| symbol case inconsistency | `GAPDH` and `gapdh` for `P04406` (documented) | upper-case gene symbols |
| duplicate hidden by case | `MYC → GAPDH` and `MYC → gapdh`, phosphorylation, otherwise identical (not documented: the description mentions one duplicate) | keep once; only visible after normalization |
| undirected edge | `EGFR – SOD1` binding with `is_directed = 0`, while `SOD1 → EGFR` binding exists as a directed row | keep both rows as separate edges with `is_directed` as given; add no reverse edges |
| contradictory flags | the same `EGFR – SOD1` row has `is_stimulation = 1` and `is_inhibition = 1` | keep the values as given (no "fixing" of source data) |
| integer-encoded booleans | all `is_*` and `consensus_*` columns are `0`/`1` | convert to Python `bool` |
| cross-species interactions | e.g. mouse `SOD1` (10090) with human `EGFR` (9606) | taxon is a node property, taken per endpoint |

After correct handling the graph has **15 nodes and 21 edges**.

Two evaluation variants are derived from the published file:

- **clean** (L3, L4): duplicates removed and symbols upper-cased; 21 rows, same columns. Tests implementation without data cleaning.
- **original** (L1, L5, L6): the file byte-for-byte as published; 23 rows. Tests whether the model handles the issues as specified.

Both variants have the same ground truth.

### Preparation script

[`evaluation/prepare_data.py`](../evaluation/prepare_data.py) downloads the pinned files from Zenodo, verifies the md5 checksums, and writes `raw/`, `clean/`, `original/` and `ground_truth.json` to [`evaluation/data/`](../evaluation/data/README.md), where the outputs are committed (MIT allows redistribution with the license notice and attribution). Output: `23 raw rows -> 15 nodes, 21 edges`. `ground_truth.json` never goes into a workspace.

**Metadata note:** the bundled `croissant.jsonld` says version `1.0.1` and cites `doi:10.5281/zenodo.16902349`, while the record is version 1.0.5 (`doi:10.5281/zenodo.21455347`). It also types `is_directed`, `is_stimulation` and `is_inhibition` as boolean but the `consensus_*` flags as integers, although all six are encoded the same way. The prompts therefore fix the types explicitly. Consider reporting the mismatch to the dataset author.

## 3. Task ladder

### 3.1 Levels

| Level | Capability tested | Starting state | Pass criterion (checked by grader) |
|---|---|---|---|
| **L0** Tool call | emits a well-formed tool call and uses the result | empty workspace | `get_cookiecutter_instructions` was called and the reported template location is `https://github.com/biocypher/biocypher-cookiecutter-template` (with or without `.git`; see 3.4) |
| **L1** Local file tools | reads files instead of guessing | original variant in `data/` | exact row count (23), column names, and number of distinct proteins (15) |
| **L2** Protocol + scaffold | follows the mandatory sequence from the system prompt | dataset in `data/` | `check_project_exists` → `get_cookiecutter_instructions` → `cookiecutter` via `run_command`, in that order; `synthetic-ppi/` exists with the template layout (3.4); no scaffold files created with `write_file`; no question asked to the user |
| **L3** Adapter implementation | writes working adapter code | reference scaffold + clean variant | hidden grader: nodes and edges match ground truth |
| **L4** Schema + graph build + tests | integrates with BioCypher end to end | reference scaffold + clean variant | L3 checks + `schema_config.yaml` contains the required types + `create_knowledge_graph.py` exits 0 + Parquet output has 15 `Protein` and 21 `PairwiseMolecularInteraction` rows with `bool` flag columns + model ran `pytest`, it passes, and the template's dummy tests were replaced by tests of the real adapter |
| **L5** Robustness | handles specified data-quality issues | reference scaffold + original variant | L4 checks against the same ground truth (issues handled as specified) |
| **L6** Full autonomous | everything in one run | original variant only | L2 + L5 checks |

L0–L1 are cheap gates: a model (or serving setup) that fails them reliably usually has a tool-calling parser or chat-template problem in the serving stack, not a reasoning problem. Fix the serving setup before interpreting anything above.

L6 versus the sum of L2–L5 shows the cost of **long-horizon** work: models that pass every level in isolation but fail L6 lose track over long contexts.

### 3.2 Seeded reference state

For L3–L5, the runner seeds the workspace with the **pristine scaffold**: the root commit of the [synthetic-ppi-reference](https://github.com/ssciwr/synthetic-ppi-reference) repository, which is the unmodified output of the cookiecutter template (commit `01d32f1`) run exactly as the MCP instructions describe, with none of its defects fixed (3.4). It goes to `synthetic-ppi/`, the dataset to `synthetic-ppi/data/`. This decouples the levels: L3 measures implementation only.

The **reference solution** (3.5) validates the grader (section 5).

### 3.3 Optional stress dimensions

Run these as separate conditions on L3/L4 only, after the base ladder:

- **Truncation:** `MCP_RESULT_MAX_CHARS=2000`. Measures whether the model narrows tool arguments after a truncated result instead of repeating the call (the system prompt asks for that).
- **Injected failure:** reference scaffold with one additional broken test. (The missing `pyarrow` dependency is already a natural injected failure in L4–L6, see 3.4.) Measures error recovery: does it read the traceback, fix the cause, re-run?
- **Change request:** a second scripted user turn after success (`Also drop all interactions of type binding, and add a node property organism: "Homo sapiens" for 9606, "Mus musculus" for 10090, "Rattus norvegicus" for 10116.`). Measures editing existing code instead of rewriting it, and multi-turn consistency.
- **Metadata available:** L5 with `croissant.jsonld` placed next to the TSV versus without. Measures whether the model uses dataset metadata, and whether it is misled by the type and version inconsistencies noted in section 2.

### 3.4 Verified toolchain facts (checked 2026-10-09)

These were checked by calling the MCP server directly, generating the scaffold with the exact command the MCP returns, and running the reference solution (3.5) through BioCypher. Re-check them when any version changes.

| Component | Version | Facts that affect the evaluation |
|---|---|---|
| BioCypher MCP (`mcp.biocypher.org/mcp`) | server `biocypher_mcp` 1.13.1 | 9 tools: `get_available_workflows`, `check_project_exists`, `get_cookiecutter_instructions`, `get_adapter_creation_workflow`, `get_phase_guidance`, `get_implementation_patterns`, `get_decision_guidance`, `get_schema_configuration_guidance`, `get_resource_management_guidance` |
| `get_cookiecutter_instructions` | – | `template_url` = `https://github.com/biocypher/biocypher-cookiecutter-template` (L0 ground truth; accept a trailing `.git`, which the returned command template uses). The instructions tell the model to **ask the user for the project name and confirm the context** before running cookiecutter, which conflicts with unattended runs, so the L2/L6 prompts give the name and say not to ask. |
| `check_project_exists` | – | Runs on the **MCP server** (returns `"project_path": "/app"`) and always returns the generic expected structure; it cannot see the workspace. Its result carries no information, so only the call order counts as a protocol check. |
| Cookiecutter template | commit `01d32f1` (HEAD on 2026-10-09) | With `project_name=synthetic-ppi` and default derivations it creates `synthetic-ppi/` with `src/synthetic_ppi/adapters/synthetic_ppi_adapter.py` (class `SyntheticPpiAdapter(data_source, **kwargs)` returning dummy data), `config/schema_config.yaml` and `config/biocypher_config.yaml` (Neo4j, offline, Biolink 3.2.1 head ontology, `output_directory: biocypher-out`), `create_knowledge_graph.py`, `tests/test_synthetic_ppi_adapter.py`, `pyproject.toml` (`requires-python >= 3.13`, `biocypher>=0.17.0`), Docker files. |
| BioCypher | 0.17.0 | Neo4j offline output is **Parquet** by default: `biocypher-out/Protein-part000.parquet` and `biocypher-out/PairwiseMolecularInteraction-part000.parquet`. Biolink 3.2.1 contains both `protein` and `pairwise molecular interaction` (`is_a: pairwise gene to gene interaction`). |

**Scaffold defects the model has to overcome** (part of what L4–L6 measure; record per run which ones the model fixed):

| Defect in the generated scaffold | Effect |
|---|---|
| `pyproject.toml` lacks `pyarrow` (BioCypher's `neo4j` extra) | `create_knowledge_graph.py` fails with `ModuleNotFoundError: No module named 'pyarrow'` until it is added |
| `create_knowledge_graph.py` uses the placeholder `data_source = "data/your_data.csv"` | the graph build fails until it points to the dataset |
| the example `get_edges()` docstring says `(source_id, target_id, edge_label, edge_type, properties)` and the example tuples follow that order | wrong for BioCypher, whose edge tuple is `(edge_id, source_id, target_id, label, properties)`; models that copy the example produce broken edges |
| template tests check the dummy implementation (`get_metadata`, `validate_data_source`, CSV input) | all 7 pass on the **unimplemented** scaffold; 5 of them fail on a correct implementation. "pytest passes" alone therefore rewards doing nothing, and a correct solution requires rewriting the tests |
| `schema_config.yaml` contains example `protein`/`gene` types with unrelated properties | must be replaced, not extended |
| `.gitignore` excludes `data/` | a model that commits its project silently leaves the dataset out |
| the post-generation hook runs `git init` and commits the scaffold under the user's git identity | in the workspace this happens with whatever git identity the sandbox has |

**BioCypher masks adapter errors in the graph output.** A deliberately naive adapter (no normalization, no deduplication, flags as `int`) yields 46 node and 23 edge tuples, yet the Parquet output still has exactly 15 nodes and 21 edges: BioCypher deduplicates by ID and keeps the first occurrence (`GAPDH` wins over `gapdh` only by row order), and it writes the flags as `int64` although the schema declares `bool`. The graph output counts therefore cannot detect data-handling errors; the data-quality checks must be run on the **adapter's tuples** (grader step 2), and the output check must include column types.

The adapter environment needs Python ≥ 3.13 (template requirement). The workspace's `run_command` environment must provide it, or the model has to create one, which costs cycles unrelated to model quality; provide it in the harness.

### 3.5 Reference solution

The repository [synthetic-ppi-reference](https://github.com/ssciwr/synthetic-ppi-reference) holds the verified reference: `git diff <root commit>` is the complete solution (adapter, schema, tests, `pyarrow` dependency, data path, un-ignored `data/`). Verified with Python 3.13.12, BioCypher 0.17.0 and pyarrow 26.0.0: 11 tests pass, the graph build writes 15 `Protein` and 21 `PairwiseMolecularInteraction` records with `bool` flag columns, and its tests catch all grader mutants.

## 4. Prompts

Rules for all prompts:

- The backend system prompt is unchanged (it is part of what is evaluated).
- The user prompt states **the task, the fixed interfaces, and the acceptance criteria**, but not the steps. Fixed interfaces (class name, method names, labels, ID scheme) are what makes automatic grading possible; leaving them open would test guessing, not ability.
- Prompts are stored as files (`prompts/L0.md` … `prompts/L6.md`) and versioned; the prompt file hash is logged with each run.
- Same wording for all models. No "think step by step", no few-shot examples unless they are added for all models as a separate condition.

The prompts are in [`evaluation/prompts/`](../evaluation/prompts/), written out in full (no placeholders) so that each file's hash pins exactly what the model saw. Their structure:

| Part | Used in | Content |
|---|---|---|
| task | all | L0: report the cookiecutter template; L1: report rows, columns, proteins; L2: scaffold `synthetic-ppi`; L3–L5: implement in the seeded `synthetic-ppi/`; L6: scaffold and implement |
| no-confirmation line | L2, L6 | "Do not ask me for confirmation; use the defaults from the instructions for all other values." (the MCP instructions tell the model to ask, see 3.4) |
| interface block | L3–L6 | class `SyntheticPpiAdapter(data_source)` as scaffolded, node and edge tuple formats, labels, ID scheme `<source>_<target>_<type>`, property names and types, schema entries |
| build/tests | L4–L6 | make `create_knowledge_graph.py` build the graph from the dataset, make the tests test the real adapter, run them |
| data-quality rules | L5, L6 | upper-case symbols; rows identical after that kept once; everything else as given |
| final lines | all | L0 `TEMPLATE:`; L1 `ROWS:`/`COLUMNS:`/`PROTEINS:`; L2 `STATUS:`; L3–L6 `STATUS:` and `TESTS: <passed>/<total>` |

The final lines make claims parseable (section 6, claim accuracy), and the runner uses them to detect a turn that ended with a question (scripted reply).

## 5. Grading

The grader runs **after** the agent finished, in a fresh process, on a copy of the workspace. It is never visible to the model during the run (otherwise models can read and game it).

1. **Load the adapter:** find the file defining `class SyntheticPpiAdapter` (expected at `synthetic-ppi/src/synthetic_ppi/adapters/synthetic_ppi_adapter.py`), import it with `importlib` from that path, and instantiate it with `data_source=<dataset file>`. Import or instantiation failure = all L3+ checks fail.
2. **Compare with `ground_truth.json`:**
   - node precision/recall on accessions (15 expected); label = `protein`; property accuracy (`genesymbol` upper case, `ncbi_tax_id` is `int`); exactly one node per accession (no `GAPDH`/`gapdh` split)
   - edge precision/recall on edge IDs (21 expected); label = `protein_protein_interaction`; property accuracy with type check (`bool`, not `int` or `str`)
   - one named check per data-quality issue, so failures are attributable: `dedup_exact` (TP53→CREB1 once), `dedup_after_case` (MYC→GAPDH once), `undirected_kept` (`P00533_P08228_binding` present with `is_directed = False`, no extra reverse edge created), `flags_unchanged` (that row keeps both `is_stimulation` and `is_inhibition` true), `bool_types`
   - no exceptions while iterating the generators; generators can be iterated twice (common bug: reading a file handle once)
3. **L4+:** parse `schema_config.yaml` and check the two entries; run `create_knowledge_graph.py` in the project's environment with a timeout; check the exit code, the row counts of `biocypher-out/Protein-part000.parquet` (15) and `biocypher-out/PairwiseMolecularInteraction-part000.parquet` (21), and that the six flag columns are `bool` (counts alone do not detect data-handling errors, see 3.4). Record which scaffold defects were fixed (`pyarrow` added, data path set, example edge order not copied, template tests replaced).
   - Run the project's tests yourself and record pass/fail and the number of tests that call `get_nodes`/`get_edges` on the real dataset format.
   - **Test strength:** run the model's tests against each grader mutant (below) dropped into the project; the fraction of mutants that make at least one test fail measures whether the model wrote meaningful tests or just made the suite green.
4. **Protocol checks** (from the event log, section 6): order of the mandatory tool calls; no `write_file` creating scaffold-template files; `pytest` called via `run_command` after the last code change.

Each check is a named boolean or ratio, stored per run. **Pass** for a level = all its checks pass. The ratios (precision, recall, property accuracy) are the **partial score**, which separates "almost right" from "nothing works".

Validate the grader before the first model run (`uv run python -m evaluation.grader validate`): the reference solution must pass all checks, and seven deliberately broken mutants (wrong ID scheme, flags left as `int`, no deduplication, symbols not upper-cased, reverse edge added for the undirected row, edge tuple in the template's wrong order, generator exhausted after one pass; [`evaluation/mutants.py`](../evaluation/mutants.py)) must each fail at least their expected checks. Current state: reference OK, all mutants detected, reference tests catch 7/7.

## 6. Metrics

### Logging

Log every run as JSONL of the workspace events (`turn_started`, `usage`, `tool_call`, `tool_result`, `text_delta`, `turn_done`/`turn_error`) plus run metadata. All metrics below are computed from these logs plus the grader output, so they can be recomputed later without re-running models.

Run metadata: model alias and underlying model, BioCypher version in the adapter environment, quantization, serving stack and version (vLLM/llama.cpp/Ollama, tool-call parser), context window setting, temperature, LiteLLM version, MCP server URL and version, cookiecutter template version, prompt file hash, dataset hash, repository commit, `CLAUDE_MAX_TOKENS`, `MCP_RESULT_MAX_CHARS`, budgets, start time, repetition index.

### Outcome

| Metric | Definition |
|---|---|
| success | all checks of the level pass |
| partial score | mean of the ratio checks (precision, recall, property accuracy) |
| termination | `done` / `budget_exhausted` / `turn_error` / `context_overflow` / `malformed_output` |
| interventions | number of scripted replies needed (0–2) |

### Efficiency

| Metric | Definition | Note |
|---|---|---|
| **cycles** | number of LLM calls = number of `usage` events | the cycle count; report for successful runs separately from failed ones |
| tool calls | number of `tool_call` events, by tool name | |
| wall time | `turn_started` → terminal event | depends on hardware; compare only within one serving setup |
| input / output tokens | sum over `usage` events | tokenizers differ between model families: compare tokens only within a family; across families use cycles, tool calls and characters |
| peak context | maximum `input` of a single call | shows how close the model gets to its context window |

### Process quality (where open-weight models typically differ)

| Metric | Definition |
|---|---|
| tool error rate | `tool_result` with `is_error` / all tool results (invalid arguments, unknown paths) |
| command failure rate | `run_command` results with non-zero exit / all `run_command` calls |
| invalid tool calls | calls to non-existent tools or with schema-invalid arguments |
| loop score | max number of identical consecutive `(tool name, args)` calls; ≥ 3 counts as a loop |
| recovery rate | fraction of failed tool results followed (within 3 calls) by a *different* call to the same tool or a fix, rather than a repeat or giving up |
| truncation response | after a truncated result: next call to the same tool has narrower arguments (yes/no) |
| protocol adherence | fraction of the protocol checks passed (L2, L6) |
| **claim accuracy** | model's `STATUS`/`TESTS` line agrees with the grader (claims `DONE` but fails = overclaim) |

Claim accuracy matters as much as success: a model that fails and says so is usable with a human in the loop; a model that fails and reports success is not.

## 7. Running the comparison

Budgets per run (same for all models, in `evaluation/config.toml`): L0–L1: 10 calls; L2: 25; L3: 40; L4–L5: 60; L6: 100 calls; 10–45 min wall time depending on the level.

Procedure:

1. For each model: run L0 and L1 three times. Stop and fix the serving setup if they fail (tool-call parser, chat template, context window).
2. Run the ladder L2–L6 with n repetitions each. Interleave models and repetitions (round robin) instead of running all repetitions of one model in a block, so that changes of the remote MCP server or network conditions spread evenly. Better: run a pinned local MCP server.
3. Fresh session and fresh workspace for every run; nothing carries over.
4. Include Claude (direct Anthropic API) as the reference upper bound, run through the same harness and prompts.

### Harness

[`evaluation/runner.py`](../evaluation/runner.py) creates a `SessionManager` session of the registry backend in-process (no HTTP and no GitHub auth), gives `run_command` a per-run Python 3.13 venv with the `workspace_env` packages, seeds the workspace, submits the level prompt, records the session events to `events.jsonl`, enforces the budgets (interrupts the turn when the call cap or time is reached), sends the scripted reply when a turn ends without the expected final line, copies the workspace, and grades it. Runs are resumable. [`evaluation/metrics.py`](../evaluation/metrics.py) computes the metrics below and the report table. Usage: [README](../README.md).

### Reporting

Per model × level:

| Model | Level | n | success rate (95 % CI) | partial score (median) | cycles, successful runs (median, IQR) | tool error rate | loops | overclaims |
|---|---|---|---|---|---|---|---|---|

- Success rates with Wilson score intervals; with n = 10 the interval is wide (e.g. 5/10 successes gives roughly 24–76 %), so only treat large differences as real, or increase n for the models you shortlist.
- pass@k (success in at least one of k runs) for the "with retries" scenario, alongside the plain success rate.
- Cycles and tokens only over successful runs (failed runs often stop early or run to the budget, which distorts medians); report the budget-exhausted share separately.
- A failure-mode table per model (counts of: tool-call format errors, protocol violations, wrong data handling per data-quality check, loops, context overflow, overclaims) is usually more informative than the success rate alone.

## 8. Limits of this evaluation

- One small task in one domain (protein–protein interactions, 23 rows). It shows orchestration limits, not general capability; results do not transfer automatically to large or messy real datasets.
- The remote MCP server and the cookiecutter template can change between runs; pin or record their versions.
- Serving configuration (quantization, tool-call parser, context length) can change outcomes as much as the model choice. Report it with every result and compare models under the same serving stack where possible.
- The fixed interfaces in the prompts make grading possible but also make the task easier than an open-ended request; L6 is the closest to real use.

## References

- Carreño, E. (2026). *Synthetic Protein Interaction Dataset* (Version 1.0.5) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21455347. License: MIT.
