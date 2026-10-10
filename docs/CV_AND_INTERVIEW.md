# Turning AutoInsight into CV lines and interview answers

## 0. The honesty rule (read first)

Interviewers for AI/ML roles will open the repo and ask you to explain any line on your CV. Claim only what you have
**run yourself**. Before you copy bullets below, tick what is true for *your* copy:

- [ ] CI is green on GitHub (Actions tab) — this also proves the Docker builds and sandbox tests work
- [ ] You ran it once with a **real LLM** (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) and the Docker sandbox, and read the agent logs
- [ ] You ran `python scripts/evaluate.py --real-llm` and saved *your* numbers (`docs/EVALUATION_LLM.md`)
- [ ] The live demo URL works (Hugging Face Space / Render)
- [ ] You ran it on one **real** dataset (e.g. the public IBM Telco churn CSV) and can discuss what it found
- [ ] You can explain every file in `src/autoinsight/` at a high level (the Q&A below is your study guide)

Bullets tagged **[needs: …]** are only true once you have ticked that item.

## 1. CV entry

**Project header (one line)**
> **AutoInsight — Multi-agent system for automated data analysis** · LangGraph, Claude/GPT APIs, Docker, FastAPI, Streamlit, scikit-learn · [github.com/tarubee19/autoinsight](https://github.com/tarubee19/autoinsight)

**Bullets — pick 3–4 (each is backed by something in the repo)**

* Designed a **LangGraph multi-agent workflow** (planner → data-engineer → analyst → modeler → reporter) that turns a raw CSV and a business question into a cleaned dataset, statistical analysis, trained model and a markdown report; self-repair implemented as **graph cycles** with an explicit retry budget and graceful fallback to vetted templates.
* Built a **defence-in-depth execution layer for LLM-written code**: AST allow-list guard → Docker sandbox (no network, read-only FS, non-root, capabilities dropped, CPU/memory/PID/time limits) → output-contract validation; sandbox output is sanitised (symlink removal, per-agent file allow-list) before the host reads or unpickles anything.
* Engineered **reliability for agent output**: reports are checked for invented numbers (every figure traced to computed results, rate shown in the report); statistics use effect sizes and Benjamini-Hochberg correction; model selection uses cross-validation with the threshold chosen on out-of-fold predictions and a leakage check.
* Wrote a **fault-injection harness** (scripted LLM that emits unsafe code, runtime errors, missing outputs, hallucinated features, invented statistics, API outages) — **7/7 scenarios recovered** with a correct final report — plus **100+ automated tests**, ruff linting and a GitHub Actions pipeline.
* Validated on three synthetic datasets with **planted ground-truth drivers**: model importance recovered 100% of the planted drivers (precision@k), the statistical agent 80–100%; held-out ROC AUC 0.74–0.84 / R² 0.93 (synthetic data — not a real-world claim).
* Shipped the trained model as a **prediction API + UI** (FastAPI `/predict`, schema-driven Streamlit form) with input validation, out-of-range/unseen-category warnings, risk bands (against the tuned threshold and base rate) and per-prediction what-if explanations.
* **[needs: live demo]** Deployed a public offline-mode demo on Hugging Face Spaces / Render from a single Docker image.
* **[needs: real-LLM run]** Benchmarked Claude/GPT agents against the vetted-template baseline: *<your numbers: attempts needed, fallbacks used, run time, API cost per run>*.

**Skills line additions:** LangGraph · multi-agent orchestration · LLM tool use/code generation · sandboxed execution (Docker) · prompt-injection-aware design · FastAPI · Streamlit · scikit-learn · statistical testing · evaluation & fault injection · pytest · GitHub Actions CI

**Shorter version (for a one-page CV)**
> *AutoInsight* — LangGraph multi-agent system: upload a CSV + a question → agents clean data, run statistics, train a model, and write a number-verified report. Sandboxed execution of LLM-written code (Docker + AST guard), self-repair loops, fault-injection tests (7/7 recovered), prediction API + UI. Python, LangGraph, FastAPI, Docker, scikit-learn.

## 2. The 60-second pitch

"Most LLM demos stop at generating text. I wanted to build something that does the work and fails safely. AutoInsight takes a CSV
and a question like 'what drives churn?'. A planner picks the target, a data-engineer agent writes and runs the cleaning code, an
analyst agent runs the statistics and charts, a trusted script trains and validates a model, and a reporter writes the business
report. The interesting part is the engineering around the model: the code the LLM writes is checked by an AST guard and run in a
locked-down Docker container, its output is validated against contracts, errors are fed back for self-repair, and if the model
keeps failing the system falls back to a vetted template. The report's numbers are checked against the computed results. I tested it by
injecting faults — unsafe code, bugs, hallucinated features, outages — and all seven scenarios recovered. The trained model is
also exposed through an API and a UI for predictions."

## 3. A 5-minute demo script

1. `autoinsight doctor` → show mode (offline vs. LLM) and sandbox status.
2. Open the UI, pick *Telecom customer churn*, **Run analysis**; narrate the stage tracker as agents finish.
3. **Report tab:** executive summary → drivers with effect sizes → "numbers traced to computed results: 71 of 71".
4. **Agent logs tab:** open the engineer's code; (with a real LLM) show a retry where the first attempt failed and how the error was fed back.
5. **Predict tab:** change *Contract* to *Two year*; the churn probability drops; point at the "what moved this score" chart.
6. Open `examples/self_repair_trace.md`: "this is what happens when the model writes unsafe code, then buggy code".
7. Close with `docs/EVALUATION.md` (planted drivers + fault injection) and the security section.

## 4. Interview questions and strong answers

**Why LangGraph instead of one big prompt or a hand-written loop?**
The workflow has typed shared state, conditional routing, and cycles (retry a node with error feedback). LangGraph expresses
exactly that, and keeps routing separate from the agents. One prompt can't execute code, validate results, or recover; a bare
`while` loop hides control flow. (Be ready to say the graph is small enough that you could also write it by hand — what it buys
is explicit state, routing and easy extension.)

**Why multiple agents? Isn't that over-engineering?**
Separation of concerns with different contracts and failure modes: cleaning has to preserve the target and row counts, analysis
has to name real columns, the report has to be grounded. Each stage can be validated, retried and replaced independently. The modeler
is deliberately *not* an LLM agent — it is a trusted script — which shows I know when not to use an LLM.

**What happens when the LLM writes bad code?**
Four gates: AST guard (rejects before execution), sandbox (crash/timeout → traceback with only the agent's own lines), output
contracts (wrong-but-running code), then the error text goes into the next prompt. After `max_code_attempts` (default 3) one
attempt uses the vetted template; the report states that a fallback was used. Show `examples/self_repair_trace.md`.

**Is running LLM-generated code safe?**
Not by the AST guard — that is a pre-filter. The boundary is the Docker sandbox: no network, read-only root, non-root user, all
capabilities dropped, no-new-privileges, memory/CPU/PID/time limits, input read-only. Caveats I can state: Docker isn't a hard boundary
against kernel exploits (use gVisor/Firecracker for hostile multi-tenant use), and the compose LLM override mounts the Docker socket,
which is root-equivalent, so it's for local use only. Public demos run offline mode, where no model-written code runs at all.

**What did you do about files the sandbox leaves behind?** *(a good "I found a bug in my own design" story)*
While documenting the project I noticed that the host copied everything from the sandbox's output directory into the run's
artifacts and later unpickled `model.joblib` from there. A sandboxed program could therefore (a) leave a symlink that resolves to a
host file when copied and served, or (b) plant a malicious pickle. Fix: `sanitize_output` removes symlinks/special files, and
`_promote` accepts only allow-listed file types and only the files each step owns; the model comes only from the trusted trainer.
There are tests for each case.

**How do you stop the report from hallucinating?**
The reporter only sees a compact facts file; afterwards every number in its text is extracted and matched, rounding-aware, to the
computed facts. Under 90% traceable → one repair attempt → deterministic writer. The rate is printed in the report.
Limits I would name myself: it checks that a number *exists* in the results, not that the sentence is *true* (a real number on the wrong
feature passes), and small integers/years are skipped. A next step would be claim-level verification (structured claims with
references to fact IDs).

**How did you evaluate it?**
Two experiments in `scripts/evaluate.py`. (1) Driver recovery: synthetic datasets with planted drivers and deliberate mess, scored by
precision@k. (2) Fault injection with a scripted LLM. I say clearly that synthetic data with strong effects shows the pipeline works,
not that it generalises, and that the next step is real datasets and a real-LLM run — which the script supports (`--real-llm`).

**Why a trusted script for modeling?**
Evaluation mistakes (leakage, tuning on the test set, thresholds chosen on test data) are exactly what free-form generation gets
wrong, and the artifact must be loadable by the API. So: stratified split, model selection by CV on the training split only, threshold
from out-of-fold predictions, one touch of the test set, permutation importance on raw columns, leakage warnings.

**Why Benjamini-Hochberg? Why effect sizes?**
Testing dozens of features produces false positives; BH controls the false discovery rate. With thousands of rows everything is
"significant", so effect sizes (Cohen's d, Cramér's V) carry the meaning; the report ranks by a unified 0–1 strength.

**What does "association, not causation" mean for this tool?**
Every report says it. The drivers are statistical associations in observational data; recommendations call for experiments before spending budget.

**How does the prediction API avoid garbage-in/garbage-out?**
Schema-driven validation: types, ranges and categories come from training. Unknown fields are ignored and reported, missing fields use
training defaults and are reported, out-of-range values and unseen categories yield warnings. Responses include the decision threshold,
base rate and risk band so the number is interpretable.

**How would you scale it?**
Replace the in-process thread pool with a queue and workers, move run storage to object storage + a database, run sandboxes on a
separate service with quotas, add per-user auth and rate limits, chunked/DuckDB processing for large files, and tracing + cost
accounting per agent. (Table in `docs/ARCHITECTURE.md`.)

**What about cost and latency?**
Only the planner, engineer, analyst and reporter call the LLM; the reporter sees compact facts, not rows; the trainer is pure
sklearn. Retries are bounded. Offline mode costs nothing. *(Measure and quote your real per-run token cost once you have run it with a key.)*

**How do you test an LLM application?**
Replace the LLM with a scripted double that implements the same protocol (`ScriptedLLM`), then assert on behaviour: what was rejected,
what the retry prompt contained, which fallbacks fired. Real-LLM runs go through the evaluation script, not unit tests.

**What is the hardest bug you hit?** *(pick one; all are real from this codebase)*
- Target detection once picked `EmployeeNumber` (an ID column) as the thing to explain on the HR dataset; fixed with an ID/eligibility filter, stop-words and an outcome-synonym list (churn ≈ leave/cancel/quit …).
- Categories literally named "None" were silently parsed as NaN by pandas; fixed with explicit missing-token handling and a test.
- A semicolon-delimited CSV with a BOM broke the sandboxed read, because each stage parsed the raw upload itself; fixed by canonicalising the upload once at the start of the run.
- The symlink/pickle output issue above.

**What would you do differently / next?**
Real-data benchmark vs. a human analyst's notebook; claim-level report verification; SHAP or counterfactuals for explanations; time-based splits when a date
column exists; a human-approval node before long jobs; OpenTelemetry traces and token accounting; stronger sandbox runtime (gVisor/Firecracker).

## 5. Extensions that add CV value (pick one, do it properly)

| Extension | What it demonstrates |
|---|---|
| Run the real-LLM benchmark, publish `docs/EVALUATION_LLM.md`, compare two models | Evaluation discipline, cost awareness |
| Tracing with LangSmith or OpenTelemetry + token/cost dashboard | Observability / LLMOps |
| Human-in-the-loop approval node (LangGraph interrupt) before the sandbox | Agent control, product thinking |
| Replace the thread pool with Redis + RQ/Celery workers | Production architecture |
| Time-aware validation + drift report for a deployed model | Applied ML maturity |
| Claim-level verification of the report (structured claims → fact IDs) | Hallucination mitigation beyond number matching |
| MCP server exposing `analyze_csv` / `predict` tools | Agent ecosystem integration |
