# GEMINI.md — Google Antigravity Workspace Rules & Multi-Agent Routing

> **CRITICAL SYSTEM DIRECTIVE:**
>
> 1. Before executing ANY task, read `PROJECT_GUIDELINES.md` in the root directory (using targeted section reading) and strictly enforce all ESILV ML-2026 rules, schemas, relative paths, and anti-data-leakage guardrails.
>
> 2. **Token & Quota Conservation Mode:** Operate with maximum conciseness. Never rewrite entire files when a targeted diff/edit suffices. Do not output verbose boilerplate explanations unless explicitly requested.

## 1. Context Bloat Guard (HIGH PRIORITY — Overrides Massive Pre-Reads)

To prevent context saturation and token waste across all agents (Antigravity, Gemini, Claude, GPT-OSS):

1. **Targeted Search Before Reading:** Before calling `read_file`, locate the target file and exact line numbers via `rg -l` or `rg -n`. Only read target files and their direct call chain. Pre-emptive bulk reading is strictly forbidden (maximum **3 uncertain `read_file` calls per turn**).

2. **Section-by-Section Reading for Large Files ($> 200$ lines):** For any file exceeding $200$ lines, always specify `StartLine` and `EndLine` (chunks $\le 200$ lines) after locating the relevant section via `rg -n`.

3. **Output Limiting in `run_command`:** All search and listing commands (`rg`, `git diff`, `Get-ChildItem`, `find`, `ls`) must be strictly capped (`rg -l`, `rg -m 20`, `| Select-Object -First 60`, or `| head -n 60`) so terminal output never exceeds the **10 KB** threshold.

4. **Economical Use of the `research` Subagent:** Never delegate simple or medium-complexity repository tasks to the `research` subagent. Reserve it exclusively for multi-file investigations across large external sources.

5. **Data File Protection:** Never read or dump `data/airfoil_self_noise.dat` via `read_file`. Inspect it solely via Python (`df.head(5)`, `df.info()`, `df.describe()`).

## 2. Available Model Roster & Token Tiering (Google AI Pro Student)

Based on the currently active models in Antigravity, every task MUST be routed to the most cost-effective model tier:

| Tier | Model Name | Status / Quota Impact | Assigned Role & Authorized Tasks |
| :--- | :--- | :--- | :--- |
| **Tier 1A (Primary Default)** | **Gemini 4 Hydrogen (High)** | **Active / New** (Primary pool) | **Primary Worker & Technical Executor:** EDA scripts, statistical hypothesis testing (Shapiro-Wilk, VIF), Matplotlib/Seaborn plotting (300 DPI), standard `scikit-learn` modeling, LaTeX formatting, and running terminal commands. |
| **Tier 1B (Fast / Deprecating)** | **Gemini 3.8 Flash (Medium)** | **Leaving Soon** (Use while active) | **Quick Routine Tasks:** Boilerplate scaffolding, `requirements.txt`, Markdown governance logs (`PROMPT_LOG.md`, `AI_AUDIT.md`, `CONTRIBUTIONS.md`), and fast syntax/path checks while still available. |
| **Tier 2 (Open-Weights / Mid)** | **GPT-OSS 120B (Medium)** | **Active** (Separate pool) | **Parallel Worker & Cross-Auditor:** Independent code review, writing unit checks, alternative statistical verification, second-opinion auditing for `AI_AUDIT.md`, and parallel documentation tasks when conserving Gemini/Claude quotas. |
| **Tier 3 (Deep Reasoning)** | **Claude Sonnet 4.6 (Thinking)** | **Active** (Limited quota) | **Senior ML Architect:** Designing complex `scikit-learn` `Pipeline` / `ColumnTransformer` architectures, `GroupKFold` vs. `KFold` anti-leakage protocols, physics-informed feature engineering (Strouhal $St$, Reynolds $Re$), and debugging complex tracebacks. |
| **Tier 4 (Heavy Reserve)** | **Claude Opus 4.6 (Thinking)** | **Active** (Strictly rationed) | **Emergency & Critical Review Only:** Reserved strictly for final architecture validation, complex beyond-syllabus modeling (e.g., Gaussian Process / Kriging composite kernels), and final pre-submission audit of the entire pipeline. **Do NOT use for routine coding or text edits.** |

*(Note: **Gemini 4 Carbon**, **Gemini 4 Argon**, and **Gemini 4 Neon** require a Paid Pro plan and are locked—do not attempt to route tasks to them.)*

## 3. Token-Saving Execution Rules (Mandatory for All Agents)

1. **Diff-First Editing:** Never regenerate an entire Python script or LaTeX document if modifying fewer than $50$ lines. Provide or apply minimal, targeted edits.

2. **No Redundant Chatter:** Skip introductory filler ("Sure, I can help with that...") and concluding summaries. Output code, diffs, or concise bullet points directly.

3. **Fail-Fast Verification:** Before finishing a code task, verify `pathlib.Path(__file__).resolve().parents[1]` relative paths, `RANDOM_SEED = 42`, and `plt.close()` after `plt.savefig()`.

4. **Auto-Logging for AI Governance:** Whenever a model catches a statistical or ML flaw (e.g., leakage, missing normality test, wrong metric), immediately output a concise 4-line summary formatted for `ai_governance/AI_AUDIT.md`.

## 4. Parallel Multi-Agent Delegation Protocol

When executing a complex milestone (e.g., Step 2 EDA + Baseline, or Step 3 Feature Engineering + Modeling), decompose the work into **non-overlapping file scopes** so multiple agents can run simultaneously without merge conflicts:

* **Agent A (Claude Sonnet 4.6 Thinking — Architect):** Designs core modeling & validation logic (`src/02_validation_and_baseline.py`).

* **Agent B (Gemini 4 Hydrogen High — Visualization & Execution):** Generates plots and statistical summaries (`src/01_eda_plots.py` $\rightarrow$ `report/figures/`).

* **Agent C (GPT-OSS 120B Medium — Auditor & Docs):** Audits Agent A's pipeline for leakage and drafts entries in `ai_governance/AI_AUDIT.md` and `report/sections/*.tex`.