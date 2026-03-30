Here is the full production upgrade plan. Review it carefully before giving me the go-ahead.                                     
                                                                                                                                     ---                                                                                                                              
  Production Upgrade Plan                                                                                                                                                                                                                                             
  Phase 1 — Quick Wins (1–3 days, additive only)                                                                                                                                                                                                                      
  Gap 3 — Hardcoded paths → config.py                                                                                                                                                                                                                                   Fix: Single config.py at project root holding all paths and constants. Every script imports from it.                             
  Why config.py over .env/dotenv: No secrets here — just structural paths and account labels. A plain Python module is simpler,
  type-checked, and importable everywhere. Dotenv is the right tool when you need to keep values out of version control; that's not
   this situation.
  Scope: New config.py (~20 lines). 2–4 line change at the top of every script. 1 hour.

  ---
  Gap 2 — No data validation → pandera schemas

  Fix: Add pandera schema validation at the output of parse_statements.py and input of feature_engineering.py. Schemas enforce:    
  date_operation is datetime in plausible range, amount is finite non-NaN, type is {DEBIT, CREDIT}, description is non-empty, row  
  count above a minimum threshold.
  Why pandera over pydantic or assert: Pydantic validates Python objects row-by-row. Assert is silent in optimized mode. Pandera   
  validates an entire DataFrame in one call with human-readable column-level error messages. Well-known in data engineering.       
  Scope: New schemas.py (~40 lines). Two call sites. 2 hours.

  ---
  Gap 14 — No authentication → streamlit-authenticator

  Fix: streamlit-authenticator with a local credentials.yaml of bcrypt-hashed passwords. 15 lines added to the top of dashboard.py.
  Why not a reverse proxy (Nginx): This is a local Windows machine. The realistic threat is a colleague on the same network. A full
   reverse proxy is disproportionate. streamlit-authenticator is proportionate and demonstrable.
  Scope: New credentials.yaml (gitignored). 15 lines added to dashboard.py. 1 hour.

  ---
  Gap 16 — No error recovery → run_pipeline.py orchestrator

  Fix: A run_pipeline.py script that runs each stage in sequence, catches per-stage exceptions, writes a data/pipeline_status.json,
   and exits non-zero on failure. The dashboard reads the status file and shows a st.warning() banner when an upstream stage       
  failed.
  Why not Celery/Airflow: Those require brokers or servers. The pipeline runs manually or on a schedule, not in response to        
  concurrent web requests. A script with subprocess.run() + a JSON status file is the honest proportionate solution.
  Scope: New run_pipeline.py (~60 lines). 10 lines added to dashboard.py. Existing scripts unchanged. 2 hours.

  ---
  Phase 2 — Architecture Fixes (1–2 weeks, modifying core files) stopped here on 2503 next phasev3

  Gap 7 — No model persistence → joblib

  Fix: Serialize trained models to a models/ directory using joblib. Dashboard loads pre-saved artifacts instead of retraining on  
  every load. Each training run also saves a metadata JSON sidecar: training date, data hash, feature list, test metrics.
  Why joblib over pickle or MLflow: joblib is already installed as a scikit-learn dependency — no new package. Better than raw     
  pickle for numpy arrays. MLflow adds a local server and database — overengineered for a solo project.
  Scope: New model_store.py (~50 lines). All training scripts gain a save_artifacts() call. dashboard.py::train_models() replaced  
  with load_artifacts(). 1 full day.

  ---
  Gap 5 — Temporal leakage → chronological split

  Fix: Replace train_test_split(shuffle=True) in train_models.py and creditworthiness.py with a hard chronological split (last 6   
  months = test). Also fix feature_engineering.py to compute monthly aggregates without leaking future rows.
  Why this matters: cashflow_forecast.py already does this correctly. The classifiers don't. Features like monthly_income are      
  computed on the full dataset before the split — future months contaminate the training signal. The inflated accuracy numbers are 
  misleading.
  Scope: ~10 lines changed in train_models.py and creditworthiness.py. More careful change in feature_engineering.py. 4 hours.     

  ---
  Gap 6 — Circular labels → manual_labels.csv

  Fix: Create data/manual_labels.csv with 150–200 hand-labeled transactions (manual effort, ~2–3 hours of your time). These become 
  the evaluation-only test set. Training still uses rule-labeled data, but test metrics are now reported against ground truth.     
  Why not label everything: Labeling 1,000+ transactions takes weeks. The honest middle ground is a held-out manually labeled
  evaluation set. This is exactly what a technical reviewer wants to see — not perfect data, but awareness of the limitation and a 
  measurement of it.
  Scope: Manual labeling (your work). ~20 lines added to train_models.py to load and evaluate against this set. 3 hours + manual   
  labeling.

  ---
  Gap 17 — .xlsx files → SQLite

  Fix: Migrate all .xlsx pipeline outputs to a single data/finance.db SQLite database. One migrate_xlsx_to_sqlite.py script handles
   the one-time migration. All scripts switch from pd.read_excel() / ExcelWriter to pd.read_sql() / df.to_sql().
  Why SQLite over PostgreSQL or DuckDB: PostgreSQL requires a running server. DuckDB is analytically excellent but the format      
  change propagates everywhere with an extra dependency. SQLite is zero-configuration, a single file, handles concurrent reads     
  safely, supports transactions, and is in the Python standard library. Well-known and immediately recognizable on a portfolio.    
  Scope: New migrate_xlsx_to_sqlite.py. All 8 pipeline scripts + dashboard.py switch to SQLite reads/writes. Largest single change 
  — 1 full day.

  ---
  Gap 9 — Negative R² → SelectKBest + honest Ridge

  Fix: Three changes to cashflow_forecast.py: (1) Apply SelectKBest with f_regression to cut features to top 10 — more features    
  than months guarantees overfitting. (2) Make Ridge Regression the primary model, not XGBoost — with 27 training samples,
  tree-based models have no statistical basis. (3) Lead the dashboard display with the naive baseline comparison as the headline   
  metric.
  Why not Prophet or ARIMA: Prophet is for daily/weekly series. ARIMA requires stationarity testing. With 42 monthly points, honest
   Ridge with feature selection is more credible than a sophisticated model with no valid evaluation set.
  Scope: ~30 lines changed in cashflow_forecast.py. Dashboard display text updated. 3 hours.

  ---
  Phase 3 — Advanced / Production Hardening (2–4 weeks)

  Gap 18 — No tests → pytest (3 layers)

  - Unit tests for pure functions: parse_amount(), assign_category(), compute_credit_score() — file tests/test_utils.py
  - Integration tests using synthetic fixture data — verify parse → feature pipeline produces correct schema — file
  tests/test_pipeline.py
  - Regression tests (model snapshot tests) — save a fixed input vector and expected output to a fixture JSON; assert loaded model 
  produces the same output on each run — file tests/test_models.py

  Why pytest over unittest: Industry standard, no boilerplate class structure, better output, rich plugin ecosystem. Regression    
  tests depend on Gap 7 (model persistence) being done first.
  Scope: New tests/ directory, 3 files, conftest.py. 1–2 days.

  ---
  Gap 8 — No drift detection → evidently

  Fix: Monthly drift report comparing current transaction distributions against 6-month baseline. run_pipeline.py calls it after   
  parsing; significant drift writes a warning to pipeline_status.json and the dashboard shows a st.error() banner.
  Why evidently over alibi-detect or nannyml: alibi-detect brings TensorFlow/PyTorch. nannyml targets deployments with label       
  delays. evidently produces HTML reports from a single function call, pip-installable, no server, explicitly designed for tabular 
  ML on DataFrames.
  Scope: New drift_check.py (~60 lines). Integrated into run_pipeline.py. 4 hours.

  ---
  Gap 10 — Prompt versioning → prompts/ directory

  Fix: Extract the prompt f-string from loan_report.py::build_prompt() to prompts/loan_report_v1.txt. Use str.format_map(ctx) to   
  substitute values. Log the template filename in every saved report header.
  Why a text file: Git-trackable. git diff prompts/loan_report_v1.txt loan_report_v2.txt shows exactly what changed between report 
  generations. 1 hour.

  ---
  Gap 11 — LLM hallucination → number validation

  Fix: validate_report_numbers(report_text, ctx) extracts all numbers from the report text with regex, checks each one near a known
   metric name against ctx values within 5% tolerance. Failures append a VALIDATION WARNING block to the saved report — no silent  
  acceptance.
  Why not JSON mode: Forcing JSON output breaks the narrative prose which is the entire purpose. Number extraction preserves       
  free-text while catching the most dangerous hallucinations: fabricated financial figures.
  Scope: New function in loan_report.py (~40 lines). 2 hours.

  ---
  Gap 12 — Local Ollama only → backend abstraction

  Fix: Replace direct ollama.chat() with a call_llm(prompt, backend, model) dispatcher. Backend selected by config.py: LLM_BACKEND 
  = "ollama" or "openai". The OpenAI path reads OPENAI_API_KEY from .env — the only case where a secret is needed.
  Why keep both: Local Ollama is a portfolio differentiator. OpenAI makes it deployable. Both should remain available.
  Scope: ~30 lines modified in loan_report.py. Add openai, python-dotenv to requirements. 2 hours.

  ---
  Gap 13 — LLM contradicts ML → hard constraints in prompt

  Fix: build_constraints(ctx) derives hard assertions from ML output (e.g. HIGH_RISK → recommendation must be DECLINE or
  CONDITIONAL, never APPROVE). Injected as a HARD CONSTRAINTS block in the prompt. validate_report_numbers() also checks the       
  consistency post-generation.
  Why prompt injection over post-generation rewriting: Rewriting produces a report where the narrative argues for one outcome but  
  the heading says another — worse than the original. Prevention via prompt + detection as backstop is how production LLM
  guardrails actually work.
  Scope: ~25 lines added to loan_report.py. 2 hours.

  ---
  Gap 15 — No logging → standard logging with rotation

  Fix: logger.py configures logging.RotatingFileHandler writing to logs/pipeline.log. All scripts replace print() with log.info() /
   log.warning() / log.error(). Logs: stage timing, transaction counts, validation warnings, model metrics, LLM call duration.     
  Why standard logging over loguru: loguru is convenient but an extra dependency for something the stdlib already handles. Standard
   logging with RotatingFileHandler is immediately recognizable.
  Scope: New logger.py (~20 lines). All scripts gain 1 import + mechanical print → log replacement. 3 hours.

  ---
  Gap 4 — No data versioning → DVC

  Fix: DVC tracks data/finance.db and models/ by SHA256 hash in lightweight .dvc pointer files committed to git. Actual data stored
   in a local cache (no cloud needed — another local folder suffices). dvc.yaml mirrors the run_pipeline.py stages.
  Why DVC over Git LFS: Git LFS stores blobs in Git history and requires paid plan for large files. DVC tracks pointers in git,    
  data separately. git checkout v1.2 restores both code and the corresponding data version.
  Depends on: Gap 7 (model persistence) and Gap 17 (SQLite) being complete first.
  Scope: dvc init + dvc add + dvc.yaml. 3 hours.

  ---
  Gap 1 — 42 rows of data → synthetic augmentation + honest framing

  Fix: This cannot be fully solved without more real data. Two concrete actions:
  1. synthetic_augmentation.py generates 200 synthetic monthly records sampled from the empirical feature distributions, used only 
  for the cashflow forecaster (the model most hurt by tiny N). Flagged with is_synthetic=True and excluded from evaluation.        
  2. Dashboard and README updated to clearly state "trained on your own financial history, not a general population model" — which 
  is an honest and defensible claim.
  Why not more sophisticated augmentation: The fundamental problem is one person's behavioral fingerprint. SMOTE and VAE-based     
  augmentation would create synthetic data that still reflects only this person's patterns. Honesty in framing is the more valuable
   portfolio signal than a technical workaround.
  Scope: New synthetic_augmentation.py (~50 lines). Dashboard text updated. README updated. 4 hours.

  ---
  Summary Table

  ┌───────┬──────────────────────┬─────────────────────────┬────────────────────────────┬─────────────┐
  │ Phase │         Gap          │           Fix           │         New Files          │   Effort    │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 1     │ Hardcoded paths      │ config.py               │ config.py                  │ 1h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 1     │ No validation        │ pandera schemas         │ schemas.py                 │ 2h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 1     │ No auth              │ streamlit-authenticator │ credentials.yaml           │ 1h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 1     │ No error recovery    │ Pipeline orchestrator   │ run_pipeline.py            │ 2h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 2     │ No model persistence │ joblib artifacts        │ model_store.py, models/    │ 1 day       │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 2     │ Temporal leakage     │ Chronological split     │ —                          │ 4h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 2     │ Circular labels      │ Manual eval set         │ data/manual_labels.csv     │ 3h + manual │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 2     │ .xlsx database       │ SQLite migration        │ migrate_xlsx_to_sqlite.py  │ 1 day       │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 2     │ Negative R²          │ SelectKBest + Ridge     │ —                          │ 3h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ No tests             │ pytest 3-layer suite    │ tests/                     │ 1–2 days    │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ No drift detection   │ evidently               │ drift_check.py             │ 4h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ Prompt versioning    │ prompts/ directory      │ prompts/loan_report_v1.txt │ 1h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ LLM hallucination    │ Number validator        │ —                          │ 2h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ Ollama only          │ Backend abstraction     │ .env.example               │ 2h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ LLM contradicts ML   │ Hard constraints        │ —                          │ 2h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ No logging           │ logging + rotation      │ logger.py, logs/           │ 3h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ No data versioning   │ DVC                     │ dvc.yaml                   │ 3h          │
  ├───────┼──────────────────────┼─────────────────────────┼────────────────────────────┼─────────────┤
  │ 3     │ 42 rows              │ Synthetic aug + framing │ synthetic_augmentation.py  │ 4h          │
  └───────┴──────────────────────┴─────────────────────────┴────────────────────────────┴─────────────┘

  ---
  Review this and tell me which phases/gaps you want to proceed with, or if you want to adjust any of the technology choices before
   I start implementing.