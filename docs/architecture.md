# System Architecture

## Components

### Data Pipeline (`src/data/`)

Handles ingestion, validation, and feature engineering.

| Module | Responsibility |
|---|---|
| `downloader.py` | Downloads the Kaggle Credit Card Fraud dataset with fallback methods (kagglehub, HTTP). Validates schema, row count, column types, and class distribution. |
| `preprocessor.py` | Feature engineering pipeline: time-of-day features, amount scaling (log + StandardScaler), interaction features, and train/test splitting with stratification and class-balance weighting. |
| `feature_store.py` | Versioned Parquet-based feature storage. Saves and loads feature sets with metadata (feature names, creation timestamp, row count). |

### Model Layer (`src/models/`)

Training, evaluation, versioning, and explainability.

| Module | Responsibility |
|---|---|
| `trainer.py` | Trains Logistic Regression, Random Forest, or XGBoost with automatic class-imbalance handling. Supports k-fold cross-validation with per-fold precision, recall, F1, and AUC-ROC. |
| `evaluator.py` | Generates evaluation reports: confusion matrix, classification report, ROC-AUC, precision-recall curves, and multi-model comparison tables. |
| `registry.py` | Persists trained models via joblib with JSON metadata (model type, version, training date, metrics). Supports listing, loading by version, and retrieving the latest model. |
| `explainer.py` | SHAP-based explainability using TreeExplainer (tree models) or KernelExplainer (linear). Produces per-prediction explanations and global summary plots. |

### Inference API (`src/api/`)

FastAPI service exposing prediction endpoints.

| Module | Responsibility |
|---|---|
| `app.py` | Application entry point. Async lifespan loads model from registry and initialises SQLite. Adds timing middleware (`X-Process-Time-Ms` header). |
| `schemas.py` | Pydantic v2 schemas: `TransactionInput` (V1-V28, Time, Amount), `PredictionOutput`, `BatchInput` (max 100), `BatchOutput`, `HealthResponse`. |
| `routes/predict.py` | POST `/api/v1/predict` (single) and POST `/api/v1/predict/batch` (vectorised). Logs every prediction to SQLite. |
| `routes/ab_test.py` | GET `/api/v1/ab-test/results` — returns current A/B test metrics and significance. |

### Streaming (`src/streaming/`)

Real-time transaction simulation and A/B testing.

| Module | Responsibility |
|---|---|
| `simulator.py` | Reads from CSV/Parquet and emits transactions to an `asyncio.Queue` at a configurable rate. Injects synthetic fraud patterns (large amounts, shifted V-features). |
| `consumer.py` | Pulls transactions from the queue and passes them to an inference callback. Tracks processed count and supports graceful shutdown. |
| `ab_router.py` | Deterministic MD5-hash-based traffic splitting. `MetricsTracker` per variant records predictions, actuals, and latencies. Chi-squared significance testing via `scipy.stats.chi2_contingency`. |
| `run_simulator.py` | Container entry point. Reads `API_URL`, `STREAM_RATE`, `DATA_PATH` from environment variables and runs the simulator + consumer loop. |

### Monitoring Dashboard (`src/dashboard/`)

Streamlit UI with auto-refresh.

| Module | Responsibility |
|---|---|
| `app.py` | Main entry point. Wide layout, 30-second auto-refresh, sidebar navigation. |
| `data.py` | Synchronous SQLite data access layer (plain `sqlite3`, not `aiosqlite`). Provides `get_recent_predictions`, `compute_overview_metrics`, `get_ab_results`, `get_high_confidence_alerts`, date filtering, and CSV export. |
| `pages/overview.py` | Big-number metric cards: transactions in 1h / 24h / 7d, fraud count, fraud rate. |
| `pages/realtime_feed.py` | Latest 50 predictions with colour-coded fraud probability. |
| `pages/performance.py` | Fraud rate over time (Plotly line chart), prediction distribution, date range filter. |
| `pages/ab_test.py` | Side-by-side model comparison with fraud rate and latency bar charts, significance indicator. |
| `pages/feature_importance.py` | SHAP summary plots and per-prediction feature bar charts. |
| `pages/alerts.py` | High-confidence fraud alerts with adjustable threshold slider and CSV download. |

### Utilities (`src/utils/`)

| Module | Responsibility |
|---|---|
| `config.py` | Loads YAML config with environment variable overrides. Singleton access via `get_config()`. |
| `logger.py` | JSON structured logging with timestamp, module, level, and message. |
| `database.py` | Async SQLite manager via aiosqlite. Schema: `predictions` table and `ab_test_results` table. |

## Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│ Kaggle CSV   │────>│ Preprocessor │────>│ Feature Store  │
│ (284,807 txn)│     │ (engineer)   │     │ (Parquet)      │
└──────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                                                  v
                                          ┌───────────────┐
                                          │ Trainer       │
                                          │ (CV + tune)   │
                                          └───────┬───────┘
                                                  │
                              ┌────────────┐      v
                              │ Evaluator  │<── Model ──>┌──────────┐
                              │ (metrics)  │             │ Registry │
                              └────────────┘             │ (joblib) │
                                                         └─────┬────┘
                                                               │
                                                               v
┌─────────────┐     ┌──────────┐     ┌─────────────────────────────────┐
│ Simulator   │────>│ Consumer │────>│ FastAPI (/predict, /batch, /ab) │
│ (async gen) │     │ (queue)  │     └──────────────┬──────────────────┘
└─────────────┘     └──────────┘                    │
                                                    v
                                              ┌───────────┐
                                              │  SQLite   │
                                              │  (preds)  │
                                              └─────┬─────┘
                                                    │
                                                    v
                                              ┌───────────┐
                                              │ Streamlit │
                                              │ Dashboard │
                                              └───────────┘
```

## Technology Decisions

### Why XGBoost for fraud detection?

Gradient-boosted trees handle the highly imbalanced class distribution well (0.17% fraud) via `scale_pos_weight`. They capture non-linear feature interactions without manual engineering and provide fast inference suitable for real-time scoring. The model also integrates cleanly with SHAP's TreeExplainer for efficient per-prediction explanations.

### Why SHAP for explainability?

SHAP provides theoretically grounded (Shapley values) per-feature attribution for every prediction. TreeExplainer runs in polynomial time on tree ensembles, making it practical for production. Global summary plots reveal which features the model relies on most, while per-prediction waterfall charts support fraud analyst review.

### Why SQLite?

SQLite keeps the project self-contained with zero external dependencies — no database server to configure or maintain. For a portfolio project this removes deployment friction. The async wrapper (`aiosqlite`) prevents blocking the FastAPI event loop, while the dashboard uses synchronous `sqlite3` since Streamlit runs a blocking execution model. In production, this layer would swap to PostgreSQL or a time-series database.

### Why separate sync/async data access?

FastAPI runs on an async event loop, so database calls use `aiosqlite` to avoid blocking. Streamlit re-runs the entire script on each interaction and does not expose an event loop, so the dashboard data layer uses plain `sqlite3`. Both read from the same database file, keeping the architecture simple.
