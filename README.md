# Fraud Detection System

![CI](https://github.com/YOUR_USERNAME/fraud-detection-system/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/YOUR_USERNAME/fraud-detection-system/branch/main/graph/badge.svg)

> Part of my Data Science Portfolio — [Skaraz Data](https://github.com/YOUR_USERNAME)

## Overview

End-to-end machine-learning system for detecting fraudulent credit card transactions. The project covers the full ML lifecycle: data ingestion, feature engineering, model training with hyperparameter tuning, real-time inference via a REST API, A/B testing of model variants, a live monitoring dashboard, and containerised deployment.

### Features

- **Data pipeline** — automated download, validation, and feature engineering for the Kaggle Credit Card Fraud dataset
- **Model training** — Logistic Regression, Random Forest, and XGBoost with cross-validation, class-imbalance handling, and Optuna hyperparameter tuning
- **Model registry** — versioned model storage with metadata (joblib + JSON)
- **Explainability** — SHAP-based per-prediction and global feature importance
- **REST API** — FastAPI endpoints for single and batch prediction with confidence scores
- **Streaming simulator** — async transaction stream with configurable fraud injection rate
- **A/B testing** — deterministic traffic splitting with chi-squared significance testing
- **Monitoring dashboard** — Streamlit UI with real-time feed, performance charts, alerts, and A/B comparison
- **Docker Compose** — multi-service deployment (API, dashboard, simulator) with health checks
- **CI/CD** — GitHub Actions pipeline with linting, testing, coverage, and Docker build verification

## Architecture

```mermaid
graph LR
    subgraph Data Pipeline
        A[Kaggle Dataset] --> B[Downloader]
        B --> C[Preprocessor]
        C --> D[Feature Store]
    end

    subgraph Model Layer
        D --> E[Trainer]
        E --> F[Evaluator]
        F --> G[Registry]
        E --> H[Explainer]
    end

    subgraph Inference
        G --> I[FastAPI]
        I --> J[/api/v1/predict]
        I --> K[/api/v1/predict/batch]
        I --> L[/api/v1/ab-test/results]
        I --> M[/health]
    end

    subgraph Streaming
        N[Simulator] -->|queue| O[Consumer]
        O --> P[A/B Router]
        P --> I
    end

    subgraph Monitoring
        Q[Streamlit Dashboard]
        Q --> R[(SQLite)]
        I --> R
    end
```

For a detailed breakdown see [docs/architecture.md](docs/architecture.md).

## Quick Start

### Local Development

```bash
# Clone
git clone git@github.com:YOUR_USERNAME/fraud-detection-system.git
cd fraud-detection-system

# Set up Python with pyenv
pyenv install 3.11
pyenv local 3.11

# Install dependencies
pip install -r requirements.txt

# Run the API server
make run
# => http://localhost:8000

# Run tests
make test

# Lint
make lint
```

### Docker Compose

```bash
# Build and start all services
make docker-up

# API:       http://localhost:8000
# Dashboard: http://localhost:8501

# View logs
make docker-logs

# Stop
make docker-down
```

## API Examples

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_version": "xgboost_v1"
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "Time": 0.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": -0.47, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": -0.39, "V23": -0.11, "V24": -0.22,
    "V25": -0.64, "V26": 0.72, "V27": -0.22, "V28": 0.03,
    "Amount": 149.62
  }'
```

```json
{
  "transaction_id": "txn_001",
  "fraud_probability": 0.023,
  "is_fraud": false,
  "confidence": 0.977,
  "model_version": "xgboost_v1",
  "explanation": null
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"transaction_id": "txn_001", "Time": 0, "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0, "Amount": 50.0}]}'
```

### A/B Test Results

```bash
curl http://localhost:8000/api/v1/ab-test/results
```

For full API documentation see [docs/api_reference.md](docs/api_reference.md).

## Configuration

All config files live in `configs/`:

| File | Purpose |
|---|---|
| `config.yaml` | Data paths, API host/port, database path, streaming rate |
| `model_params.yaml` | Hyperparameters for each model type |
| `ab_test.yaml` | A/B test toggle, model variants, traffic split ratio |

Environment variables override YAML values using dot-path notation (e.g. `MODEL__DEFAULT_MODEL=random_forest`).

## Project Structure

```
fraud-detection-system/
├── src/
│   ├── data/               # Data ingestion and feature engineering
│   │   ├── downloader.py   #   Dataset download and validation
│   │   ├── preprocessor.py #   Feature engineering pipeline
│   │   └── feature_store.py#   Versioned Parquet feature storage
│   ├── models/             # Training, evaluation, and serving
│   │   ├── trainer.py      #   Model training with cross-validation
│   │   ├── evaluator.py    #   Evaluation and comparison reports
│   │   ├── registry.py     #   Versioned model registry
│   │   └── explainer.py    #   SHAP-based explainability
│   ├── api/                # FastAPI inference service
│   │   ├── app.py          #   Application entry point
│   │   ├── schemas.py      #   Pydantic request/response models
│   │   └── routes/         #   Endpoint handlers
│   ├── streaming/          # Real-time transaction processing
│   │   ├── simulator.py    #   Async transaction generator
│   │   ├── consumer.py     #   Stream consumer
│   │   ├── ab_router.py    #   A/B traffic routing
│   │   └── run_simulator.py#   Container entry point
│   ├── dashboard/          # Streamlit monitoring UI
│   │   ├── app.py          #   Dashboard entry point
│   │   ├── data.py         #   Sync SQLite data access
│   │   └── pages/          #   Dashboard pages (6 views)
│   └── utils/              # Shared utilities
│       ├── config.py       #   YAML config with env overrides
│       ├── logger.py       #   JSON structured logging
│       └── database.py     #   Async SQLite manager
├── tests/                  # 567 tests (pytest)
├── configs/                # YAML configuration files
├── data/sample/            # 1000-row sample dataset for CI
├── docs/                   # Architecture and API docs
├── scripts/                # Demo and utility scripts
├── Dockerfile              # API container (multi-stage)
├── Dockerfile.dashboard    # Dashboard container
├── Dockerfile.simulator    # Simulator container
├── docker-compose.yml      # Multi-service orchestration
├── Makefile                # Build/run shortcuts
├── requirements.txt        # Python dependencies
└── .github/workflows/ci.yml # CI/CD pipeline
```

## Technology Stack

| Category | Technology |
|---|---|
| ML | scikit-learn, XGBoost, imbalanced-learn, Optuna |
| Explainability | SHAP |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Dashboard | Streamlit, Plotly |
| Data | pandas, NumPy, PyArrow |
| Database | SQLite (aiosqlite for API, sqlite3 for dashboard) |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions, Ruff, pytest, Codecov |

## License

MIT
