# API Reference

Base URL: `http://localhost:8000`

## Endpoints

### Health Check

```
GET /health
```

Returns service status and the currently loaded model version.

**Response** `200 OK`

```json
{
  "status": "healthy",
  "model_version": "xgboost_v1"
}
```

---

### Single Prediction

```
POST /api/v1/predict
```

Score a single transaction for fraud.

**Query Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `include_explanation` | bool | `false` | Include SHAP feature attributions in response |

**Request Body** — `application/json`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `transaction_id` | string | yes | Unique identifier |
| `Time` | float | yes | Seconds elapsed from first transaction |
| `V1` – `V28` | float | yes | PCA-transformed features |
| `Amount` | float | yes | >= 0 |

Extra fields are rejected (`extra='forbid'`).

**Example Request**

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

**Response** `200 OK`

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

**Response Fields**

| Field | Type | Description |
|---|---|---|
| `transaction_id` | string | Echoed from request |
| `fraud_probability` | float | 0.0 – 1.0, probability of fraud |
| `is_fraud` | bool | `true` if probability >= threshold (default 0.5) |
| `confidence` | float | `max(p, 1-p)` — model certainty |
| `model_version` | string | Version of the model that produced the score |
| `explanation` | object or null | SHAP values per feature (when requested) |

---

### Batch Prediction

```
POST /api/v1/predict/batch
```

Score up to 100 transactions in a single request. Uses vectorised inference for better throughput.

**Request Body** — `application/json`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `transactions` | array of TransactionInput | yes | Max length 100 |

**Example Request**

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "txn_001",
        "Time": 0, "V1": 0, "V2": 0, "V3": 0, "V4": 0,
        "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0,
        "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0,
        "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0,
        "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0,
        "V25": 0, "V26": 0, "V27": 0, "V28": 0,
        "Amount": 50.0
      }
    ]
  }'
```

**Response** `200 OK`

```json
{
  "predictions": [
    {
      "transaction_id": "txn_001",
      "fraud_probability": 0.012,
      "is_fraud": false,
      "confidence": 0.988,
      "model_version": "xgboost_v1",
      "explanation": null
    }
  ]
}
```

---

### A/B Test Results

```
GET /api/v1/ab-test/results
```

Returns current A/B test metrics and statistical significance.

**Response (A/B active)** `200 OK`

```json
{
  "enabled": true,
  "model_a": {
    "count": 523,
    "fraud_rate": 0.019,
    "mean_latency_ms": 12.4,
    "accuracy": 0.97
  },
  "model_b": {
    "count": 519,
    "fraud_rate": 0.021,
    "mean_latency_ms": 14.1,
    "accuracy": 0.96
  },
  "significance": {
    "chi2_statistic": 1.23,
    "p_value": 0.267,
    "is_significant": false,
    "alpha": 0.05
  },
  "split_ratio": 0.5
}
```

**Response (A/B inactive)** `200 OK`

```json
{
  "enabled": false,
  "message": "A/B testing is not active"
}
```

---

## Error Responses

### 422 Validation Error

Returned when the request body fails Pydantic validation.

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "Amount"],
      "msg": "Input should be greater than or equal to 0",
      "input": -10.0
    }
  ]
}
```

Common causes:
- Missing required field (e.g. `transaction_id`, any V-feature)
- Negative `Amount`
- Extra fields not in the schema
- Non-numeric values for numeric fields
- Batch exceeds 100 transactions

### 500 Internal Server Error

Returned on unexpected server errors (model not loaded, database failure).

```json
{
  "detail": "Internal server error"
}
```

---

## Response Headers

| Header | Description |
|---|---|
| `X-Process-Time-Ms` | Server-side processing time in milliseconds (added by timing middleware) |

---

## Configuration

The API reads from `configs/config.yaml` at startup:

```yaml
api:
  host: 0.0.0.0
  port: 8000
model:
  registry_path: models/
  default_model: xgboost
database:
  path: data/predictions.db
```

Environment variables can override any config key using double-underscore notation (e.g. `API__PORT=9000`).
