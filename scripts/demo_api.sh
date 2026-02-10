#!/usr/bin/env bash
# ---------------------------------------------------------------
# Demo script — sample API calls against the Fraud Detection API
# Usage:  bash scripts/demo_api.sh [BASE_URL]
# Default BASE_URL: http://localhost:8000
# ---------------------------------------------------------------

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "=== Fraud Detection API Demo ==="
echo "Target: $BASE_URL"
echo

# ---------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------
echo "--- Health Check ---"
curl -s "$BASE_URL/health" | python -m json.tool
echo

# ---------------------------------------------------------------
# 2. Single prediction
# ---------------------------------------------------------------
echo "--- Single Prediction ---"
curl -s -X POST "$BASE_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo_txn_001",
    "Time": 0.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": -0.47, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": -0.39, "V23": -0.11, "V24": -0.22,
    "V25": -0.64, "V26": 0.72, "V27": -0.22, "V28": 0.03,
    "Amount": 149.62
  }' | python -m json.tool
echo

# ---------------------------------------------------------------
# 3. Single prediction with SHAP explanation
# ---------------------------------------------------------------
echo "--- Prediction with Explanation ---"
curl -s -X POST "$BASE_URL/api/v1/predict?include_explanation=true" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo_txn_002",
    "Time": 50000.0,
    "V1": 1.19, "V2": 0.27, "V3": 0.17, "V4": 0.45,
    "V5": 0.06, "V6": -0.08, "V7": 0.09, "V8": 0.09,
    "V9": -0.26, "V10": -0.17, "V11": 1.61, "V12": 1.07,
    "V13": 0.49, "V14": -0.14, "V15": 0.64, "V16": 0.46,
    "V17": -0.11, "V18": -0.18, "V19": -0.15, "V20": -0.07,
    "V21": -0.23, "V22": -0.64, "V23": 0.07, "V24": -0.06,
    "V25": 0.08, "V26": 0.25, "V27": 0.03, "V28": 0.01,
    "Amount": 2.69
  }' | python -m json.tool
echo

# ---------------------------------------------------------------
# 4. Batch prediction (3 transactions)
# ---------------------------------------------------------------
echo "--- Batch Prediction ---"
curl -s -X POST "$BASE_URL/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "demo_batch_001",
        "Time": 0, "V1": 0, "V2": 0, "V3": 0, "V4": 0,
        "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0,
        "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0,
        "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0,
        "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0,
        "V25": 0, "V26": 0, "V27": 0, "V28": 0,
        "Amount": 10.0
      },
      {
        "transaction_id": "demo_batch_002",
        "Time": 100, "V1": -5.0, "V2": 3.0, "V3": -8.0, "V4": 5.0,
        "V5": -2.0, "V6": -3.0, "V7": -7.0, "V8": 1.0, "V9": -3.0,
        "V10": -5.0, "V11": 3.0, "V12": -6.0, "V13": 1.0, "V14": -8.0,
        "V15": 0, "V16": -6.0, "V17": -7.0, "V18": -2.0, "V19": 1.0,
        "V20": 0.5, "V21": 0.3, "V22": -0.1, "V23": -0.2, "V24": 0.4,
        "V25": 0.1, "V26": -0.5, "V27": 0.2, "V28": -0.1,
        "Amount": 3499.99
      },
      {
        "transaction_id": "demo_batch_003",
        "Time": 200, "V1": 1.0, "V2": -0.5, "V3": 0.8, "V4": 0.3,
        "V5": 0.1, "V6": 0.2, "V7": -0.1, "V8": 0.05, "V9": 0.3,
        "V10": 0.1, "V11": -0.2, "V12": 0.4, "V13": -0.1, "V14": 0.2,
        "V15": -0.3, "V16": 0.1, "V17": -0.05, "V18": 0.15, "V19": 0.1,
        "V20": -0.08, "V21": 0.02, "V22": 0.01, "V23": -0.03, "V24": 0.1,
        "V25": 0.05, "V26": -0.02, "V27": 0.04, "V28": 0.01,
        "Amount": 25.50
      }
    ]
  }' | python -m json.tool
echo

# ---------------------------------------------------------------
# 5. A/B test results
# ---------------------------------------------------------------
echo "--- A/B Test Results ---"
curl -s "$BASE_URL/api/v1/ab-test/results" | python -m json.tool
echo

echo "=== Demo complete ==="
