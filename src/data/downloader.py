"""Automated download and validation of the Kaggle Credit Card Fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected dataset constants
EXPECTED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]
EXPECTED_COL_COUNT = 31
EXPECTED_ROW_COUNT = 284_807
EXPECTED_FRAUD_RATE = 0.00172
FRAUD_RATE_TOLERANCE = 0.002


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str


@dataclass
class ValidationResult:
    """Aggregated result of all validation checks."""

    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, message: str) -> None:
        self.checks.append(ValidationCheck(name=name, passed=passed, message=message))

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"[{status}] {c.name}: {c.message}")
        return "\n".join(lines)


class DatasetDownloader:
    """Download and validate the Credit Card Fraud Detection dataset."""

    KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
    DATASET_FILENAME = "creditcard.csv"

    def download(self, destination: Path) -> Path:
        """Download the dataset to *destination* directory.

        Tries ``kagglehub`` first, then falls back to a direct URL.
        Returns the path to the downloaded CSV file.
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / self.DATASET_FILENAME

        if target.exists():
            logger.info("Dataset already exists at %s, skipping download", target)
            return target

        # Attempt 1: kagglehub
        try:
            import kagglehub

            logger.info("Downloading via kagglehub …")
            dataset_path = kagglehub.dataset_download(self.KAGGLE_DATASET)
            src = Path(dataset_path) / self.DATASET_FILENAME
            if src.exists():
                import shutil

                shutil.copy2(src, target)
                logger.info("Dataset saved to %s", target)
                return target
        except Exception as exc:
            logger.warning("kagglehub download failed: %s", exc)

        # Attempt 2: direct URL via requests
        try:
            import requests

            url = (
                "https://storage.googleapis.com/download.tensorflow.org/"
                "data/creditcard.csv"
            )
            logger.info("Downloading from fallback URL …")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Dataset saved to %s", target)
            return target
        except Exception as exc:
            logger.warning("Fallback URL download failed: %s", exc)

        raise RuntimeError(
            "Unable to download the dataset. Please download manually from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and "
            f"place creditcard.csv in {destination}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, filepath: Path) -> ValidationResult:
        """Run comprehensive validation checks on the dataset file."""
        result = ValidationResult()
        filepath = Path(filepath)

        # 1. File existence and readability
        if not filepath.exists():
            result.add("file_exists", False, f"File not found: {filepath}")
            return result
        result.add("file_exists", True, f"File found at {filepath}")

        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            result.add("file_readable", False, f"Cannot read CSV: {exc}")
            return result
        result.add("file_readable", True, "File is readable")

        # 2. Column count
        col_count = len(df.columns)
        result.add(
            "column_count",
            col_count == EXPECTED_COL_COUNT,
            f"Found {col_count} columns, expected {EXPECTED_COL_COUNT}",
        )

        # 3. Expected columns present
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        result.add(
            "schema_columns",
            len(missing_cols) == 0,
            f"Missing columns: {missing_cols}"
            if missing_cols
            else "All expected columns present",
        )

        # 4. Column dtypes – all should be numeric
        non_numeric = [
            col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
        ]
        result.add(
            "column_dtypes",
            len(non_numeric) == 0,
            f"Non-numeric columns: {non_numeric}"
            if non_numeric
            else "All columns are numeric",
        )

        # 5. Row count
        row_count = len(df)
        row_ok = abs(row_count - EXPECTED_ROW_COUNT) < 100  # allow small tolerance
        result.add(
            "row_count",
            row_ok,
            f"Found {row_count} rows, expected ~{EXPECTED_ROW_COUNT}",
        )

        # 6. Missing values
        total_missing = int(df.isna().sum().sum())
        result.add(
            "missing_values",
            total_missing == 0,
            f"{total_missing} missing values found"
            if total_missing
            else "No missing values",
        )

        # 7. Class distribution
        if "Class" in df.columns:
            fraud_rate = df["Class"].mean()
            rate_ok = abs(fraud_rate - EXPECTED_FRAUD_RATE) < FRAUD_RATE_TOLERANCE
            result.add(
                "class_distribution",
                rate_ok,
                f"Fraud rate: {fraud_rate:.4%} (expected ~{EXPECTED_FRAUD_RATE:.3%})",
            )
        else:
            result.add("class_distribution", False, "Class column missing")

        # 8. Amount non-negative
        if "Amount" in df.columns:
            neg_amounts = int((df["Amount"] < 0).sum())
            result.add(
                "amount_range",
                neg_amounts == 0,
                f"{neg_amounts} negative Amount values"
                if neg_amounts
                else "All Amount values non-negative",
            )
        else:
            result.add("amount_range", False, "Amount column missing")

        return result

    # ------------------------------------------------------------------
    # Sample creation
    # ------------------------------------------------------------------

    def create_sample(
        self,
        source: Path,
        destination: Path,
        n_rows: int = 1000,
        random_state: int = 42,
    ) -> Path:
        """Create a stratified sample preserving the class distribution.

        Parameters
        ----------
        source:
            Path to the full dataset CSV.
        destination:
            Directory where the sample will be saved.
        n_rows:
            Number of rows in the sample.
        random_state:
            Random seed for reproducibility.
        """
        df = pd.read_csv(source)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        fraud = df[df["Class"] == 1]
        non_fraud = df[df["Class"] == 0]

        # Proportional sampling
        fraud_ratio = len(fraud) / len(df)
        n_fraud = max(1, round(n_rows * fraud_ratio))
        n_non_fraud = n_rows - n_fraud

        sample_fraud = fraud.sample(
            n=min(n_fraud, len(fraud)), random_state=random_state
        )
        sample_non_fraud = non_fraud.sample(
            n=min(n_non_fraud, len(non_fraud)), random_state=random_state
        )

        sample = (
            pd.concat([sample_fraud, sample_non_fraud], ignore_index=True)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

        target = destination / "sample_transactions.csv"
        sample.to_csv(target, index=False)
        logger.info(
            "Sample saved to %s (%d rows, %d fraud)", target, len(sample), n_fraud
        )
        return target
