"""Versioned feature storage backed by Parquet files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VersionInfo:
    """Metadata for a single feature-store version."""

    version: str
    columns: List[str]
    dtypes: Dict[str, str]
    row_count: int
    created_at: str


class FeatureStore:
    """Save and load versioned feature sets as compressed Parquet files.

    Directory layout::

        base_path/
            <version>/
                features.parquet
                metadata.json
    """

    _PARQUET_NAME = "features.parquet"
    _META_NAME = "metadata.json"

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, version: Optional[str] = None) -> str:
        """Persist a DataFrame as a versioned Parquet file.

        Parameters
        ----------
        df:
            The feature dataframe to store.
        version:
            Human-readable version tag.  When *None* a timestamp-based
            version is auto-generated (``YYYYMMDD_HHMMSS``).

        Returns
        -------
        str
            The version string that was written.
        """
        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        version_dir = self.base_path / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write parquet with snappy compression
        parquet_path = version_dir / self._PARQUET_NAME
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

        # Write metadata
        meta = VersionInfo(
            version=version,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            row_count=len(df),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        meta_path = version_dir / self._META_NAME
        meta_path.write_text(json.dumps(asdict(meta), indent=2))

        logger.info(
            "Saved version '%s' (%d rows, %d cols)", version, len(df), len(df.columns)
        )
        return version

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, version: str = "latest") -> pd.DataFrame:
        """Load a feature set by version.

        Parameters
        ----------
        version:
            Exact version string or ``"latest"`` to resolve the most
            recent version by directory name (lexicographic sort).

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError
            If the requested version (or any version for ``"latest"``)
            does not exist.
        ValueError
            If metadata validation fails after loading.
        """
        if version == "latest":
            version = self._resolve_latest()

        version_dir = self.base_path / version
        parquet_path = version_dir / self._PARQUET_NAME
        meta_path = version_dir / self._META_NAME

        if not parquet_path.exists():
            raise FileNotFoundError(f"No parquet file at {parquet_path}")

        df = pd.read_parquet(parquet_path, engine="pyarrow")

        # Validate against metadata if present
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            expected_cols = meta.get("columns", [])
            expected_rows = meta.get("row_count")

            if list(df.columns) != expected_cols:
                raise ValueError(
                    f"Column mismatch: parquet has {list(df.columns)}, "
                    f"metadata expects {expected_cols}"
                )
            if expected_rows is not None and len(df) != expected_rows:
                raise ValueError(
                    f"Row count mismatch: parquet has {len(df)}, "
                    f"metadata expects {expected_rows}"
                )

        logger.info(
            "Loaded version '%s' (%d rows, %d cols)", version, len(df), len(df.columns)
        )
        return df

    # ------------------------------------------------------------------
    # List versions
    # ------------------------------------------------------------------

    def list_versions(self) -> List[VersionInfo]:
        """Return metadata for all stored versions, sorted oldest-first."""
        if not self.base_path.exists():
            return []

        versions: List[VersionInfo] = []
        for child in sorted(self.base_path.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / self._META_NAME
            if not meta_path.exists():
                continue
            raw = json.loads(meta_path.read_text())
            versions.append(
                VersionInfo(
                    version=raw["version"],
                    columns=raw["columns"],
                    dtypes=raw["dtypes"],
                    row_count=raw["row_count"],
                    created_at=raw["created_at"],
                )
            )
        return versions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_latest(self) -> str:
        """Return the version string of the most recent save."""
        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Feature store path does not exist: {self.base_path}"
            )

        dirs = sorted(
            [
                d.name
                for d in self.base_path.iterdir()
                if d.is_dir() and (d / self._META_NAME).exists()
            ]
        )
        if not dirs:
            raise FileNotFoundError("No versions found in feature store")
        return dirs[-1]
