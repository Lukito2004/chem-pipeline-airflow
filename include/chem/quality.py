"""Data-quality checks for pipeline inputs."""
from __future__ import annotations
import pandas as pd
from rdkit import Chem


class DataQualityError(Exception):
    """Raised when an input file fails validation."""


def validate_input(df: pd.DataFrame, name: str, min_valid_ratio: float = 0.8) -> dict:
    """Validate a SMILES input file; raise DataQualityError if it's bad."""
    if "smiles" not in df.columns:
        raise DataQualityError(f"{name}: missing 'smiles' column")
    if df.empty:
        raise DataQualityError(f"{name}: file is empty")

    total = len(df)
    valid = sum(1 for s in df["smiles"]
                if isinstance(s, str) and Chem.MolFromSmiles(s) is not None)
    ratio = valid / total
    if ratio < min_valid_ratio:
        raise DataQualityError(
            f"{name}: only {ratio:.0%} valid SMILES (need ≥{min_valid_ratio:.0%})")
    return {"file": name, "rows": total, "valid": valid, "valid_ratio": ratio}