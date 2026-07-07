"""Helpers for discovering dataset ids from S3 keys."""
from __future__ import annotations
import re

# input/<id>_scaffolds.csv -> <id> (ids may contain underscores)
SCAFFOLD_RE = re.compile(r"(?:^|/)(?P<id>[^/]+)_scaffolds\.csv$")


def dataset_ids_from_keys(keys: list[str]) -> list[str]:
    """Extract sorted, unique dataset ids from a list of S3 keys."""
    return sorted({m.group("id") for k in keys if (m := SCAFFOLD_RE.search(k))})