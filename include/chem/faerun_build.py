"""Optional step 5: Faerun interactive graph.

Projects Morgan-fingerprint space to 2D (t-SNE, falling back to PCA for tiny
sets) and renders an interactive Faerun HTML scatter coloured by K-means
cluster, with molecule structures drawn on hover.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from include.chem.clustering import fingerprint_matrix

log = logging.getLogger(__name__)


def _project_2d(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n < 5:
        return PCA(n_components=2).fit_transform(X) if n >= 2 else np.zeros((n, 2))
    try:
        perplexity = min(30, max(2, n // 3))
        return TSNE(n_components=2, perplexity=perplexity,
                    init="pca", random_state=42).fit_transform(X)
    except Exception as e:
        log.warning("t-SNE failed (%s); falling back to PCA", e)
        return PCA(n_components=2).fit_transform(X)


def build(df: pd.DataFrame, out_basename: str) -> str | None:
    """Write <out_basename>.html (+ .js) and return the HTML path, or None."""
    try:
        from faerun import Faerun
    except Exception as e:
        log.warning("Faerun unavailable (%s); skipping graph", e)
        return None

    coords = _project_2d(fingerprint_matrix(df["smiles"].tolist()))

    f = Faerun(view="front", clear_color="#111111", coords=False)
    f.add_scatter(
        "molecules",
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "c": df["cluster"].tolist(),
            "labels": df["smiles"].tolist(),
        },
        shader="smoothCircle", point_scale=5, colormap="tab10",
        has_legend=True, categorical=True, legend_title="K-means cluster",
    )
    f.plot(out_basename, template="smiles")
    return f"{out_basename}.html"