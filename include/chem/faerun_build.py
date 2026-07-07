"""Optional step 5: Faerun interactive graph.

Projects Morgan-fingerprint space to 2D (t-SNE, falling back to PCA for tiny
sets) and renders an interactive Faerun HTML scatter coloured by K-means
cluster, with molecule structures drawn on hover.
"""
from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Compatibility problems: Faerun 0.4.x calls matplotlib.cm.get_cmap, removed in
# matplotlib >= 3.9. Restoring it from the new colormap registry.
import matplotlib.cm
from matplotlib import colormaps as _mpl_colormaps
if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = _mpl_colormaps.get_cmap

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
    # Write file_name + path separately so the HTML references the data JS by a
    # realtive name (molecules.js), not an absolute build-time path.
    out_dir, name = os.path.split(out_basename)
    f.plot(name, path=out_dir or ".", template="smiles")
    return f"{out_basename}.html"