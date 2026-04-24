"""LCZ raster plotting utilities.

Public API:
    lcz_colormap()  → (cmap, norm)  — ListedColormap for LCZ uint8 rasters
    save_lcz_map(raster, title, save_path, dpi=150)  — save PNG with bottom legend
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_src = Path(__file__).parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from utils.constants import lcz_dict


def lcz_colormap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """Return (cmap, norm) for LCZ uint8 rasters (values 0–17).

    Index 0 → white (nodata), indices 1–17 → standard LCZ class colours.
    """
    colors = ["#ffffff"] + [lcz_dict[i]["color"] for i in range(1, 18)]
    cmap = mcolors.ListedColormap(colors)
    bounds = list(range(0, 19))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def save_lcz_map(
    raster: np.ndarray,
    title: str,
    save_path: Path | str,
    dpi: int = 150,
) -> None:
    """Save an LCZ prediction map as a PNG with legend at the bottom.

    Parameters
    ----------
    raster:
        uint8 array, values 0 (nodata/white) and 1–17 (LCZ classes).
    title:
        Figure title string.
    save_path:
        Output PNG path.
    dpi:
        Resolution (default 150).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cmap, norm = lcz_colormap()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(raster, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    present = [i for i in range(1, 18) if np.any(raster == i)]
    patches = [
        mpatches.Patch(color=lcz_dict[i]["color"], label=f"{i}: {lcz_dict[i]['name']}")
        for i in present
    ]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
