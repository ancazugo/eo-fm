"""T6 — post-processing: block-majority argmax -> min-zone dissolve -> map.

Raw per-pixel argmax is a diagnostic; the deliverable map product dissolves
same-prediction connected block components (reusing the Stage 4a adjacency,
the same mechanism as ``lcz_labels.classify.form_zones``) and reassigns
components smaller than ``min_zone_area_ha`` to the modal prediction among
their non-small neighbours — a single documented pass, not an iterated fixpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from rasterio.features import rasterize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from lcz_labels.export import raster_grid


def dissolve_small_zones(
    pred_lcz: np.ndarray,
    block_ids: np.ndarray,
    areas_m2: np.ndarray,
    adjacency_df,
    *,
    min_zone_area_ha: float = 15.0,
) -> np.ndarray:
    """Single-pass min-zone dissolve over predicted (not pseudo-) labels.

    Connected components of same-prediction adjacent blocks form candidate
    zones; components below ``min_zone_area_ha`` are reassigned to the modal
    prediction among their adjacent NON-small blocks (a single tower-block
    prediction inside dissolved lowrise fabric is treated as map noise here,
    the mirror image of ``form_zones``'s zone_grade flag on pseudo-labels).
    Unpredicted blocks (``pred_lcz == 0``) are left untouched.
    """
    n = len(block_ids)
    pos = {b: i for i, b in enumerate(block_ids)}
    valid = pred_lcz > 0

    edges = adjacency_df[
        adjacency_df["block_a"].isin(pos) & adjacency_df["block_b"].isin(pos)
    ].copy()
    if edges.empty:
        return pred_lcz.copy()
    pa = edges["block_a"].map(pos).to_numpy()
    pb = edges["block_b"].map(pos).to_numpy()

    same = (pred_lcz[pa] == pred_lcz[pb]) & valid[pa] & valid[pb]
    graph = coo_matrix((np.ones(int(same.sum())), (pa[same], pb[same])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)

    comp_area = np.zeros(comp.max() + 1 if n else 0)
    np.add.at(comp_area, comp[valid], areas_m2[valid])
    small = valid & (comp_area[comp] / 1.0e4 < min_zone_area_ha)

    out = pred_lcz.copy()
    if not small.any():
        return out

    neighbours: dict[int, list[int]] = {i: [] for i in np.where(small)[0]}
    for a, b in zip(pa, pb):
        if small[a] and valid[b] and not small[b]:
            neighbours[a].append(int(pred_lcz[b]))
        if small[b] and valid[a] and not small[a]:
            neighbours[b].append(int(pred_lcz[a]))
    n_reassigned = 0
    for i, votes in neighbours.items():
        if votes:
            vals, counts = np.unique(votes, return_counts=True)
            out[i] = int(vals[np.argmax(counts)])
            n_reassigned += 1
    logger.info(f"dissolve: {int(small.sum())} small-zone blocks, "
                f"{n_reassigned} reassigned to a neighbour's prediction")
    return out


def write_map_product(
    pred_lcz: np.ndarray, blocks_gdf, aoi_name: str, lcz_config, out_dir: str | Path,
) -> dict[str, Path]:
    """GeoParquet + rendered GeoTIFF of the final (post-processed) prediction."""
    transform, utm, (h, w) = raster_grid(aoi_name, lcz_config)
    g = blocks_gdf if str(blocks_gdf.crs) == utm else blocks_gdf.to_crs(utm)
    labelled = pred_lcz > 0

    kw = dict(out_shape=(h, w), transform=transform, fill=0, all_touched=False)
    if labelled.any():
        raster = rasterize(
            zip(g.geometry.values[labelled], pred_lcz[labelled].astype("uint8").tolist()),
            dtype="uint8", **kw,
        )
    else:
        raster = np.zeros((h, w), dtype="uint8")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = out_dir / f"map_{aoi_name}.tif"
    with rasterio.open(
        tif_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint8",
        crs=utm, transform=transform, nodata=0, compress="lzw",
    ) as dst:
        dst.write(raster, 1)

    gdf_out = g.copy()
    gdf_out["pred_lcz"] = pred_lcz
    parquet_path = out_dir / f"map_{aoi_name}.parquet"
    gdf_out.to_parquet(parquet_path)
    logger.info(f"[{aoi_name}] map product: {int(labelled.sum())}/{len(g)} blocks -> "
                f"{tif_path.name}, {parquet_path.name}")
    return {"raster": tif_path, "vector": parquet_path}
