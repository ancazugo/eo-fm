"""Build the WUDAPT training GeoPackages for the teacher-student arm (E1/E2).

Reads the lcz_wudapt QC pool (``patches_wudapt_rxr.gpkg``) and writes gpkgs in
the ``--pseudo-gpkg`` schema (``patch_id, dataset, LCZ_class, weight``) that
``patch_classification.py`` appends to the So2Sat training set:

* ``wudapt_e1.gpkg`` -- every WUDAPT patch that survives the leakage filters,
  at its own QC weight (time x accuracy x area x neighbour-conflict). No teacher.
* ``wudapt_e2.gpkg`` -- the same, minus the patches the teacher vetoes
  (``--teacher-checkpoint``).

Leakage filters, applied to both:

1. ``wudapt_split == train`` only. WUDAPT val/test stay out of training so they
   remain a secondary readout.
2. **Culture-10 exclusion.** Drop any patch inside a culture-10 city's GUPPD
   polygon, or within ``--buffer-km`` of it, or within ``--buffer-km`` of any
   So2Sat validation/testing patch (those extend past the GUPPD polygon in
   places). lcz_wudapt already forces culture-city AOIs to test, but a
   neighbouring AOI can still reach into the same urban area.
3. **So2Sat wins.** Drop any patch that intersects a So2Sat patch of any split.
4. Only patches with an extracted npy for ``--output-name``/``--year``.

Teacher veto (E2). The teacher scores each surviving patch with the same input
pipeline it was trained with (channel stats from the checkpoint, nodata masking,
dihedral TTA). A WUDAPT label is vetoed when the teacher's probability for it
is below tau, where tau is the ``--veto-quantile`` of the teacher's probability
for the TRUE label on So2Sat validation. So "vetoed" means: the teacher finds
this label less plausible than it finds all but that fraction of genuine
held-out labels. With 2017 embeddings and 2019-2024 labels, some vetoes are
real change since 2017, which is exactly what should be dropped. The veto rate
by label year is written out to check that.

Outputs in ``--out-dir``: the two gpkgs, ``wudapt_teacher_scores.parquet`` (every
candidate with its teacher probabilities and filter flags) and
``wudapt_build_report.txt``.

Example:
    python src/build_wudapt_train_gpkg.py \\
        --wudapt-gpkg ${DATA_DIR}/input/WUDAPT/patches_wudapt_rxr.gpkg \\
        --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \\
        --output-name GeoTessera_v2 --year 2017 --embedding-name tesserav2 \\
        --teacher-checkpoint <run>/mobilenet_small_GeoTessera_v2_global-best.pt \\
        --family mobilenet --preset small --out-dir data/wudapt_teacher_student
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).parent))

from datasets.so2sat import GPKG_OPEN, PatchItem, build_patch_index  # noqa: E402
from utils.city_split import SO2SAT_CULTURE_CITIES  # noqa: E402

KEEP_COLS = ["patch_id", "dataset", "LCZ_class", "weight", "aoi", "region",
             "label_year", "oa", "nbr_conflict", "nbr_dist_m", "w_time"]


def _metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """World-wide metric CRS for buffering; distortion is irrelevant at 1.3 km
    next to the per-city spread, and one CRS keeps a single STRtree."""
    return gdf.to_crs("EPSG:3857")


def culture_exclusion(cand: gpd.GeoDataFrame, so2sat: gpd.GeoDataFrame,
                      bounds_gpkg: Path, buffer_km: float) -> np.ndarray:
    """True where a candidate is inside/near a culture-10 city."""
    b = gpd.read_file(bounds_gpkg, layer="so2sat_guppd_bounds_gdf", **GPKG_OPEN)
    names = {c.replace("_", " ") for c in SO2SAT_CULTURE_CITIES}
    b = b[b["JRC_NAME_MAIN"].isin(names)]
    if len(b) != len(SO2SAT_CULTURE_CITIES):
        raise SystemExit(f"expected {len(SO2SAT_CULTURE_CITIES)} culture polygons, "
                         f"found {len(b)}: {sorted(b.JRC_NAME_MAIN)}")
    # Buffer each shape in its own UTM so 1.3 km means 1.3 km at any latitude,
    # then bring the buffered shapes back to one CRS for the tree.
    zones = []
    for _, row in b.iterrows():
        one = gpd.GeoDataFrame(geometry=[row.geometry], crs=b.crs)
        utm = one.estimate_utm_crs()
        zones.append(one.to_crs(utm).buffer(buffer_km * 1000).to_crs(4326).iloc[0])
    evalp = so2sat[so2sat["dataset"].isin(["validation", "testing"])]
    for utm, grp in evalp.groupby(evalp.geometry.centroid.x.floordiv(6)):
        g = gpd.GeoDataFrame(geometry=grp.geometry.values, crs=so2sat.crs)
        crs = g.estimate_utm_crs()
        zones.extend(g.to_crs(crs).buffer(buffer_km * 1000).to_crs(4326).values)
    tree = STRtree(zones)
    hit = np.zeros(len(cand), dtype=bool)
    idx_c, _ = tree.query(cand.geometry.values, predicate="intersects")
    hit[np.unique(idx_c)] = True
    return hit


def so2sat_overlap(cand: gpd.GeoDataFrame, so2sat: gpd.GeoDataFrame) -> np.ndarray:
    """True where a candidate intersects any So2Sat patch (So2Sat always wins).

    ``touches`` is excluded: WUDAPT's lattice can abut a So2Sat square along an
    edge without sharing any ground.
    """
    tree = STRtree(so2sat.geometry.values)
    idx_c, idx_s = tree.query(cand.geometry.values, predicate="intersects")
    a = cand.geometry.values[idx_c].intersection(so2sat.geometry.values[idx_s]).area
    hit = np.zeros(len(cand), dtype=bool)
    hit[np.unique(idx_c[a > 0])] = True
    return hit


def teacher_probs(items: list[PatchItem], args) -> np.ndarray:
    """(N, 17) softmax from the teacher, through its own training input path."""
    import torch
    from torch.utils.data import DataLoader

    from datasets.registry import get_nodata_predicate
    from datasets.so2sat import PatchDataset
    from infer_roi import load_model_and_normalize
    from training.evaluate import predict_probs
    from utils.runtime import resolve_dequantize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalize = load_model_and_normalize(
        args.teacher_checkpoint, args.family, args.embedding_name, device,
        preset=args.preset, patch_size=args.patch_size,
    )
    dequantize_fn, _ = resolve_dequantize(args.embedding_name)
    mean, std = normalize if normalize is not None else (None, None)
    ds = PatchDataset(
        items, args.patch_size, dequantize_fn=dequantize_fn,
        nodata_mode=args.nodata_mode,
        nodata_predicate=get_nodata_predicate(args.embedding_name),
        normalize="channel" if normalize is not None else "none",
        channel_mean=mean, channel_std=std,
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=args.num_workers)
    with torch.no_grad():
        return predict_probs(model, loader, device, tta=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--wudapt-gpkg", type=Path, required=True)
    p.add_argument("--so2sat-dir", type=Path, required=True)
    p.add_argument("--output-name", required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--embedding-name", required=True)
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--buffer-km", type=float, default=1.3)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, default=None)
    p.add_argument("--family", default="mobilenet")
    p.add_argument("--preset", default="small")
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--nodata-mode", default="mask", choices=["zero", "mask"])
    p.add_argument("--veto-quantile", type=float, default=0.10,
                   help="tau = this quantile of the teacher's p(true label) on So2Sat val")
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    def say(msg: str) -> None:
        logger.info(msg)
        report.append(msg)

    pool = gpd.read_file(args.wudapt_gpkg, **GPKG_OPEN)
    say(f"WUDAPT pool: {len(pool):,} patches, {pool.aoi.nunique()} AOIs")
    cand = pool[pool["wudapt_split"].isin(args.splits)].reset_index(drop=True)
    say(f"  wudapt_split in {args.splits}: {len(cand):,}")

    index = build_patch_index(args.so2sat_dir, args.output_name, args.year)
    unlab = index.get("unlabeled", {})
    has_npy = cand["patch_id"].astype(str).isin(unlab).to_numpy()

    so2sat = gpd.read_file(args.so2sat_dir / "patches_reference_rxr.gpkg",
                           columns=["dataset"], **GPKG_OPEN).to_crs(cand.crs)
    near_culture = culture_exclusion(cand, so2sat, args.so2sat_dir / "so2sat_guppd_bounds.gpkg",
                                     args.buffer_km)
    on_so2sat = so2sat_overlap(cand, so2sat)
    cand["has_npy"], cand["near_culture"], cand["on_so2sat"] = has_npy, near_culture, on_so2sat
    keep = has_npy & ~near_culture & ~on_so2sat
    say(f"  no npy: {(~has_npy).sum():,} | culture-10 +{args.buffer_km} km: "
        f"{near_culture.sum():,} | overlaps So2Sat: {on_so2sat.sum():,}")
    say(f"  E1 kept: {keep.sum():,} "
        f"(AOIs dropped for culture proximity: "
        f"{sorted(cand.loc[near_culture, 'aoi'].unique())[:12]}...)")

    e1 = cand[keep].reset_index(drop=True)
    e1["dataset"] = "unlabeled"
    e1[KEEP_COLS + ["geometry"]].to_file(args.out_dir / "wudapt_e1.gpkg", driver="GPKG")
    say(f"  E1 weight: mean {e1.weight.mean():.3f}, median {e1.weight.median():.3f}; "
        f"per class: {e1.LCZ_class.value_counts().sort_index().to_dict()}")

    scores = cand.drop(columns="geometry").copy()
    if args.teacher_checkpoint is not None:
        items = [PatchItem(unlab[str(pid)], int(c) - 1, "train")
                 for pid, c in zip(e1.patch_id, e1.LCZ_class)]
        probs = teacher_probs(items, args)
        lab = e1.LCZ_class.to_numpy().astype(int)
        p_label = probs[np.arange(len(e1)), lab - 1]

        # Calibration: the teacher's p(true label) on So2Sat validation, i.e. the
        # culture-10 held-out half it was early-stopped on.
        val = gpd.read_file(args.so2sat_dir / "patches_reference_rxr.gpkg",
                            columns=["patch_id", "dataset", "LCZ_class"],
                            ignore_geometry=True, **GPKG_OPEN)
        val = val[val["dataset"] == "validation"]
        vidx = index.get("validation", {})
        val = val[val["patch_id"].astype(str).isin(vidx)]
        vitems = [PatchItem(vidx[str(pid)], int(c) - 1, "val")
                  for pid, c in zip(val.patch_id, val.LCZ_class)]
        vprobs = teacher_probs(vitems, args)
        vlab = val.LCZ_class.to_numpy().astype(int)
        v_true = vprobs[np.arange(len(val)), vlab - 1]
        tau = float(np.quantile(v_true, args.veto_quantile))
        say(f"Teacher on So2Sat val ({len(val):,}): OA {np.mean(vprobs.argmax(1) + 1 == vlab):.3f}, "
            f"p(true) median {np.median(v_true):.3f}; tau = q{args.veto_quantile:.2f} = {tau:.4f}")

        veto = p_label < tau
        agree = probs.argmax(1) + 1 == lab
        say(f"Teacher on E1 candidates: agrees with the WUDAPT label on {agree.mean():.1%}; "
            f"p(label) median {np.median(p_label):.3f}; vetoed {veto.sum():,} ({veto.mean():.1%})")
        tab = pd.DataFrame({"LCZ": lab, "veto": veto, "year": e1.label_year.to_numpy(),
                            "region": e1.region.to_numpy()})
        say("Veto rate by class: " + ", ".join(
            f"{c}:{r:.0%}" for c, r in tab.groupby("LCZ").veto.mean().items()))
        say("Veto rate by label year: " + ", ".join(
            f"{y}:{r:.0%}(n={n})" for y, (r, n) in
            tab.groupby("year").veto.agg(["mean", "size"]).iterrows() if n >= 100))
        say("Veto rate by region: " + ", ".join(
            f"{g}:{r:.0%}" for g, r in tab.groupby("region").veto.mean().items()))

        e2 = e1[~veto].reset_index(drop=True)
        e2[KEEP_COLS + ["geometry"]].to_file(args.out_dir / "wudapt_e2.gpkg", driver="GPKG")
        say(f"  E2 kept: {len(e2):,}; per class: "
            f"{e2.LCZ_class.value_counts().sort_index().to_dict()}")

        tscore = pd.DataFrame(probs, columns=[f"p{c}" for c in range(1, 18)])
        tscore.insert(0, "patch_id", e1.patch_id.to_numpy())
        tscore["p_label"], tscore["teacher_top1"], tscore["vetoed"] = p_label, probs.argmax(1) + 1, veto
        scores = scores.merge(tscore, on="patch_id", how="left")
        scores.attrs["tau"] = tau

    scores.to_parquet(args.out_dir / "wudapt_teacher_scores.parquet")
    (args.out_dir / "wudapt_build_report.txt").write_text("\n".join(report) + "\n")
    logger.info(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
