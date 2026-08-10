"""Convert osm-rasterizer output into a properly labelled LCZ 1-17 raster.

``osm-rasterizer`` (see docs/osm_lcz_tag_mapping.md) writes pixel values that are
**1-based indices into the feature order**, not LCZ class codes: value 8 means
"the 8th feature, roads_minor", not LCZ 8 (Large Low-Rise).  This script maps
those proxy indices onto the So2Sat convention used everywhere else in the repo
(built 1-10 map to themselves, natural letters A-G map to 11-17) and writes a
uint8 GeoTIFF with the official WUDAPT colour table embedded, so QGIS renders it
with no styling at all.

Two proxy classes cannot be resolved per-pixel.  ``buildings_lowrise`` is LCZ 3
*or* 6, ``buildings_midrise`` 2 or 5, ``buildings_highrise`` 1 or 4 — compact vs
open is a neighbourhood density property, and LCZ 9 is purely one.  They are
split by the Building Surface Fraction (Stewart & Oke 2012): a moving-window
fraction of building pixels, binned at 0.20 / 0.40.  LCZ 9 is confined to
low-rise, since Stewart & Oke define it as 1-3 storeys.

Both osm-rasterizer outputs are accepted and auto-detected:

  * ``--single-layer`` (1 band)  — proxy indices read directly
  * multi-band (one 0/1 band per feature) — composed here in band order, which
    reproduces the same "last feature wins" priority without inheriting any
    ``--fill-nodata`` artefact

**On ``--fill-nodata``:** it fills unmapped pixels from the nearest labelled
neighbour, which for a sparsely mapped city invents most of the raster.  On the
Nairobi run only 28.4% of the AOI carried any OSM feature, and the fill inflated
``roads_minor`` from 4.15% to 53.42% and the building union from 3.36% to 13.79%
— which in turn pushed the BSF "compact" bin from 28.5% to 89.7%.  The fill is
not recoverable, so this script detects it, refuses to derive density from it
(unless ``--force``), and offers ``--fill-distance-m`` to redo the fill *after*
relabelling, where it no longer corrupts the density logic.

Example:
    python src/osm_lcz_relabel.py lcz_labels_multiband.tif \\
        -o nairobi_lcz.tif --png --dpi 200 \\
        --density bsf --bsf-window-m 100 --fill-distance-m 100 \\
        --title "Nairobi - OSM-derived LCZ proxy (BSF 100 m)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from loguru import logger
from scipy.ndimage import binary_erosion, distance_transform_edt, uniform_filter

sys.path.insert(0, str(Path(__file__).parent))

from utils.constants import lcz_dict

# The curated feature order of docs/osm_lcz_tag_mapping.md Command A.  Used only
# when the raster carries no CATEGORIES/BAND_NAMES tag to declare its own.
FEATURE_ORDER: tuple[str, ...] = (
    "low_plants", "scrub", "scattered_trees", "dense_trees", "bare_soil",
    "bare_rock", "rail", "roads_minor", "roads_major", "paved", "heavy_industry",
    "buildings_lowrise", "large_lowrise", "lightweight", "buildings_midrise",
    "buildings_highrise", "water",
)

# proxy feature -> LCZ code, for the features a single pixel can decide alone.
UNAMBIGUOUS: dict[str, int] = {
    "low_plants": 14,       # D
    "scrub": 13,            # C
    "scattered_trees": 12,  # B
    "dense_trees": 11,      # A
    "bare_soil": 16,        # F
    "bare_rock": 15,        # E
    "rail": 15,             # E
    "roads_minor": 15,      # E
    "roads_major": 15,      # E
    "paved": 15,            # E
    "heavy_industry": 10,
    "large_lowrise": 8,
    "lightweight": 7,
    "water": 17,            # G
}

# proxy feature -> (sparse, open, compact) LCZ code, chosen by BSF bin.
AMBIGUOUS: dict[str, tuple[int, int, int]] = {
    "buildings_lowrise": (9, 6, 3),
    "buildings_midrise": (5, 5, 2),
    "buildings_highrise": (4, 4, 1),
}

BUILDING_FEATURES = frozenset(AMBIGUOUS) | {"large_lowrise", "lightweight"}

# Buffered lines are 1-2 px wide at 10 m, so a 3x3 erosion empties them; nodata
# fill turns them into blobs with a large interior.  rail is NOT a probe: its tag
# set includes landuse=railway polygons, which score 0.575 even unfilled.
FILL_PROBES = ("roads_minor", "roads_major")
FILL_INTERIOR_THRESHOLD = 0.30

# Excluded as donors by --fill-from areal: exactly the features Command A buffers
# from line geometries (the ones carrying a `line_width` option).  A buffered
# network is pervasive, so under nearest-neighbour fill it wins almost every
# contest and swamps the map — filling Nairobi from all donors takes LCZ 15 from
# 16.9% to 54.3%, and in Cairo, whose Nile Delta irrigation canals are mapped as
# `waterway=drain`/`ditch`, water goes 10.1% -> 29.7%.  Keying on "was this burned
# from a line?" rather than on a per-city name list is what makes the two behave
# the same.  Buildings stay donors: they are small polygons, not a network, and
# the zone around a building genuinely is built.
LINEAR_FEATURES = ("scattered_trees", "rail", "roads_minor", "roads_major",
                   "paved", "water")


def read_feature_order(src: rasterio.DatasetReader, override: str | None) -> list[str]:
    """Resolve the proxy-index -> feature-name order for an osm-rasterizer raster.

    The two output modes tag themselves differently and a "try one tag, fall back
    to the other" scheme fails silently: a single-layer raster carries
    ``BAND_NAMES='landcover'`` (the composed layer's name) with the real feature
    list in ``CATEGORIES``, while a multi-band raster puts the list in
    ``BAND_NAMES`` and has no ``CATEGORIES``.  So the tag is chosen by band count.
    """
    if override:
        names = [n.strip() for n in override.split(",") if n.strip()]
        logger.info(f"Feature order from --feature-order ({len(names)} features).")
    else:
        tag = "CATEGORIES" if src.count == 1 else "BAND_NAMES"
        raw = src.tags().get(tag, "")
        names = [n.strip() for n in raw.split(",") if n.strip()]
        if names:
            logger.info(f"Feature order from the {tag} tag ({len(names)} features).")
        elif src.count in (1, len(FEATURE_ORDER)):
            names = list(FEATURE_ORDER)
            logger.warning(
                f"No {tag} tag; assuming the documented Command A order "
                f"({len(names)} features). Pass --feature-order to override."
            )
        else:
            raise SystemExit(
                f"No {tag} tag and {src.count} bands does not match the documented "
                f"{len(FEATURE_ORDER)}-feature order. Pass --feature-order."
            )

    if src.count > 1 and len(names) != src.count:
        raise SystemExit(
            f"Feature order has {len(names)} names but the raster has {src.count} bands."
        )

    known = set(UNAMBIGUOUS) | set(AMBIGUOUS)
    if unknown := [n for n in names if n not in known]:
        raise SystemExit(
            f"Unknown feature name(s): {', '.join(unknown)}\n"
            f"Known features: {', '.join(sorted(known))}"
        )
    return names


def build_lut(names: Sequence[str]) -> np.ndarray:
    """Build the ``(n_features + 1, 3)`` proxy-index x density-bin -> LCZ table.

    Row 0 stays 0 (nodata).  Columns are the density bins in ascending order:
    0 = sparse, 1 = open, 2 = compact.  Unambiguous features carry the same code
    in all three columns, so a single fancy-index resolves the whole raster and
    ``--density fixed`` is just column 1.
    """
    lut = np.zeros((len(names) + 1, 3), np.uint8)
    for i, name in enumerate(names, start=1):
        if name in AMBIGUOUS:
            lut[i, :] = AMBIGUOUS[name]
        else:
            lut[i, :] = UNAMBIGUOUS[name]
    return lut


def compose_proxy(
    src: rasterio.DatasetReader, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(proxy indices, building mask)``.

    Multi-band rasters are composed one band at a time — ``proxy[band > 0] = i``
    in band order, so later features overwrite earlier ones exactly as
    ``--single-layer`` does — which keeps peak memory at one band rather than the
    whole stack.
    """
    if src.count == 1:
        proxy = src.read(1)
        if proxy.max() > len(names):
            raise SystemExit(
                f"Raster holds proxy index {proxy.max()} but the feature order has "
                f"only {len(names)} features — the tags look stale. Pass --feature-order."
            )
        building_idx = [i for i, n in enumerate(names, start=1) if n in BUILDING_FEATURES]
        # A building overwritten by water is lost here; on a single-layer input
        # that is unavoidable and, at typical water coverage, negligible for BSF.
        return proxy, np.isin(proxy, building_idx)

    proxy = np.zeros(src.shape, np.uint8)
    building = np.zeros(src.shape, bool)
    for i, name in enumerate(names, start=1):
        band = src.read(i) > 0
        proxy[band] = i
        if name in BUILDING_FEATURES:
            building |= band
    return proxy, building


def detect_fill(proxy: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    """Interior fraction of each buffered-line probe class, as a nodata-fill test.

    A 6-12 m buffered line at 10 m resolution is 1-2 px across, so its 3x3
    erosion is empty by construction; nearest-neighbour fill grows it into blobs.
    Measured on Nairobi: roads_minor 0.000 unfilled vs 0.877 filled, roads_major
    0.012 vs 0.649 — a wide gap around the 0.30 threshold.
    """
    structure = np.ones((3, 3), bool)
    interior = {}
    for probe in FILL_PROBES:
        if probe not in names:
            continue
        mask = proxy == names.index(probe) + 1
        n = int(mask.sum())
        if n:
            interior[probe] = float(binary_erosion(mask, structure).sum()) / n
    return interior


def density_bins(
    building: np.ndarray, window_px: int, t_open: float, t_compact: float
) -> np.ndarray:
    """Bin the Building Surface Fraction into 0 = sparse, 1 = open, 2 = compact.

    ``mode="nearest"`` matters: the default ``constant`` treats the outside as
    zero and fabricates a low-BSF ring half a window wide around the raster.
    """
    bsf = uniform_filter(building.astype(np.float32), size=window_px, mode="nearest")
    return np.digitize(bsf, [t_open, t_compact]).astype(np.uint8)


def fill_gaps(
    lcz: np.ndarray, max_dist_px: float, donors: np.ndarray | None = None
) -> np.ndarray:
    """Fill nodata from the nearest labelled pixel, up to ``max_dist_px``.

    ``donors`` restricts which labelled pixels may be copied from; pixels outside
    it keep their own label but never supply one.  Only nodata is overwritten.
    """
    gaps = lcz == 0
    if not gaps.any():
        return lcz
    if donors is None:
        donors = lcz > 0
    dist, idx = distance_transform_edt(~donors, return_distances=True, return_indices=True)
    return np.where(gaps & (dist <= max_dist_px), lcz[tuple(idx)], lcz)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    h = value.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def write_lcz_tif(
    path: Path, lcz: np.ndarray, src: rasterio.DatasetReader, meta: dict[str, str]
) -> None:
    """Write the LCZ raster with the full WUDAPT colour table embedded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": lcz.shape[0], "width": lcz.shape[1],
        "count": 1, "dtype": "uint8", "crs": src.crs, "transform": src.transform,
        "nodata": 0, "compress": "lzw", "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(lcz, 1)
        dst.set_band_description(1, "lcz")
        # All 17 entries always, so the file renders identically regardless of
        # which classes a given AOI happens to contain.
        dst.write_colormap(1, {
            0: (255, 255, 255, 0),
            **{i: (*_hex_to_rgb(lcz_dict[i]["color"]), 255) for i in range(1, 18)},
        })
        dst.update_tags(LCZ_CONVENTION="So2Sat 1-17", **meta)
        dst.update_tags(1, CLASSES="; ".join(
            f"{i}={lcz_dict[i]['alt_code']}:{lcz_dict[i]['name']}" for i in range(1, 18)
        ))


def summarise(lcz: np.ndarray) -> str:
    """Per-class area table, plus the share of the AOI left unmapped."""
    mapped = int((lcz > 0).sum())
    lines = [f"{'LCZ':>4} {'':>3}  {'Class':<22} {'Pixels':>12} {'% mapped':>9}"]
    for code in range(1, 18):
        n = int((lcz == code).sum())
        if not n:
            continue
        lines.append(
            f"{code:>4} {lcz_dict[code]['alt_code']:>3}  {lcz_dict[code]['name']:<22} "
            f"{n:>12,} {100 * n / mapped:>8.2f}%"
        )
    nodata_pct = 100 * (1 - mapped / lcz.size)
    lines.append(f"\nnodata: {lcz.size - mapped:,} px ({nodata_pct:.2f}% of the AOI)")
    return "\n".join(lines)


def save_png(
    lcz: np.ndarray, src: rasterio.DatasetReader, path: Path, title: str, dpi: int
) -> None:
    from rasterio.warp import transform_bounds

    from utils.plot_lcz import save_lcz_map

    extent = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    save_lcz_map(lcz, title, path, dpi=dpi, extent=extent)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Relabel osm-rasterizer proxy indices into an LCZ 1-17 raster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path,
                   help="osm-rasterizer GeoTIFF (single-layer or multi-band).")

    g = p.add_argument_group("Output")
    g.add_argument("-o", "--output", type=Path, default=None,
                   help="Output GeoTIFF (default: <input stem>_lcz.tif).")
    g.add_argument("--png", nargs="?", const="AUTO", default=None,
                   help="Also write a PNG (bare flag: <output stem>.png).")
    g.add_argument("--dpi", type=int, default=150, help="PNG resolution.")
    g.add_argument("--title", default=None, help="PNG title.")

    g = p.add_argument_group("Classification")
    g.add_argument("--density", choices=["auto", "bsf", "fixed"], default="auto",
                   help="How to split compact/open/sparse. auto = bsf unless a "
                        "nodata fill is detected, in which case fixed. fixed maps "
                        "low/mid/high-rise to the open classes 6/5/4.")
    g.add_argument("--bsf-window-m", type=float, default=100.0,
                   help="BSF moving-window size in metres (LCZ neighbourhood scale).")
    g.add_argument("--bsf-open", type=float, default=0.20,
                   help="BSF at or above which a building is at least open.")
    g.add_argument("--bsf-compact", type=float, default=0.40,
                   help="BSF at or above which a building is compact.")
    g.add_argument("--feature-order", default=None,
                   help="Comma-separated feature names overriding the raster tags.")

    g = p.add_argument_group("Post-processing")
    g.add_argument("--fill-distance-m", type=float, default=0.0,
                   help="Fill nodata from the nearest label up to this distance "
                        "(0 = off). Runs after relabelling. Needs a 2 x H x W "
                        "int32 index array (~190 MB for a 4186x5676 grid).")
    g.add_argument("--fill-from", choices=["areal", "all"], default="areal",
                   help="Which labels may be copied into gaps. areal excludes the "
                        f"road/rail features ({', '.join(LINEAR_FEATURES)}), whose "
                        "pervasive network otherwise swamps the fill.")

    g = p.add_argument_group("Safety")
    g.add_argument("--force", action="store_true",
                   help="Allow --density bsf on a raster whose nodata was filled.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    output = args.output or args.input.with_name(f"{args.input.stem}_lcz.tif")

    with rasterio.open(args.input) as src:
        names = read_feature_order(src, args.feature_order)
        logger.info(f"{args.input.name}: {src.count} band(s), {src.shape} @ "
                    f"{src.res[0]:g} m, {src.crs}")
        proxy, building = compose_proxy(src, names)

        interior = detect_fill(proxy, names)
        filled = any(v >= FILL_INTERIOR_THRESHOLD for v in interior.values())
        if filled:
            detail = ", ".join(f"{k} {v:.3f}" for k, v in interior.items())
            logger.warning(
                f"This raster looks like it was written with --fill-nodata "
                f"(buffered-line interior fraction: {detail}). Most of its labels "
                f"were invented by nearest-neighbour fill and the building mask is "
                f"inflated, so BSF-derived density would be badly wrong."
            )

        density = args.density
        if density == "auto":
            density = "fixed" if filled else "bsf"
            logger.info(f"--density auto resolved to '{density}'.")
        elif density == "bsf" and filled and not args.force:
            raise SystemExit(
                "Refusing --density bsf on a nodata-filled raster: the fill inflates "
                "the building mask and skews BSF towards 'compact' (measured on "
                "Nairobi: 28.5% -> 89.7%). Re-run osm-rasterizer without "
                "--fill-nodata (use --fill-distance-m here instead), or pass --force."
            )

        lut = build_lut(names)
        if density == "bsf":
            window_px = max(1, round(args.bsf_window_m / src.res[0]))
            logger.info(f"BSF window {args.bsf_window_m:g} m = {window_px} px; "
                        f"bins at {args.bsf_open:g} / {args.bsf_compact:g}")
            bins = density_bins(building, window_px, args.bsf_open, args.bsf_compact)
            lcz = lut[proxy, bins]
        else:
            logger.info("Fixed density: low/mid/high-rise -> LCZ 6/5/4 (open).")
            lcz = lut[proxy, 1]

        if args.fill_distance_m > 0:
            donors = None
            if args.fill_from == "areal":
                linear_idx = [i for i, n in enumerate(names, start=1)
                              if n in LINEAR_FEATURES]
                donors = (lcz > 0) & ~np.isin(proxy, linear_idx)
            max_px = args.fill_distance_m / src.res[0]
            before = int((lcz == 0).sum())
            lcz = fill_gaps(lcz, max_px, donors)
            logger.info(f"Filled {before - int((lcz == 0).sum()):,} nodata px within "
                        f"{args.fill_distance_m:g} m (donors: {args.fill_from}).")

        write_lcz_tif(output, lcz, src, {
            "SOURCE": str(args.input),
            "DENSITY_MODE": density,
            "BSF_WINDOW_M": f"{args.bsf_window_m:g}" if density == "bsf" else "n/a",
            "FEATURE_ORDER": ",".join(names),
        })
        logger.info(f"Wrote {output}")

        if args.png is not None:
            png = output.with_suffix(".png") if args.png == "AUTO" else Path(args.png)
            title = args.title or f"OSM-derived LCZ proxy — {args.input.stem}"
            save_png(lcz, src, png, title, args.dpi)
            logger.info(f"Wrote {png}")

    print(f"\n{summarise(lcz)}")


if __name__ == "__main__":
    main()
