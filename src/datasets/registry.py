"""Registry of embedding types: metadata used across the pipelines.

``in_channels`` feeds model construction; the ``zarr_*`` filename metadata
drives tile discovery without opening files (datasets.tiles.build_tile_index,
find_tiles_for_roi). CLI ``--embedding-name`` choices are derived from the
registry keys.

``nodata_predicate`` marks per-pixel missing data. Each family stores nodata
differently and none of them uses NaN, so the rule cannot be global — see
:func:`get_nodata_predicate` for the measurements behind each entry.

**Provenance** (``product``/``version``/``source``/``status``, Task 1.5.2) is
what makes two entries comparable or not. ``in_channels`` alone does not: Task
1.5.1 measured ``tesserav1.1`` and ``tesserav1.1_global`` to be *different
feature bases* — same scene, matched-channel correlation ≈ 0, a linear 128→128
map recovering 89% of one from the other — while both declare 128 channels, so
``build_model`` accepts either and nothing would flag a model or normalizer
built on one and applied to the other. :func:`is_comparable` and
:func:`check_checkpoint_provenance` are the guards; see RESULTS.md
"Task 1.5.1 — Tessera product identity".

Keys are never renamed. They appear in extraction paths on disk and in the W&B
config of every historical run, so a rename orphans both.
"""

import re
from pathlib import Path

import numpy as np
from loguru import logger

# Vocabularies. Kept explicit so a typo in an entry is a test failure rather
# than a silently unique provenance that compares equal to nothing.
PRODUCTS = {"tessera", "alphaearth", "esd", "sentinel", "osm", "aux"}
VERSIONS = {"v1", "v1.1", "v2", "coop", None}
SOURCES = {
    "percity_geotessera", "global_0.1deg", "source_coop", "gee_zarr", "local_tif",
}
STATUSES = {"canonical", "supported", "deprecated", "untested"}

PROVENANCE_FIELDS = ("product", "version", "source", "status")

EMBEDDING_REGISTRY: dict[str, dict] = {
    "tessera": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera 128-band Sentinel-1/2 embeddings",
        "product": "tessera",
        "version": "v1",
        "source": "gee_zarr",
        # Untested against the CURRENT code, not never used: 42 W&B runs used
        # this product (Task 1.5.4), but they predate the open_tile ordering bug
        # (path.is_dir() tested before path.suffix == ".zarr", which misrouted
        # every zarr tile to the Tessera NPY reader). Fixed in Task 1.0 and
        # nothing has been re-run through it since, so it should not look
        # available until someone exercises it.
        "status": "untested",
        # Zarr fast-path: tiles are named by center coords, 0.1° × 0.1° grid.
        # e.g. grid_0.15_52.05_2024.zarr → center (0.15, 52.05)
        "zarr_tile_size": 0.1,
        "zarr_filename_pattern": r"grid_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)_\d+\.(zarr|tif)",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": True,
    },
    "alpha_earth": {
        "in_channels": 64,
        "resolution": 10,
        "description": "AlphaEarth 64-band satellite embeddings",
        "product": "alphaearth",
        "version": "v1",
        "source": "gee_zarr",
        # Same open_tile ordering bug as `tessera`, same caveat: used
        # historically, not verified against the fixed code.
        "status": "untested",
        # Zarr fast-path: tiles are named by bottom-left corner, 0.1° × 0.1° grid.
        # e.g. gse_2.2_48.8_2021.zarr → bottom-left (2.2, 48.8)
        "zarr_tile_size": 0.1,
        "zarr_filename_pattern": r"gse_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)_\d+\.(zarr|tif)",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": False,
    },
    "seamless": {
        "in_channels": 72,  # 12 temporal months × 6 VQ levels (band 13 is QA, excluded)
        "resolution": 30,
        "description": "Embedded Seamless Data (ESD) quantized embeddings, dequantized to 72 channels",
        "product": "esd",
        "version": None,
        "source": "local_tif",
        "status": "canonical",
    },
    "alpha_earth_coop": {
        "in_channels": 64,
        "resolution": 10,
        "description": "AlphaEarth coop GeoTIFF tiles (source.coop), 64-band Int8 quantized",
        "nodata_all_channels_eq": -128.0,
        "product": "alphaearth",
        "version": "coop",
        "source": "source_coop",
        "status": "canonical",
    },
    "tesserav1.1": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v1.1 int8+scales tiles, dequantized to 128-band float32",
        "nodata_all_channels_eq": 0.0,
        "product": "tessera",
        "version": "v1.1",
        # A DIFFERENT ARCHIVE from tesserav1.1_global despite the shared version
        # label, and a different feature basis (Task 1.5.1). 51 cities only:
        # 27,229 of the 352,366 cultural-split training patches.
        "source": "percity_geotessera",
        "status": "supported",
    },
    "tesserav1.1_global": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v1.1 global tiles (global_0.1_degree_representation + tiff_all), 128-band float32",
        "nodata_all_channels_eq": 0.0,
        "product": "tessera",
        "version": "v1.1",
        "source": "global_0.1deg",
        # Canonical Tessera: 97.3% coverage of the cultural-split training set,
        # the only Tessera entry that can carry a global-split claim.
        "status": "canonical",
    },
    "tesserav2": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v2 global tiles (large_student), int8+scales dequantized to 128-band float32",
        "nodata_all_channels_eq": 0.0,
        "product": "tessera",
        "version": "v2",
        "source": "global_0.1deg",
        "status": "supported",         # 37% So2Sat coverage; version comparison only
    },
    "osm_evidence": {
        "in_channels": 15,
        "resolution": 10,
        "description": "Binary OSM evidence layers (buildings + height buckets, "
                       "roads, rail, landuse groups, vegetation, water, bare, "
                       "completeness mask) rasterized by build_osm_rasters.py. "
                       "--embedding-dir = .../osm_evidence/tiles",
        "product": "osm",
        "version": None,
        "source": "local_tif",
        # Code retained, generated rasters deleted when the OSM fusion pilot was
        # put on hold — nothing can run against this until they are rebuilt.
        "status": "untested",
        # 0.5° tiles named by bottom-left corner: osm_{lon}_{lat}.tif
        "zarr_tile_size": 0.5,
        "zarr_filename_pattern": r"osm_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)\.(zarr|tif)",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": False,
    },
    "aux_struct": {
        "in_channels": 4,
        "resolution": 10,
        "description": "Auxiliary structural bands, normalised to ~[0,1]: GHSL ANBH/50, "
                       "built fraction, non-residential built fraction, ETH canopy height/50. "
                       "--embedding-dir = .../aux_struct/merged_aux (precompute_aux_tiles.py)",
        "product": "aux",
        "version": None,
        "source": "local_tif",
        "status": "supported",
        # 0.5° precomputed tiles named by bottom-left corner: aux_{lon}_{lat}.tif
        "zarr_tile_size": 0.5,
        "zarr_filename_pattern": r"aux_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)\.(zarr|tif)",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": False,
    },
}


def get_nodata_predicate(name: str):
    """Return ``fn(arr) -> (H, W) bool`` marking invalid pixels, or None.

    ``arr`` is the ``(C, H, W)`` array exactly as it comes off disk — before
    ``nan_to_num`` and before any dequantization — so the sentinel is tested in
    the units it is actually stored in.

    Per-family rules, each measured over the So2Sat training split in Phase 0
    (see RESULTS.md, "Incidental findings" and the GATE 0 amendments):

    * ``alpha_earth_coop`` — invalid where all 64 int8 channels equal -128.
      0.41-0.68% of pixels. -128 NEVER occurs in a single channel alone, so a
      per-channel test would be wrong; and these are not NaN, so ``nan_to_num``
      never caught them. Under the (correct) dequantization each becomes a
      vector of L2 norm 8.06 instead of 1.0.
    * ``tesserav1.1`` / ``tesserav1.1_global`` / ``tesserav2`` — invalid where
      all 128 channels are exactly 0 (0.05-0.15% of pixels). Tessera also has
      ~0.9% *per-channel* quantization zeros affecting 61% of pixels, which are
      valid data: testing per-channel here would discard most of the dataset.
    * ``seamless`` — none. Measured 0.0000% all-zero pixels.
    * ``sentinel1`` / ``sentinel2`` — none. The GeoTIFFs set ``nodata=None``,
      carry no NaNs and have no all-zero pixels in any of the three splits, so
      they are treated as fully valid rather than searched for a mask at read
      time.

    NaN in any channel counts as invalid for every family, even though no
    family currently contains any — it is the one rule that is always right.
    """
    sentinel = EMBEDDING_REGISTRY.get(name, {}).get("nodata_all_channels_eq")

    def predicate(arr: np.ndarray) -> np.ndarray:
        invalid = np.isnan(arr).any(axis=0)
        if sentinel is not None:
            invalid |= (arr == sentinel).all(axis=0)
        return invalid

    return predicate


def get_in_channels(name: str) -> int:
    """Return the number of input channels for a given embedding name."""
    if name not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding '{name}'. Choose from: {list(EMBEDDING_REGISTRY)}")
    return EMBEDDING_REGISTRY[name]["in_channels"]


# ── Provenance (Task 1.5.2) ──────────────────────────────────────────────────

def provenance(name: str | list[str] | tuple[str, ...]) -> dict:
    """Provenance of one embedding, or of a fused combination.

    Fused runs (several ``--embedding-name`` values) get each field joined with
    ``+`` in the given order, so the result is still a flat dict of strings that
    a checkpoint and a W&B config can carry unchanged.

    Returns ``{"embedding_name", "product", "version", "source", "status"}``,
    with ``version: None`` rendered as the string ``"none"`` — these values are
    written into checkpoints and compared as strings.
    """
    names = [name] if isinstance(name, str) else list(name)
    for n in names:
        if n not in EMBEDDING_REGISTRY:
            raise ValueError(
                f"Unknown embedding '{n}'. Choose from: {list(EMBEDDING_REGISTRY)}"
            )
    out = {"embedding_name": "+".join(names)}
    for field in PROVENANCE_FIELDS:
        out[field] = "+".join(
            str(EMBEDDING_REGISTRY[n].get(field) or "none") for n in names
        )
    return out


def is_comparable(a: str, b: str) -> bool:
    """True only if two registry entries are the same product, version and source.

    ``status`` is deliberately excluded: it records how much we trust an entry,
    not what the data is, and promoting an entry to ``canonical`` must not
    change whether an existing checkpoint still matches it.
    """
    pa, pb = provenance(a), provenance(b)
    return all(pa[f] == pb[f] for f in ("product", "version", "source"))


def check_checkpoint_provenance(ckpt: dict, embedding_name) -> None:
    """Refuse a checkpoint built on a different product than the one requested.

    This is the failure mode that produces a plausible wrong number instead of
    an error: ``tesserav1.1`` and ``tesserav1.1_global`` both declare 128
    channels, so a checkpoint from one loads cleanly against the other and
    predicts confident nonsense (Task 1.5.1 — matched-channel correlation ≈ 0).

    Mismatch raises. **Absence only warns**: every checkpoint written before
    Task 1.5.3 carries no provenance at all, including the one the GATE 1
    bit-identity regression replays, and refusing those would break
    reproducibility of the pre-Phase-1 path for no safety gain.
    """
    want = provenance(embedding_name)
    have = {f: ckpt.get(f) for f in PROVENANCE_FIELDS if ckpt.get(f) is not None}

    if not have:
        logger.warning(
            "Checkpoint carries no provenance metadata (written before Task "
            f"1.5.3). Assuming it matches '{want['embedding_name']}' — verify "
            "against the run's W&B config if the numbers matter."
        )
        return

    differs = {f: (have[f], want[f]) for f in ("product", "version", "source")
               if f in have and have[f] != want[f]}
    if differs:
        detail = "; ".join(f"{f}: checkpoint '{h}' vs requested '{w}'"
                           for f, (h, w) in differs.items())
        raise ValueError(
            f"Checkpoint provenance does not match --embedding-name "
            f"'{want['embedding_name']}'. {detail}. The checkpoint was trained on "
            f"'{ckpt.get('embedding_name', '<unrecorded>')}'. These are different "
            "products even where the channel count matches, so the prediction "
            "would be meaningless — re-run with the embedding it was trained on."
        )


def find_tiles_for_roi(
    directory: str | Path,
    roi: tuple[float, float, float, float],
    embedding_name: str,
) -> list[Path]:
    """Return paths of tiles in *directory* that overlap *roi*.

    Bounds are derived from filenames — no file is opened.

    Args:
        directory: Folder containing ``.zarr`` or ``.tif`` tile files.
        roi: ``(west, south, east, north)`` bounding box in EPSG:4326.
        embedding_name: Key in :data:`EMBEDDING_REGISTRY` that carries
            filename-pattern metadata (``zarr_filename_pattern`` etc.).

    Returns:
        Sorted list of :class:`~pathlib.Path` objects for matching tiles.

    Raises:
        ValueError: If *embedding_name* is unknown or has no filename-pattern
            metadata.
    """
    if embedding_name not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding '{embedding_name}'. Choose from: {list(EMBEDDING_REGISTRY)}")

    meta = EMBEDDING_REGISTRY[embedding_name]
    pattern = meta.get("zarr_filename_pattern")
    tile_size = meta.get("zarr_tile_size")
    is_center = meta.get("zarr_filename_is_center")

    if pattern is None or tile_size is None or is_center is None:
        supported = sorted(
            k for k, m in EMBEDDING_REGISTRY.items() if m.get("zarr_filename_pattern")
        )
        raise ValueError(
            f"Embedding '{embedding_name}' has no filename-pattern metadata. "
            f"Embeddings supporting find_tiles_for_roi: {supported}."
        )

    directory = Path(directory)
    regex = re.compile(pattern)
    half = tile_size / 2

    qminx, qminy, qmaxx, qmaxy = roi
    matching: list[Path] = []

    for path in sorted(directory.glob("*.zarr")) + sorted(directory.glob("*.tif")):
        m = regex.match(path.name)
        if m is None:
            continue
        lon = float(m.group("lon"))
        lat = float(m.group("lat"))
        if is_center:
            tminx, tminy, tmaxx, tmaxy = lon - half, lat - half, lon + half, lat + half
        else:
            tminx, tminy, tmaxx, tmaxy = lon, lat, lon + tile_size, lat + tile_size
        if tmaxx > qminx and tminx < qmaxx and tmaxy > qminy and tminy < qmaxy:
            matching.append(path)

    return sorted(matching)
