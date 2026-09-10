"""Render embedding rasters as RGB through a *fitted, reusable* colour model.

The old `plot_embeddings.py` showed raw channels 0, 1 and 2 and stretched them
against the current image. That answers "what do three arbitrary dimensions look
like here", and nothing else: the channels are arbitrary, and the stretch changes
per image so two cities are never comparable.

Here a colour model is fitted **once** on a stratified sample of pixels and then
applied to any ROI, so a colour means the same thing in Nairobi as in London and
in 2017 as in 2025. Three methods, one contract:

* ``pca``  — a 3-component linear projection. Exact, invertible, reportable
  (the loadings and explained variance are a result, not just a picture), and
  the only one whose colour distance is metric.
* ``umap`` / ``tsne`` — fitted with ``n_components=3`` on the pixel sample, then
  **distilled into a small MLP** ``f: R^C -> R^3``. Distillation is what makes
  them usable on a raster at all: t-SNE has no out-of-sample transform, and
  ``umap.transform`` would need ~9M nearest-neighbour queries for one city at
  10 m (hours) and is not Lipschitz, so adjacent near-identical pixels can land
  far apart and produce speckle that *reads as spatial texture* but is not. The
  MLP is smooth by construction, so that failure mode cannot occur, and applying
  it is a matmul chain — seconds.

Read UMAP/t-SNE colours qualitatively: those axes are **non-metric**, so "these
two areas differ" is supported and "twice as different" is not. The fitted head's
held-out R^2 is stored in the model and printed on fit; if it is low, the raster
is not showing the manifold and you should not pretend otherwise.

Colour models are **per embedding family** — different channel counts, different
bases — so colours compare across cities and years within one embedding, never
between two.

Fit a model, then apply it:

    python src/embedding_rgb.py fit \\
        --so2sat-dir $DATA_DIR/input/So2Sat-LCZ42/v4 \\
        --embedding-name tesserav2 --output-name GeoTessera_v2 --year 2017 \\
        --method pca --model-dir $DATA_DIR/output/lcz-classification/embedding_viz/colour_models

    python src/embedding_rgb.py apply \\
        --model .../colour_models/colour_tesserav2_pca.npz \\
        --embedding-dir /tessera/v2/large_student/2017 --year 2017 \\
        --city Nairobi --output nairobi_v2_pca.tif
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from datasets.registry import EMBEDDING_REGISTRY, get_nodata_predicate  # noqa: E402
from datasets.so2sat import build_so2sat_items  # noqa: E402
from datasets.tiles import build_tile_index, meters_to_out_res, open_and_clip, setup_output  # noqa: E402
from utils.runtime import resolve_dequantize  # noqa: E402

METHODS = ("pca", "umap", "tsne")
# Percentiles defining the persisted stretch. Fixed on the fit sample, never
# recomputed per image -- recomputing is exactly what destroys comparability.
STRETCH_PCT = (2.0, 98.0)
# Pixels transformed at a time. A 4185x5676 city ROI is 23.7M pixels; casting
# that to float64 in one go is ~12 GB per intermediate.
CHUNK_PX = 1_000_000
# Long-edge floor for `--upscale auto`. A 33x33 patch is a thumbnail at its own
# size; 512 px puts it on screen without inventing any values.
BARE_MIN_SIDE = 512

# Frame drawn around a bare image by ``--border``, as (R, G, B) and a width in
# FILE pixels. Matches R/embedding_raster.R's BARE_BORDER_COL ("grey15") so a
# patch coloured here and one drawn there carry the same frame.
BARE_BORDER_RGB = (38, 38, 38)
BARE_BORDER_PX = 6


# ── Colour models ─────────────────────────────────────────────────────────────

class ColourModel:
    """Embedding vectors -> 3 channels, with a stretch fixed at fit time.

    Subclasses implement ``_fit_project`` / ``_project``. Everything else --
    centring, the stretch, saving, loading -- is shared so a raster written
    through any method is directly comparable to the same method elsewhere.
    """

    method = "base"

    def __init__(self, meta: dict | None = None):
        self.meta = meta or {}
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    # -- fitting ------------------------------------------------------------

    def _center(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit(self, X: np.ndarray, *, scale: bool = False, seed: int = 42) -> "ColourModel":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        # Centred-but-unscaled by default: z-scoring turns PCA on the covariance
        # into PCA on the correlation matrix, which up-weights low-variance
        # channels. For near-unit-norm embeddings that is a real change.
        self.scale_ = X.std(axis=0) + 1e-8 if scale else np.ones(X.shape[1])
        self.meta["scaled"] = bool(scale)
        Z = self._fit_project(self._center(X), seed=seed)
        self._fit_stretch(Z)
        return self

    def _fit_project(self, Xc: np.ndarray, *, seed: int) -> np.ndarray:
        raise NotImplementedError

    def _project(self, Xc: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _fit_stretch(self, Z: np.ndarray) -> None:
        """Per-axis percentile stretch. Overridden by PCA."""
        self.lo_ = np.percentile(Z, STRETCH_PCT[0], axis=0)
        self.hi_ = np.percentile(Z, STRETCH_PCT[1], axis=0)

    # -- applying -----------------------------------------------------------

    def transform(self, X: np.ndarray, chunk: int = CHUNK_PX) -> np.ndarray:
        """(N, C) -> (N, 3) float in [0, 1], clipped to the persisted stretch.

        Chunked: a city ROI at 10 m is tens of millions of pixels, and casting
        all of them to float64 at once would be tens of GB. The result is
        identical either way -- every operation here is per-row.
        """
        X = np.asarray(X)
        n = len(X)
        out = np.empty((n, 3), dtype=np.float32)
        span = np.maximum(self.hi_ - self.lo_, 1e-12)
        for i in range(0, n, chunk):
            block = np.asarray(X[i:i + chunk], dtype=np.float64)
            Z = self._project(self._center(block))
            out[i:i + chunk] = np.clip((Z - self.lo_) / span, 0.0, 1.0)
        return out

    def to_uint8(self, X: np.ndarray, chunk: int = CHUNK_PX) -> np.ndarray:
        return np.round(self.transform(X, chunk=chunk) * 255).astype(np.uint8)

    # -- persistence --------------------------------------------------------

    def _arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": np.array(self.method),
            "mean": self.mean_, "scale": self.scale_,
            "lo": self.lo_, "hi": self.hi_,
            **self._arrays(),
        }
        for k, v in self.meta.items():
            payload[f"meta_{k}"] = np.array(v)
        np.savez(path, **payload)
        logger.info(f"Colour model → {path}")

    @staticmethod
    def load(path: Path) -> "ColourModel":
        d = np.load(path, allow_pickle=False)
        method = str(d["method"])
        cls = {"pca": PCAColour, "umap": MLPColour, "tsne": MLPColour}[method]
        m = cls()
        m.method = method
        m.mean_, m.scale_ = d["mean"], d["scale"]
        m.lo_, m.hi_ = d["lo"], d["hi"]
        m._load_arrays(d)
        m.meta = {k[5:]: d[k].item() for k in d.files if k.startswith("meta_")}
        return m

    def _load_arrays(self, d) -> None:
        raise NotImplementedError


class PCAColour(ColourModel):
    """Three principal components. Linear, so it applies anywhere exactly."""

    method = "pca"

    def _fit_project(self, Xc: np.ndarray, *, seed: int) -> np.ndarray:
        pca = PCA(n_components=3, random_state=seed).fit(Xc)
        comp = pca.components_
        # Fix the sign convention, or a refit silently inverts the map's colours
        # and nobody can tell why: force each component's largest-magnitude
        # loading positive.
        flip = np.sign(comp[np.arange(3), np.argmax(np.abs(comp), axis=1)])
        flip[flip == 0] = 1.0
        self.components_ = comp * flip[:, None]
        self.evr_ = pca.explained_variance_ratio_
        self.meta["explained_variance_ratio"] = ",".join(f"{v:.4f}" for v in self.evr_)
        logger.info(f"PCA colour model: explained variance {self.evr_.round(4).tolist()}")
        return self._project(Xc)

    def _project(self, Xc: np.ndarray) -> np.ndarray:
        return Xc @ self.components_.T

    def _fit_stretch(self, Z: np.ndarray) -> None:
        """One common divisor across the three PCs.

        Stretching each PC independently to full range inflates PC3 -- typically
        a small fraction of PC1's variance -- into a vivid blue channel, and the
        image then looks far more structured than the data is. Centring each PC
        on its own median while dividing all three by PC1's range keeps the
        relative variance visible, which is the honest picture.
        """
        med = np.median(Z, axis=0)
        span = np.percentile(Z[:, 0], STRETCH_PCT[1]) - np.percentile(Z[:, 0], STRETCH_PCT[0])
        half = max(span, 1e-12) / 2.0
        self.lo_, self.hi_ = med - half, med + half

    def _arrays(self):
        return {"components": self.components_, "evr": self.evr_}

    def _load_arrays(self, d):
        self.components_, self.evr_ = d["components"], d["evr"]


class MLPColour(ColourModel):
    """A manifold layout (UMAP or t-SNE), distilled into an MLP.

    The manifold is fitted on the sample; the MLP then learns to reproduce its
    3-D coordinates from the raw embedding. Applying the MLP is what makes this
    usable on a full raster, and it is *smooth*, which the underlying transform
    is not.

    Weights are stored as plain arrays and the forward pass is reimplemented
    here, so applying a model never depends on the sklearn version that fitted
    it.
    """

    def __init__(self, meta: dict | None = None, method: str = "umap", **fit_kw):
        super().__init__(meta)
        self.method = method
        self.fit_kw = fit_kw

    def _fit_manifold(self, Xr: np.ndarray, seed: int) -> np.ndarray:
        if self.method == "umap":
            import umap

            return umap.UMAP(
                n_components=3, random_state=seed, verbose=True,
                n_neighbors=self.fit_kw.get("n_neighbors", 30),
                min_dist=self.fit_kw.get("min_dist", 0.1),
            ).fit_transform(Xr)
        from sklearn.manifold import TSNE

        # barnes_hut is the only method that scales, and it supports up to 3
        # components -- exactly the ceiling we need.
        return TSNE(
            n_components=3, method="barnes_hut", random_state=seed, verbose=1,
            perplexity=self.fit_kw.get("perplexity", 40.0),
        ).fit_transform(Xr)

    def _fit_project(self, Xc: np.ndarray, *, seed: int) -> np.ndarray:
        from sklearn.neural_network import MLPRegressor

        n_manifold = int(self.fit_kw.get("manifold_max", 100_000))
        rng = np.random.default_rng(seed)
        idx = (rng.choice(len(Xc), n_manifold, replace=False)
               if len(Xc) > n_manifold else np.arange(len(Xc)))

        # Reduce first, exactly as the patch-level scatter does, so the raster
        # and the scatter are built the same way.
        pre_dim = min(int(self.fit_kw.get("pre_pca", 50)), Xc.shape[1])
        pre = PCA(n_components=pre_dim, random_state=seed).fit(Xc[idx])
        # Store the basis, not the estimator: the projection is replayed by hand
        # below, so a loaded model never depends on the sklearn version that
        # fitted it (and cannot trip over attributes sklearn sets internally).
        self.pre_mean_, self.pre_components_ = pre.mean_, pre.components_
        Xr = self._pre(Xc[idx])

        logger.info(f"Fitting {self.method.upper()}-3 on {len(Xr)} pixels …")
        Y = np.asarray(self._fit_manifold(Xr, seed), dtype=np.float64)

        # Normalise the target so the MLP trains on a sane scale; folded back in
        # at apply time.
        self.y_mean_, self.y_std_ = Y.mean(axis=0), Y.std(axis=0) + 1e-8
        Yn = (Y - self.y_mean_) / self.y_std_

        n_hold = max(1, int(0.1 * len(Xr)))
        tr, ho = slice(n_hold, None), slice(0, n_hold)
        logger.info(f"Distilling into an MLP ({len(Xr) - n_hold} train / {n_hold} held out) …")
        mlp = MLPRegressor(
            hidden_layer_sizes=tuple(self.fit_kw.get("hidden", (256, 128))),
            activation="relu", random_state=seed, max_iter=400,
            early_stopping=True, n_iter_no_change=15,
        ).fit(Xr[tr], Yn[tr])

        self.W_ = [np.asarray(w, dtype=np.float64) for w in mlp.coefs_]
        self.b_ = [np.asarray(b, dtype=np.float64) for b in mlp.intercepts_]

        # Held-out fidelity: how faithfully the raster reproduces the layout.
        # Reported rather than assumed -- a low value means the picture is not
        # the manifold.
        pred = self._head(Xr[ho])
        ss_res = ((Yn[ho] - pred) ** 2).sum()
        ss_tot = ((Yn[ho] - Yn[ho].mean(axis=0)) ** 2).sum()
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
        med = float(np.median(np.linalg.norm(Yn[ho] - pred, axis=1)))
        self.meta["holdout_r2"] = round(r2, 4)
        self.meta["holdout_median_err"] = round(med, 4)
        logger.info(f"Distillation fidelity: held-out R^2={r2:.4f}, median error={med:.4f} "
                    f"(in units of the manifold's own SD)")
        if r2 < 0.7:
            logger.warning(
                f"Held-out R^2 {r2:.3f} is low: the MLP is not reproducing the "
                f"{self.method.upper()} layout well, so the raster's colours are "
                "only loosely that layout. Try more pixels or a wider head.")
        return self._project(Xc)

    def _pre(self, Xc: np.ndarray) -> np.ndarray:
        return (Xc - self.pre_mean_) @ self.pre_components_.T

    def _head(self, Xr: np.ndarray) -> np.ndarray:
        h = Xr
        for i, (w, b) in enumerate(zip(self.W_, self.b_)):
            h = h @ w + b
            if i < len(self.W_) - 1:
                h = np.maximum(h, 0.0)      # relu; output layer is identity
        return h

    def _project(self, Xc: np.ndarray) -> np.ndarray:
        return self._head(self._pre(Xc)) * self.y_std_ + self.y_mean_

    def _arrays(self):
        out = {
            "pre_mean": self.pre_mean_, "pre_components": self.pre_components_,
            "y_mean": self.y_mean_, "y_std": self.y_std_,
            "n_layers": np.array(len(self.W_)),
        }
        for i, (w, b) in enumerate(zip(self.W_, self.b_)):
            out[f"W{i}"], out[f"b{i}"] = w, b
        return out

    def _load_arrays(self, d):
        self.pre_mean_, self.pre_components_ = d["pre_mean"], d["pre_components"]
        self.y_mean_, self.y_std_ = d["y_mean"], d["y_std"]
        n = int(d["n_layers"])
        self.W_ = [d[f"W{i}"] for i in range(n)]
        self.b_ = [d[f"b{i}"] for i in range(n)]


def build_model(method: str, **fit_kw) -> ColourModel:
    if method == "pca":
        return PCAColour()
    return MLPColour(method=method, **fit_kw)


# ── Pixel sampling ────────────────────────────────────────────────────────────

def sample_pixels(
    items: list,
    dequantize_fn,
    nodata_predicate,
    *,
    n_patches: int,
    px_per_patch: int,
    scheme: str = "balanced",
    seed: int = 42,
) -> np.ndarray:
    """Draw a pixel sample from extracted patches. Returns ``(N, C)``.

    Two-stage on purpose: choose patches, then take only a handful of pixels
    from each. Pixels within one patch are strongly autocorrelated, so 1100 of
    them are nowhere near 1100 independent samples -- spreading the budget over
    more patches buys far more information for the same cost.

    ``scheme="balanced"`` allocates over the city x LCZ grid, so the components
    spread the classes rather than being dominated by whichever class covers the
    most ground. ``scheme="proportional"`` samples patches uniformly, which is
    what you want if the colours should be variance-optimal for what is actually
    on screen. They are different objectives; the choice is recorded in the model.
    """
    rng = np.random.default_rng(seed)
    if scheme == "balanced":
        keyed: dict[tuple, list] = {}
        for it in items:
            keyed.setdefault((it.city, it.label), []).append(it)
        cells = list(keyed.values())
        per = max(1, int(np.ceil(n_patches / max(len(cells), 1))))
        chosen, shortfall = [], 0
        for c in cells:
            take = min(per, len(c))
            chosen.extend(c[int(i)] for i in rng.choice(len(c), take, replace=False))
            shortfall += per - take
        # Redistribute what thin cells could not supply, rather than silently
        # under-sampling. The membership set is built ONCE: rebuilding it per
        # candidate makes this a 349k x 20k scan and the fit never finishes.
        if shortfall > 0:
            taken = {id(it) for it in chosen}
            pool = [it for it in items if id(it) not in taken]
            if pool:
                extra = rng.choice(len(pool), min(shortfall, len(pool)), replace=False)
                chosen.extend(pool[int(i)] for i in extra)
        logger.info(f"Balanced sample: {len(cells)} city x class cells, "
                    f"{per}/cell -> {len(chosen)} patches")
    else:
        idx = rng.choice(len(items), min(n_patches, len(items)), replace=False)
        chosen = [items[int(i)] for i in idx]
        logger.info(f"Proportional sample: {len(chosen)} patches")

    out = []
    for it in chosen:
        raw = np.load(it.path)
        valid = None
        if nodata_predicate is not None:
            valid = ~nodata_predicate(raw)
        arr = np.nan_to_num(raw.astype(np.float32), nan=0.0)
        if dequantize_fn is not None:
            arr = dequantize_fn(arr)
        c, h, w = arr.shape
        flat = arr.reshape(c, -1).T                     # (H*W, C)
        if valid is not None:
            flat = flat[valid.reshape(-1)]
        if not len(flat):
            continue
        take = min(px_per_patch, len(flat))
        out.append(flat[rng.choice(len(flat), take, replace=False)])
    X = np.concatenate(out, axis=0)
    logger.info(f"Pixel sample: {X.shape[0]:,} pixels x {X.shape[1]} channels")
    return X


def sample_tile_pixels(
    embedding_dir: Path, embedding_name: str, year: str,
    dequantize_fn, nodata_predicate, *, n_tiles: int, n_px: int, seed: int = 42,
) -> np.ndarray:
    """Pixels from random source tiles, to cover terrain So2Sat never shows.

    The So2Sat frame is entirely urban and LCZ-labelled. Without this, forest,
    desert, ice and open ocean sit outside the fitted space and every rural ROI
    clips hard against the stretch bounds.
    """
    paths, _ = build_tile_index(embedding_dir, embedding_name, year)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), min(n_tiles, len(paths)), replace=False)
    per = max(1, n_px // max(len(pick), 1))
    from datasets.tiles import open_tile

    out = []
    for i in pick:
        try:
            da = open_tile(Path(paths[int(i)]))
        except Exception as e:
            logger.warning(f"tile {paths[int(i)]}: {e} — skipping")
            continue
        raw = da.values
        valid = ~nodata_predicate(raw) if nodata_predicate is not None else None
        arr = np.nan_to_num(raw.astype(np.float32), nan=0.0)
        if dequantize_fn is not None:
            arr = dequantize_fn(arr)
        flat = arr.reshape(arr.shape[0], -1).T
        if valid is not None:
            flat = flat[valid.reshape(-1)]
        if not len(flat):
            continue
        out.append(flat[rng.choice(len(flat), min(per, len(flat)), replace=False)])
    if not out:
        logger.warning("No tile pixels sampled; the model will be urban-only.")
        return np.empty((0, 0), dtype=np.float32)
    X = np.concatenate(out, axis=0)
    logger.info(f"Tile supplement: {X.shape[0]:,} pixels from {len(out)} tiles")
    return X


def subspace_angles_check(X: np.ndarray, seed: int = 42) -> list[float]:
    """Fit PCA-3 on two disjoint halves and report the principal angles.

    Turns "I picked N pixels" into a number: near-zero angles mean the sample is
    large enough that the basis is stable, and adding more would not move it.
    """
    from scipy.linalg import subspace_angles

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    Xc = X - X.mean(axis=0)
    pa = PCA(n_components=3, random_state=seed).fit(Xc[a]).components_
    pb = PCA(n_components=3, random_state=seed).fit(Xc[b]).components_
    ang = np.degrees(subspace_angles(pa.T, pb.T)).tolist()
    logger.info(f"Split-half principal angles (deg): {[round(v, 3) for v in ang]}")
    return ang


# ── Applying to an ROI ────────────────────────────────────────────────────────

def _localise_stretch(model, tiles, bbox_4326, dequantize_fn, nodata_predicate,
                      max_px: int = 400_000, seed: int = 0) -> None:
    """Replace the model's persisted stretch with one fitted to this ROI."""
    rng = np.random.default_rng(seed)
    vals = []
    for t in tiles:
        got = open_and_clip(t, bbox_4326)
        if got is None:
            continue
        arr = got[0]
        valid = (~nodata_predicate(arr) if nodata_predicate is not None
                 else np.ones(arr.shape[1:], dtype=bool))
        d = dequantize_fn(arr) if dequantize_fn is not None else arr
        flat = np.nan_to_num(d.reshape(d.shape[0], -1).T, nan=0.0)[valid.reshape(-1)]
        if not len(flat):
            continue
        take = min(max_px // max(len(tiles), 1), len(flat))
        vals.append(flat[rng.choice(len(flat), take, replace=False)])
    if not vals:
        logger.warning("--local-stretch: no pixels sampled, keeping the model stretch")
        return
    Z = model._project(model._center(np.concatenate(vals).astype(np.float64)))
    model.lo_ = np.percentile(Z, STRETCH_PCT[0], axis=0)
    model.hi_ = np.percentile(Z, STRETCH_PCT[1], axis=0)
    model.meta["stretch"] = "local (this ROI only; NOT comparable to other maps)"
    logger.info(f"--local-stretch: re-fitted on {len(np.concatenate(vals)):,} ROI pixels")


def _raw_rgb(sel: np.ndarray) -> np.ndarray:
    """Three raw channels ``(3, H, W)`` -> uint8 RGB, on a per-image stretch.

    The stretch is refitted on every image, which is exactly what a fitted
    colour model exists to avoid: two pictures of the same place are not
    comparable this way, and neither are two places. It is kept because "show me
    channels 0, 1 and 2" is a real question about an embedding, and because it
    is the only mode that needs no fitted model at all.
    """
    lo = np.nanpercentile(sel.reshape(3, -1), STRETCH_PCT[0], axis=1)
    hi = np.nanpercentile(sel.reshape(3, -1), STRETCH_PCT[1], axis=1)
    vals = np.clip((sel - lo[:, None, None]) /
                   np.maximum(hi - lo, 1e-12)[:, None, None], 0, 1)
    return np.round(vals * 255).astype(np.uint8)


def apply_to_roi(
    model: ColourModel,
    embedding_dir: Path,
    embedding_name: str,
    year: str,
    bbox_4326: tuple[float, float, float, float],
    output_path: Path,
    *,
    dequantize_fn=None,
    nodata_predicate=None,
    out_crs: str | None = None,
    raw_bands: tuple[int, int, int] | None = None,
    local_stretch: bool = False,
    caption: bool = False,
) -> Path:
    """Colour every pixel of an ROI and write an 8-bit 3-band GeoTIFF + PNG."""
    import rasterio.warp
    from rasterio.warp import Resampling

    paths, tree = build_tile_index(embedding_dir, embedding_name, year)
    from shapely.geometry import box

    hits = tree.query(box(*bbox_4326))
    tiles = [Path(paths[int(i)]) for i in np.atleast_1d(hits)]
    if not tiles:
        raise SystemExit(f"No {embedding_name} tiles intersect {bbox_4326}")
    logger.info(f"{len(tiles)} tiles intersect the ROI")

    res_m = EMBEDDING_REGISTRY[embedding_name].get("resolution", 10)
    first = None
    for t in tiles:                      # first openable tile fixes the CRS
        got = open_and_clip(t, bbox_4326)
        if got is not None:
            first = got
            break
    if first is None:
        raise SystemExit("No tile yielded data for this ROI")
    resolved_crs = out_crs or first[1]
    cx = (bbox_4326[0] + bbox_4326[2]) / 2
    cy = (bbox_4326[1] + bbox_4326[3]) / 2
    out_res = meters_to_out_res(res_m, first[1], resolved_crs, cx, cy)
    transform, H, W = setup_output(bbox_4326, resolved_crs, out_res)
    logger.info(f"Output raster {H}x{W} @ {out_res:.2f} in {resolved_crs}")

    if local_stretch and raw_bands is None:
        # Re-derive the stretch from THIS ROI. A global model deliberately keeps
        # one stretch for every city, which is what makes two maps comparable --
        # but it also means a single city occupies only a slice of the range and
        # looks washed out. This trades that comparability for contrast, so the
        # resulting image must not be compared with any other.
        _localise_stretch(model, tiles, bbox_4326, dequantize_fn, nodata_predicate)

    rgb = np.zeros((3, H, W), dtype=np.uint8)
    seen = np.zeros((H, W), dtype=bool)

    for t in tiles:
        got = open_and_clip(t, bbox_4326)
        if got is None:
            continue
        arr, crs, tr = got
        c, th, tw = arr.shape

        if raw_bands is not None:
            sel = arr[list(raw_bands)]
            tile_rgb = _raw_rgb(sel)
            tile_valid = np.isfinite(sel).all(axis=0)
        else:
            # The nodata test runs on the array as read, before dequantization:
            # the registry's sentinel is defined in stored units. Seamless also
            # changes channel count on dequantize (13 -> 72), so the mask has to
            # be taken first either way.
            tile_valid = (~nodata_predicate(arr) if nodata_predicate is not None
                          else np.ones((th, tw), dtype=bool))
            vals = dequantize_fn(arr) if dequantize_fn is not None else arr
            flat = np.nan_to_num(vals.reshape(vals.shape[0], -1).T, nan=0.0)
            tile_rgb = model.to_uint8(flat).T.reshape(3, th, tw)

        dst = np.zeros((3, H, W), dtype=np.uint8)
        dst_valid = np.zeros((1, H, W), dtype=np.uint8)
        rasterio.warp.reproject(
            source=tile_rgb, destination=dst,
            src_transform=tr, src_crs=crs,
            dst_transform=transform, dst_crs=resolved_crs,
            resampling=Resampling.nearest,
        )
        rasterio.warp.reproject(
            source=tile_valid[None].astype(np.uint8), destination=dst_valid,
            src_transform=tr, src_crs=crs,
            dst_transform=transform, dst_crs=resolved_crs,
            resampling=Resampling.nearest,
        )
        m = dst_valid[0] > 0
        np.copyto(rgb, dst, where=np.broadcast_to(m, rgb.shape))
        seen |= m

    logger.info(f"Coloured {seen.mean():.1%} of the output raster")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Nodata / uncovered pixels get an explicit alpha rather than a colour:
    # colouring sentinels produces vivid false edges along coastlines.
    alpha = (seen * 255).astype(np.uint8)
    with rasterio.open(
        str(output_path), "w", driver="GTiff", height=H, width=W, count=4,
        dtype="uint8", crs=resolved_crs, transform=transform, photometric="RGB",
        compress="deflate",
    ) as dst:
        for i in range(3):
            dst.write(rgb[i], i + 1)
        dst.write(alpha, 4)
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha,
        ]
        dst.update_tags(**{f"colour_{k}": str(v) for k, v in model.meta.items()},
                        colour_method=model.method)
    logger.info(f"Saved GeoTIFF → {output_path}")

    png = output_path.with_suffix(".png")
    if caption:
        _save_captioned_png(rgb, alpha, png, model)
    else:
        save_bare_png(rgb, alpha, png)
    return output_path


def colour_array(
    path: Path,
    model: ColourModel | None,
    *,
    embedding_name: str,
    dequantize_fn=None,
    nodata_predicate=None,
    raw_bands: tuple[int, int, int] | None = None,
    stored_channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Colour one extracted patch or grid array. Returns ``(rgb (3,H,W), alpha (H,W))``.

    These are the ``.npy`` the pipelines actually train on, written by
    ``extract_so2sat_embeddings.py`` / ``extract_grid_embeddings.py``. The
    load / mask / dequantize order below mirrors :func:`sample_pixels` exactly,
    and that is not incidental tidiness -- it is how the colour model was
    fitted, so any departure gives colours that quietly disagree with every
    other image from the same model. Two parts of it are easy to get wrong:

    * the nodata test runs on the array **as read**, before dequantization,
      because the registry's sentinel is defined in stored units;
    * the extracted files are **not** uniformly dequantized. A Tessera v2 patch
      is already real-valued, while an AlphaEarth coop patch holds the stored
      integers as float32 (measured range -66..72) and must go through
      ``resolve_dequantize`` here just as it did during the fit.
    """
    raw = np.load(path)
    if raw.ndim != 3:
        raise SystemExit(f"{path.name}: expected a 3-D (C, H, W) array, got {raw.shape}")
    # Every extractor writes channels-first, so that is the assumption; a
    # channels-last file is accepted when its last axis is the only one that can
    # be the channels. Guessing beyond that would silently transpose an image.
    if stored_channels is not None and raw.shape[0] != stored_channels:
        if raw.shape[-1] == stored_channels:
            raw = np.moveaxis(raw, -1, 0)
        else:
            raise SystemExit(
                f"{path.name}: shape {raw.shape} has no axis of "
                f"{stored_channels} channels, which is what {embedding_name} stores")

    valid = ~nodata_predicate(raw) if nodata_predicate is not None else None
    arr = np.nan_to_num(raw.astype(np.float32), nan=0.0)
    if dequantize_fn is not None:
        arr = dequantize_fn(arr)
    c, h, w = arr.shape

    if raw_bands is not None:
        if max(raw_bands) >= c:
            raise SystemExit(f"--bands {raw_bands} out of range for {c} channels")
        sel = arr[list(raw_bands)]
        rgb = _raw_rgb(sel)
        finite = np.isfinite(sel).all(axis=0)
    else:
        if c != len(model.mean_):
            raise SystemExit(
                f"{path.name} has {c} channels, the colour model has "
                f"{len(model.mean_)} -- model and embedding do not match")
        rgb = model.to_uint8(arr.reshape(c, -1).T).T.reshape(3, h, w)
        finite = np.ones((h, w), dtype=bool)

    # Nodata gets an explicit alpha rather than a colour, as apply_to_roi does:
    # colouring sentinels produces vivid false edges.
    keep = finite if valid is None else (finite & valid)
    logger.info(f"{path.name}: {c} channels, {h}x{w} px, {keep.mean():.1%} valid")
    return rgb, (keep * 255).astype(np.uint8)


def _upscale_factor(shape: tuple[int, int], upscale, min_side: int) -> int:
    """Integer pixel-repeat factor. ``"auto"`` reaches ``min_side`` on the long edge."""
    if upscale in (None, 1, "1"):
        return 1
    if upscale != "auto":
        k = int(upscale)
        if k < 1:
            raise SystemExit("--upscale must be 'auto' or a positive integer")
        return k
    return max(1, int(np.ceil(min_side / max(max(shape), 1))))


def _draw_border(img: np.ndarray, width: int, colour=BARE_BORDER_RGB) -> np.ndarray:
    """Paint an opaque frame into the outermost ``width`` pixels, in place.

    Drawn INTO the image rather than around it, so the file keeps the size the
    upscale gave it and one file pixel still maps to a known array pixel. The
    cost is the outer ring of data, which on an upscaled image is a fraction of
    one array pixel; at ``--upscale 1`` it is a real pixel per side, which is
    why the border is not on by default.
    """
    w = min(width, min(img.shape[0], img.shape[1]) // 2)
    if w < 1:
        return img
    px = np.array([*colour, 255], dtype=img.dtype)
    img[:w, :], img[-w:, :], img[:, :w], img[:, -w:] = px, px, px, px
    return img


def save_bare_png(rgb: np.ndarray, alpha: np.ndarray, path: Path,
                  upscale="auto", min_side: int = BARE_MIN_SIDE,
                  border: int = 0) -> None:
    """Write the array and nothing else: one file pixel per array pixel.

    ``imshow`` + ``savefig`` would resample onto the figure's dpi grid, pad by
    the axes margins and round the output size -- a picture of a plot of the
    data. ``imsave`` writes the data. A 33x33 patch is unreadable at its own
    size, so ``upscale="auto"`` repeats pixels up to ``min_side``; nearest
    neighbour only, because smoothing an embedding image invents values that
    are not in it.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.image import imsave

    img = np.dstack([np.moveaxis(rgb, 0, -1), alpha[..., None]])
    k = _upscale_factor(img.shape[:2], upscale, min_side)
    if k > 1:
        img = img.repeat(k, axis=0).repeat(k, axis=1)
    if border:
        img = _draw_border(img, border)
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(str(path), img)
    logger.info(f"Saved PNG -> {path} ({img.shape[1]}x{img.shape[0]} px"
                + (f", {k}x nearest-neighbour upscale)" if k > 1 else ")"))


def _save_captioned_png(rgb: np.ndarray, alpha: np.ndarray, path: Path,
                        model: ColourModel) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.dstack([np.moveaxis(rgb, 0, -1), alpha[..., None]])
    fig, ax = plt.subplots(figsize=(10, 10 * rgb.shape[1] / max(rgb.shape[2], 1)))
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    # A raw-band image is not the model's colour at all: both `apply` and
    # `image` hand this function a placeholder PCAColour for that mode, so its
    # note is the caption. Titling it "PCA" would name a basis it never used.
    raw = str(model.meta.get("note", "")).startswith("raw bands")
    if raw:
        caption = str(model.meta["note"])
    else:
        caption = f"{model.method.upper()} embedding colour"
        note = model.meta.get("explained_variance_ratio")
        if note:
            caption += f" — explained variance {note}"
        elif "holdout_r2" in model.meta:
            caption += (f" — distilled, held-out R²={model.meta['holdout_r2']}; "
                        "axes are non-metric, read colours qualitatively")
    # Say so on the image itself when the stretch was re-fitted here: the whole
    # point of the persisted stretch is cross-city comparability, and a local
    # one silently looks better while no longer being comparable.
    if str(model.meta.get("stretch", "")).startswith("local"):
        caption += "  [LOCAL stretch — not comparable to other maps]"
    ax.set_title(caption, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved captioned PNG → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cmd_fit(args) -> None:
    dequantize_fn, _ = resolve_dequantize(args.embedding_name, force=args.dequantize)
    predicate = get_nodata_predicate(args.embedding_name) if not args.no_mask else None

    items, _ = build_so2sat_items(
        args.so2sat_dir, args.output_name, args.year, global_split=True,
        global_gpkg=args.global_gpkg,
    )
    train = [it for it in items if it.split == "train"]
    logger.info(f"{len(train)} train patches available as the sampling frame")

    X = sample_pixels(
        train, dequantize_fn, predicate,
        n_patches=args.n_patches, px_per_patch=args.px_per_patch,
        scheme=args.sample_scheme, seed=args.seed,
    )
    if args.tile_dir is not None and args.tile_frac > 0:
        extra = sample_tile_pixels(
            args.tile_dir, args.embedding_name, args.year, dequantize_fn, predicate,
            n_tiles=args.tile_count,
            n_px=int(len(X) * args.tile_frac), seed=args.seed,
        )
        if extra.size:
            X = np.concatenate([X, extra], axis=0)
            logger.info(f"Sample with tile supplement: {X.shape}")

    if args.method == "pca":
        subspace_angles_check(X, seed=args.seed)

    meta = {
        "embedding_name": args.embedding_name, "output_name": args.output_name,
        "year": str(args.year), "seed": args.seed, "scheme": args.sample_scheme,
        "n_pixels": int(len(X)), "in_channels": int(X.shape[1]),
        "masked": predicate is not None,
        "note": "colours compare across cities/years within this embedding only",
    }
    model = build_model(
        args.method, manifold_max=args.manifold_max, pre_pca=args.pre_pca,
        n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
        perplexity=args.tsne_perplexity, hidden=tuple(args.hidden),
    )
    model.meta.update(meta)
    model.fit(X, scale=args.scale, seed=args.seed)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.model_dir / f"colour_{args.embedding_name}_{args.method}.npz")


def _cmd_apply(args) -> None:
    model = ColourModel.load(args.model) if args.model else None
    name = args.embedding_name or (model.meta.get("embedding_name") if model else None)
    if name is None:
        raise SystemExit("--embedding-name is required when no model is given")
    dequantize_fn, _ = resolve_dequantize(name, force=args.dequantize)
    predicate = get_nodata_predicate(name) if not args.no_mask else None

    bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox else _city_bbox(args.city)
    raw_bands = tuple(int(b) for b in args.bands.split(",")) if args.colour_mode == "raw" else None
    if raw_bands is not None:
        model = model or PCAColour()
        model.meta.setdefault("note", f"raw bands {raw_bands}, per-image stretch")

    apply_to_roi(
        model, args.embedding_dir, name, args.year, bbox, args.output,
        dequantize_fn=dequantize_fn, nodata_predicate=predicate,
        out_crs=args.out_crs, raw_bands=raw_bands, local_stretch=args.local_stretch,
        caption=args.caption,
    )


def _cmd_image(args) -> None:
    model = ColourModel.load(args.model) if args.model else None
    name = args.embedding_name or (model.meta.get("embedding_name") if model else None)
    if name is None:
        raise SystemExit("--embedding-name is required when no model is given")
    if name not in EMBEDDING_REGISTRY:
        raise SystemExit(f"unknown embedding {name!r}; one of {sorted(EMBEDDING_REGISTRY)}")

    dequantize_fn, ch_override = resolve_dequantize(name, force=args.dequantize)
    predicate = get_nodata_predicate(name) if not args.no_mask else None
    raw_bands = tuple(int(b) for b in args.bands.split(",")) if args.colour_mode == "raw" else None
    if raw_bands is None and model is None:
        raise SystemExit("--model is required for --colour-mode model")
    if raw_bands is not None and len(raw_bands) != 3:
        raise SystemExit("--bands takes exactly three channel indices")
    # The registry's in_channels is the count AFTER dequantization, so a
    # non-None override (seamless, 13 stored -> 72) means the file's own channel
    # count is not the registry's and cannot be used to find the channel axis.
    stored = None if ch_override is not None else EMBEDDING_REGISTRY[name]["in_channels"]

    rgb, alpha = colour_array(
        args.input, model, embedding_name=name, dequantize_fn=dequantize_fn,
        nodata_predicate=predicate, raw_bands=raw_bands, stored_channels=stored,
    )
    if args.caption:
        if raw_bands is not None:
            model = model or PCAColour()
            model.meta["note"] = f"raw bands {raw_bands}, per-image stretch"
        _save_captioned_png(rgb, alpha, args.output, model)
    else:
        save_bare_png(rgb, alpha, args.output, upscale=args.upscale,
                      border=args.border)


def _city_bbox(city: str) -> tuple[float, float, float, float]:
    csv = _src.parent / "data" / "so2sat_guppd_bounds.csv"
    df = pd.read_csv(csv)
    row = df[df["JRC_NAME_MAIN"].str.lower() == city.lower()]
    if row.empty:
        raise SystemExit(f"city {city!r} not in {csv}")
    r = row.iloc[0]
    return float(r["minx"]), float(r["miny"]), float(r["maxx"]), float(r["maxy"])


def _cmd_annotate(args) -> None:
    """Add rgb_<method>_{r,g,b} to a projection parquet from a colour model.

    The parquet's own pca_1..3 are a *patch-level* basis; the raster uses a
    *pixel-level* one. Colouring a scatter by the former would not match the
    map. Running each patch's pooled vector through the pixel model does match,
    for every method -- for PCA it is the exact identity PCA(mean(px)) ==
    mean(PCA(px)) on centred data, and for UMAP/t-SNE the distilled head simply
    evaluates on the pooled vector.

    Caveat worth keeping in mind when reading the result: a pooled vector is not
    a pixel, so the colour says where the patch's *mean* lands in pixel-colour
    space.
    """
    from utils.pooling_features import compose_features

    model = ColourModel.load(args.model)
    df = pd.read_parquet(args.parquet)

    if args.features:
        feats = np.load(args.features)
    else:
        # Rebuild from the block cache in the parquet's own row order: splits
        # concatenated train, val, test, which is what build_metadata replicates.
        parts = []
        for split in ("train", "val", "test"):
            try:
                parts.append(compose_features(args.cache_dir, args.cache_key,
                                              args.pooling, split))
            except FileNotFoundError:
                continue
        if not parts:
            raise SystemExit(
                f"No cached blocks for key {args.cache_key!r} under {args.cache_dir}. "
                "Pass --features, or run the projection/bake-off first.")
        feats = np.concatenate(parts, axis=0)

    if len(feats) != len(df):
        raise SystemExit(
            f"features {len(feats)} != parquet rows {len(df)} — the cache key and "
            "the parquet are from different runs.")
    if feats.shape[1] != len(model.mean_):
        raise SystemExit(
            f"feature width {feats.shape[1]} != colour model's {len(model.mean_)} "
            "channels — pooled features must be one vector per channel (gap).")

    rgb = model.to_uint8(feats)
    cols = [f"rgb_{model.method}_{ch}" for ch in "rgb"]
    for i, c in enumerate(cols):
        df[c] = rgb[:, i]
    df.to_parquet(args.parquet, index=False)
    logger.info(f"Annotated {args.parquet.name} with {', '.join(cols)} ({len(df)} rows)")

    # Carry the colours over to the R subsample, which is a uid-keyed subset of
    # this table rather than a row-aligned one.
    sample = args.parquet.with_name(args.parquet.stem + "_sample.parquet")
    if sample.exists() and "uid" in df.columns:
        sub = pd.read_parquet(sample).drop(columns=cols, errors="ignore")
        sub = sub.merge(df[["uid", *cols]], on="uid", how="left")
        sub.to_parquet(sample, index=False)
        logger.info(f"Propagated to {sample.name} ({len(sub)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="Fit a colour model on a stratified pixel sample.")
    f.add_argument("--so2sat-dir", required=True, type=Path)
    f.add_argument("--output-name", required=True, help="Extraction folder, e.g. GeoTessera_v2.")
    f.add_argument("--embedding-name", required=True, choices=sorted(EMBEDDING_REGISTRY))
    f.add_argument("--year", required=True)
    f.add_argument("--global-gpkg", type=Path, default=None)
    f.add_argument("--method", choices=METHODS, default="pca")
    f.add_argument("--model-dir", required=True, type=Path)
    f.add_argument("--n-patches", type=int, default=60_000)
    f.add_argument("--px-per-patch", type=int, default=8,
                   help="Pixels per chosen patch. Small on purpose: pixels within "
                        "one patch are highly autocorrelated.")
    f.add_argument("--sample-scheme", choices=["balanced", "proportional"], default="balanced")
    f.add_argument("--tile-dir", type=Path, default=None,
                   help="Source tile dir for the non-urban supplement. Without it the "
                        "model only ever saw cities.")
    f.add_argument("--tile-count", type=int, default=40)
    f.add_argument("--tile-frac", type=float, default=0.2)
    f.add_argument("--scale", action="store_true",
                   help="z-score channels before fitting (correlation PCA). Off by "
                        "default: it up-weights low-variance channels.")
    f.add_argument("--manifold-max", type=int, default=100_000)
    f.add_argument("--pre-pca", type=int, default=50)
    f.add_argument("--umap-n-neighbors", type=int, default=30)
    f.add_argument("--umap-min-dist", type=float, default=0.1)
    f.add_argument("--tsne-perplexity", type=float, default=40.0)
    f.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    f.add_argument("--dequantize", action="store_true")
    f.add_argument("--no-mask", action="store_true")
    f.add_argument("--seed", type=int, default=42)
    f.set_defaults(func=_cmd_fit)

    a = sub.add_parser("apply", help="Colour an ROI through a fitted model.")
    a.add_argument("--model", type=Path, default=None)
    a.add_argument("--embedding-dir", required=True, type=Path)
    a.add_argument("--embedding-name", default=None, choices=sorted(EMBEDDING_REGISTRY))
    a.add_argument("--year", required=True)
    a.add_argument("--city", default=None)
    a.add_argument("--bbox", default=None, help="west,south,east,north in EPSG:4326.")
    a.add_argument("--output", required=True, type=Path)
    a.add_argument("--out-crs", default=None)
    a.add_argument("--colour-mode", choices=["model", "raw"], default="model",
                   help="raw reproduces the retired plot_embeddings.py: three raw "
                        "channels stretched per image, not comparable between images.")
    a.add_argument("--bands", default="0,1,2")
    a.add_argument("--caption", action="store_true",
                   help="Title the sidecar PNG with the model's provenance. Off by "
                        "default, so the PNG is just the image -- note that a "
                        "--local-stretch image then says so nowhere but the GeoTIFF tags.")
    a.add_argument("--local-stretch", action="store_true")
    a.add_argument("--dequantize", action="store_true")
    a.add_argument("--no-mask", action="store_true")
    a.set_defaults(func=_cmd_apply)

    i = sub.add_parser(
        "image", help="Colour one extracted patch/grid .npy into a bare PNG.")
    i.add_argument("--input", required=True, type=Path,
                   help="Extracted (C, H, W) .npy, e.g. .../GeoTessera_v2/2017/patch_006296.npy")
    i.add_argument("--output", required=True, type=Path)
    i.add_argument("--model", type=Path, default=None,
                   help="Fitted colour model; required unless --colour-mode raw.")
    i.add_argument("--embedding-name", default=None, choices=sorted(EMBEDDING_REGISTRY),
                   help="Defaults to the model's own embedding_name.")
    i.add_argument("--colour-mode", choices=["model", "raw"], default="model",
                   help="raw stretches three channels per image; not comparable "
                        "between images, and needs no fitted model.")
    i.add_argument("--bands", default="0,1,2", help="Channels for --colour-mode raw.")
    i.add_argument("--upscale", default="auto",
                   help="Integer pixel repeat, or 'auto' to reach "
                        f"{BARE_MIN_SIDE} px on the long edge. '1' writes the array 1:1.")
    i.add_argument("--border", nargs="?", type=int, const=BARE_BORDER_PX, default=0,
                   metavar="PX",
                   help="Frame the image with a dark rule, painted into the "
                        f"outer PX file pixels (default {BARE_BORDER_PX}). Off "
                        "by default: at --upscale 1 it costs a real pixel a side.")
    i.add_argument("--caption", action="store_true",
                   help="Title the image with the model's provenance instead of "
                        "writing it bare.")
    i.add_argument("--dequantize", action="store_true")
    i.add_argument("--no-mask", action="store_true")
    i.set_defaults(func=_cmd_image)

    n = sub.add_parser("annotate-parquet",
                       help="Add rgb_* columns to a projection parquet.")
    n.add_argument("--model", required=True, type=Path)
    n.add_argument("--parquet", required=True, type=Path)
    n.add_argument("--features", type=Path, default=None,
                   help="Pooled features .npy in cache order. Usually unnecessary: "
                        "give --cache-dir/--cache-key instead and they are rebuilt.")
    n.add_argument("--cache-dir", type=Path, default=None)
    n.add_argument("--cache-key", default=None,
                   help="Block cache key, e.g. global_AlphaEarthCoop (no pooling suffix).")
    n.add_argument("--pooling", default="gap",
                   help="Recipe to compose. Must match the colour model's width; "
                        "gap is the only one that is one vector per channel.")
    n.set_defaults(func=_cmd_annotate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
