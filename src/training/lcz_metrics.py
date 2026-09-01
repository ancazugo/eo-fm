"""LCZ-specific confusion-matrix metrics (WUDAPT / LCZ Generator suite).

The generic suite in :mod:`training.evaluate` treats the 17 LCZ types as
unordered labels, which is correct but blunt: confusing LCZ 1 with LCZ 2
(compact high-rise vs compact mid-rise) and confusing LCZ 1 with LCZ G (water)
count the same. Bechtel et al. 2020 defines a similarity matrix so the second
mistake can be scored as worse than the first, and the LCZ Generator factsheets
report the resulting metrics — so these are what make our numbers comparable to
the community's.

All four metrics are pure functions of a dense 17x17 confusion matrix. They are
deliberately *not* streaming torchmetrics: they need the whole matrix, they are
cheap once it exists, and keeping them as plain functions makes them trivial to
unit-test against the identity-matrix reductions.

``kappa`` stays the primary metric for the comparison against the patch ladder;
``kappa_w`` is reported alongside it, never instead of it. The *gap* between the
two is itself a result: a large gap means the residual errors are within-family
and forgivable, a small gap means built<->natural confusion, which is real
failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Row/column order of the similarity matrix and of every metric here: LCZ
# 1..10 then A..G, mapped to indices 0..16. This matches the project-wide
# ``1-17 -> 0-16`` label convention (see utils.constants.lcz_dict, whose
# ``alt_code`` carries the 11->A .. 17->G mapping).
LCZ_ORDER = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
             "A", "B", "C", "D", "E", "F", "G")

# LCZ 1-10 are the built types, A-G the natural ones. Index-based so callers
# never have to re-derive the off-by-one.
URBAN_INDICES = tuple(range(0, 10))
NATURAL_INDICES = tuple(range(10, 17))

DEFAULT_SIMILARITY_CSV = (
    Path(__file__).resolve().parents[2] / "docs" / "lcz_class_similarity.csv"
)


def load_similarity_matrix(path: Path | str | None = None) -> np.ndarray:
    """Load the Bechtel et al. 2020 LCZ similarity matrix as a (17, 17) array.

    Every property is asserted rather than silently coerced. A transposed or
    alphabetically-sorted CSV still produces plausible-looking OAw values, so
    the class-ordering check is the one that actually earns its keep: it pins
    that LCZ 1 is more similar to LCZ 4 (both high-rise) than to LCZ B
    (scattered trees).
    """
    path = Path(path) if path is not None else DEFAULT_SIMILARITY_CSV
    if not path.exists():
        raise FileNotFoundError(f"LCZ similarity matrix not found: {path}")

    # ``comment="#"`` so the provenance header can live in the file itself --
    # without a recorded source and revision, OAw is not comparable to
    # published factsheet values.
    rows: list[list[str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([c.strip() for c in line.split(",")])

    if not rows:
        raise ValueError(f"LCZ similarity matrix is empty: {path}")

    header = rows[0][1:]
    labels = [r[0] for r in rows[1:]]
    if tuple(header) != LCZ_ORDER:
        raise ValueError(
            f"LCZ similarity matrix column order is {header}, expected "
            f"{list(LCZ_ORDER)}. Row/column order must be 1..10, A..G so it "
            "matches the 1-17 -> 0-16 label convention."
        )
    if tuple(labels) != LCZ_ORDER:
        raise ValueError(
            f"LCZ similarity matrix row order is {labels}, expected "
            f"{list(LCZ_ORDER)}."
        )

    w = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float64)

    if w.shape != (17, 17):
        raise ValueError(f"LCZ similarity matrix must be (17, 17), got {w.shape}")
    if not np.allclose(w, w.T):
        raise ValueError("LCZ similarity matrix must be symmetric")
    if not np.allclose(np.diag(w), 1.0):
        raise ValueError("LCZ similarity matrix must have a unit diagonal")
    if w.min() < 0.0 or w.max() > 1.0:
        raise ValueError(
            f"LCZ similarity matrix must lie in [0, 1], got "
            f"[{w.min()}, {w.max()}]"
        )
    # The ordering check the docstring promises. LCZ 1 vs 4 differ only in
    # openness; LCZ 1 vs B differ in land cover and surface objects too.
    if not w[0, 3] > w[0, 11]:
        raise ValueError(
            "LCZ similarity matrix fails the class-ordering check: "
            f"W[1,4]={w[0, 3]} should exceed W[1,B]={w[0, 11]}. The CSV is "
            "probably transposed or alphabetically sorted."
        )
    return w


def oa_weighted(cm: np.ndarray, w: np.ndarray) -> float:
    """Similarity-weighted overall accuracy (Bechtel et al. 2020).

    ``WA = (1/N) * sum_ij w_ij c_ij``. Plain OA is the ``W = I`` special case.
    """
    total = cm.sum()
    if total == 0:
        return float("nan")
    return float((w * cm).sum() / total)


def kappa_weighted(cm: np.ndarray, w: np.ndarray) -> float:
    """Chance-corrected analogue of :func:`oa_weighted`.

    Reduces to Cohen's kappa when ``W = I``. ``torchmetrics`` cannot do this:
    its ``weights`` argument only accepts ``linear``/``quadratic``, both of
    which assume the classes are ordinal, and LCZ's 17 types are not.
    """
    n = cm.sum()
    if n == 0:
        return float("nan")
    expected = np.outer(cm.sum(1), cm.sum(0)) / n
    po = (w * cm).sum() / n
    pe = (w * expected).sum() / n
    if np.isclose(pe, 1.0):
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def oa_urban(cm: np.ndarray) -> float:
    """OA restricted to the built types, LCZ 1-10, in rows *and* columns.

    This is the 10x10 sub-matrix reduction the LCZ Generator factsheets report:
    both axes are restricted, so pixels that cross the built/natural boundary
    in either direction leave the denominator entirely. That confusion is
    scored by :func:`oa_built_natural` instead -- the two metrics are
    complementary, and reporting OAu alone would hide it.
    """
    sub = cm[np.ix_(URBAN_INDICES, URBAN_INDICES)]
    total = sub.sum()
    if total == 0:
        return float("nan")
    return float(np.trace(sub) / total)


def oa_built_natural(cm: np.ndarray) -> float:
    """OA after collapsing the 17 types to built (1-10) vs natural (A-G)."""
    total = cm.sum()
    if total == 0:
        return float("nan")
    u, n = list(URBAN_INDICES), list(NATURAL_INDICES)
    correct = cm[np.ix_(u, u)].sum() + cm[np.ix_(n, n)].sum()
    return float(correct / total)


def lcz_metrics_from_cm(
    cm: np.ndarray,
    w: np.ndarray | None = None,
    *,
    prefix: str = "test",
    suffix: str = "",
) -> dict[str, float]:
    """The whole WUDAPT suite from one dense confusion matrix.

    ``cm`` must be dense and square over all 17 classes with ``cm[i, j]`` =
    true ``i`` predicted ``j``; a matrix built with ``labels=present`` (as
    sklearn defaults to) will silently mis-index against ``W``.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (17, 17):
        raise ValueError(
            f"lcz_metrics_from_cm needs a dense (17, 17) matrix, got {cm.shape}. "
            "Build it with labels=range(num_classes), not labels=present."
        )
    if w is None:
        w = load_similarity_matrix()
    return {
        f"{prefix}_oau{suffix}": oa_urban(cm),
        f"{prefix}_oabu{suffix}": oa_built_natural(cm),
        f"{prefix}_oaw{suffix}": oa_weighted(cm, w),
        f"{prefix}_kappa_w{suffix}": kappa_weighted(cm, w),
    }
