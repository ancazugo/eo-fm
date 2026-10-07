"""WUDAPT AOIs must reuse the So2Sat city bbox so the 3x3 grids coincide.

Matched by the SMOD_ID suffix of the AOI directory, not by name: the name
lookup missed every "{City}__{SMOD_ID}" AOI and fell back to the label extent,
building a different checkerboard whose train tiles covered ~half of Nairobi's
So2Sat evaluation patches.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from create_city_grids import _load_city_bboxes, _lookup_city_bbox  # noqa: E402

CSV = Path(__file__).resolve().parents[1] / "data" / "so2sat_guppd_bounds.csv"


def test_wudapt_aoi_gets_the_so2sat_bbox():
    b = _load_city_bboxes(CSV)
    assert _lookup_city_bbox("Nairobi__30_9135", b) == _lookup_city_bbox("Nairobi", b)
    assert _lookup_city_bbox("Osaka_Kyoto__30_9031", b) is not None


def test_a_name_clash_with_another_smod_id_does_not_borrow_the_bbox():
    b = _load_city_bboxes(CSV)
    assert _lookup_city_bbox("Nairobi__30_1", b) is None
    assert _lookup_city_bbox("Abeokuta__30_9896", b) is None
