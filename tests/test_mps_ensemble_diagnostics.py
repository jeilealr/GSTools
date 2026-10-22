"""Stable category schemas and count guards in the optional diagnostic tool."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gstools.mps import TrainingImage, Zone

SPEC = importlib.util.spec_from_file_location(
    "ensembles",
    Path(__file__).resolve().parents[1] / "scripts/validate_mps_ensembles.py",
)
ensembles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ensembles)


@pytest.mark.parametrize("reverse", [False, True])
def test_absent_category_schema_is_order_independent(reverse):
    fields = [np.zeros((3, 3)), np.ones((3, 3))]
    if reverse:
        fields.reverse()
    rows = [ensembles.metrics(field, True, [0, 1, 2]) for field in fields]
    assert rows[0].keys() == rows[1].keys()
    for field, row in zip(fields, rows):
        absent = 1 if not field.any() else 0
        for metric in ("fraction", "largest", "span_x"):
            assert row[f"{metric}_{absent}"] == 0.0
            assert row[f"{metric}_2"] == 0.0


def test_category_universe_includes_zone_only_category():
    ti = TrainingImage(np.zeros((3, 3)))
    zone = Zone(TrainingImage(np.full((3, 3), 7)), where=np.ones((3, 3), bool))
    assert ensembles.category_universe(ti, [zone], None) == [0, 7]


@pytest.mark.parametrize("count", ["0", "-1"])
def test_invalid_count_does_not_create_study(tmp_path, monkeypatch, count):
    out = tmp_path / "study"
    monkeypatch.setattr(
        sys, "argv", ["ensembles", "--count", count, "--out", str(out)]
    )
    with pytest.raises(SystemExit) as exc:
        ensembles.main()
    assert exc.value.code == 2
    assert not out.exists()
