"""Herten input fidelity, geometry, categorical smoke, and gallery placement."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

GALLERY = Path(__file__).resolve().parents[1] / "examples/13_mps"
SPEC = importlib.util.spec_from_file_location(
    "herten", GALLERY / "07_herten_nonstationary.py"
)
herten = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(herten)


def test_supplied_section_and_crop():
    path = GALLERY / "input/herten_s1_section.npz"
    assert herten.file_hash(path) == herten.INPUT_HASH
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"section", "spacing", "source"}
        assert data["section"].shape == (320, 140)
        assert data["section"].dtype == np.int8
        assert data["spacing"].shape == () and float(data["spacing"]) == 0.05
        np.testing.assert_array_equal(
            np.unique(data["section"]), np.arange(10)
        )
    crop, spacing = herten.load_input()
    assert crop.shape == (160, 140) and spacing == 0.05
    codes, counts = np.unique(crop, return_counts=True)
    np.testing.assert_array_equal(codes, [1, 2, 3, 4, 5, 7, 8, 9])
    np.testing.assert_array_equal(
        counts, [3011, 4046, 1386, 6940, 5146, 1001, 719, 151]
    )


def test_rotation_is_radians_and_constant_along_z():
    rotation = herten.rotation_map((16, 14))
    assert rotation.shape == (16, 14, 1) and np.isfinite(rotation).all()
    np.testing.assert_array_equal(rotation[0], np.zeros((14, 1)))
    np.testing.assert_allclose(rotation[-1], np.deg2rad(-12))
    np.testing.assert_array_equal(
        rotation, np.broadcast_to(rotation[:, :1], rotation.shape)
    )


@pytest.mark.parametrize("rotated", [False, True])
def test_small_herten_smoke_is_repeatable_with_cleanup(rotated):
    # Explicit smoke profile, not a replacement for the 160x140 reference run.
    crop, _ = herten.load_input()
    kwargs = dict(shape=(9, 8), threads=2, n_neighbors=6, scan_fraction=0.02)
    first = herten.simulate_case(crop, rotated, **kwargs)
    second = herten.simulate_case(crop, rotated, **kwargs)
    assert first.shape == (9, 8)
    assert np.isfinite(first).all() and np.isin(first, np.unique(crop)).all()
    np.testing.assert_array_equal(first, np.rint(first))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "bad",
    [
        np.full((2, 2), 1.2),
        np.full((2, 2), np.nan),
        np.zeros((2, 2)),
        np.ones((3, 2)),
    ],
)
def test_raw_field_validation_rejects_invalid_output(bad):
    with pytest.raises(ValueError):
        herten.validate_field(bad, (2, 2), [1, 2])


def test_gallery_numbering_and_no_private_dependencies():
    pages = sorted(GALLERY.glob("[0-9]*.py"))
    assert [p.name[:3] for p in pages] == [f"{n:02d}_" for n in range(1, 20)]
    assert [p.name for p in pages[5:9]] == [
        "06_channel_strebelle_nonstat.py",
        "07_herten_nonstationary.py",
        "08_channel_strebelle_radial.py",
        "09_multivariate_context.py",
    ]
    assert (GALLERY / "18_categorical_penalty.py").is_file()
    source = Path(herten.__file__).read_text()
    for marker in (
        "gmd_claim4",
        "results_pp1",
        "BITEXACT_OK",
        "post-hoc",
        "pre-registered",
    ):
        assert marker not in source
    ast.parse(source)


def test_preview_does_not_simulate_and_uses_shared_extent(monkeypatch):
    import matplotlib.pyplot as plt

    def unexpected(*args, **kwargs):
        raise AssertionError("preview must not run Direct Sampling")

    monkeypatch.setattr(herten, "simulate_case", unexpected)
    fig = herten.preview()
    try:
        for ax in fig.axes[:3]:
            image = ax.images[0]
            np.testing.assert_array_equal(image.get_extent(), [0, 8, 0, 7])
            assert image.origin == "lower"
            assert ax.get_xlabel() == "x (m)" and ax.get_ylabel() == "z (m)"
        assert all(
            ax.images[0].norm is fig.axes[0].images[0].norm
            for ax in fig.axes[:3]
        )
        manifest = json.loads(
            (GALLERY / "output/herten_nonstationary.json").read_text()
        )
        assert manifest["controls"]["threshold"] == 0.05
    finally:
        plt.close(fig)
