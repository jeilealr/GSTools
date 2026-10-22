"""Exact artifact selection, isolation, and failure reporting for paired builds."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pairing",
    Path(__file__).resolve().parents[1] / "scripts/validate_ds_pairing.py",
)
pairing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pairing)


def test_install_resolves_exact_wheels_and_offline_dependencies():
    cmd = pairing.install_command(
        "python", "core.whl", "front.whl", "house", "floor"
    )
    assert cmd[:5] == ["python", "-m", "pip", "--isolated", "install"]
    assert cmd[5:7] == ["core.whl", "front.whl"]
    assert "pytest" in cmd and "--no-deps" not in cmd
    assert cmd[-5:] == [
        "--no-index",
        "--find-links",
        "house",
        "--constraint",
        "floor",
    ]


def test_subprocess_isolation(monkeypatch):
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTEST_PLUGINS",
        "PYTEST_ADDOPTS",
    ):
        monkeypatch.setenv(key, "injected")
    env = pairing.isolated_env()
    assert all(
        key not in env
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTEST_PLUGINS",
            "PYTEST_ADDOPTS",
        )
    )
    assert (
        env["PYTHONNOUSERSITE"] == env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    )


def test_wheel_selection_rejects_ambiguity(tmp_path):
    with pytest.raises(ValueError):
        pairing.one_wheel(tmp_path, "gstools-*.whl")
    first = tmp_path / "gstools-1.whl"
    first.touch()
    assert pairing.one_wheel(tmp_path, "gstools-*.whl") == first
    (tmp_path / "gstools-2.whl").touch()
    with pytest.raises(ValueError):
        pairing.one_wheel(tmp_path, "gstools-*.whl")


def test_output_ownership_and_existing_environment(tmp_path):
    arbitrary = tmp_path / "owner-data"
    arbitrary.mkdir()
    with pytest.raises(ValueError):
        pairing.safe_output(arbitrary, reuse=True)
    owned = pairing.safe_output(tmp_path / "owned")
    assert pairing.safe_output(owned, reuse=True) == owned
    (owned / "venv").mkdir()
    with pytest.raises(ValueError):
        pairing.safe_output(owned, reuse=True)
    assert (owned / "venv").is_dir()


def test_symlink_output_is_rejected(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        pairing.safe_output(link / "new")


def test_reused_wheel_provenance_is_hash_bound(tmp_path):
    wheel = tmp_path / "old.whl"
    wheel.write_bytes(b"old artifact")
    assert pairing.reuse_provenance(wheel, None) == "unknown"
    assert pairing.reuse_provenance(wheel, {"sha256": "wrong"}) == "unknown"
    source = {"head": "recorded-build-head"}
    assert (
        pairing.reuse_provenance(
            wheel, {"sha256": pairing.sha256(wheel), "source": source}
        )
        == source
    )


def test_failure_manifest_precedes_install(tmp_path, monkeypatch):
    core, front, out = (tmp_path / name for name in ("core", "front", "out"))
    for path in (
        core / "Cargo.toml",
        core / "scripts/check_mps_wheel.py",
        core / "scripts/check_mps_boundaries.py",
        front / "pyproject.toml",
    ):
        path.parent.mkdir(exist_ok=True, parents=True)
        path.touch()
    (front / "tests").mkdir()
    pairing.safe_output(out)
    (out / "wheels").mkdir()
    for name in ("gstools_core-1.whl", "gstools-1.whl"):
        (out / "wheels" / name).write_bytes(b"wheel")
    monkeypatch.setattr(pairing.venv.EnvBuilder, "create", lambda *args: None)
    monkeypatch.setattr(pairing, "git", lambda *args: "current-head")

    def fail(*args, **kwargs):
        running = json.loads((out / "manifest.json").read_text())
        assert running["status"] == "running"
        raise RuntimeError("dependency unavailable")

    monkeypatch.setattr(pairing, "run", fail)
    with pytest.raises(RuntimeError, match="dependency unavailable"):
        pairing.main(
            [
                "--core",
                str(core),
                "--frontend",
                str(front),
                "--out",
                str(out),
                "--no-build",
            ]
        )
    manifest = json.loads((out / "manifest.json").read_text())
    assert (
        manifest["status"] == "failed" and manifest["failed_step"] == "install"
    )
    assert manifest["artifacts"]["core"]["source"] == "unknown"
    assert manifest["dependency_profile"] == "isolated"


@pytest.mark.parametrize("name", ["venv", "wheels", "logs", "work"])
def test_symlink_output_children_are_rejected(tmp_path, name):
    owned = pairing.safe_output(tmp_path / "owned")
    (owned / name).symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink output child"):
        pairing.safe_output(owned, reuse=True)


def test_timeout_output_is_preserved(tmp_path, monkeypatch):
    import subprocess

    def expired(*args, **kwargs):
        assert kwargs["timeout"] == 4
        raise subprocess.TimeoutExpired("child", 4)

    monkeypatch.setattr(pairing.subprocess, "run", expired)
    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        pairing.run(["child"], log, timeout=4)
    assert "TIMEOUT after 4s" in log.read_text()
