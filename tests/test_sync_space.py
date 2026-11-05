"""scripts/sync_space.py: what is staged for the Space, and the pinned package."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("sync_space", ROOT / "scripts" / "sync_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync_space = load_script()


def test_stage_copies_the_space_files_and_pins_the_package(tmp_path):
    dest = tmp_path / "space"
    files = sync_space.stage(dest, "v1.0.0")

    assert {"app.py", "stubs.py", "requirements.txt", "README.md"} <= set(files)
    assert "configs/systems.yaml" in files and "grammar_graph/concepts.yaml" in files
    staged = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert staged == set(files)
    assert not any(p.startswith(("data/", "outputs/")) for p in staged)

    requirements = (dest / "requirements.txt").read_text(encoding="utf-8")
    assert "explain-to-learn@v1.0.0" in requirements and "@main" not in requirements
    card = (dest / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n") and "app_file: app.py" in card


def test_stage_without_ref_keeps_main(tmp_path):
    sync_space.stage(tmp_path / "space", None)
    assert "@main" in (tmp_path / "space" / "requirements.txt").read_text(encoding="utf-8")


def test_pin_package_fails_loudly_if_the_requirement_changed():
    with pytest.raises(ValueError, match="no longer contains"):
        sync_space.pin_package("explain-to-learn==1.0\n", "v1.0.0")


def test_upload_needs_a_space_and_a_ref():
    with pytest.raises(SystemExit):
        sync_space.main(["--ref", "v1.0.0", "--space", ""])
    with pytest.raises(SystemExit):
        sync_space.main(["--space", "user/space"])
