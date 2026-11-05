"""The data statistics figures of the report assets script, on synthetic edits."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from explain_to_learn.edits import build_edits, write_jsonl  # noqa: E402

ROOT = Path(__file__).parents[1]
PNG = b"\x89PNG"


def load_script(name, directory="scripts"):
    spec = importlib.util.spec_from_file_location(name, ROOT / directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module


assets = load_script("make_report_assets")


def test_data_statistics_and_figures(raw_dir, tmp_path):
    # conftest.raw_dir: six W&I files (an A2 essay with four edits and a C2 essay with
    # one), one LOCNESS file (one native edit) and three FCE files (two B2 essays each).
    edits = build_edits(raw_dir)

    assert assets.edits_per_essay(edits) == {
        "A2": [4] * 6,
        "B2": [1] * 6,
        "C2": [1] * 6,
        "native": [1],
    }
    shares = assets.operation_shares(edits)
    assert shares["A2"] == {"M": 0.0, "R": 0.75, "U": 0.25}
    assert shares["B2"] == {"M": 0.0, "R": 0.5, "U": 0.5}
    assert assets.top_categories(edits, n=2)["A2"] == [("VERB:SVA", 0.5), ("VERB:FORM", 0.25)]
    categories, levels, matrix = assets.category_level_shares(edits)
    assert levels == ["A2", "B2", "C2", "native"]
    assert categories[0] == "VERB:SVA"
    assert matrix.sum(axis=0) == pytest.approx([1.0] * 4)

    edits_path = tmp_path / "edits.jsonl"
    write_jsonl(edits, edits_path)
    assets.main(["--figures-dir", str(tmp_path / "figures"), "data", "--edits", str(edits_path)])
    written = sorted(path.name for path in (tmp_path / "figures").iterdir())
    assert written == sorted(assets.DATA_FIGURES)
    for name in written:
        assert (tmp_path / "figures" / name).read_bytes().startswith(PNG)


def test_data_step_without_an_edit_table_writes_nothing(tmp_path, capsys):
    assets.main(["--figures-dir", str(tmp_path / "figures"), "data", "--edits", "missing.jsonl"])

    assert "not found" in capsys.readouterr().out
    assert not (tmp_path / "figures").exists()
