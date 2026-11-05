"""Notebooks are committed without outputs, except charts (which show aggregate counts)."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_keeps_only_chart_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        for output in cell["outputs"]:
            assert output["output_type"] == "display_data"
            assert set(output["data"]) == {"image/png"}
