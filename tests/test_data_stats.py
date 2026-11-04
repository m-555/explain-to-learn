import subprocess
import sys
from pathlib import Path

from explain_to_learn.edits import build_edits, write_jsonl

SCRIPT = Path(__file__).parents[1] / "scripts" / "data_stats.py"


def test_data_stats_writes_counts_only(raw_dir, tmp_path):
    edits_path, out_path = tmp_path / "edits.jsonl", tmp_path / "stats.md"
    write_jsonl(build_edits(raw_dir), edits_path)

    subprocess.run(
        [sys.executable, str(SCRIPT), "--edits", str(edits_path), "--out", str(out_path)],
        check=True,
    )
    report = out_path.read_text(encoding="utf-8")

    assert "| VERB:SVA |" in report
    assert "| native |" in report
    assert "goverment" not in report and "She do" not in report
