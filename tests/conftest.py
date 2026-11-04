"""Synthetic fixtures in the BEA-2019 JSON and M2 formats.

All sentences are invented for the tests; none come from a corpus.
"""

import json
from pathlib import Path

import pytest

from explain_to_learn.edits import CORPUS_FILES

# Two W&I essays and the M2 blocks that json_to_m2.py would write for them.
WI_ESSAYS = [
    {
        "id": "ID-1",
        "userid": "user-1",
        "cefr": "A2.ii",
        "text": "She don't like to going. It’s my dog.\nWe was very very happy.",
        "edits": [[0, [[4, 9, "doesn't"], [18, 23, "go"], [40, 43, "were"], [44, 49, ""]]]],
    },
    {
        "id": "ID-2",
        "userid": "user-2",
        "cefr": "C2+",
        "text": "I has a cat.",
        "edits": [[0, [[2, 5, "have"]]]],
    },
]
WI_M2 = """\
S She do n't like to going .
A 1 2|||R:VERB:SVA|||does|||REQUIRED|||-NONE-|||0
A 5 6|||R:VERB:FORM|||go|||REQUIRED|||-NONE-|||0

S It 's my dog .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0

S We was very very happy .
A 1 2|||R:VERB:SVA|||were|||REQUIRED|||-NONE-|||0
A 3 4|||U:ADV||||||REQUIRED|||-NONE-|||0

S I has a cat .
A 1 2|||R:VERB:SVA|||have|||REQUIRED|||-NONE-|||0
"""

LOCNESS_ESSAYS = [
    {
        "id": "ID-3",
        "cefr": "N",
        "text": "The goverment is wrong.",
        "edits": [[0, [[4, 13, "government"]]]],
    }
]
LOCNESS_M2 = """\
S The goverment is wrong .
A 1 2|||R:SPELL|||government|||REQUIRED|||-NONE-|||0
"""

# One FCE script with two answers; the second edit of the first answer is a
# detection-only UNK edit, which the edit table skips.
FCE_ESSAYS = [
    {
        "id": "TR1*0100*2000*01",
        "q": "1",
        "l1": "Other",
        "age": "16-20",
        "answer-s": "3.1",
        "script-s": "30",
        "text": "I am agree with you.",
        "edits": [[0, [[2, 5, "", "UV"], [16, 19, None]]]],
    },
    {
        "id": "TR1*0100*2000*01",
        "q": "2",
        "l1": "Other",
        "age": "16-20",
        "answer-s": "3.1",
        "script-s": "30",
        "text": "Thank you for you letter.",
        "edits": [[0, [[14, 17, "your", "RD"]]]],
    },
]
FCE_M2 = """\
S I am agree with you .
A 1 2|||U:VERB||||||REQUIRED|||-NONE-|||0
A 4 5|||UNK|||you|||REQUIRED|||-NONE-|||0

S Thank you for you letter .
A 3 4|||R:DET|||your|||REQUIRED|||-NONE-|||0
"""


def write_pair(json_path: Path, m2_path: Path, essays: list[dict], m2_text: str) -> None:
    """Write one JSON file and its M2 file."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    m2_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text("".join(json.dumps(e) + "\n" for e in essays), encoding="utf-8")
    m2_path.write_text(m2_text, encoding="utf-8")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A data/raw directory with every corpus file, filled with synthetic essays.

    Essay IDs get a per-file prefix so that each file holds different essays.
    """
    fixtures = {
        "wi": (WI_ESSAYS, WI_M2),
        "locness": (LOCNESS_ESSAYS, LOCNESS_M2),
        "fce": (FCE_ESSAYS, FCE_M2),
    }
    for file_idx, (corpus, _, json_file, m2_file) in enumerate(CORPUS_FILES):
        essays, m2_text = fixtures[corpus]
        essays = [{**e, "id": f"{file_idx}-{e['id']}"} for e in essays]
        write_pair(tmp_path / json_file, tmp_path / m2_file, essays, m2_text)
    return tmp_path
