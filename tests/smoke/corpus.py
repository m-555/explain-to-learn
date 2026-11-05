"""Synthetic corpus for the runbook smoke test, in the layout of `scripts/download_data.sh`.

Every sentence is invented for this test; none comes from a corpus. Each sentence is
written once, with its edits marked inline as `[original->correction|TYPE]`, for example
`She [do->does|R:VERB:SVA] n't like fish .` (tokens separated by spaces, as in M2). From
that one line this module writes both formats of the BEA-2019 releases:

- the JSON file: one essay per line with its ID, CEFR level, raw text (tokens joined
  as a learner writes them) and character-offset edits;
- the M2 file: the tokenised sentences with their typed edits for annotator 0, and a
  `noop` line for a sentence without edits, as the official `json_to_m2.py` writes them.

So the two files always agree in the way `explain_to_learn.edits.align_essays` checks.
The GEE fixture (`gee/data/human_annotation_data/en_anonymized.json`) explains a few of
the dev sentences in GEE's annotation format, plus one record with a sentence that is not
in the corpus and one whose explanations and edits differ in number.

Regenerate the committed files after changing the essays:

    python tests/smoke/corpus.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
GEE_FILE = Path("gee/data/human_annotation_data/en_anonymized.json")

EDIT = re.compile(r"\[([^\[\]]*?)->([^\[\]|]*?)\|([^\[\]]+?)\]")
ATTACH_LEFT = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re", "'ve", "'ll", "'m", "'d"}
NOOP = "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"
# FCE's own error codes for the JSON file (the M2 file has the ERRANT types).
FCE_CODES = {
    "R:PREP": "RT",
    "R:NOUN:NUM": "FN",
    "M:VERB:FORM": "MV",
    "U:VERB": "UV",
    "M:DET": "MD",
    "R:PRON": "RA",
    "R:VERB:TENSE": "TV",
    "M:PUNCT": "MP",
    "R:VERB:SVA": "AGV",
}

# Each essay: ID, CEFR label (W&I and LOCNESS) and paragraphs of marked sentences.
# `curly` writes the apostrophes of the JSON text as ’, which json_to_m2.py normalises.
ESSAYS: dict[str, list[dict]] = {
    "wi+locness/{}/A.train": [
        {
            "id": "1-910001",
            "cefr": "A2.i",
            "paragraphs": [
                [
                    "My name is Tom and I [has->have|R:VERB:SVA] two "
                    "[brother->brothers|R:NOUN:NUM] .",
                    "We [lives->live|R:VERB:SVA] in a small town near [->the|M:DET] sea .",
                ],
                ["Every day I go to [the->|U:DET] school by bus .", "I like my teachers ."],
            ],
        },
        {
            "id": "1-910002",
            "cefr": "A1.ii",
            "paragraphs": [
                [
                    "Yesterday I [go->went|R:VERB:TENSE] to the park with my "
                    "[freind->friend|R:SPELL] .",
                    "The weather was very nice .",
                    "We [play->played|R:VERB:TENSE] football [in->on|R:PREP] the grass .",
                ]
            ],
        },
        {
            "id": "1-910003",
            "cefr": "A2.ii",
            "curly": True,
            "paragraphs": [
                [
                    "I like [cook->cooking|R:VERB:FORM] with my mother .",
                    "She [do->does|R:VERB:SVA] n't like fish , but I love it .",
                ],
                [
                    "[english->English|R:ORTH] is my favourite subject at school .",
                    "It 's [a->an|R:DET] easy language for me .",
                ],
            ],
        },
    ],
    "wi+locness/{}/B.train": [
        {
            "id": "1-920001",
            "cefr": "B1.i",
            "paragraphs": [
                [
                    "Last summer I [have->had|R:VERB:TENSE] a holiday in [->a|M:DET] "
                    "small village .",
                    "The people there were very friendly .",
                    "I am interested [about->in|R:PREP] their traditions and food .",
                ]
            ],
        },
        {
            "id": "1-920002",
            "cefr": "B2.ii",
            "paragraphs": [
                [
                    "However [->,|M:PUNCT] many students prefer to study at home .",
                    "It is [more easy->easier|R:ADJ:FORM] to concentrate when nobody is talking .",
                    "You must [to->|U:VERB:FORM] try both places before you decide .",
                ]
            ],
        },
        {
            "id": "1-920003",
            "cefr": "B1.ii",
            "paragraphs": [
                [
                    "My brother [want->wants|R:VERB:SVA] [->to|M:VERB:FORM] become a doctor .",
                    "He [catched->caught|R:VERB:INFL] a cold last week , so he stayed at home .",
                    "Our parents are proud [on->of|R:PREP] him .",
                ]
            ],
        },
    ],
    "wi+locness/{}/C.train": [
        {
            "id": "1-930001",
            "cefr": "C1.i",
            "paragraphs": [
                [
                    "The [goverment->government|R:SPELL] should invest more money in "
                    "public transport .",
                    "Despite [of->|U:PREP] the high costs , the benefits would be considerable .",
                    "[->It|M:PRON] is often argued that cars [is->are|R:VERB:SVA] essential "
                    "for rural families .",
                ]
            ],
        },
        {
            "id": "1-930002",
            "cefr": "C2",
            "paragraphs": [
                [
                    "On Sundays [always I->I always|R:WO] visit my grandparents .",
                    "They live in [an->a|R:DET] house with a large garden .",
                    "The garden is full of flowers in spring .",
                ]
            ],
        },
    ],
    "wi+locness/{}/A.dev": [
        {
            "id": "1-910101",
            "cefr": "A2.ii",
            "paragraphs": [
                [
                    "My sister [like->likes|R:VERB:SVA] [dog->dogs|R:NOUN:NUM] very much .",
                    "She [have->has|R:VERB:SVA] [a->an|R:DET] old cat too .",
                    "Every morning she [give->gives|R:VERB:SVA] it some milk .",
                ]
            ],
        },
        {
            "id": "1-910102",
            "cefr": "A1.ii",
            "paragraphs": [
                [
                    "I [goed->went|R:VERB:INFL] to the market on Saturday .",
                    "I [buyed->bought|R:VERB:INFL] [->some|M:DET] apples and [a->|U:DET] bread .",
                ],
                ["The market is near [of->|U:PREP] my house .", "It is a nice place ."],
            ],
        },
        {
            "id": "1-910103",
            "cefr": "A2.i",
            "paragraphs": [
                [
                    "We [was->were|R:VERB:SVA] happy at the party .",
                    "[Me->I|R:PRON] and my friend danced all night .",
                    "We want [->to|M:VERB:FORM] go again next year .",
                ]
            ],
        },
    ],
    "wi+locness/{}/B.dev": [
        {
            "id": "1-920101",
            "cefr": "B1.ii",
            "paragraphs": [
                [
                    "Last year I [have visited->visited|R:VERB:TENSE] London with my class .",
                    "We stayed in a hotel [in->near|R:PREP] the river .",
                    "The museums were [interested->interesting|R:ADJ] and free .",
                    "I would like [->to|M:VERB:FORM] go back one day .",
                ]
            ],
        },
        {
            "id": "1-920102",
            "cefr": "B2.i",
            "paragraphs": [
                [
                    "Nowadays [->,|M:PUNCT] many people work from home .",
                    "This [have->has|R:VERB:SVA] both advantages and "
                    "[disadvantage->disadvantages|R:NOUN:NUM] .",
                    "Workers save time because they do not need [->to|M:VERB:FORM] travel .",
                ],
                ["On the other hand , they can feel lonely ."],
            ],
        },
        {
            "id": "1-920103",
            "cefr": "B1.i",
            "paragraphs": [
                [
                    "My favourite sport is [the->|U:DET] tennis .",
                    "I [play->have played|R:VERB:TENSE] it since I was ten .",
                    "My coach [say->says|R:VERB:SVA] that I am good [in->at|R:PREP] it .",
                ]
            ],
        },
    ],
    "wi+locness/{}/C.dev": [
        {
            "id": "1-930101",
            "cefr": "C1.ii",
            "paragraphs": [
                [
                    "The [enviroment->environment|R:SPELL] is one of the main "
                    "[issue->issues|R:NOUN:NUM] of our time .",
                    "Governments [has->have|R:VERB:SVA] to act now [->,|M:PUNCT] otherwise "
                    "it will be too late .",
                    "We should also discuss [about->|U:PREP] the role of individuals .",
                ]
            ],
        },
        {
            "id": "1-930102",
            "cefr": "C2",
            "paragraphs": [
                [
                    "Social media [have->has|R:VERB:SVA] changed the way we communicate .",
                    "It allows people to stay in touch with friends abroad .",
                    "However [->,|M:PUNCT] it can also cause [the->|U:DET] stress .",
                ]
            ],
        },
    ],
    "wi+locness/{}/N.dev": [
        {
            "id": "7-9900101",
            "cefr": "N",
            "paragraphs": [
                [
                    "The [arguement->argument|R:SPELL] for nuclear power is not convincing .",
                    "[Their->There|R:PRON] are many risks involved .",
                    "Supporters claim that it [produce->produces|R:VERB:SVA] no emissions .",
                    "Nuclear waste [,->|U:PUNCT] lasts for thousands of years .",
                ]
            ],
        },
        {
            "id": "7-9900102",
            "cefr": "N",
            "paragraphs": [
                [
                    "Many people believe [,->|U:PUNCT] that university should be free .",
                    "Education [are->is|R:VERB:SVA] a right , not a privilege .",
                    "Students should not leave with large debts .",
                ]
            ],
        },
    ],
    "fce/{}/fce.train": [
        {
            "id": "TR9001*0100*2000*01",
            "q": "1",
            "paragraphs": [
                [
                    "I am writing to complain [for->about|R:PREP] the concert last Saturday .",
                    "The show started two [hour->hours|R:NOUN:NUM] late and the sound was "
                    "terrible .",
                    "I would like [->to|M:VERB:FORM] get my money back .",
                ]
            ],
        },
        {
            "id": "TR9001*0100*2000*01",
            "q": "2",
            "paragraphs": [
                [
                    "I [am->|U:VERB] agree with your idea about the class trip .",
                    "We could meet at [->the|M:DET] station on Friday morning .",
                    # Detected but not corrected: FCE marks it, the edit table skips it.
                    "It would be [a->a|UNK] great day for everyone .",
                ]
            ],
        },
    ],
    "fce/{}/fce.dev": [
        {
            "id": "TR9002*0100*2000*01",
            "q": "1",
            "paragraphs": [
                [
                    "Thank you for [you->your|R:PRON] letter .",
                    "I [enjoy->enjoyed|R:VERB:TENSE] the party very much .",
                ]
            ],
        }
    ],
    "fce/{}/fce.test": [
        {
            "id": "TR9003*0100*2000*01",
            "q": "1",
            "paragraphs": [
                [
                    "In my opinion [->,|M:PUNCT] sport is good for young people .",
                    "It [help->helps|R:VERB:SVA] them to stay healthy .",
                ]
            ],
        }
    ],
}

# GEE records: source, target, (operation, original, corrected, explanation, GEE type),
# and the labels the GEE annotator put on an explanation (by explanation number).
GEE_RECORDS = [
    {
        "source": "My sister like dog very much.",
        "target": "My sister likes dogs very much.",
        "edits": [
            (
                "replace",
                "like",
                "likes",
                "The word 'like' changes to 'likes' because 'my sister' is one person.",
                "subject-verb agreement",
            ),
            (
                "replace",
                "dog",
                "dogs",
                "The word 'dog' changes to 'dogs' because she likes dogs in general.",
                "noun number",
            ),
        ],
        "all_correct": True,
    },
    {
        "source": "We was happy at the party.",
        "target": "We were happy at the party.",
        "edits": [
            (
                "replace",
                "was",
                "were",
                "The word 'was' changes to 'were' because the party is in the past.",
                "verb tense",
            )
        ],
        "labels": {0: ["Wrong error explanation"]},
    },
    {
        "source": "Nowadays many people work from home.",
        "target": "Nowadays, many people work from home.",
        "edits": [
            (
                "insert",
                "",
                ",",
                "A comma is inserted after 'Nowadays' because it introduces the sentence.",
                "punctuation",
            )
        ],
        "all_correct": True,
    },
    {
        "source": "The cat sleeps in the sofa every afternoon.",
        "target": "The cat sleeps on the sofa every afternoon.",
        "edits": [
            (
                "replace",
                "in",
                "on",
                "The word 'in' changes to 'on' because we sit on a sofa.",
                "preposition",
            )
        ],
        "all_correct": True,
    },
    {
        "source": "My coach say that I am good in it.",
        "target": "My coach says that I am good at it.",
        # Two edits but one explanation: the comparison skips the record.
        "edits": [
            (
                "replace",
                "say",
                "says",
                "The word 'say' changes to 'says' because 'my coach' is one person.",
                "subject-verb agreement",
            ),
            ("replace", "in", "at", "", ""),
        ],
    },
]


def parse_sentence(marked: str) -> tuple[list[str], list[tuple[int, int, str, str]]]:
    """Tokens and edits `(start, end, correction, type)` of one marked sentence."""
    tokens: list[str] = []
    edits = []
    position = 0
    for match in EDIT.finditer(marked):
        tokens += marked[position : match.start()].split()
        original, correction, error_type = match.groups()
        start = len(tokens)
        tokens += original.split()
        edits.append((start, len(tokens), " ".join(correction.split()), error_type.strip()))
        position = match.end()
    tokens += marked[position:].split()
    return tokens, edits


def join_tokens(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Text as a learner writes it, and each token's character span in it."""
    text, spans = "", []
    for token in tokens:
        if text and token not in ATTACH_LEFT:
            text += " "
        spans.append((len(text), len(text) + len(token)))
        text += token
    return text, spans


def m2_block(tokens: list[str], edits: list[tuple[int, int, str, str]]) -> str:
    """One M2 sentence block; a sentence without edits gets a `noop` line."""
    lines = ["S " + " ".join(tokens)]
    for start, end, correction, error_type in edits:
        lines.append(f"A {start} {end}|||{error_type}|||{correction}|||REQUIRED|||-NONE-|||0")
    if not edits:
        lines.append(NOOP)
    return "\n".join(lines)


def essay_files(essay: dict, corpus: str) -> tuple[dict, list[str]]:
    """The JSON record and the M2 blocks of one essay."""
    paragraphs, json_edits, blocks = [], [], []
    offset = 0
    for paragraph in essay["paragraphs"]:
        sentences = []
        for marked in paragraph:
            tokens, edits = parse_sentence(marked)
            sentence, spans = join_tokens(tokens)
            base = offset + sum(len(s) + 1 for s in sentences)
            for start, end, correction, error_type in edits:
                begin = base + (spans[start][0] if start < len(spans) else len(sentence))
                finish = base + spans[end - 1][1] if end > start else begin
                corrected = join_tokens(correction.split())[0]
                if corpus != "fce":
                    json_edits.append([begin, finish, corrected])
                elif error_type == "UNK":
                    json_edits.append([begin, finish, None])
                else:
                    json_edits.append([begin, finish, corrected, FCE_CODES.get(error_type, "X")])
            sentences.append(sentence)
            blocks.append(m2_block(tokens, edits))
        paragraphs.append(" ".join(sentences))
        offset += len(paragraphs[-1]) + 1
    text = "\n".join(paragraphs)
    if essay.get("curly"):
        text = text.replace("'", "’")
    if corpus == "fce":
        record = {
            "id": essay["id"],
            "q": essay["q"],
            "l1": "Other",
            "age": "21-25",
            "answer-s": "3.3",
            "script-s": "30",
            "text": text,
            "edits": [[0, json_edits]],
        }
    else:
        record = {"id": essay["id"]}
        if corpus == "wi":
            record["userid"] = "user-" + essay["id"]
        record |= {"cefr": essay["cefr"], "text": text, "edits": [[0, json_edits]]}
    return record, blocks


def gee_record(number: int, spec: dict) -> dict:
    """One record of GEE's annotation file (`gee.parse_record` reads it)."""
    explanations = [
        f"{explanation}<br>Error type: {gee_type}"
        for _, _, _, explanation, gee_type in spec["edits"]
        if explanation
    ]
    text = (
        f"<b>Source:</b> {spec['source']}<br><b>Target:</b> {spec['target']}<br><br>"
        "<b>Explanation:</b><br>" + "<br>".join(explanations)
    )
    edits = "<br>".join(repr([op, old, new]) for op, old, new, _, _ in spec["edits"])
    record = {
        "text": text,
        "edits": edits,
        "id": number,
        "annotator": 1,
        "annotation_id": 900 + number,
        "created_at": "2026-01-01T00:00:00.000000Z",
        "updated_at": "2026-01-01T00:00:00.000000Z",
        "lead_time": 10.0,
    }
    if spec.get("all_correct"):
        record["all_correct"] = "There is no error in the explanation(s) above."
    plain = re.sub(r"</?b>", "", text.replace("<br>", "\n"))
    labels = []
    for index, names in spec.get("labels", {}).items():
        explanation = spec["edits"][index][3]
        start = plain.index(explanation)
        labels.append(
            {
                "start": "/text()[1]",
                "startOffset": 0,
                "end": "/text()[1]",
                "endOffset": len(explanation),
                "globalOffsets": {"start": start, "end": start + len(explanation)},
                "text": explanation,
                "labels": names,
            }
        )
    if labels:
        record["label"] = labels
    return record


def write_corpus(raw_dir: str | Path = RAW_DIR) -> list[Path]:
    """Write every corpus file and the GEE fixture under `raw_dir`; returns their paths."""
    raw_dir = Path(raw_dir)
    written = []
    for pattern, essays in ESSAYS.items():
        corpus = "fce" if pattern.startswith("fce") else "wi"
        records, blocks = [], []
        for essay in essays:
            kind = "locness" if essay.get("cefr") == "N" else corpus
            record, essay_blocks = essay_files(essay, kind)
            records.append(record)
            blocks += essay_blocks
        json_path = raw_dir / (pattern.format("json") + ".json")
        m2_path = raw_dir / (pattern.format("m2") + ".gold.bea19.m2")
        for path in (json_path, m2_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        json_path.write_text(lines, encoding="utf-8")
        m2_path.write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
        written += [json_path, m2_path]
    gee_path = raw_dir / GEE_FILE
    gee_path.parent.mkdir(parents=True, exist_ok=True)
    records = [gee_record(number, spec) for number, spec in enumerate(GEE_RECORDS, start=1)]
    gee_path.write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n", "utf-8")
    written.append(gee_path)
    return written


if __name__ == "__main__":
    for path in write_corpus():
        print(f"Wrote {path}")
