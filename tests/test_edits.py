import pytest
from conftest import FCE_ESSAYS, FCE_M2, WI_ESSAYS, WI_M2, write_pair

from explain_to_learn.edits import (
    Essay,
    M2Edit,
    align_essays,
    apply_edits,
    build_edits,
    cefr_level,
    essay_edits,
    parse_m2,
    read_essays,
    read_jsonl,
    write_jsonl,
)


@pytest.fixture
def wi_files(tmp_path):
    json_path, m2_path = tmp_path / "wi.json", tmp_path / "wi.m2"
    write_pair(json_path, m2_path, WI_ESSAYS, WI_M2)
    return json_path, m2_path


def test_parse_m2_reads_sentences_and_edits(wi_files):
    sentences = parse_m2(wi_files[1])

    assert len(sentences) == 4
    assert sentences[0].tokens == ("She", "do", "n't", "like", "to", "going", ".")
    assert sentences[0].edits[0] == M2Edit(1, 2, "R:VERB:SVA", "does", 0)
    assert sentences[1].edits[0].error_type == "noop"
    assert sentences[1].edits[0].correction == ""
    assert sentences[2].edits[1] == M2Edit(3, 4, "U:ADV", "", 0)


def test_parse_m2_without_trailing_blank_line(tmp_path):
    path = tmp_path / "one.m2"
    path.write_text("S A cat .\nA 0 1|||R:DET|||The|||REQUIRED|||-NONE-|||0", encoding="utf-8")

    assert parse_m2(path)[0].edits[0].correction == "The"


def test_apply_edits_handles_replace_insert_and_delete():
    tokens = ["We", "was", "very", "very", "happy", "."]
    edits = [
        M2Edit(3, 4, "U:ADV", "", 0),
        M2Edit(1, 2, "R:VERB:SVA", "were", 0),
        M2Edit(5, 5, "M:PUNCT", "!", 0),
    ]

    assert apply_edits(tokens, edits) == ["We", "were", "very", "happy", "!", "."]


def test_cefr_level_mapping():
    assert cefr_level("A2.ii") == "A2"
    assert cefr_level("C2+") == "C2"
    assert cefr_level("N") == "native"
    with pytest.raises(ValueError):
        cefr_level("X1")


def test_read_essays_wi_and_fce(wi_files, tmp_path):
    wi = read_essays(wi_files[0], "wi")
    fce_path = tmp_path / "fce.json"
    write_pair(fce_path, tmp_path / "fce.m2", FCE_ESSAYS, FCE_M2)
    fce = read_essays(fce_path, "fce")

    assert [(e.essay_id, e.cefr_level) for e in wi] == [("ID-1", "A2"), ("ID-2", "C2")]
    assert [e.essay_id for e in fce] == ["TR1*0100*2000*01-1", "TR1*0100*2000*01-2"]
    assert {e.cefr_level for e in fce} == {"B2"}


def test_align_essays_assigns_sentences_to_essays(wi_files):
    essays = read_essays(wi_files[0], "wi")
    aligned = list(align_essays(essays, parse_m2(wi_files[1])))

    assert [(essay.essay_id, len(sents)) for essay, sents in aligned] == [("ID-1", 3), ("ID-2", 1)]


def test_align_essays_rejects_mismatched_text(wi_files):
    essays = [Essay("ID-1", "A2", "Completely different text.")]

    with pytest.raises(ValueError, match="ID-1"):
        list(align_essays(essays, parse_m2(wi_files[1])))


def test_align_essays_rejects_leftover_sentences(wi_files):
    essays = read_essays(wi_files[0], "wi")[:1]

    with pytest.raises(ValueError, match="left"):
        list(align_essays(essays, parse_m2(wi_files[1])))


def test_essay_edits_fields(wi_files):
    essay, sentences = next(align_essays(read_essays(wi_files[0], "wi"), parse_m2(wi_files[1])))
    rows = essay_edits("wi", "dev", essay, sentences)

    assert [r.edit_id for r in rows] == ["wi-ID-1-0-0", "wi-ID-1-0-1", "wi-ID-1-2-0", "wi-ID-1-2-1"]
    first = rows[0]
    assert first.original == "She do n't like to going ."
    assert first.corrected == "She does n't like to go ."
    assert (first.start, first.end) == (1, 2)
    assert (first.original_text, first.corrected_text) == ("do", "does")
    assert (first.operation, first.error_category) == ("R", "VERB:SVA")
    assert (first.split, first.cefr_level, first.essay_key) == ("dev", "A2", "wi-ID-1")
    deletion = rows[3]
    assert (deletion.original_text, deletion.corrected_text) == ("very", "")
    assert deletion.corrected == "We were very happy ."


def test_essay_edits_skip_unk_but_keep_edit_index(tmp_path):
    json_path, m2_path = tmp_path / "fce.json", tmp_path / "fce.m2"
    write_pair(json_path, m2_path, FCE_ESSAYS, FCE_M2)
    aligned = align_essays(read_essays(json_path, "fce"), parse_m2(m2_path))
    rows = [row for essay, sents in aligned for row in essay_edits("fce", "train", essay, sents)]

    assert [r.edit_id for r in rows] == ["fce-TR1*0100*2000*01-1-0-0", "fce-TR1*0100*2000*01-2-0-0"]
    assert all(r.error_type != "UNK" for r in rows)


def test_build_edits_ids_are_unique_and_stable(raw_dir):
    first = build_edits(raw_dir)
    second = build_edits(raw_dir)

    assert [e.edit_id for e in first] == [e.edit_id for e in second]
    assert len({e.edit_id for e in first}) == len(first)
    assert {e.corpus for e in first} == {"wi", "locness", "fce"}
    assert {e.cefr_level for e in first if e.corpus == "locness"} == {"native"}


def test_build_edits_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        build_edits(tmp_path)


def test_jsonl_round_trip(raw_dir, tmp_path):
    edits = build_edits(raw_dir)
    path = tmp_path / "out" / "edits.jsonl"
    write_jsonl(edits, path)

    assert read_jsonl(path) == edits
