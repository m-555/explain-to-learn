"""Config loading, variant selection and argument checks of train/train_lora.py.

The training libraries are imported only inside the training functions, so these tests
need neither a GPU nor Unsloth.
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "train" / "train_lora.py"


def load_script():
    spec = importlib.util.spec_from_file_location("train_lora", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_lora = load_script()


def test_merge_replaces_nested_keys_without_changing_the_base():
    base = {"lora": {"rank": 16, "alpha": 32}, "seed": 1}
    merged = train_lora.merge(base, {"lora": {"rank": 8}})

    assert merged == {"lora": {"rank": 8, "alpha": 32}, "seed": 1}
    assert base == {"lora": {"rank": 16, "alpha": 32}, "seed": 1}


def test_default_variant_matches_the_planned_settings():
    pytest.importorskip("yaml")
    config = train_lora.load_config(train_lora.CONFIG_PATH, "r16")

    assert config["variant"] == "r16"
    assert config["model"]["load_in_4bit"] is False
    assert (config["lora"]["rank"], config["lora"]["alpha"], config["lora"]["dropout"]) == (
        16,
        32,
        0.05,
    )
    assert set(config["lora"]["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    training = config["training"]
    assert (training["learning_rate"], training["lr_scheduler"], training["epochs"]) == (
        2e-4,
        "cosine",
        2,
    )
    assert config["logging"]["report_to"] == "none"


@pytest.mark.parametrize("variant, rank", [("r8", 8), ("r16", 16), ("r32", 32)])
def test_rank_variants_keep_alpha_at_twice_the_rank(variant, rank):
    pytest.importorskip("yaml")
    config = train_lora.load_config(train_lora.CONFIG_PATH, variant)

    assert config["lora"]["rank"] == rank
    assert config["lora"]["alpha"] == 2 * rank
    assert config["model"]["load_in_4bit"] is False


def test_qlora_variant_changes_only_quantisation():
    pytest.importorskip("yaml")
    default = train_lora.load_config(train_lora.CONFIG_PATH, "r16")
    qlora = train_lora.load_config(train_lora.CONFIG_PATH, "qlora_r16")

    assert qlora["model"]["load_in_4bit"] is True
    qlora["model"]["load_in_4bit"] = False
    assert qlora | {"variant": "r16"} == default


def test_unknown_variant_lists_the_choices():
    pytest.importorskip("yaml")
    with pytest.raises(ValueError, match="r16"):
        train_lora.load_config(train_lora.CONFIG_PATH, "r64")


def test_run_name_defaults_to_system_and_variant(tmp_path):
    args = train_lora.parse_args(["--system", "d", "--output-dir", str(tmp_path)])

    assert args.run_name == "d-r16"
    assert args.backend == "unsloth"


def test_push_needs_a_repository_and_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    base = ["--system", "c", "--output-dir", str(tmp_path), "--push-to-hub"]

    with pytest.raises(SystemExit):
        train_lora.parse_args(base)
    with pytest.raises(SystemExit):
        train_lora.parse_args([*base, "--hub-repo", "user/adapter"])
    monkeypatch.setenv("HF_TOKEN", "token")
    assert train_lora.parse_args([*base, "--hub-repo", "user/adapter"]).hub_repo == "user/adapter"


def test_existing_adapter_is_not_overwritten(tmp_path):
    (tmp_path / "c-r16").mkdir()
    (tmp_path / "c-r16" / "adapter_config.json").write_text("{}")

    with pytest.raises(SystemExit):
        train_lora.parse_args(["--system", "c", "--output-dir", str(tmp_path)])


class FakeTokenizer:
    """Chat template joins the contents; one token per word."""

    def apply_chat_template(self, messages, tokenize):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens):
        return {"input_ids": text.split()}


def test_count_long_examples():
    rows = [
        {"prompt": [{"content": "one two"}], "completion": [{"content": "three"}]},
        {"prompt": [{"content": "one two three four"}], "completion": [{"content": "five"}]},
    ]

    assert train_lora.count_long_examples(FakeTokenizer(), rows, max_length=3) == 1


def test_git_commit_names_the_commit_and_uncommitted_changes(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.org"]
    subprocess.run([*git, "init", "-q"], check=True)
    (tmp_path / "config.yaml").write_text("rank: 16\n")
    subprocess.run([*git, "add", "config.yaml"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "Add config"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert train_lora.git_commit(tmp_path) == head
    (tmp_path / "config.yaml").write_text("rank: 8\n")
    assert train_lora.git_commit(tmp_path) == head + "-dirty"
    assert train_lora.git_commit(tmp_path / "missing") == "unknown"
