"""Config, settings and argument checks of train/train_dpo.py.

The training libraries are imported only inside `train`, so these tests need neither a
GPU nor TRL. The one test that builds TRL's `DPOConfig` runs only where TRL is installed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def train_dpo(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "train"))  # train_dpo.py imports train_lora.py
    spec = importlib.util.spec_from_file_location("train_dpo", ROOT / "train" / "train_dpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("train_lora", None)


@pytest.fixture
def data_dir(tmp_path):
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text("{}\n")
    return tmp_path


def test_helpers_are_shared_with_the_lora_script(train_dpo):
    for helper in ("count_long_examples", "library_versions", "load_config", "push_adapter"):
        assert getattr(train_dpo, helper).__module__ == "train_lora"


def test_default_variant_starts_from_the_system_d_adapter(train_dpo):
    yaml = pytest.importorskip("yaml")
    config = train_dpo.load_config(train_dpo.CONFIG_PATH, "beta01")
    system_d = yaml.safe_load((ROOT / "configs" / "systems.yaml").read_text())["systems"]["D"]

    assert config["model"]["name"] == system_d["model"]
    assert config["model"]["adapter"] == system_d["adapter"]
    training = config["training"]
    assert (training["beta"], training["learning_rate"], training["loss_type"]) == (
        0.1,
        5e-6,
        "sigmoid",
    )


def test_beta_variant_changes_only_beta(train_dpo):
    pytest.importorskip("yaml")
    default = train_dpo.load_config(train_dpo.CONFIG_PATH, "beta01")
    stronger = train_dpo.load_config(train_dpo.CONFIG_PATH, "beta03")

    assert stronger["training"]["beta"] == 0.3
    stronger["training"]["beta"] = 0.1
    assert stronger | {"variant": "beta01"} == default


def test_dpo_settings(train_dpo, data_dir, tmp_path):
    pytest.importorskip("yaml")
    config = train_dpo.load_config(train_dpo.CONFIG_PATH, "beta01")
    args = train_dpo.parse_args(["--data-dir", str(data_dir), "--output-dir", str(tmp_path)])
    args.report_to = "none"
    settings = train_dpo.dpo_settings(config, args)

    assert settings["beta"] == 0.1 and settings["loss_type"] == ["sigmoid"]
    assert settings["bf16"] is True and settings["max_length"] == 1024
    assert settings["max_steps"] == -1 and settings["save_strategy"] == "no"
    assert settings["output_dir"].endswith("d-dpo-beta01")


def test_dpo_settings_are_valid_trl_arguments(train_dpo, data_dir, tmp_path):
    pytest.importorskip("yaml")
    trl = pytest.importorskip("trl")
    config = train_dpo.load_config(train_dpo.CONFIG_PATH, "beta01")
    args = train_dpo.parse_args(["--data-dir", str(data_dir), "--output-dir", str(tmp_path)])
    args.report_to = "none"
    settings = train_dpo.dpo_settings(config, args) | {"bf16": False, "use_cpu": True}

    dpo_config = trl.DPOConfig(**settings)
    assert dpo_config.beta == 0.1 and dpo_config.loss_type == ["sigmoid"]


def test_run_name_defaults_to_the_variant(train_dpo, data_dir, tmp_path):
    args = train_dpo.parse_args(
        ["--variant", "beta03", "--data-dir", str(data_dir), "--output-dir", str(tmp_path)]
    )

    assert args.run_name == "d-dpo-beta03"


def test_arguments_are_checked(train_dpo, data_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    base = ["--data-dir", str(data_dir), "--output-dir", str(tmp_path)]

    with pytest.raises(SystemExit):
        train_dpo.parse_args([*base, "--push-to-hub"])
    with pytest.raises(SystemExit):
        train_dpo.parse_args([*base, "--push-to-hub", "--hub-repo", "user/adapter"])
    with pytest.raises(SystemExit):
        train_dpo.parse_args(["--data-dir", str(tmp_path / "empty"), "--output-dir", str(tmp_path)])
    (tmp_path / "d-dpo-beta01").mkdir()
    (tmp_path / "d-dpo-beta01" / "adapter_config.json").write_text("{}")
    with pytest.raises(SystemExit):
        train_dpo.parse_args(base)


class FakeTokenizer:
    """Chat template joins the contents; one token per word."""

    def apply_chat_template(self, messages, tokenize):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens):
        return {"input_ids": text.split()}


def test_long_pairs_counts_each_side(train_dpo):
    rows = [
        {
            "prompt": [{"content": "one two"}],
            "chosen": [{"content": "three four"}],
            "rejected": [{"content": "three"}],
        }
    ]

    assert train_dpo.long_pairs(FakeTokenizer(), rows, max_length=3) == {
        "chosen": 1,
        "rejected": 0,
    }
