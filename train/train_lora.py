"""Train one LoRA adapter for system C or D.

Reads one variant of `train/lora_config.yaml` and the SFT examples written by
`python -m explain_to_learn.train_data`. Training uses Unsloth by default, or plain
TRL + PEFT with `--backend peft`. The adapter is saved to `outputs/adapters/<run>/`
(git-ignored) with `run_info.json`: settings, data sizes, losses, peak GPU memory, run
time and library versions. Logging to Weights & Biases or MLflow is off unless chosen.
`--push-to-hub` uploads the adapter to a Hugging Face Hub repository, using the token
in the HF_TOKEN environment variable.

Runs are seeded: the same config, data and seed give the same data order and LoRA
initialisation. GPU kernels are not bit-for-bit deterministic, so losses can still
differ slightly between runs.

Usage:
    python train/train_lora.py --variant r16 --system d
    python train/train_lora.py --variant r16 --system d --max-steps 20     # quick check
    python train/train_lora.py --variant r16 --system d --report-to wandb \
        --push-to-hub --hub-repo <user>/explain-to-learn-d-r16
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("lora_config.yaml")
SYSTEMS = ("c", "d")
LIBRARIES = ("torch", "transformers", "trl", "peft", "unsloth", "bitsandbytes")


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `base` with the keys of `override` replaced, nested dicts merged."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, variant: str) -> dict[str, Any]:
    """Load the defaults and apply one named variant on top of them."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    variants = raw["variants"]
    if variant not in variants:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(variants)}")
    return merge(raw["defaults"], variants[variant] or {}) | {"variant": variant}


def count_long_examples(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> int:
    """Count examples whose prompt and completion are longer than `max_length` tokens."""
    count = 0
    for row in rows:
        text = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
        count += len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_length
    return count


def library_versions() -> dict[str, str]:
    """Installed versions of the main training libraries."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def load_with_unsloth(config: dict[str, Any]) -> tuple[Any, Any, None]:
    """Load the base model with Unsloth and add LoRA adapters to it."""
    import torch
    from unsloth import FastLanguageModel

    model_cfg, lora = config["model"], config["lora"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=model_cfg["load_in_4bit"],
        load_in_16bit=not model_cfg["load_in_4bit"],
        use_exact_model_name=True,  # the named checkpoint, not an Unsloth copy of it
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        use_gradient_checkpointing=config["training"]["gradient_checkpointing"],
        random_state=config["seed"],
    )
    return model, tokenizer, None


def load_with_peft(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Load the base model with Transformers; TRL adds the LoRA adapters from the config."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg, lora = config["model"], config["lora"]
    quantization = None
    if model_cfg["load_in_4bit"]:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], dtype=torch.bfloat16, quantization_config=quantization, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    peft_config = LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, peft_config


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Train one adapter, save it with `run_info.json` and return the run information."""
    import torch

    if args.backend == "unsloth":
        import unsloth  # noqa: F401  (must be imported before Transformers and TRL)
    from datasets import load_dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found; see docs/gpu_setup.md.")
    set_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    model_cfg, training = config["model"], config["training"]
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", config["logging"]["project"])
    elif args.report_to == "mlflow":
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", config["logging"]["project"])

    load = load_with_unsloth if args.backend == "unsloth" else load_with_peft
    model, tokenizer, peft_config = load(config)

    data_dir = args.data_dir / f"system_{args.system}"
    files = {split: str(data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    data = load_dataset("json", data_files=files).select_columns(["prompt", "completion"])
    long_examples = sum(
        count_long_examples(tokenizer, data[split].to_list(), model_cfg["max_seq_length"])
        for split in files
    )

    sft_config = SFTConfig(
        output_dir=str(args.checkpoint_dir / args.run_name),
        run_name=args.run_name,
        seed=config["seed"],
        data_seed=config["seed"],
        num_train_epochs=training["epochs"],
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=training["batch_size"],
        per_device_eval_batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["lr_scheduler"],
        warmup_steps=training["warmup_ratio"],  # a value below 1 is a share of all steps
        weight_decay=training["weight_decay"],
        optim=training["optimizer"],
        bf16=True,
        gradient_checkpointing=args.backend == "peft",  # Unsloth sets up its own
        max_length=model_cfg["max_seq_length"],
        completion_only_loss=True,  # loss on the JSON answer only, not on the prompt
        logging_steps=training["logging_steps"],
        eval_strategy="epoch",
        save_strategy="no",
        report_to=args.report_to,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    start = time.perf_counter()
    result = trainer.train()
    train_minutes = (time.perf_counter() - start) / 60
    eval_loss = trainer.evaluate()["eval_loss"]

    adapter_dir = args.output_dir / args.run_name
    trainer.model.save_pretrained(str(adapter_dir))
    info = {
        "run": args.run_name,
        "system": args.system,
        "variant": config["variant"],
        "backend": args.backend,
        "config": config,
        "examples": {
            "train": len(data["train"]),
            "validation": len(data["validation"]),
            "longer_than_max_seq_length": long_examples,
        },
        "train_loss": result.training_loss,
        "eval_loss": eval_loss,
        "train_minutes": round(train_minutes, 1),
        "peak_gpu_memory_gb": {
            "allocated": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "reserved": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        },
        "gpu": torch.cuda.get_device_name(0),
        "versions": library_versions(),
    }
    (adapter_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def push_adapter(adapter_dir: Path, repo_id: str, private: bool) -> str:
    """Upload the adapter folder to a Hugging Face Hub model repository."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        commit_message=f"Upload adapter {adapter_dir.name}",
    )
    return f"https://huggingface.co/{repo_id}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and check the command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for system C or D.")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--variant", default="r16", help="a key under `variants` in the config")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sft"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adapters"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--run-name", help="default: <system>-<variant>, e.g. d-r16")
    parser.add_argument("--backend", choices=("unsloth", "peft"), default="unsloth")
    parser.add_argument("--report-to", choices=("none", "wandb", "mlflow"))
    parser.add_argument("--max-steps", type=int, help="stop early, for a quick check")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", help="Hub repository ID, e.g. <user>/explain-to-learn-d-r16")
    parser.add_argument("--public", action="store_true", help="make a new Hub repository public")
    args = parser.parse_args(argv)

    args.run_name = args.run_name or f"{args.system}-{args.variant}"
    if args.push_to_hub and not args.hub_repo:
        parser.error("--push-to-hub needs --hub-repo")
    if args.push_to_hub and not os.environ.get("HF_TOKEN"):
        parser.error("--push-to-hub needs a Hugging Face token in HF_TOKEN")
    if (args.output_dir / args.run_name / "adapter_config.json").exists():
        parser.error(f"{args.output_dir / args.run_name} already exists; choose another --run-name")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.variant)
    args.report_to = args.report_to or config["logging"]["report_to"]

    info = train(config, args)
    adapter_dir = args.output_dir / args.run_name
    memory = info["peak_gpu_memory_gb"]
    print(f"Adapter: {adapter_dir}")
    print(f"Train loss {info['train_loss']:.4f}, validation loss {info['eval_loss']:.4f}")
    print(f"Peak GPU memory: {memory['reserved']} GB reserved, {memory['allocated']} GB allocated")
    print(f"Training time: {info['train_minutes']} min")
    if info["examples"]["longer_than_max_seq_length"]:
        print(f"Truncated examples: {info['examples']['longer_than_max_seq_length']}")
    if args.push_to_hub:
        print(f"Pushed to {push_adapter(adapter_dir, args.hub_repo, private=not args.public)}")


if __name__ == "__main__":
    main()
