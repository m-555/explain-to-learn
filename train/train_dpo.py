"""DPO on top of the system D adapter, from the teachers' preference pairs.

Reads one variant of `train/dpo_config.yaml` and the pairs written by
`python -m explain_to_learn.preference` (`outputs/dpo/train.jsonl` and
`validation.jsonl`). The base model is loaded in bf16 with Transformers and the system D
adapter on top of it with PEFT, as trainable. TRL's DPOTrainer then copies that adapter
into a frozen "ref" adapter and uses it as the reference model, so the reference is
system D itself and no second model is loaded.

The trained adapter is saved to `outputs/adapters/<run>/` (git-ignored), without the
reference copy, with `run_info.json`: settings, data sizes, losses, reward accuracy and
margin on the validation pairs, peak GPU memory, run time and library versions. It loads
on the base model like the C and D adapters. Loading the config, the library versions,
counting long examples and the Hub upload reuse the helpers of `train/train_lora.py`.

Checked against TRL 1.15.0 and PEFT 0.21.2, the versions pinned in the `train` extra:
`DPOConfig` takes `beta`, `loss_type` (a list) and `max_length` (there is no separate
prompt length); `DPOTrainer(model, ref_model=None, args, train_dataset, eval_dataset,
processing_class)` reads conversational `prompt`, `chosen` and `rejected` message lists;
for a PEFT model with a trained adapter and no `ref_model` it adds the "ref" copy, which
needs PEFT 0.20 or newer. TRL 1.15.0 computes the log-probabilities with a Triton kernel,
so DPO needs a CUDA GPU; Triton comes with PyTorch's CUDA builds for Linux (WSL2 or the
training image, docs/gpu_setup.md).

Usage:
    python train/train_dpo.py --variant beta01
    python train/train_dpo.py --variant beta01 --max-steps 10     # quick check
    python train/train_dpo.py --variant beta01 --push-to-hub \
        --hub-repo <user>/explain-to-learn-d-dpo
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# train_lora.py sits next to this script, so Python finds it when the script runs.
from train_lora import count_long_examples, library_versions, load_config, push_adapter

CONFIG_PATH = Path(__file__).with_name("dpo_config.yaml")
SPLITS = ("train", "validation")


def dpo_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments of TRL's `DPOConfig` for one run."""
    training = config["training"]
    return {
        "output_dir": str(args.checkpoint_dir / args.run_name),
        "run_name": args.run_name,
        "seed": config["seed"],
        "data_seed": config["seed"],
        "num_train_epochs": training["epochs"],
        "max_steps": args.max_steps or -1,
        "per_device_train_batch_size": training["batch_size"],
        "per_device_eval_batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation"],
        "learning_rate": training["learning_rate"],
        "lr_scheduler_type": training["lr_scheduler"],
        "warmup_steps": training["warmup_ratio"],  # a value below 1 is a share of all steps
        "weight_decay": training["weight_decay"],
        "optim": training["optimizer"],
        "beta": training["beta"],
        "loss_type": [training["loss_type"]],
        "max_length": config["model"]["max_length"],
        "bf16": True,
        "gradient_checkpointing": training["gradient_checkpointing"],
        "logging_steps": training["logging_steps"],
        "eval_strategy": "epoch",
        "save_strategy": "no",
        "report_to": args.report_to,
    }


def long_pairs(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> dict[str, int]:
    """Number of chosen and of rejected answers whose prompt + answer exceed `max_length`."""
    return {
        side: count_long_examples(
            tokenizer, [{"prompt": r["prompt"], "completion": r[side]} for r in rows], max_length
        )
        for side in ("chosen", "rejected")
    }


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run DPO, save the adapter with `run_info.json` and return the run information."""
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found; see docs/gpu_setup.md.")
    set_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    model_cfg = config["model"]
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", config["logging"]["project"])
    elif args.report_to == "mlflow":
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", config["logging"]["project"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)

    files = {split: str(args.data_dir / f"{split}.jsonl") for split in SPLITS}
    data = load_dataset("json", data_files=files).select_columns(["prompt", "chosen", "rejected"])
    longer = {
        split: long_pairs(tokenizer, data[split].to_list(), model_cfg["max_length"])
        for split in SPLITS
    }

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(**dpo_settings(config, args)),
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
    )
    start = time.perf_counter()
    result = trainer.train()
    train_minutes = (time.perf_counter() - start) / 60
    evaluation = trainer.evaluate()

    adapter_dir = args.output_dir / args.run_name
    trainer.model.save_pretrained(str(adapter_dir), selected_adapters=["default"])
    info = {
        "run": args.run_name,
        "variant": config["variant"],
        "adapter_trained_from": args.adapter,
        "config": config,
        "pairs": {split: len(data[split]) for split in SPLITS},
        "longer_than_max_length": longer,
        "train_loss": result.training_loss,
        "eval_loss": evaluation["eval_loss"],
        "eval_reward_accuracy": evaluation.get("eval_rewards/accuracies"),
        "eval_reward_margin": evaluation.get("eval_rewards/margins"),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and check the command-line arguments."""
    parser = argparse.ArgumentParser(description="DPO on top of the system D adapter.")
    parser.add_argument("--variant", default="beta01", help="a key under `variants` in the config")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--adapter", help="the adapter to start from (default: from the config)")
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/dpo"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adapters"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--run-name", help="default: d-dpo-<variant>, e.g. d-dpo-beta01")
    parser.add_argument("--report-to", choices=("none", "wandb", "mlflow"))
    parser.add_argument("--max-steps", type=int, help="stop early, for a quick check")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", help="Hub repository ID, e.g. <user>/explain-to-learn-d-dpo")
    parser.add_argument("--public", action="store_true", help="make a new Hub repository public")
    args = parser.parse_args(argv)

    args.run_name = args.run_name or f"d-dpo-{args.variant}"
    if args.push_to_hub and not args.hub_repo:
        parser.error("--push-to-hub needs --hub-repo")
    if args.push_to_hub and not os.environ.get("HF_TOKEN"):
        parser.error("--push-to-hub needs a Hugging Face token in HF_TOKEN")
    if (args.output_dir / args.run_name / "adapter_config.json").exists():
        parser.error(f"{args.output_dir / args.run_name} already exists; choose another --run-name")
    missing = [s for s in SPLITS if not (args.data_dir / f"{s}.jsonl").is_file()]
    if missing:
        parser.error(f"no {' or '.join(missing)} pairs in {args.data_dir}; run preference.py first")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.variant)
    args.report_to = args.report_to or config["logging"]["report_to"]
    args.adapter = args.adapter or config["model"]["adapter"]

    info = train(config, args)
    adapter_dir = args.output_dir / args.run_name
    memory = info["peak_gpu_memory_gb"]
    print(f"Adapter: {adapter_dir}")
    print(f"Train loss {info['train_loss']:.4f}, validation loss {info['eval_loss']:.4f}")
    print(f"Validation reward accuracy {info['eval_reward_accuracy']}")
    print(f"Peak GPU memory: {memory['reserved']} GB reserved, {memory['allocated']} GB allocated")
    print(f"Training time: {info['train_minutes']} min")
    if any(count for split in info["longer_than_max_length"].values() for count in split.values()):
        print(f"Truncated answers: {info['longer_than_max_length']}")
    if args.push_to_hub:
        print(f"Pushed to {push_adapter(adapter_dir, args.hub_repo, private=not args.public)}")


if __name__ == "__main__":
    main()
