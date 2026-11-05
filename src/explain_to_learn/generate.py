"""Generation: run a base or LoRA-tuned model on edits and parse its JSON output.

The output for each edit is a JSON object with the fields `concept`, `explanation`
and `hint` (see `prompts.py`).

Backends
--------
- `OpenAIBackend`: any OpenAI-compatible chat completions endpoint (vLLM, the
  llama.cpp server, Ollama). Needs only the standard library.
- `TransformersBackend`: a local Hugging Face model, optionally with a PEFT (LoRA)
  adapter. Needs the `model` extra and a CUDA build of PyTorch.

`generate` calls a backend with deterministic decoding by default, retries once when
the reply is not valid JSON, and records errors instead of raising, so that a failed
edit counts against coverage rather than stopping a run.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from explain_to_learn.edits import Edit
from explain_to_learn.io import read_rows
from explain_to_learn.prompts import build_messages, prompt_level

OUTPUT_FIELDS = ("concept", "explanation", "hint")
RETRY_MESSAGE = (
    "Your reply was not a valid JSON object with the keys concept, explanation and "
    "hint. Reply again with only that JSON object."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL = "<|channel|>final<|message|>"


@dataclass(frozen=True)
class Decoding:
    """Decoding settings; temperature 0 means greedy (deterministic) decoding."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    seed: int = 0


class Backend(Protocol):
    """Anything that turns chat messages into the model's reply text."""

    def complete(self, messages: list[dict], decoding: Decoding) -> str: ...


@dataclass
class Generation:
    """Result for one input: raw reply, parsed fields (or None) and timing."""

    raw: str
    parsed: dict | None
    attempts: int
    latency_s: float
    error: str | None = None


# --- Parsing -----------------------------------------------------------------------


def _strip_reasoning(text: str) -> str:
    """Remove thinking blocks and keep only the final channel of a harmony reply."""
    if _HARMONY_FINAL in text:
        text = text.split(_HARMONY_FINAL)[-1]
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = _THINK_BLOCK.sub("", text)
    for marker in ("<|return|>", "<|end|>", "<|im_end|>", "<|eot_id|>"):
        text = text.replace(marker, "")
    return text


def validate_output(obj: Any) -> dict | None:
    """Return the three output fields as stripped strings, or None if invalid.

    Valid means: a JSON object with all three keys, string (or null) values, a
    non-empty concept, and a non-empty explanation or hint. Extra keys are dropped.
    """
    if not isinstance(obj, dict) or any(key not in obj for key in OUTPUT_FIELDS):
        return None
    out = {}
    for key in OUTPUT_FIELDS:
        value = obj[key]
        if value is None:
            value = ""
        if not isinstance(value, str):
            return None
        out[key] = value.strip()
    if not out["concept"] or not (out["explanation"] or out["hint"]):
        return None
    return out


def _json_objects(text: str) -> Iterable[Any]:
    """Every JSON value that starts at a `{` in `text`, in order."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        yield obj


def parse_output(text: str | None) -> dict | None:
    """Extract and validate {"concept", "explanation", "hint"} from a model reply.

    Handles thinking tags, gpt-oss harmony channels, code fences and text around the
    JSON. Returns the first valid object, or None if there is none.
    """
    if not text:
        return None
    text = _strip_reasoning(text)
    candidates = [*_FENCE.findall(text), text]
    for candidate in candidates:
        for obj in _json_objects(candidate):
            parsed = validate_output(obj)
            if parsed is not None:
                return parsed
    return None


# --- Backends ----------------------------------------------------------------------


@dataclass
class OpenAIBackend:
    """Chat completions on an OpenAI-compatible server (vLLM, llama.cpp, Ollama)."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "none"
    timeout_s: float = 120.0
    extra_body: dict = field(default_factory=dict)

    def request_body(self, messages: list[dict], decoding: Decoding) -> dict:
        """The JSON body sent to `/chat/completions`."""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "max_tokens": decoding.max_new_tokens,
            "seed": decoding.seed,
            **self.extra_body,
        }

    def complete(self, messages: list[dict], decoding: Decoding) -> str:
        body = json.dumps(self.request_body(messages, decoding)).encode("utf-8")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            reply = json.loads(response.read().decode("utf-8"))
        return reply["choices"][0]["message"].get("content") or ""


class TransformersBackend:
    """A local Hugging Face causal LM, with an optional PEFT (LoRA) adapter."""

    def __init__(self, model: str, adapter: str | None = None, dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def complete(self, messages: list[dict], decoding: Decoding) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        sampling = (
            {"do_sample": True, "temperature": decoding.temperature, "top_p": decoding.top_p}
            if decoding.temperature > 0
            else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        )
        self.torch.manual_seed(decoding.seed)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=decoding.max_new_tokens, **sampling
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def make_backend(system: dict) -> Backend:
    """Build the backend named in a system definition from `configs/systems.yaml`.

    A server backend sends the key in the `API_KEY` environment variable, if set, as
    `scripts/serve_model.sh` expects.
    """
    kind = system["backend"]
    if kind == "openai":
        return OpenAIBackend(
            model=system.get("served_model", system["model"]),
            base_url=system.get("base_url", "http://localhost:8000/v1"),
            api_key=os.environ.get("API_KEY", "none"),
            timeout_s=system.get("timeout_s", 120.0),
            extra_body=system.get("extra_body") or {},
        )
    if kind == "transformers":
        return TransformersBackend(
            system["model"], system.get("adapter"), system.get("dtype", "bfloat16")
        )
    raise ValueError(f"Unknown backend {kind!r}; use 'openai' or 'transformers'")


def load_systems(path: str | Path = "configs/systems.yaml") -> dict[str, dict]:
    """Read the system definitions, with the shared defaults merged into each."""
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = config.get("defaults", {})
    systems = {}
    for name, system in config["systems"].items():
        merged = {**defaults, **system}
        merged["decoding"] = {**defaults.get("decoding", {}), **system.get("decoding", {})}
        systems[name] = merged
    return systems


# --- Running -----------------------------------------------------------------------


def generate(
    backend: Backend, messages: list[dict], decoding: Decoding | None = None, retries: int = 1
) -> Generation:
    """Call the backend; on invalid JSON ask again up to `retries` times.

    Errors from the backend (for example a dropped connection) are recorded in
    `Generation.error` and the result has `parsed=None`.
    """
    decoding = decoding or Decoding()
    start = time.perf_counter()
    raw, error, attempts = "", None, 0
    conversation = list(messages)
    while attempts <= retries:
        attempts += 1
        try:
            raw = backend.complete(conversation, decoding)
            error = None
        except Exception as exc:  # recorded, so that one failure does not stop a run
            raw, error = "", f"{type(exc).__name__}: {exc}"
            continue
        parsed = parse_output(raw)
        if parsed is not None:
            return Generation(raw, parsed, attempts, time.perf_counter() - start)
        conversation = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": RETRY_MESSAGE},
        ]
    return Generation(raw, None, attempts, time.perf_counter() - start, error or "invalid_json")


def edit_item(
    edit: Edit, mode: str, graph: Any = None, grounded: bool = False, **extra: Any
) -> tuple[dict, list[dict]]:
    """The generation record and chat messages for one edit.

    `graph` is any object with the `GrammarGraph` methods `concept_for` and
    `snippet`; it is optional for ungrounded systems. A grounded system gets no
    snippet for an error type without a concept, which `grounded` in the record
    shows. `extra` fields (for example `system` and `run`) are added to the record.
    """
    level = prompt_level(edit.cefr_level)
    concept_id = graph.concept_for(edit.error_type) if graph is not None else None
    snippet = graph.snippet(concept_id, level) if concept_id else None
    record = {
        **extra,
        "edit_id": edit.edit_id,
        "level": level,
        "mode": mode,
        "grounded": bool(grounded and snippet),
        "concept_id": concept_id,
        "concept_name": snippet["name"] if snippet else None,
    }
    return record, build_messages(edit, level, mode, snippet if grounded else None)


def done_ids(path: str | Path) -> set[str]:
    """Edit IDs already written to an output file, so a long run can resume."""
    if not Path(path).is_file():
        return set()
    return {row["edit_id"] for row in read_rows(path)}


def run_batch(
    backend: Backend,
    items: Iterable[tuple[dict, list[dict]]],
    out_path: str | Path,
    decoding: Decoding | None = None,
    retries: int = 1,
    workers: int = 1,
    on_result: Callable[[dict], None] | None = None,
) -> int:
    """Generate for (record, messages) pairs and append one JSON line per record.

    Each output line is the record (which must contain `edit_id`) plus `raw`,
    `parsed`, `attempts`, `latency_s` and `error`. Use `workers > 1` with a server
    backend so that it can batch requests. Returns the number of lines written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def run_one(item: tuple[dict, list[dict]]) -> dict:
        record, messages = item
        return {**record, **asdict(generate(backend, messages, decoding, retries))}

    written = 0
    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as pool:
        for row in pool.map(run_one, items):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if on_result is not None:
                on_result(row)
    return written
