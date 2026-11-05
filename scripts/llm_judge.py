"""Rate the items of a rating sheet with an LLM judge that follows eval/rubric.md.

Each item is sent to an OpenAI-compatible chat completions endpoint, for example the
local vLLM server started with `scripts/serve_model.sh`. The full rubric is the system
message, and the judge must answer with one JSON object holding the four ratings and the
flag. Answers that are not valid JSON with allowed values are retried, then left empty
and counted. The output has the same rating columns as the human sheets, so
`scripts/agreement.py` and `explain_to_learn.human_eval` read it directly. A small
`.json` file next to it records the judge model and settings.

Usage:
    python scripts/llm_judge.py --model <model-id> \
        --sheet outputs/human_eval/rating_sheet.csv --out outputs/human_eval/ratings_llm_judge.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from explain_to_learn.human_eval import (
    DIMENSIONS,
    FLAG,
    FLAG_VALUES,
    RATING_COLUMNS,
    SCALE,
    SHEET_COLUMNS,
    read_csv,
    write_csv,
)

ITEM_FIELDS = SHEET_COLUMNS[1:]  # everything a rater sees except the item ID
INSTRUCTIONS = (
    "You are an experienced teacher of English as a foreign language. Rate the feedback "
    "item in the next message by following this rubric exactly.\n\n{rubric}\n\n"
    "Answer with one JSON object and nothing else. It must have the keys "
    + ", ".join(f'"{dim}"' for dim in DIMENSIONS)
    + ' (each "no", "partly" or "yes") and "'
    + FLAG
    + '" ("yes" or "no").'
)

Complete = Callable[[list[dict[str, str]]], str]


def judge_messages(rubric: str, item: dict[str, str]) -> list[dict[str, str]]:
    """The chat messages for one item: rubric as instructions, item fields as input."""
    lines = [f"{field}: {item.get(field, '')}" for field in ITEM_FIELDS]
    return [
        {"role": "system", "content": INSTRUCTIONS.format(rubric=rubric)},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_judgement(text: str) -> dict[str, str]:
    """Check the judge's answer: one JSON object with every rating in its allowed values.

    Raises `ValueError` otherwise. Values are returned lower-case; other keys are ignored.
    """
    answer = json.loads(text)
    if not isinstance(answer, dict):
        raise ValueError("the answer is not a JSON object")
    ratings = {}
    for column, allowed in [*((dim, SCALE) for dim in DIMENSIONS), (FLAG, FLAG_VALUES)]:
        value = str(answer.get(column, "")).strip().lower()
        if value not in allowed:
            raise ValueError(f"{column} must be one of {', '.join(allowed)}, got {value!r}")
        ratings[column] = value
    return ratings


def judge_item(
    complete: Complete, rubric: str, item: dict[str, str], retries: int
) -> dict[str, str]:
    """Ratings for one item; after `retries` failed retries, empty ratings and the error."""
    error = ""
    for _ in range(retries + 1):
        try:
            return parse_judgement(complete(judge_messages(rubric, item))) | {"comment": ""}
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            error = str(exc)
    return {column: "" for column in (*DIMENSIONS, FLAG)} | {"comment": f"judge failed: {error}"}


def endpoint_client(
    base_url: str,
    model: str,
    api_key: str,
    json_mode: bool,
    seed: int,
    timeout: float = 120,
    max_tokens: int = 200,
) -> Complete:
    """A function that sends chat messages to an OpenAI-compatible endpoint.

    For a reasoning model such as gpt-oss, the reasoning counts towards `max_tokens`.
    """

    def complete(messages: list[dict[str, str]]) -> str:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"]

    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate a rating sheet with an LLM judge.")
    parser.add_argument("--model", required=True, help="model name served by the endpoint")
    parser.add_argument("--sheet", type=Path, default=Path("outputs/human_eval/rating_sheet.csv"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/human_eval/ratings_llm_judge.csv")
    )
    parser.add_argument("--rubric", type=Path, default=Path("eval/rubric.md"))
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="answer length limit; raise it for gpt-oss"
    )
    parser.add_argument("--no-json-mode", action="store_true", help="do not send response_format")
    parser.add_argument("--limit", type=int, help="rate only the first N items, for a check")
    args = parser.parse_args()

    rubric = args.rubric.read_text(encoding="utf-8")
    items = read_csv(args.sheet)[: args.limit]
    complete = endpoint_client(
        args.base_url,
        args.model,
        args.api_key,
        not args.no_json_mode,
        args.seed,
        max_tokens=args.max_tokens,
    )
    rows = []
    for number, item in enumerate(items, start=1):
        rows.append({"item_id": item["item_id"]} | judge_item(complete, rubric, item, args.retries))
        if number % 50 == 0:
            print(f"{number}/{len(items)} items rated")
    write_csv(rows, args.out, ("item_id", *RATING_COLUMNS))

    failed = sum(row["comment"].startswith("judge failed") for row in rows)
    settings = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "json_mode": not args.no_json_mode,
        "rubric_sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest(),
        "items": len(rows),
        "failed": failed,
    }
    args.out.with_suffix(".json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"{len(rows)} items rated, {failed} failed -> {args.out}")


if __name__ == "__main__":
    main()
