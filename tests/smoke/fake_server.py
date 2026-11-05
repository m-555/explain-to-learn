"""A fake OpenAI-compatible chat completions server for the runbook smoke test.

It stands in for the vLLM servers of `scripts/serve_model.sh`, so that every script of
the runbook runs unchanged, without a GPU or a model. It answers `POST
/v1/chat/completions` and `GET /v1/models` with the standard library only, and tells
the three kinds of prompt apart by their system message:

- explanation prompts (`prompts.SYSTEM_PROMPT`, used by systems A to D, R and the
  pipeline): a JSON object built from the edit described in the user message. It quotes
  the edited words as the mode asks, names the graph concept when the prompt has one
  (otherwise the error category), and for system R arrives in a code fence. A fixed
  share of prompts, chosen by a hash, gets a reply that is not JSON: some only on the
  first attempt, so the retry path runs, and some on every attempt, so that coverage,
  the silver filter and the rating sheet meet replies that failed;
- judge prompts (`scripts/llm_judge.py`): the four ratings and the flag, chosen by a
  hash of the item;
- correction prompts (`pipeline.CORRECTION_PROMPT`): the sentence with a few fixed
  replacements, so that correction scores are not all zero.

Any other prompt gets HTTP 400, so a changed prompt fails the smoke test loudly. Every
answer depends only on the request, so runs are repeatable. The server keeps the kind
and model of each request (never its text) in `requests_seen`.

Usage, to try the runbook commands by hand against it:
    python tests/smoke/fake_server.py --port 8001
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from explain_to_learn.pipeline import CORRECTION_PROMPT
from explain_to_learn.prompts import SYSTEM_PROMPT

JUDGE_START = "You are an experienced teacher of English as a foreign language."
RATINGS = ("no", "partly", "yes")
FENCED_MODELS = {"openai/gpt-oss-20b"}  # system R's replies arrive in a code fence
NOT_JSON = "I am not sure which rule applies here."
# Replacements of the fake corrector; the first four occur in the smoke corpus.
CORRECTIONS = {
    "We was": "We were",
    "She have": "She has",
    "enviroment": "environment",
    "arguement": "argument",
    "She don't": "She doesn't",
    "to going": "to go",
}


def digest(*parts: str) -> bytes:
    """A stable hash of the parts, for choices that must not change between runs."""
    return hashlib.sha256("\n".join(parts).encode("utf-8")).digest()


def prompt_fields(user_message: str) -> dict[str, str]:
    """The `Name: value` lines of an explanation prompt, keyed by name without spaces."""
    fields = {}
    for line in user_message.splitlines():
        name, separator, value = line.partition(": ")
        if separator:
            fields[name.strip()] = value.strip()
    return fields


def explanation(messages: list[dict[str, str]], model: str) -> str:
    """The reply to an explanation prompt (see the module docstring)."""
    user = messages[1]["content"]  # the first user message, also in a retry
    choice = digest(model, user)[0] % 12
    if choice == 0 or (choice == 1 and len(messages) == 2):
        return NOT_JSON
    fields = prompt_fields(user)
    concept = fields.get("Concept") or fields["Error type"].partition(":")[2]
    action = fields["Correction"].split(" (tokens")[0]  # e.g. replace "do" with "does"
    words = re.findall(r'"(.*?)"', action)
    explanation_text = hint = ""
    if fields["Mode"] == "explain":
        if action.startswith("replace"):
            explanation_text = f'Write "{words[1]}" instead of "{words[0]}" here.'
        elif action.startswith("add"):
            explanation_text = f'Add "{words[0]}" here.'
        else:
            explanation_text = f'Remove "{words[0]}" here.'
        explanation_text += f" This is about {concept}."
    elif action.startswith("add"):
        # Point at the word next to the gap, as metrics.faithfulness expects of a hint.
        before, _, after = fields["In context"].partition("[+")
        neighbour = before.split()[-1] if before.split() else after.split("]")[1].split()[0]
        hint = f'Something is missing next to "{neighbour}"; think about {concept}.'
    else:
        hint = f'Look again at "{words[0]}" and think about {concept}.'
    reply = json.dumps(
        {"concept": concept, "explanation": explanation_text, "hint": hint}, ensure_ascii=False
    )
    return f"```json\n{reply}\n```" if model in FENCED_MODELS else reply


def judgement(messages: list[dict[str, str]]) -> str:
    """The reply to a judge prompt: four ratings and the flag, as JSON."""
    item = messages[-1]["content"]
    if not prompt_fields(item).get("feedback"):
        ratings = ["no"] * 4  # empty feedback, as the rubric says
    else:
        hashed = digest("judge", item)
        ratings = [RATINGS[hashed[i] % 3] for i in range(4)]
    names = ("rule_correct", "error_located", "helps_fix", "fits_level")
    return json.dumps(dict(zip(names, ratings, strict=True)) | {"invented_rule": "no"})


def correction(messages: list[dict[str, str]]) -> str:
    """The reply to a correction prompt: the sentence with the fixed replacements."""
    sentence = messages[-1]["content"]
    for wrong, right in CORRECTIONS.items():
        sentence = sentence.replace(wrong, right)
    return sentence


def answer(body: dict[str, Any]) -> tuple[str, str]:
    """The kind of prompt and the reply text; raises ValueError for an unknown prompt."""
    messages = body["messages"]
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    if system == SYSTEM_PROMPT:
        return "explanation", explanation(messages, body.get("model", ""))
    if system.startswith(JUDGE_START):
        return "judge", judgement(messages)
    if system == CORRECTION_PROMPT:
        return "correction", correction(messages)
    raise ValueError("unknown prompt: the system message matches no known script")


class FakeServer(ThreadingHTTPServer):
    """The HTTP server, with the kind and model of every request it answered."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, Handler)
        self.requests_seen: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


class Handler(BaseHTTPRequestHandler):
    """Answers the two endpoints the project's clients use."""

    server: FakeServer

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self.send_json(200, {"object": "list", "data": []})
        else:
            self.send_json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": f"no route {self.path}"}})
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            kind, reply = answer(body)
        except (ValueError, KeyError, IndexError) as error:
            self.send_json(400, {"error": {"message": str(error)}})
            return
        settings = sorted(key for key in body if key not in ("model", "messages"))
        with self.server.lock:
            self.server.requests_seen.append(
                {"kind": kind, "model": body.get("model"), "settings": settings}
            )
        self.send_json(
            200,
            {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        """Print nothing per request; the requests quote learner text."""


def start(host: str = "127.0.0.1", port: int = 0) -> FakeServer:
    """Start the server in a background thread; port 0 picks a free port."""
    server = FakeServer((host, port))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake OpenAI-compatible server for tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = FakeServer((args.host, args.port))
    print(f"Fake model server on {server.base_url} (Ctrl+C to stop)", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
