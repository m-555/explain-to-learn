#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server in the official vllm/vllm-openai Docker image.
#
# Usage:
#   bash scripts/serve_model.sh MODEL [NAME=ADAPTER ...]
#
# MODEL is a Hugging Face model ID (for example the base model of systems A-D, or the
# teacher model). Each NAME=ADAPTER pair serves a LoRA adapter under NAME; ADAPTER is a
# local adapter directory (mounted read-only) or a Hugging Face Hub repository ID.
# Clients select the base model by its ID and an adapter by its NAME, in the `model`
# field of a request to http://localhost:$PORT/v1.
#
# Examples:
#   bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507
#   bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507 \
#     system-c=outputs/adapters/c-r16 system-d=outputs/adapters/d-r16
#
# Settings (environment variables):
#   PORT                    host port (default 8000)
#   VLLM_IMAGE              image and tag (default vllm/vllm-openai:v0.33.0)
#   MAX_MODEL_LEN           context length in tokens (default 4096)
#   GPU_MEMORY_UTILIZATION  share of GPU memory vLLM may use (default 0.90)
#   MAX_LORA_RANK           largest adapter rank to accept (default 32)
#   HF_CACHE                host Hugging Face cache (default ~/.cache/huggingface)
#   HF_TOKEN                passed to the container for gated or private models
#   API_KEY                 if set, clients must send it as a bearer token
set -euo pipefail

if (( $# < 1 )); then
  sed -n '2,12p' "$0"
  exit 1
fi

MODEL="$1"
shift
PORT="${PORT:-8000}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.33.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE"
docker_args=(
  --rm --gpus all --ipc=host
  -p "${PORT}:8000"
  -v "${HF_CACHE}:/root/.cache/huggingface"
)
vllm_args=(
  --model "$MODEL"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(--env HF_TOKEN)
fi
if [[ -n "${API_KEY:-}" ]]; then
  vllm_args+=(--api-key "$API_KEY")
fi

if (( $# > 0 )); then
  vllm_args+=(--enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras "$#" --lora-modules)
  for pair in "$@"; do
    name="${pair%%=*}"
    adapter="${pair#*=}"
    if [[ "$name" == "$pair" || -z "$name" || -z "$adapter" ]]; then
      echo "Adapters must be given as NAME=ADAPTER, got: $pair" >&2
      exit 1
    fi
    if [[ -d "$adapter" ]]; then
      docker_args+=(-v "$(realpath "$adapter"):/adapters/${name}:ro")
      adapter="/adapters/${name}"
    fi
    vllm_args+=("${name}=${adapter}")
  done
fi

echo "Serving $MODEL on http://localhost:${PORT}/v1 with $VLLM_IMAGE"
exec docker run "${docker_args[@]}" "$VLLM_IMAGE" "${vllm_args[@]}"
