# Demo

A Gradio app that gives feedback on one sentence at the learner's CEFR level:

1. **Correction.** An open grammatical error correction model rewrites the sentence. By
   default this is [CoEdIT large](https://huggingface.co/grammarly/coedit-large); with
   `--corrector model` the explanation model is prompted to correct instead.
2. **Edits.** [ERRANT](https://github.com/chrisjbryant/errant) splits the correction
   into typed edits.
3. **Grounding.** Each edit's error type is looked up in the grammar graph, which gives
   the concept and its rule worded for the learner's level.
4. **Feedback.** The explanation model (system D: the LoRA-tuned model with the graph
   snippet) writes an explanation, or a hint in hint mode. The prompt is built by the
   same code as in the evaluation (`generate.edit_item`).

`app.py` is the interface, and `pipeline.py` holds the steps. `--fake` replaces every
step with a stub (rule-based correction, a three-concept graph, placeholder feedback),
so the interface runs without a GPU, model files or ERRANT.

## Run locally on the RTX 5090

Set up the environment as in [docs/gpu_setup.md](../docs/gpu_setup.md), then install
the demo's packages from the repository root:

```bash
pip install -e ".[demo,data,graph,model]"
python -m spacy download en_core_web_sm
```

Preview the interface without models:

```bash
python demo/app.py --fake
```

Open <http://localhost:7860>. WSL2 forwards `localhost` to Windows, so a browser on the
Windows side works.

Run system D on the GPU, with the models loaded in the app:

```bash
python demo/app.py --adapter outputs/adapters/d-r16
```

The base model takes about 8 GB in bf16 and CoEdIT large about 3 GB, so both fit on the
32 GB card. To use the vLLM server instead, for example while it also serves evaluation
runs, start it with the adapter and point the app at it. Lower vLLM's memory share so
that the correction model still fits on the GPU:

```bash
GPU_MEMORY_UTILIZATION=0.8 bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507 system-d=outputs/adapters/d-r16
```

```bash
python demo/app.py --backend openai --model system-d
```

### Settings

Each option can be given on the command line or as an environment variable.

| Option | Variable | Default | Meaning |
| --- | --- | --- | --- |
| `--fake` | `DEMO_FAKE=1` | off | Stubs instead of models |
| `--backend` | `DEMO_BACKEND` | `transformers` | `transformers` (model in the app) or `openai` (a server) |
| `--model` | `DEMO_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Base model, or the served model name for `openai` |
| `--adapter` | `DEMO_ADAPTER` | none | LoRA adapter directory or Hub ID (`transformers` only) |
| `--base-url` | `DEMO_BASE_URL` | `http://localhost:8000/v1` | Server address for `openai` |
| `--corrector` | `DEMO_CORRECTOR` | `gec` | `gec` (CoEdIT) or `model` (prompt the explanation model) |
| `--gec-model` | `DEMO_GEC_MODEL` | `grammarly/coedit-large` | Correction model |
| `--graph` | `DEMO_GRAPH` | `grammar_graph` | Folder with `concepts.yaml` and `errant_map.yaml` |
| `--port` | `DEMO_PORT` | `7860` | Port of the app |

With `--backend openai`, the key in the `API_KEY` environment variable is sent to the
server, matching `API_KEY` in `scripts/serve_model.sh`.

## Deploy to a Hugging Face Space

A Space runs the app on Hugging Face's servers. It needs the app files, the grammar
graph, and the adapter on the Hugging Face Hub.

1. Push the system D adapter to the Hub. The training script does this with
   `--push-to-hub --hub-repo <user>/explain-to-learn-d-r16` and the token in `HF_TOKEN`.
2. Create a Space at <https://huggingface.co/new-space> with the Gradio SDK and the
   hardware you want (see below).
3. Copy the files into the Space repository, from the root of this repository:

   ```bash
   git clone https://huggingface.co/spaces/<user>/explain-to-learn space
   cp demo/app.py demo/pipeline.py demo/requirements.txt space/
   cp -r grammar_graph space/
   ```

4. Replace `space/README.md` with a Space card whose YAML header configures the app:

   ```yaml
   ---
   title: Explain-to-Learn
   sdk: gradio
   sdk_version: 6.31.0
   python_version: "3.12"
   app_file: app.py
   short_description: Level-matched feedback on English grammar edits
   ---
   ```

5. In the Space settings, under "Variables and secrets", add the variable
   `DEMO_ADAPTER=<user>/explain-to-learn-d-r16`. If the adapter repository is private,
   also add a secret `HF_TOKEN` with read access. Set `DEMO_FAKE=1` to run the preview
   instead.
6. Commit and push the Space repository (`git add`, `git commit`, `git push`). The Space
   builds from `requirements.txt` and starts the app.

`requirements.txt` installs this package from GitHub at `main`. Pin it to a release tag
or commit, so the Space keeps working when the repository changes.

### CPU or ZeroGPU

| | CPU basic | ZeroGPU |
| --- | --- | --- |
| Hardware | 2 vCPU, 16 GB RAM, free | Half of an NVIDIA RTX Pro 6000 Blackwell (48 GB) while a request runs |
| Who can host | Any account | Gradio Spaces only; free accounts can host a few ZeroGPU Spaces and PRO accounts more (see the [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) for current limits) |
| Good for | `DEMO_FAKE=1` preview, or `DEMO_BACKEND=openai` with a model server hosted elsewhere | The full pipeline with the model in the app |

- **CPU basic.** A 4B model generating on two CPU cores takes far too long for an
  interactive demo. Use the Space for the preview, or let it call a model server you
  host (`DEMO_BACKEND=openai`, `DEMO_BASE_URL`, `DEMO_MODEL=system-d`, and the
  `API_KEY` secret). The correction model and ERRANT still run on the CPU.
- **ZeroGPU.** The Space loads the models when it starts, and a GPU is attached only
  while a request runs. `app.py` marks the request handler with `@spaces.GPU`, which
  ZeroGPU needs; elsewhere the `spaces` package is absent and the handler runs as is.
  ZeroGPU supports a fixed list of PyTorch versions, so `requirements.txt` pins
  `torch==2.13.0`.

## Licences

The app code is MIT-licensed. The models it runs come with their own terms: the base
model (Qwen3-4B-Instruct-2507) is Apache-2.0, CoEdIT large is CC BY-NC 4.0, and the
adapter is trained on data derived from W&I+LOCNESS and FCE, which may be used for
non-commercial research and education only. A public Space with these models is
therefore for non-commercial use.
