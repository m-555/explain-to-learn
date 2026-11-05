# Demo

A Gradio app with two tabs.

**Feedback** gives feedback on a short text at the learner's CEFR level:

1. **Correction.** With "Correct the text automatically" on, an open grammatical error
   correction model corrects each sentence. By default this is
   [Unbabel/gec-t5_small](https://huggingface.co/Unbabel/gec-t5_small); with
   `--corrector llm` the explanation model's base model is prompted to correct instead.
   With the option off, the user types the corrected text as well, and no correction
   model is used.
2. **Edits.** [ERRANT](https://github.com/chrisjbryant/errant) splits each correction
   into typed edits, in the same format as the edit table.
3. **Grounding.** Each edit's error type is looked up in the grammar graph, which gives
   the concept and its rule worded for the learner's level.
4. **Feedback.** The explanation model (system D by default: the LoRA-tuned model with
   the graph snippet) writes an explanation, or a hint in hint mode.

**Teacher summary** takes a learner's text (corrected automatically or by hand) and
groups its edits by grammar concept, in a suggested teaching order: a concept comes
after the concepts it builds on that also have errors. No explanation model is used.

Both tabs run `explain_to_learn.pipeline` and `explain_to_learn.summary`, the same code
as the `explain-to-learn` command. `app.py` is the interface. `stubs.py` holds the
stubs for preview mode (`--fake`): rule-based correction and edits, and placeholder
feedback, on the real grammar graph. Preview mode needs no GPU, model files or ERRANT.

## Run locally on the RTX 5090

Set up the environment as in [docs/gpu_setup.md](../docs/gpu_setup.md), then install
the demo's packages from the repository root:

```bash
pip install -e ".[demo,data,graph,generate,model]"
python -m spacy download en_core_web_sm
```

Preview the interface without models:

```bash
python demo/app.py --fake
```

Open <http://localhost:7860>. WSL2 forwards `localhost` to Windows, so a browser on the
Windows side works.

Run system D on the GPU, with the models loaded in the app. The system's model and
adapter come from `configs/systems.yaml` (`outputs/adapters/d-r16` after training):

```bash
python demo/app.py
```

The base model takes about 8 GB in bf16 and the correction model about 0.25 GB, so both
fit on the 32 GB card. To use the vLLM server instead, for example while it also serves
evaluation runs, start it with the adapter and point the app at it:

```bash
bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507 system-d=outputs/adapters/d-r16
```

```bash
python demo/app.py --backend openai
```

The app then asks the server at the system's `base_url` for the adapter named
`served_model` (`system-d`). The correction model stays in the app, on the GPU if
PyTorch sees one.

### Settings

Each option can be given on the command line or as an environment variable. Options
that are not set keep the system's settings in `configs/systems.yaml`.

| Option | Variable | Default | Meaning |
| --- | --- | --- | --- |
| `--fake` | `DEMO_FAKE=1` | off | Stubs instead of models |
| `--config` | `DEMO_CONFIG` | `configs/systems.yaml` | Systems and correctors |
| `--system` | `DEMO_SYSTEM` | `D` | Explanation system in the config |
| `--backend` | `DEMO_BACKEND` | from the config | `transformers` (model in the app) or `openai` (a server) |
| `--model` | `DEMO_MODEL` | from the config | Base model, or the served model name for `openai` |
| `--adapter` | `DEMO_ADAPTER` | from the config | LoRA adapter directory or Hub ID (`transformers` only) |
| `--base-url` | `DEMO_BASE_URL` | from the config | Server address for `openai` |
| `--corrector` | `DEMO_CORRECTOR` | `gec` | Corrector in the config: `gec` (Unbabel/gec-t5_small) or `llm` |
| `--graph` | `DEMO_GRAPH` | `grammar_graph` | Folder with `concepts.yaml` and `errant_map.yaml` |
| `--port` | `DEMO_PORT` | `7860` | Port of the app |

With `--backend openai` or `--corrector llm`, the key in the `API_KEY` environment
variable is sent to the server, matching `API_KEY` in `scripts/serve_model.sh`.

## Deploy to a Hugging Face Space

A Space runs the app on Hugging Face's servers. It needs the app files, the grammar
graph, the systems config, and the adapter on the Hugging Face Hub.

1. Push the system D adapter to the Hub. The training script does this with
   `--push-to-hub --hub-repo <user>/explain-to-learn-d-r16` and the token in `HF_TOKEN`.
2. Create a Space at <https://huggingface.co/new-space> with the Gradio SDK and the
   hardware you want (see below).
3. Copy the files into the Space repository, from the root of this repository:

   ```bash
   git clone https://huggingface.co/spaces/<user>/explain-to-learn space
   cp demo/app.py demo/stubs.py demo/requirements.txt space/
   cp -r grammar_graph space/
   mkdir -p space/configs && cp configs/systems.yaml space/configs/
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
  host (`DEMO_BACKEND=openai`, `DEMO_BASE_URL`, and the `API_KEY` secret). The
  correction model (60M parameters) and ERRANT run on the CPU, and the teacher summary
  needs nothing else.
- **ZeroGPU.** The Space loads the models when it starts, and a GPU is attached only
  while a request runs. `app.py` marks both request handlers with `@spaces.GPU`, which
  ZeroGPU needs; elsewhere the `spaces` package is absent and the handlers run as they
  are. ZeroGPU supports a fixed list of PyTorch versions, so `requirements.txt` pins
  `torch==2.13.0`.

## Licences

The app code is MIT-licensed. The models it runs come with their own terms: the base
model (Qwen3-4B-Instruct-2507) is Apache-2.0; the correction model
(Unbabel/gec-t5_small) is Apache-2.0 according to its model card, whose tags name the
cLang-8 data (CC BY-NC-SA 4.0); and the adapter is trained on data derived from
W&I+LOCNESS and FCE, which may be used for non-commercial research and education only.
A public Space with these models is therefore for non-commercial use.
