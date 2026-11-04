---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
language:
  - en
license: other
tags:
  - lora
  - grammatical-error-explanation
  - language-learning
  - education
---

# Model card: Explain-to-Learn LoRA adapters

LoRA adapters for [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
that turn one grammar correction in an English learner's sentence into a short
explanation or a hint, matched to the learner's CEFR level. Two adapters are trained
with the same settings and data:

| Adapter | System | Input |
| --- | --- | --- |
| `c-r16` | C | Sentence, edit, error type, level and mode |
| `d-r16` | D | The same, plus the rule for the edit's concept from a teacher-built grammar graph |

Code, data preparation and evaluation: <https://github.com/m-555/explain-to-learn>.

> **Status:** the adapters are not trained yet. Sections that depend on training or
> evaluation say "Not available yet" and will be filled in from the run records.

## Model details

- **Developed by:** Mohsen Mirzaei
- **Model type:** LoRA adapter for a 4B-parameter decoder-only instruction model
- **Language:** English (feedback for learners of English)
- **Base model licence:** Apache-2.0
- **Adapter licence:** non-commercial research and education only, because the training
  data is derived from W&I+LOCNESS and FCE (see [Licences](#licences))
- **Repository:** <https://github.com/m-555/explain-to-learn>

## Intended use

- **Primary use:** research on grammar feedback for language learners: whether
  grounding in a teacher-built grammar graph makes a small fine-tuned model's
  explanations more correct and better matched to the learner's level.
- **Users:** researchers in NLP for education, and teachers trying the demo.
- **Setting:** feedback on corrections that are already known (a teacher's or a
  correction model's), shown to a learner as an explanation or a hint.

### Out of scope

- Grading, placement or any other decision about a learner.
- Unsupervised use with children; the training data is teen and adult exam-style
  writing.
- Languages other than English, and feedback on content, organisation or style.
- Finding errors: the model explains a given correction and does not check whether the
  correction is right.

## How to use

Input and output format: `src/explain_to_learn/prompts.py` builds the chat messages and
the expected answer, one JSON object with `concept`, `explanation` and `hint`. In explain
mode `hint` is empty; in hint mode `explanation` is empty.

Serve the base model with both adapters on an OpenAI-compatible vLLM server:

```bash
bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507 system-c=<user>/explain-to-learn-c-r16 system-d=<user>/explain-to-learn-d-r16
```

Or load an adapter with Transformers and PEFT:

```python
from explain_to_learn.generate import Decoding, TransformersBackend, generate

backend = TransformersBackend(
    "Qwen/Qwen3-4B-Instruct-2507", adapter="<user>/explain-to-learn-d-r16"
)
result = generate(backend, messages, Decoding())  # messages from generate.edit_item
print(result.parsed)
```

## Training data

Silver explanations: a larger teacher model (gpt-oss-20b, prompted with the grammar
graph) wrote an explanation or a hint for a seeded sample of edits from the training
essays of W&I+LOCNESS and FCE. Outputs were kept only if they parsed, named the graph
concept and named the edited words as the mode requires. See
[DATA_CARD.md](DATA_CARD.md) for the sources, filtering and splits.

| | Value |
| --- | --- |
| Training examples | Not available yet |
| Validation examples (5% of essays, held out by essay) | Not available yet |
| Share of examples in hint mode | Not available yet |
| Silver hand-check: share of sampled items marked correct (95% interval) | Not available yet |

## Training procedure

Supervised fine-tuning with LoRA, loss on the JSON answer only. Settings from
`train/lora_config.yaml` (variant `r16`):

| Setting | Value |
| --- | --- |
| Base weights | bf16, frozen |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q, k, v, o, gate, up and down projections |
| Learning rate and schedule | 2e-4, cosine, 3% warm-up |
| Epochs | 2 |
| Effective batch size | 32 (8 x 4 gradient accumulation) |
| Maximum sequence length | 1024 tokens |
| Seed | 42 |

Variants trained for comparison: rank 8 and 32 (alpha = 2 x rank) and 4-bit QLoRA at
rank 16.

### Training hardware and software

| | Value |
| --- | --- |
| GPU | One NVIDIA GeForce RTX 5090 (32 GB), Windows 10 host, WSL2 Ubuntu 24.04 |
| Software | PyTorch 2.14.1 (CUDA 13.0), Unsloth 2026.11.2, Transformers 5.17.0, TRL 1.15.0, PEFT 0.21.2 |
| Training time per adapter | Not available yet |
| Peak GPU memory | Not available yet |

## Evaluation

Systems A (base model, prompted), B (base model with the graph snippet), C and D are
compared on 300 gold edits from the W&I+LOCNESS development set, with gold corrections:

- **Automatic metrics:** coverage, concept accuracy, faithfulness and level fit, with
  paired bootstrap confidence intervals.
- **Blind teacher ratings:** 100 edits x 4 systems, shuffled with system names hidden,
  on the rubric in `eval/rubric.md` (rule correct, error located, helps the learner fix
  it, fits the level, invented rule).
- **Agreement:** a second teacher rates 50 items (Cohen's kappa), and an LLM judge with
  the same rubric is compared with the human ratings.

### Results

Not available yet.

## Limitations

- English only, on teen and adult exam-style writing.
- The main evaluation uses gold corrections; in real use, correction errors carry
  through to the explanations.
- The silver training data comes from a larger model and may carry its errors and
  biases; the hand-check covers a sample only.
- The grammar graph has about 50 concepts. Error types without a concept get no
  grounding, and their feedback can be generic.
- One main rater; agreement is measured on a subset.
- The model can still state wrong or invented rules. Feedback should be checked by a
  teacher before it is used with learners.

## Licences

- **Base model:** Qwen3-4B-Instruct-2507, Apache-2.0.
- **Training data:** derived from W&I+LOCNESS and FCE (BEA-2019 versions), which may be
  used for non-commercial research and education only. The adapters inherit this
  restriction.
- **Teacher model:** gpt-oss-20b, Apache-2.0.
- **Code:** MIT.

## Citation

Not available yet.
