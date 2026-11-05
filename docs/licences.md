# Licences

Every dataset, model, library and tool the project uses, its licence, and what that
means for each thing this repository releases. Each licence was read at its source
(the licence file shipped with the data, the model card's licence field, or the
repository's licence file) in November 2026. Where no licence could be found, the table
says so; the absence of a licence means no permission is granted.

## Datasets

| Dataset | Licence | Source | What it means here |
| --- | --- | --- | --- |
| Cambridge English Write & Improve (W&I), BEA-2019 v2.1 | Non-commercial research and education. Excerpts of under 100 words may be published. Publications must cite Yannakoudakis et al. (2018). | `licence.wi.txt` in the [BEA-2019 download](https://www.cl.cam.ac.uk/research/nl/bea2019st/) | Main source of edits and levels. Never redistributed; the repository holds IDs and aggregate counts only. |
| LOCNESS, BEA-2019 v2.1 | Non-commercial use only. Publications must credit the Centre for English Corpus Linguistics (CECL), UCLouvain, and a copy should be sent to the address in the licence. No part may be given to a third party without CECL's authorisation. | `license.locness.txt` in the same download | Native-writer essays in the dev split. Never redistributed and never quoted, not even as short excerpts. |
| FCE, BEA-2019 v2.1 | Non-commercial research and education. Excerpts of under 100 words may be published. Publications must cite Yannakoudakis et al. (2011). | `licence.txt` in the [BEA-2019 download](https://www.cl.cam.ac.uk/research/nl/bea2019st/) | Extra training edits. Never redistributed. |
| GEE data (Song et al., 2024) | No licence: the [repository](https://github.com/Yixiao-Song/GEE-with-LLMs) has no licence file and its README states no terms. | – | Downloaded locally as a comparison point and never committed. The tests use an invented file in the same format. |
| cLang-8 (training data of the correction model) | Data CC BY-NC-SA 4.0; code Apache-2.0. | [README of google-research-datasets/clang8](https://github.com/google-research-datasets/clang8) | Not used directly; it matters for the correction model below. |

## Models

| Model | Licence | Source | Role |
| --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 | Apache-2.0 | [model card and LICENSE](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE) | Base model of systems A-D |
| gpt-oss-20b | Apache-2.0 | [model card](https://huggingface.co/openai/gpt-oss-20b) | Reference system R, silver training data, LLM judge |
| Unbabel/gec-t5_small | Apache-2.0 according to its model card, which names cLang-8, CoNLL-2013 and CoNLL-2014 as its data | [model card](https://huggingface.co/Unbabel/gec-t5_small) | Correction step of the demo and the end-to-end pipeline |
| spaCy `en_core_web_sm` | MIT | [model card](https://huggingface.co/spacy/en_core_web_sm) | Tokenisation and tagging for ERRANT |

The terms of the CoNLL-2013 and CoNLL-2014 data named by the correction model's card
were not checked here; the reading below does not depend on them, because cLang-8
already sets the stricter condition.

## Libraries and tools

| Component | Licence | Source |
| --- | --- | --- |
| ERRANT | MIT | [LICENSE.md](https://github.com/chrisjbryant/errant/blob/main/LICENSE.md) |
| spaCy | MIT | [LICENSE](https://github.com/explosion/spaCy/blob/master/LICENSE) |
| PyTorch | BSD-style PyTorch licence; the wheel also bundles third-party components (PyPI lists the full expression) | [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Transformers, TRL, PEFT, Accelerate, Datasets | Apache-2.0 | [transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [trl](https://github.com/huggingface/trl/blob/main/LICENSE), [peft](https://github.com/huggingface/peft/blob/main/LICENSE), [accelerate](https://github.com/huggingface/accelerate/blob/main/LICENSE), [datasets](https://github.com/huggingface/datasets/blob/main/LICENSE) |
| Unsloth | Apache-2.0 | [LICENSE](https://github.com/unslothai/unsloth/blob/main/LICENSE) |
| bitsandbytes | MIT | [LICENSE](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/LICENSE) |
| vLLM | Apache-2.0 | [LICENSE](https://github.com/vllm-project/vllm/blob/main/LICENSE) |
| llama.cpp | MIT | [LICENSE](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) |
| Gradio | Apache-2.0 | [LICENSE](https://github.com/gradio-app/gradio/blob/main/LICENSE) |
| NetworkX | BSD-3-Clause | [LICENSE.txt](https://github.com/networkx/networkx/blob/main/LICENSE.txt) |
| RDFLib | BSD-3-Clause | [LICENSE](https://github.com/RDFLib/rdflib/blob/main/LICENSE) |
| PyYAML | MIT | [LICENSE](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| NumPy | BSD-3-Clause (with bundled components under compatible licences) | [LICENSE.txt](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Matplotlib | Matplotlib licence (based on the PSF licence) | [PyPI](https://pypi.org/project/matplotlib/) |
| JupyterLab | BSD-3-Clause | [PyPI](https://pypi.org/project/jupyterlab/) |
| pytest, Ruff | MIT | [pytest](https://pypi.org/project/pytest/), [ruff](https://pypi.org/project/ruff/) |
| ACL style files: `acl_natbib.bst` | LaTeX Project Public License (stated in the file's header) | [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files) |
| ACL style files: `acl.sty` | No licence stated; ACL distributes it as the official template for papers in its venues | [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files) |

Both ACL files are included in `report/` unchanged.

## What follows for each thing this repository releases

**Code (MIT).** Everything under `src/`, `scripts/`, `train/`, `demo/` and `tests/` is
the author's work under the MIT licence in [LICENSE](../LICENSE). It contains no corpus
text: the test corpora are invented.

**Grammar graph (MIT).** `grammar_graph/` is written by the author and released with
the code. Its examples are written by the author, not taken from the corpora.

**IDs and aggregate results.** `data/splits/` (essay and edit IDs) and `eval/results/`
(counts, metrics, ratings summaries) contain no learner text. They let others rebuild
the data after downloading the corpora under their own licences.

**LoRA adapters: non-commercial research and education only.** The base model is
Apache-2.0, but the adapters are trained on silver explanations of W&I and FCE edits.
Both licences restrict use to non-commercial research and education and exclude
"information derived from the dataset" from products that are sold. The adapters are
therefore released for non-commercial research and education only, as
[MODEL_CARD.md](../MODEL_CARD.md) states (`license: other`).

**Silver data, gold explanations and ratings: not released.** They quote learner
sentences, so they stay in `outputs/` and are never committed or uploaded. Only their
summaries are published.

**Demo and Hugging Face Space: non-commercial.** The demo runs the adapters above, so it
is non-commercial for that reason alone. Its correction model's card says Apache-2.0,
but the card names cLang-8 as training data, which is CC BY-NC-SA 4.0. Whether a
model trained on non-commercial data inherits the restriction is not settled; the safe
reading, used here, is that it does. The Space is therefore a non-commercial research
demo, as [demo/README.md](../demo/README.md) states.

**GEE.** Its data has no licence, so it is never redistributed. The comparison
publishes only aggregate scores, and the report cites the GEE paper.

**Excerpts in the README and the report.** W&I and FCE allow excerpts of under 100
words with a citation. LOCNESS allows no redistribution. Showcase examples therefore
come from W&I only, each with a short window of context around the edit
(`scripts/select_showcase_examples.py`), and the README and the report cite
Yannakoudakis et al. (2018).

**Obligations of the report and any other publication.**

- Cite Yannakoudakis et al. (2018) for W&I, Yannakoudakis et al. (2011) for FCE, and
  Bryant et al. (2019) for the BEA-2019 releases.
- Credit CECL, UCLouvain for LOCNESS, and send a copy of the publication to the
  address given in the LOCNESS licence.
- Cite ERRANT, the GEE paper and the BEA 2025 shared task where they are used.
