---
title: Explain-to-Learn
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
short_description: Level-matched feedback on English grammar edits
license: other
---

# Explain-to-Learn

Short explanations or hints for grammar corrections in English learners' writing,
matched to the learner's CEFR level and grounded in a teacher-built grammar graph.
Choose a level and a mode (explain or hint), enter a sentence, and the demo corrects
it, finds the edits, and explains each one. The teacher-summary tab groups a text's
errors by grammar concept in a suggested teaching order.

Code, evaluation and documentation: <https://github.com/m-555/explain-to-learn>

For non-commercial research and education only: the explanation adapter is trained on
data derived from W&I+LOCNESS and FCE, and the correction model on cLang-8
(CC BY-NC-SA 4.0). See the
[licence notes](https://github.com/m-555/explain-to-learn/blob/main/docs/licences.md).
