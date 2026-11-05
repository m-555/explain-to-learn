"""Local annotation app for the teacher's work on the evaluation and training data.

Tasks, one tab each (see `docs/annotation_guide.md`):

- gold explanations: reference explanations and hints for the 300 gold edits;
- blind ratings: the rating sheet, rated on the rubric without system names;
- second rater: the 50-item subset sheet, rated under the second rater's name;
- silver hand-check: mark a sample of silver training items correct or not;
- preference pairs: choose the better of two outputs for the same edit.

Every task saves after each item to `outputs/annotations/` (git-ignored, because the
records quote learner text) and resumes at the first item not yet saved.

Usage:
    python -m explain_to_learn.annotation
"""
