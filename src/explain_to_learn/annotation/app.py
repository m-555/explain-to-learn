"""Gradio interface of the annotation app: one tab per task.

Every tab works the same way: press Start (after typing a rater name where one is
needed), and the task opens at the first item not yet saved. Each item has a progress
line, the item itself, a form, and two buttons: Back, and Save and next. The tasks
(`gold.py`, `ratings.py`, `handcheck.py`, `preferences.py`) hold all the logic and are
tested without a browser; this module only lays out the components and connects them.

Keyboard shortcuts (`SHORTCUTS_JS`) act on the visible tab. A digit chooses the option
whose label starts with that digit, in the first question of the item that has no answer
yet. Enter saves and goes to the next item, `b` goes back, and Ctrl+Enter (Cmd+Enter on
a Mac) saves even while typing in a text box. Digits, Enter and `b` are ignored while
typing, so they never replace text.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import gradio as gr

from explain_to_learn.annotation.gold import GoldTask
from explain_to_learn.annotation.handcheck import CRITERION, MARK_CHOICES, HandcheckTask
from explain_to_learn.annotation.preferences import CHOICES, PreferenceTask
from explain_to_learn.annotation.ratings import (
    FLAG_CHOICES,
    LABELS,
    QUESTIONS,
    SCALE_CHOICES,
    RatingTask,
    read_rubric,
)
from explain_to_learn.annotation.task import Task
from explain_to_learn.human_eval import DIMENSIONS, FLAG

NOT_STARTED = "Press Start first."
SHORTCUT_HELP = (
    "**Keys:** a number chooses the option that starts with it, in the first open question "
    "· **Enter** save and next · **b** back · **Ctrl+Enter** save while typing"
)
SHORTCUTS_JS = """
() => {
  if (window.annotationShortcuts) return;
  window.annotationShortcuts = true;
  const visible = (element) => element.offsetParent !== null;
  const typing = (element) =>
    element.closest("textarea, select, [contenteditable='true'], " +
                    "input:not([type=radio]):not([type=checkbox])");
  const press = (scope, name, event) => {
    const element = scope.querySelector("." + name);
    const button = element &&
      (element.matches("button") ? element : element.querySelector("button"));
    if (button) { button.click(); event.preventDefault(); }
  };
  document.addEventListener("keydown", (event) => {
    const scope = [...document.querySelectorAll(".shortcut-scope")].find(visible);
    if (!scope || event.altKey) return;
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      return press(scope, "shortcut-save", event);
    }
    if (event.ctrlKey || event.metaKey || typing(event.target)) return;
    if (event.key === "Enter" && !event.target.closest("button")) {
      return press(scope, "shortcut-save", event);
    }
    if (event.key === "b") return press(scope, "shortcut-back", event);
    if (!/^[0-9]$/.test(event.key)) return;
    const open = [...scope.querySelectorAll(".shortcut-choice")]
      .find((group) => visible(group) && !group.querySelector("input:checked"));
    const option = open && [...open.querySelectorAll("label")]
      .find((label) => label.textContent.trim().startsWith(event.key + " "));
    if (option) { option.click(); event.preventDefault(); }
  });
}
"""

FormUpdates = Callable[[Task, list[Any]], list[Any]]


def nav_buttons() -> tuple[gr.Button, gr.Button]:
    """The Back and Save-and-next buttons, marked for the keyboard shortcuts."""
    with gr.Row():
        back = gr.Button("Back (b)", elem_classes="shortcut-back")
        save = gr.Button("Save and next (Enter)", variant="primary", elem_classes="shortcut-save")
    return back, save


def connect(
    task_state: gr.State,
    index_state: gr.State,
    start: gr.Button,
    start_inputs: list[gr.components.Component],
    make_task: Callable[..., Task],
    work: gr.Column,
    status: gr.Markdown,
    panels: list[gr.Markdown],
    form: list[gr.components.Component],
    form_updates: FormUpdates | None = None,
) -> None:
    """Connect a tab's Start, Back and Save-and-next buttons to its task.

    `make_task` gets the values of `start_inputs`. `form_updates` can turn the form
    values into component updates, for example to set a dropdown's choices as well.
    """
    back, save = nav_buttons()
    outputs = [index_state, status, *panels, *form]

    def show(task: Task, index: int) -> list[Any]:
        values = task.values(index)
        if form_updates is not None:
            values = form_updates(task, values)
        return [index, task.status(index), *task.view(index), *values]

    def begin(*inputs: Any) -> list[Any]:
        try:
            task = make_task(*inputs)
        except (OSError, ValueError, ImportError) as exc:
            raise gr.Error(str(exc)) from exc
        return [task, gr.update(visible=True), *show(task, task.start_index())]

    def go_back(task: Task | None, index: int) -> list[Any]:
        if task is None:
            raise gr.Error(NOT_STARTED)
        return show(task, max(index - 1, 0))

    def save_and_next(task: Task | None, index: int, *values: Any) -> list[Any]:
        if task is None:
            raise gr.Error(NOT_STARTED)
        try:
            task.save(index, *values)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return show(task, min(index + 1, len(task) - 1))

    start.click(begin, start_inputs, [task_state, work, *outputs])
    back.click(go_back, [task_state, index_state], outputs)
    save.click(save_and_next, [task_state, index_state, *form], outputs)


def gold_tab(args: argparse.Namespace) -> None:
    """Task a: reference explanations and hints for the gold edits."""
    task_state, index_state = gr.State(None), gr.State(0)
    drafts = ", ".join(f"`{path}`" for path in args.drafts) or "none"
    gr.Markdown(
        f"Gold edits from `{args.gold_ids}`, model drafts: {drafts}. "
        f"Saved to `{args.out_dir / 'gold_explanations.jsonl'}`."
    )
    start = gr.Button("Start", variant="primary")
    with gr.Column(visible=False, elem_classes="shortcut-scope") as work:
        status = gr.Markdown()
        with gr.Row():
            with gr.Column():
                item, concept_info, draft_info = gr.Markdown(), gr.Markdown(), gr.Markdown()
            with gr.Column():
                concept = gr.Dropdown(label="Concept", choices=[])
                explanation = gr.Textbox(label="Reference explanation", lines=3)
                hint = gr.Textbox(label="Hint", lines=2)
                use_draft = gr.Button(
                    "Fill empty boxes from the model draft", size="sm", visible=bool(args.drafts)
                )
                notes = gr.Textbox(label="Notes", lines=2)
                not_rule = gr.Checkbox(label="Not explainable by a rule (say why in the notes)")
                from_draft = gr.Checkbox(label="Started from the model draft", interactive=False)
        connect(
            task_state,
            index_state,
            start,
            [],
            lambda: GoldTask(args.gold_ids, args.edits, args.graph, args.out_dir, args.drafts),
            work,
            status,
            [item, concept_info, draft_info],
            [concept, explanation, hint, notes, not_rule, from_draft],
            lambda task, values: [
                gr.update(choices=task.concept_choices(), value=values[0]),
                *values[1:],
            ],
        )
        gr.Markdown(SHORTCUT_HELP)

    def concept_changed(task: GoldTask | None, index: int, concept_id: str | None) -> str:
        return task.concept_view(index, concept_id) if task else ""

    def fill(
        task: GoldTask | None, index: int, text: str, hint_text: str, used: bool
    ) -> tuple[str, str, bool]:
        if task is None:
            raise gr.Error(NOT_STARTED)
        return task.fill_from_draft(index, text, hint_text, used)

    concept.input(concept_changed, [task_state, index_state, concept], concept_info)
    use_draft.click(
        fill,
        [task_state, index_state, explanation, hint, from_draft],
        [explanation, hint, from_draft],
    )


def rating_tab(args: argparse.Namespace, sheet: Any, rater: str, intro: str) -> None:
    """Tasks b and c: rate a blind rating sheet under a rater's name."""
    task_state, index_state = gr.State(None), gr.State(0)
    gr.Markdown(f"{intro} Sheet: `{sheet}`. Saved to `{args.out_dir}/ratings_<name>.csv`.")
    with gr.Row():
        name = gr.Textbox(label="Rater name", value=rater, scale=3)
        start = gr.Button("Start", variant="primary", scale=1)
    with gr.Column(visible=False, elem_classes="shortcut-scope") as work:
        status = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                item = gr.Markdown()
                with gr.Accordion("Rubric", open=False):
                    gr.Markdown(read_rubric(args.rubric))
            with gr.Column(scale=2):
                ratings = [
                    gr.Radio(
                        SCALE_CHOICES,
                        label=LABELS[dim],
                        info=QUESTIONS[dim],
                        elem_classes="shortcut-choice",
                    )
                    for dim in DIMENSIONS
                ]
                flag = gr.Radio(
                    FLAG_CHOICES,
                    label=LABELS[FLAG],
                    info=QUESTIONS[FLAG],
                    elem_classes="shortcut-choice",
                )
                comment = gr.Textbox(label="Comment (needed if you leave a rating open)")
        connect(
            task_state,
            index_state,
            start,
            [name],
            lambda rater_name: RatingTask(sheet, args.out_dir, rater_name),
            work,
            status,
            [item],
            [*ratings, flag, comment],
        )
        gr.Markdown(SHORTCUT_HELP)


def silver_tab(args: argparse.Namespace) -> None:
    """Task d: mark a sample of silver training items correct or not."""
    task_state, index_state = gr.State(None), gr.State(0)
    gr.Markdown(
        f"Items from `{args.handcheck}`. Saved to `{args.out_dir / 'silver_handcheck.csv'}`."
        f"\n\n{CRITERION}"
    )
    start = gr.Button("Start", variant="primary")
    with gr.Column(visible=False, elem_classes="shortcut-scope") as work:
        status = gr.Markdown()
        with gr.Row():
            item = gr.Markdown()
            with gr.Column():
                mark = gr.Radio(MARK_CHOICES, label="Mark", elem_classes="shortcut-choice")
                note = gr.Textbox(label="Note (optional)", lines=2)
        connect(
            task_state,
            index_state,
            start,
            [],
            lambda: HandcheckTask(args.handcheck, args.out_dir),
            work,
            status,
            [item],
            [mark, note],
        )
        gr.Markdown(SHORTCUT_HELP)


def preferences_tab(args: argparse.Namespace) -> None:
    """Task e: choose the better of two outputs for the same edit."""
    task_state, index_state = gr.State(None), gr.State(0)
    sources = ", ".join(f"`{path}`" for _, path in args.pair) or "none (start with two --pair)"
    gr.Markdown(
        f"Run files: {sources}. The two outputs are shown without their source names. "
        f"Saved to `{args.out_dir / 'preferences.jsonl'}`."
    )
    start = gr.Button("Start", variant="primary")
    with gr.Column(visible=False, elem_classes="shortcut-scope") as work:
        status = gr.Markdown()
        item = gr.Markdown()
        with gr.Row():
            first, second = gr.Markdown(), gr.Markdown()
        choice = gr.Radio(CHOICES, label="Which is better?", elem_classes="shortcut-choice")
        connect(
            task_state,
            index_state,
            start,
            [],
            lambda: PreferenceTask(args.pair, args.edits, args.graph, args.out_dir, args.seed),
            work,
            status,
            [item, first, second],
            [choice],
        )
        gr.Markdown(SHORTCUT_HELP)


def build_app(args: argparse.Namespace) -> gr.Blocks:
    """The app with one tab for each task in `args.task`."""
    with gr.Blocks(title="Explain-to-Learn annotation") as app:
        gr.Markdown(
            "# Explain-to-Learn annotation\n\nEach item is saved when you press "
            "**Save and next**, and every task starts again at the first item not yet "
            "saved. See `docs/annotation_guide.md`."
        )
        with gr.Tabs():
            if "gold" in args.task:
                with gr.Tab("Gold explanations"):
                    gold_tab(args)
            if "ratings" in args.task:
                with gr.Tab("Blind ratings"):
                    rating_tab(args, args.sheet, args.rater, "Rate every item of the sheet.")
            if "second-rater" in args.task:
                with gr.Tab("Second rater"):
                    rating_tab(
                        args,
                        args.second_sheet,
                        args.second_rater,
                        "Type your name, then rate the 50 items of the second sheet.",
                    )
            if "silver" in args.task:
                with gr.Tab("Silver hand-check"):
                    silver_tab(args)
            if "preferences" in args.task:
                with gr.Tab("Preference pairs"):
                    preferences_tab(args)
        app.load(None, None, None, js=SHORTCUTS_JS)
    return app
