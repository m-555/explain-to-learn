"""Cohen's kappa on hand-computed cases, and the agreement table."""

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "agreement.py"


def load_script():
    spec = importlib.util.spec_from_file_location("agreement", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agreement = load_script()


def test_plain_kappa_two_categories():
    # 50 items: both yes 20, yes/no 5, no/yes 10, both no 15.
    # p_o = 35/50 = 0.7; p_e = 0.5 * 0.6 + 0.5 * 0.4 = 0.5; kappa = (0.7 - 0.5) / 0.5 = 0.4.
    first = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    second = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15

    assert agreement.cohen_kappa(first, second, [0, 1]) == pytest.approx(0.4)


def test_quadratic_kappa_three_categories():
    # Pairs (0,0), (1,2), (2,2), (2,1). Weights (i - j)^2 / 4.
    # Observed disagreement: (1/4 + 1/4) / 4 = 1/8.
    # Both raters' shares are (1/4, 1/4, 1/2); expected disagreement = 22/64.
    # Weighted kappa = 1 - (8/64) / (22/64) = 7/11.
    first, second = [0, 1, 2, 2], [0, 2, 2, 1]

    assert agreement.cohen_kappa(first, second, [0, 1, 2], quadratic=True) == pytest.approx(7 / 11)
    # Plain kappa: p_o = 1/2, p_e = 1/16 + 1/16 + 1/4 = 3/8, kappa = (1/8) / (5/8) = 1/5.
    assert agreement.cohen_kappa(first, second, [0, 1, 2]) == pytest.approx(0.2)


def test_quadratic_weights_punish_large_disagreements_more():
    near = agreement.cohen_kappa([0, 1, 2, 2], [1, 1, 2, 2], [0, 1, 2], quadratic=True)
    far = agreement.cohen_kappa([0, 1, 2, 2], [2, 1, 2, 2], [0, 1, 2], quadratic=True)

    assert near > far


def test_perfect_and_undefined_agreement():
    assert agreement.cohen_kappa([0, 1, 2], [0, 1, 2], [0, 1, 2]) == 1
    assert math.isnan(agreement.cohen_kappa([2, 2], [2, 2], [0, 1, 2]))
    assert math.isnan(agreement.cohen_kappa([], [], [0, 1, 2]))


def ratings(values: dict[str, int | None], flag: int | None = 0) -> dict:
    return {dim: values.get(dim) for dim in agreement.DIMENSIONS} | {"invented_rule": flag}


def test_agreement_rows_use_only_items_rated_by_both():
    first = {
        "item-1": ratings({"rule_correct": 2}),
        "item-2": ratings({"rule_correct": 0}),
        "item-3": ratings({"rule_correct": 1}),
        "item-4": ratings({"rule_correct": 2}),
    }
    second = {
        "item-1": ratings({"rule_correct": 2}),
        "item-2": ratings({"rule_correct": 0}),
        "item-3": ratings({"rule_correct": None}),
    }

    rows = {row[0]: row for row in agreement.agreement_rows(first, second)}
    assert rows["rule_correct"][:3] == ["rule_correct", 2, 1.0]
    assert rows["rule_correct"][3] == pytest.approx(1.0)
    assert rows["error_located"][1] == 0
    assert rows["invented_rule"][4] is None

    table = agreement.markdown("Rater 1 vs rater 2", agreement.agreement_rows(first, second))
    assert "| rule_correct | 2 | 1.00 | 1.00 | 1.00 |" in table
    assert "| error_located | 0 | n/a | n/a | n/a |" in table
    assert "| invented_rule | 3 | 1.00 | n/a | – |" in table
