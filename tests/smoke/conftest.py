"""The smoke tests run only when asked for, with `pytest -m smoke` (or `make smoke`).

They run the whole experiment runbook as subprocesses and take about a minute, so a plain
`pytest` leaves them out. A run with any `-m` expression selects them as usual.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "smoke: runs the experiment runbook end to end")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.option.markexpr:
        return
    smoke = [item for item in items if item.get_closest_marker("smoke")]
    if smoke:
        config.hook.pytest_deselected(items=smoke)
        items[:] = [item for item in items if not item.get_closest_marker("smoke")]
