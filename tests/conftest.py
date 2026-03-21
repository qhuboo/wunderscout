import logging
import pytest
import pickle
from pathlib import Path

DATA = Path(__file__).parent / "data"


def pytest_configure():
    """PytestHook: Runs FIRST — before any tests are collected."""
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


@pytest.fixture
def frames():
    with open(DATA / "frames.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def team_heatmaps():
    with open(DATA / "team_heatmaps.pkl", "rb") as f:
        return pickle.load(f)
