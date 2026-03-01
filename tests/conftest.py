import logging
import pytest
import pickle
from pathlib import Path

DATA = Path(__file__).parent / "data"


def pytest_configure():
    """PytestHook: Runs FIRST — before any tests are collected."""
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)


@pytest.fixture
def tracking_result():
    with open(DATA / "tracking_result.pkl", "rb") as f:
        return pickle.load(f)
