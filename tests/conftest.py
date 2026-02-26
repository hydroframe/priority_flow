from __future__ import annotations

import shutil
from pathlib import Path

import pytest


CORRECT_OUTPUT_DIR = Path(__file__).parent / "correct_output"
TEST_OUTPUT_DIR = Path(__file__).parent


@pytest.fixture
def test_output_dir(request) -> Path:
    """
    Create a per-test output directory under ``tests/correct_output`` and return it.

    The directory is named after the current test node, e.g.
    ``tests/correct_output/test_name``.
    """
    test_name = request.node.name
    output_dir = TEST_OUTPUT_DIR / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def cleanup_test_output_dir(test_output_dir: Path):
    """
    Clean up the per-test output directory created by ``test_output_dir``.

    Use this fixture together with ``test_output_dir`` in a test to ensure
    that the directory is removed after the test completes.
    """
    try:
        yield
    finally:
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
