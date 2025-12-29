"""
Pytest configuration and fixtures for BPE tokenizer tests.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import pytest


class Snapshot:
    """Snapshot testing utility for arbitrary data types using pickle format."""

    def __init__(
        self,
        snapshot_dir: str = "tests/_snapshots",
        default_force_update: bool = False,
        default_test_name: str | None = None,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.default_force_update = default_force_update
        self.default_test_name = default_test_name

    def _get_snapshot_path(self, test_name: str) -> Path:
        return self.snapshot_dir / f"{test_name}.pkl"

    def assert_match(
        self,
        actual: Any,
        test_name: str | None = None,
        force_update: bool | None = None,
    ):
        """
        Assert that the actual data matches the snapshot.

        Args:
            actual: Single object or dictionary of named objects
            test_name: The name of the test (used for the snapshot file)
            force_update: If True, update the snapshot instead of comparing
        """
        if force_update is None:
            force_update = self.default_force_update
        if test_name is None:
            assert self.default_test_name is not None, "Test name must be provided or set as default"
            test_name = self.default_test_name

        snapshot_path = self._get_snapshot_path(test_name)

        if force_update:
            # Update mode: save the actual data as the new snapshot
            with open(snapshot_path, "wb") as f:
                pickle.dump(actual, f)
            return

        # Load the snapshot
        with open(snapshot_path, "rb") as f:
            expected_data = pickle.load(f)

        if isinstance(actual, dict):
            for key in actual:
                if key not in expected_data:
                    raise AssertionError(f"Key '{key}' not found in snapshot for {test_name}")
                assert actual[key] == expected_data[key], (
                    f"Data for key '{key}' does not match snapshot for {test_name}"
                )
        else:
            assert actual == expected_data, f"Data does not match snapshot for {test_name}"


@pytest.fixture
def snapshot(request):
    """
    Fixture providing snapshot testing functionality.

    Usage:
        def test_my_function(snapshot):
            result = my_function()
            snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    # Create the snapshot handler with default settings
    snapshot_handler = Snapshot(default_force_update=force_update, default_test_name=request.node.name)

    return snapshot_handler
