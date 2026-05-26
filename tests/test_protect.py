"""Tests for the protect module."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

import protect as p


# ============================================================================
# TransformLog
# ============================================================================


class TestTransformLog:
    def test_add_appends_entry(self):
        log = p.TransformLog()
        log.add(function="noise", columns=["income"], params={"scale": 0.05}, rows_affected=100)
        assert len(log.entries) == 1
        assert log.entries[0]["function"] == "noise"
        assert log.entries[0]["columns"] == ["income"]

    def test_to_text_lists_operations(self):
        log = p.TransformLog()
        log.add(function="noise", columns=["income"], params={"scale": 0.05}, rows_affected=100)
        log.add(function="winsorize", columns=["age"], params={"limits": (0.01, 0.99)}, rows_affected=50)
        text = log.to_text()
        assert "noise" in text
        assert "winsorize" in text
        assert "income" in text

    def test_to_json_roundtrip(self):
        log = p.TransformLog()
        log.add(function="bin", columns=["age"], params={"bins": 10}, rows_affected=200)
        data = json.loads(log.to_json())
        assert len(data["entries"]) == 1
        assert data["entries"][0]["function"] == "bin"

    def test_summary_counts_by_function(self):
        log = p.TransformLog()
        log.add(function="noise", columns=["a"], params={}, rows_affected=10)
        log.add(function="noise", columns=["b"], params={}, rows_affected=20)
        log.add(function="bin", columns=["c"], params={}, rows_affected=30)
        summary = log.summary()
        assert summary["by_function"]["noise"] == 2
        assert summary["by_function"]["bin"] == 1
        assert summary["total_operations"] == 3
