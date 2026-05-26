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


# ============================================================================
# Helpers
# ============================================================================


class TestHelpers:
    def test_resolve_random_state_int_seeds(self):
        rng1 = p.protect._resolve_random_state(42)
        rng2 = p.protect._resolve_random_state(42)
        assert rng1.integers(0, 100) == rng2.integers(0, 100)

    def test_resolve_random_state_none_is_random(self):
        rng = p.protect._resolve_random_state(None)
        assert isinstance(rng, np.random.Generator)

    def test_validate_columns_accepts_str(self, small_df):
        cols = p.protect._validate_columns(small_df, "age")
        assert cols == ["age"]

    def test_validate_columns_accepts_list(self, small_df):
        cols = p.protect._validate_columns(small_df, ["age", "income"])
        assert cols == ["age", "income"]

    def test_validate_columns_raises_on_missing(self, small_df):
        with pytest.raises(KeyError, match="not in DataFrame"):
            p.protect._validate_columns(small_df, "nonexistent")

    def test_select_share_returns_correct_count(self, small_df):
        rng = np.random.default_rng(42)
        mask = p.protect._select_share(small_df, share=0.5, unit_id=None, rng=rng)
        assert mask.sum() == 10

    def test_select_share_by_unit_selects_whole_units(self, small_df):
        rng = np.random.default_rng(42)
        mask = p.protect._select_share(small_df, share=0.5, unit_id="pid", rng=rng)
        for pid, grp in small_df.groupby("pid"):
            mask_for_unit = mask[grp.index]
            assert mask_for_unit.nunique() == 1

    def test_select_share_zero_is_no_op(self, small_df):
        rng = np.random.default_rng(42)
        mask = p.protect._select_share(small_df, share=0.0, unit_id="pid", rng=rng)
        assert not mask.any()

    def test_select_share_one_selects_all(self, small_df):
        rng = np.random.default_rng(42)
        mask = p.protect._select_share(small_df, share=1.0, unit_id="pid", rng=rng)
        assert mask.all()

    def test_apply_per_unit_consistency(self, small_df):
        rng = np.random.default_rng(42)

        def draw(_): return rng.normal()

        result = p.protect._apply_per_unit(small_df, "pid", draw)
        for pid, grp in small_df.groupby("pid"):
            vals = result[grp.index]
            assert vals.nunique() == 1
