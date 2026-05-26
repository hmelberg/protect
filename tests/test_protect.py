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


# ============================================================================
# noise
# ============================================================================


class TestNoise:
    def test_auto_scale_works_without_explicit_scale(self, small_df):
        out = p.noise(small_df, "income", random_state=42)
        assert not (out["income"] == small_df["income"]).all()

    def test_gaussian_changes_values(self, small_df):
        out = p.noise(small_df, "income", scale=1000, random_state=42)
        assert not (out["income"] == small_df["income"]).all()

    def test_does_not_mutate_input(self, small_df):
        original = small_df["income"].copy()
        p.noise(small_df, "income", scale=1000, random_state=42)
        pd.testing.assert_series_equal(small_df["income"], original)

    def test_reproducible(self, small_df):
        out1 = p.noise(small_df, "income", scale=1000, random_state=42)
        out2 = p.noise(small_df, "income", scale=1000, random_state=42)
        pd.testing.assert_frame_equal(out1, out2)

    def test_share_zero_is_no_op(self, small_df):
        out = p.noise(small_df, "income", scale=1000, share=0.0, random_state=42)
        pd.testing.assert_series_equal(out["income"], small_df["income"])

    def test_unit_id_consistency(self, small_df):
        out = p.noise(small_df, "income", scale=1000, unit_id="pid", random_state=42)
        for pid, grp in out.groupby("pid"):
            assert grp["income"].nunique() == 1

    def test_discrete_method_produces_integer_steps(self, small_df):
        out = p.noise(small_df, "income", scale=3, method="discrete", random_state=42)
        diffs = (out["income"] - small_df["income"]).dropna()
        assert (diffs == diffs.astype(int)).all()
        assert diffs.abs().max() <= 3

    def test_multiplicative_method(self, small_df):
        out = p.noise(small_df, "income", scale=0.1, method="multiplicative", random_state=42)
        ratio = out["income"] / small_df["income"]
        assert ratio.between(0.5, 1.5).all()

    def test_group_mean_replaces_with_group_mean(self, small_df):
        out = p.noise(small_df, "income", scale=5, method="group_mean")
        assert out["income"].nunique() <= small_df["income"].nunique()

    def test_direction_up_only_increases(self, small_df):
        out = p.noise(small_df, "income", scale=1000, direction="up", random_state=42)
        diffs = out["income"] - small_df["income"]
        assert (diffs >= 0).all()

    def test_clip_bounds_output(self, small_df):
        out = p.noise(small_df, "income", scale=100000, clip=(0, 200000), random_state=42)
        assert out["income"].between(0, 200000).all()

    def test_raises_on_missing_column(self, small_df):
        with pytest.raises(KeyError):
            p.noise(small_df, "nonexistent", scale=1.0)
