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


# ============================================================================
# jitter
# ============================================================================


class TestJitter:
    def test_auto_scale_works_without_explicit_scale(self, small_df):
        out = p.jitter(small_df, "income", random_state=42)
        assert not (out["income"] == small_df["income"]).all()

    def test_auto_scale_date_column(self, panel_df):
        out = p.jitter(panel_df, "visit", random_state=42)
        diff = (out["visit"] - panel_df["visit"]).abs()
        assert (diff <= pd.Timedelta("1 day")).all()

    def test_jitter_numeric_changes_values(self, small_df):
        out = p.jitter(small_df, "income", scale=100, random_state=42)
        assert not (out["income"] == small_df["income"]).all()

    def test_jitter_numeric_bounded(self, small_df):
        out = p.jitter(small_df, "income", scale=100, random_state=42)
        diff = (out["income"] - small_df["income"]).abs()
        assert (diff < 100).all()

    def test_jitter_date_column(self, panel_df):
        out = p.jitter(panel_df, "visit", scale="3 days", random_state=42)
        diff = (out["visit"] - panel_df["visit"]).abs()
        assert (diff <= pd.Timedelta("3 days")).all()

    def test_jitter_gaussian_distribution(self, small_df):
        out = p.jitter(small_df, "income", scale=50, distribution="gaussian", random_state=42)
        assert not (out["income"] == small_df["income"]).all()

    def test_jitter_unit_id_consistency(self, panel_df):
        out = p.jitter(panel_df, "income", scale=100, unit_id="pid", random_state=42)
        for pid, grp in out.groupby("pid"):
            assert grp["income"].nunique() == 1


# ============================================================================
# winsorize
# ============================================================================


class TestWinsorize:
    def test_percentile_method_caps_extremes(self, small_df):
        out = p.winsorize(small_df, "income", limits=(0.1, 0.9), method="percentile")
        lo = small_df["income"].quantile(0.1)
        hi = small_df["income"].quantile(0.9)
        assert out["income"].min() >= lo
        assert out["income"].max() <= hi

    def test_value_method_top_codes(self, small_df):
        out = p.winsorize(small_df, "age", limits=(None, 60), method="value")
        assert out["age"].max() <= 60

    def test_value_method_bottom_codes(self, small_df):
        out = p.winsorize(small_df, "age", limits=(25, None), method="value")
        assert out["age"].min() >= 25

    def test_iqr_method_caps_outliers(self, small_df):
        out = p.winsorize(small_df, "income", limits=(1.5, 1.5), method="iqr")
        q1, q3 = small_df["income"].quantile([0.25, 0.75])
        iqr = q3 - q1
        assert out["income"].min() >= q1 - 1.5 * iqr
        assert out["income"].max() <= q3 + 1.5 * iqr

    def test_per_group_caps(self, small_df):
        out = p.winsorize(small_df, "income", limits=(0.1, 0.9), method="percentile", by="diagnosis")
        assert len(out) == len(small_df)
        assert "income" in out.columns


# ============================================================================
# bin
# ============================================================================


class TestBin:
    def test_int_bins_produces_n_intervals(self, small_df):
        out = p.bin(small_df, "income", bins=4)
        assert out["income"].nunique() <= 4

    def test_explicit_edges(self, small_df):
        out = p.bin(small_df, "age", bins=[0, 30, 50, 100], method="manual")
        labels = set(out["age"].unique())
        assert all(isinstance(x, str) for x in labels)

    def test_midpoint_labels(self, small_df):
        out = p.bin(small_df, "age", bins=[0, 30, 50, 100], method="manual", labels="midpoint")
        assert pd.api.types.is_numeric_dtype(out["age"])

    def test_min_count_merges_sparse_bins(self, panel_df):
        out = p.bin(panel_df, "cost", bins=20, method="quantile", min_count=20)
        counts = out["cost"].value_counts()
        assert counts.min() >= 20

    def test_does_not_mutate_input(self, small_df):
        original = small_df["income"].copy()
        p.bin(small_df, "income", bins=4)
        pd.testing.assert_series_equal(small_df["income"], original)


# ============================================================================
# year, month
# ============================================================================


class TestYearMonth:
    def test_year_returns_integer(self, panel_df):
        out = p.year(panel_df, "dob")
        assert pd.api.types.is_integer_dtype(out["dob"])
        assert (out["dob"] == panel_df["dob"].dt.year).all()

    def test_year_as_date(self, panel_df):
        out = p.year(panel_df, "dob", as_date=True)
        assert pd.api.types.is_datetime64_any_dtype(out["dob"])
        assert (out["dob"].dt.month == 1).all()
        assert (out["dob"].dt.day == 1).all()

    def test_year_bin_5(self, panel_df):
        out = p.year(panel_df, "dob", bin=5)
        assert out["dob"].apply(lambda x: "-" in x).all()

    def test_month_returns_string(self, panel_df):
        out = p.month(panel_df, "visit")
        assert out["visit"].iloc[0].count("-") == 1

    def test_month_bin_3_groups_into_quarters(self, panel_df):
        out = p.month(panel_df, "visit", bin=3)
        per_year_bins = out["visit"].apply(lambda x: x.split("-")[0]).nunique()
        assert per_year_bins > 0


# ============================================================================
# diff
# ============================================================================


class TestDiff:
    def test_diff_from_first_per_unit(self, panel_df):
        out = p.diff(panel_df, "visit", ref="first_per_unit", unit_id="pid")
        for pid, grp in panel_df.groupby("pid"):
            first_visit_idx = grp.index.min()
            assert out.loc[first_visit_idx, "visit"] == 0
        assert (out["visit"] >= 0).all()

    def test_diff_from_min(self, panel_df):
        out = p.diff(panel_df, "visit", ref="min")
        assert out["visit"].min() == 0

    def test_diff_from_scalar_date(self, panel_df):
        ref_date = pd.Timestamp("2020-01-01")
        out = p.diff(panel_df, "visit", ref=ref_date)
        expected = (panel_df["visit"] - ref_date).dt.days
        pd.testing.assert_series_equal(out["visit"], expected.astype(int), check_names=False)

    def test_diff_months_unit(self, panel_df):
        out = p.diff(panel_df, "visit", ref="min", unit="months")
        days_out = p.diff(panel_df, "visit", ref="min", unit="days")
        assert out["visit"].max() < days_out["visit"].max()

    def test_diff_requires_unit_id_for_per_unit_ref(self, panel_df):
        with pytest.raises(ValueError, match="unit_id"):
            p.diff(panel_df, "visit", ref="first_per_unit")


# ============================================================================
# shorten
# ============================================================================


class TestShorten:
    def test_keep_n_characters(self, panel_df):
        out = p.shorten(panel_df, "zip", keep=3)
        assert out["zip"].str.len().max() <= 3

    def test_sep_truncates_at_separator(self, panel_df):
        out = p.shorten(panel_df, "icd", sep=".")
        for val in out["icd"].unique():
            assert "." not in val

    def test_min_count_cascades(self, panel_df):
        out = p.shorten(panel_df, "icd", sep=".", min_count=5)
        counts = out["icd"].value_counts()
        assert len(counts) > 0

    def test_per_value_rules(self, panel_df):
        out = p.shorten(panel_df, "icd", keep=1, per_value={"I10": "keep_full"})
        assert len(out) == len(panel_df)


# ============================================================================
# collapse
# ============================================================================


class TestCollapse:
    def test_mapping_mode(self, panel_df):
        out = p.collapse(panel_df, "country", mapping={"LI": "Other Europe", "AD": "Other Europe"})
        assert "Other Europe" in out["country"].values

    def test_rare_below_collapses_rare(self, panel_df):
        out = p.collapse(panel_df, "country", rare_below=5)
        counts = out["country"].value_counts()
        real_values = counts.drop("Other", errors="ignore")
        assert (real_values >= 5).all()

    def test_keep_top_n(self, panel_df):
        out = p.collapse(panel_df, "country", keep_top=3)
        assert out["country"].nunique() <= 4

    def test_keep_prop(self, panel_df):
        out = p.collapse(panel_df, "country", keep_prop=0.05)
        counts = out["country"].value_counts(normalize=True)
        real = counts.drop("Other", errors="ignore")
        assert (real >= 0.05).all()

    def test_multiple_modes_raises(self, panel_df):
        with pytest.raises(ValueError, match="exactly one"):
            p.collapse(panel_df, "country", rare_below=5, keep_top=3)

    def test_custom_other_label(self, panel_df):
        out = p.collapse(panel_df, "country", rare_below=5, other_label="Rare")
        assert "Rare" in out["country"].values or out["country"].value_counts().min() >= 5


# ============================================================================
# pseudonymize
# ============================================================================


class TestPseudonymize:
    def test_random_method_returns_df_and_key(self, panel_df):
        result = p.pseudonymize(panel_df, "pid", method="random", random_state=42)
        assert isinstance(result, tuple)
        df_out, key = result
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(key, dict)

    def test_random_consistent_within_run(self, panel_df):
        df_out, _ = p.pseudonymize(panel_df, "pid", method="random", random_state=42)
        groups = df_out.groupby(panel_df["pid"])["pid"].nunique()
        assert (groups == 1).all()

    def test_random_changes_across_runs(self, panel_df):
        out1, _ = p.pseudonymize(panel_df, "pid", method="random", random_state=42)
        out2, _ = p.pseudonymize(panel_df, "pid", method="random", random_state=43)
        assert not (out1["pid"] == out2["pid"]).all()

    def test_hash_deterministic(self, panel_df):
        out1, _ = p.pseudonymize(panel_df, "pid", method="hash", salt="secret")
        out2, _ = p.pseudonymize(panel_df, "pid", method="hash", salt="secret")
        pd.testing.assert_series_equal(out1["pid"], out2["pid"])

    def test_hash_different_salts_differ(self, panel_df):
        out1, _ = p.pseudonymize(panel_df, "pid", method="hash", salt="salt1")
        out2, _ = p.pseudonymize(panel_df, "pid", method="hash", salt="salt2")
        assert not (out1["pid"] == out2["pid"]).all()

    def test_return_key_false(self, panel_df):
        result = p.pseudonymize(panel_df, "pid", method="random", return_key=False, random_state=42)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# insert
# ============================================================================


class TestInsert:
    def test_row_level_default(self, panel_df):
        out = p.insert(panel_df, share=0.1, random_state=42)
        assert len(out) > len(panel_df)

    def test_row_level_n(self, panel_df):
        out = p.insert(panel_df, n=20, random_state=42)
        assert len(out) == len(panel_df) + 20

    def test_unit_level_adds_new_units(self, panel_df):
        out = p.insert(panel_df, share=0.1, level="unit", unit_id="pid", random_state=42)
        n_original = panel_df["pid"].nunique()
        n_new = out["pid"].nunique()
        assert n_new > n_original

    def test_unit_level_decoys_have_new_pids(self, panel_df):
        out = p.insert(panel_df, share=0.1, level="unit", unit_id="pid", random_state=42)
        new_pids = set(out["pid"]) - set(panel_df["pid"])
        assert len(new_pids) > 0

    def test_unit_level_requires_unit_id(self, panel_df):
        with pytest.raises(ValueError, match="unit_id"):
            p.insert(panel_df, share=0.1, level="unit")

    def test_warns_above_threshold(self, panel_df):
        with pytest.warns(UserWarning, match="share"):
            p.insert(panel_df, share=0.1, random_state=42)


# ============================================================================
# eliminate
# ============================================================================


class TestEliminate:
    def test_where_drops_matching_rows(self, small_df):
        out = p.eliminate(small_df, where=small_df["age"] > 50)
        assert (out["age"] <= 50).all()

    def test_share_drops_rows(self, small_df):
        out = p.eliminate(small_df, share=0.3, random_state=42)
        assert len(out) < len(small_df)

    def test_share_unit_level_drops_units(self, panel_df):
        out = p.eliminate(panel_df, share=0.3, level="unit", unit_id="pid", random_state=42)
        for pid, grp in out.groupby("pid"):
            n_in_orig = (panel_df["pid"] == pid).sum()
            assert len(grp) == n_in_orig

    def test_rare_below_masks_rare_values(self, panel_df):
        out = p.eliminate(panel_df, rare_below=3, columns=["country"])
        counts = panel_df["country"].value_counts()
        rare = counts[counts < 3].index.tolist()
        for r in rare:
            assert (out["country"] != r).all()

    def test_no_args_raises(self, small_df):
        with pytest.raises(ValueError, match="mode"):
            p.eliminate(small_df)

    def test_multiple_modes_raises(self, small_df):
        with pytest.raises(ValueError, match="exactly one"):
            p.eliminate(small_df, where=small_df["age"] > 50, share=0.1)

    def test_columns_only_masks_to_nan(self, small_df):
        out = p.eliminate(small_df, columns=["income"])
        assert out["income"].isna().all()

    def test_unit_level_requires_unit_id(self, panel_df):
        with pytest.raises(ValueError, match="unit_id"):
            p.eliminate(panel_df, share=0.3, level="unit")


# ============================================================================
# swap
# ============================================================================


class TestSwap:
    def test_rank_row_swap_changes_some_values(self, panel_df):
        out = p.swap(panel_df, "cost", method="rank", level="row", share=0.5, random_state=42)
        assert not (out["cost"].values == panel_df["cost"].values).all()
        assert sorted(out["cost"].values) == sorted(panel_df["cost"].values)

    def test_random_unit_swap_swaps_whole_records(self, panel_df):
        out = p.swap(panel_df, "income", method="random", level="unit",
                     unit_id="pid", share=0.5, random_state=42)
        for pid, grp in out.groupby("pid"):
            assert grp["income"].nunique() == 1

    def test_shuffle_within_group_preserves_set(self, panel_df):
        out = p.swap(panel_df, "cost", method="shuffle", level="row",
                     by="icd", random_state=42)
        for icd, grp in panel_df.groupby("icd"):
            orig = sorted(grp["cost"].values)
            new = sorted(out.loc[grp.index, "cost"].values)
            assert orig == new

    def test_unit_level_requires_unit_id(self, panel_df):
        with pytest.raises(ValueError, match="unit_id"):
            p.swap(panel_df, "income", method="random", level="unit", share=0.5)

    def test_reproducible(self, panel_df):
        out1 = p.swap(panel_df, "cost", method="rank", share=0.5, random_state=42)
        out2 = p.swap(panel_df, "cost", method="rank", share=0.5, random_state=42)
        pd.testing.assert_frame_equal(out1, out2)


# ============================================================================
# suppress (table)
# ============================================================================


class TestSuppressTable:
    def test_min_n_suppresses_small_cells(self):
        tab = pd.DataFrame({"value": [10, 2, 30, 1, 40]},
                           index=["A", "B", "C", "D", "E"])
        counts = pd.Series([10, 2, 30, 1, 40], index=tab.index)
        out = p.suppress(tab, min_n=5, counts=counts)
        assert pd.isna(out.loc["B", "value"])
        assert pd.isna(out.loc["D", "value"])
        assert out.loc["A", "value"] == 10

    def test_round_table(self):
        tab = pd.Series([13, 27, 41, 58, 73], index=list("ABCDE"))
        out = p.suppress(tab, round=10)
        assert all(v % 10 == 0 for v in out.dropna())

    def test_fuzzy_count_ranges(self):
        tab = pd.Series([2, 7, 15, 25, 50], index=list("ABCDE"))
        out = p.suppress(tab, ranges=[(1, 4), (5, 9), (10, 19), (20, 99)])
        assert out["A"] == "1-4"
        assert out["B"] == "5-9"
        assert out["D"] == "20-99"

    def test_dominance_rule_suppresses_when_top_contributors_dominate(self):
        tab = pd.DataFrame({"value": [1000, 500]}, index=["X", "Y"])
        contributions = {"X": [800, 100, 100], "Y": [100, 100, 100, 100, 100]}
        out = p.suppress(tab, dominance=(1, 0.7), contributions=contributions)
        assert pd.isna(out.loc["X", "value"])
        assert out.loc["Y", "value"] == 500


# ============================================================================
# suppress (regression + plot)
# ============================================================================


class TestSuppressRegression:
    def test_widen_alpha(self):
        sm = pytest.importorskip("statsmodels.api")
        X = sm.add_constant(np.arange(100, dtype=float))
        y = 2 * X[:, 1] + np.random.default_rng(42).normal(0, 1, 100)
        result = sm.OLS(y, X).fit()
        out = p.suppress(result, widen_alpha=0.01)
        ci99 = np.asarray(result.conf_int(alpha=0.01))
        new_ci = np.asarray(out.conf_int())
        np.testing.assert_allclose(new_ci, ci99, rtol=1e-6)

    def test_redact_intercept_below_threshold(self):
        sm = pytest.importorskip("statsmodels.api")
        X = sm.add_constant(np.arange(20, dtype=float))
        y = 2 * X[:, 1] + np.random.default_rng(42).normal(0, 1, 20)
        result = sm.OLS(y, X).fit()
        out = p.suppress(result, redact_intercept=5, group_counts={"const": 3})
        assert np.isnan(out.params["const"]) or out.params["const"] is None


class TestSuppressPlot:
    def test_hexbin_suppresses_sparse(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        result = p.suppress((x, y), hexbin=True, gridsize=20, min_count=10)
        assert "x_centers" in result
        assert "y_centers" in result
        assert "counts" in result


# ============================================================================
# risk
# ============================================================================


class TestRisk:
    def test_risk_report_basic_metrics(self, panel_df):
        report = p.risk(panel_df, quasi_ids=["sex", "zip"], unit_id="pid")
        assert isinstance(report, p.RiskReport)
        assert report.k_min >= 1
        assert report.distinct_combos > 0

    def test_units_at_risk(self, panel_df):
        report = p.risk(panel_df, quasi_ids=["sex", "zip", "country"], unit_id="pid")
        assert report.units_at_risk > 0

    def test_l_diversity_with_sensitive(self, panel_df):
        report = p.risk(panel_df, quasi_ids=["sex"], sensitive=["icd"], unit_id="pid")
        assert report.l_min is not None
        assert report.l_min > 0

    def test_describe_returns_text(self, panel_df):
        report = p.risk(panel_df, quasi_ids=["sex", "zip"], unit_id="pid")
        text = report.describe()
        assert isinstance(text, str)
        assert "k" in text.lower()

    def test_diff_two_reports(self, panel_df):
        r1 = p.risk(panel_df, quasi_ids=["sex", "zip"], unit_id="pid")
        # zip is a 5-digit string in the fixture; coerce to numeric so bin works
        df_num = panel_df.copy()
        df_num["zip"] = df_num["zip"].astype(int)
        df2 = p.bin(df_num, "zip", bins=2)
        r2 = p.risk(df2, quasi_ids=["sex", "zip"], unit_id="pid")
        d = r1.diff(r2)
        assert "k_min" in d
