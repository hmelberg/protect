"""protect — statistical disclosure control for tabular data and results.

See docs/specs/ for design, README.md for usage, BACKGROUND.md for the SDC primer.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

# Group A exports TransformLog; Group B adds noise + jitter. Other verbs
# (winsorize, bin, year, month, diff, shorten, collapse, pseudonymize, insert,
# eliminate, swap, suppress, risk, RiskReport, protect, profile) are forthcoming.
__all__ = [
    "TransformLog",
    "noise",
    "jitter",
    "winsorize",
    "bin",
    "year",
    "month",
    "diff",
    "shorten",
    "collapse",
    "pseudonymize",
    "insert",
    "eliminate",
    "swap",
    "suppress",
]


# ============================================================================
# TransformLog
# ============================================================================


@dataclass
class TransformLog:
    """Audit trail for protection operations.

    Returned by `protect()` and optionally by individual verbs with `audit=True`.
    Designed as documentation for HIPAA Expert Determination, GDPR records,
    and microdata.no method reporting.
    """
    entries: list[dict] = field(default_factory=list)

    def add(self, *, function: str, columns: Sequence[str] | None = None,
            params: dict | None = None, rows_affected: int | None = None,
            units_affected: int | None = None, notes: str | None = None) -> None:
        """Append an operation entry with timestamp and audit metadata."""
        self.entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "function": function,
            "columns": list(columns) if columns else None,
            "params": params or {},
            "rows_affected": rows_affected,
            "units_affected": units_affected,
            "notes": notes,
        })

    def to_text(self) -> str:
        """Return a human-readable text rendering of all entries."""
        lines = []
        for i, e in enumerate(self.entries, 1):
            cols = ", ".join(e["columns"]) if e["columns"] else "-"
            params = ", ".join(f"{k}={v!r}" for k, v in e["params"].items())
            lines.append(
                f"{i}. {e['function']}({cols}) [{params}] "
                f"rows={e['rows_affected']}, units={e['units_affected']}"
            )
            if e.get("notes"):
                lines.append(f"   note: {e['notes']}")
        return "\n".join(lines) if lines else "(empty log)"

    def to_json(self) -> str:
        """Return all entries as a JSON string."""
        return json.dumps({"entries": self.entries}, default=str, indent=2)

    def summary(self) -> dict:
        """Return aggregate counts: total operations and operations per function."""
        by_function: dict[str, int] = {}
        for e in self.entries:
            by_function[e["function"]] = by_function.get(e["function"], 0) + 1
        return {
            "total_operations": len(self.entries),
            "by_function": by_function,
        }

    def __len__(self) -> int:
        return len(self.entries)


# ============================================================================
# Helpers
# ============================================================================


def _resolve_random_state(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Convert int seed / Generator / None to a Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_columns(data: pd.DataFrame, columns: str | Sequence[str]) -> list[str]:
    """Normalize columns argument to a list and verify each is in `data`."""
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise KeyError(f"Columns {missing} not in DataFrame")
    return columns


def _select_share(
    data: pd.DataFrame,
    share: float,
    unit_id: str | None,
    rng: np.random.Generator,
) -> pd.Series:
    """Boolean mask aligned to `data.index` selecting `share` of units (or rows).

    If `unit_id` is given, selection is at unit granularity: a whole unit's
    rows are all True or all False. Otherwise, rows are selected independently.
    """
    if share <= 0:
        return pd.Series(False, index=data.index)
    if share >= 1:
        return pd.Series(True, index=data.index)

    if unit_id is None:
        n = len(data)
        n_select = int(round(n * share))
        choice = rng.choice(n, size=n_select, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[choice] = True
        return pd.Series(mask, index=data.index)

    units = data[unit_id].unique()
    n_select = int(round(len(units) * share))
    selected = set(rng.choice(units, size=n_select, replace=False))
    return data[unit_id].isin(selected)


def _apply_per_unit(
    data: pd.DataFrame,
    unit_id: str,
    fn: Callable[[Any], Any],
) -> pd.Series:
    """Apply `fn` once per unit, broadcast to all rows of that unit.

    `fn` is called with the unit's id and returns a scalar; the result is
    indexed back to `data.index`.
    """
    units = data[unit_id].unique()
    draws = {u: fn(u) for u in units}
    return data[unit_id].map(draws)


def _check_unit_invariant(
    data: pd.DataFrame,
    columns: Sequence[str],
    unit_id: str,
) -> None:
    """Warn if any declared invariant column varies within `unit_id`."""
    for col in columns:
        n_distinct = data.groupby(unit_id)[col].nunique()
        violating = n_distinct[n_distinct > 1]
        if len(violating) > 0:
            warnings.warn(
                f"Column {col!r} varies within {len(violating)} units "
                f"(declared invariant); first offender: {violating.index[0]!r}",
                stacklevel=2,
            )


# ============================================================================
# Value-level verbs
# ============================================================================


def noise(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    scale: float | str = "auto",
    method: str = "gaussian",
    share: float = 1.0,
    direction: str = "both",
    clip: tuple[float, float] | None = None,
    by: str | None = None,
    unit_id: str | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Add noise to numeric columns.

    Parameters
    ----------
    data : DataFrame
    columns : str | list of str
        Numeric column(s) to perturb.
    scale : float | 'auto', default 'auto'
        SD (gaussian/laplace), spread (uniform), max step (discrete), proportion
        (multiplicative), or integer group size (group_mean). With 'auto',
        scale is 0.05 x column_std per column (or 3 for discrete, 0.05 for
        multiplicative, 3 for group_mean group-size).
    method : {'gaussian', 'laplace', 'uniform', 'discrete', 'multiplicative', 'group_mean'}
    share : float in [0, 1], default 1.0
        Fraction of units (or rows when unit_id is None) to perturb.
    direction : {'both', 'up', 'down'}, default 'both'
        Asymmetric noise; clipped to non-negative or non-positive when not 'both'.
    clip : (lo, hi) | None
        Post-noise clipping.
    by : str | None
        Grouping for method='group_mean' (sort within group before grouping
        into k-tuples).
    unit_id : str | None
        When set, noise is drawn once per unit and broadcast.
    random_state : int | Generator | None

    Returns
    -------
    DataFrame
        Copy of `data` with perturbed columns.
    """
    rng = _resolve_random_state(random_state)
    columns = _validate_columns(data, columns)
    out = data.copy()

    if method == "group_mean":
        k = 3 if scale == "auto" else int(scale)
        for col in columns:
            out[col] = _noise_group_mean(out, col, k, by=by)
        return out

    select_mask = _select_share(data, share, unit_id, rng)
    n_total = len(data)

    # When share=0 nothing is selected; return the copy untouched so dtypes
    # are preserved (important for integer columns).
    if not select_mask.any():
        return out

    for col in columns:
        col_scale = _resolve_noise_scale(out[col], scale, method)

        if unit_id is not None:
            unit_noise = _apply_per_unit(
                data, unit_id, lambda _u, _s=col_scale: _draw_noise(rng, method, _s, 1)[0]
            )
            noise_arr = unit_noise.values
        else:
            noise_arr = _draw_noise(rng, method, col_scale, n_total)

        if direction == "up":
            noise_arr = np.abs(noise_arr)
        elif direction == "down":
            noise_arr = -np.abs(noise_arr)

        noise_arr = np.where(select_mask.values, noise_arr, 0)

        if method == "multiplicative":
            new = out[col].values * (1 + noise_arr)
        else:
            new = out[col].values + noise_arr

        if clip is not None:
            new = np.clip(new, clip[0], clip[1])

        out[col] = new

    return out


def _resolve_noise_scale(series: pd.Series, scale, method: str) -> float:
    """Compute the effective scale, handling 'auto'."""
    if scale != "auto":
        return float(scale)
    if method == "multiplicative":
        return 0.05
    if method == "discrete":
        return 3.0
    sd = float(series.std())
    if sd == 0 or np.isnan(sd):
        return 1.0
    return 0.05 * sd


def _draw_noise(rng: np.random.Generator, method: str, scale: float, n: int) -> np.ndarray:
    """Draw an array of noise samples by method."""
    if method == "gaussian":
        return rng.normal(0, scale, size=n)
    if method == "laplace":
        return rng.laplace(0, scale, size=n)
    if method == "uniform":
        return rng.uniform(-scale, scale, size=n)
    if method == "discrete":
        s = int(scale)
        return rng.integers(-s, s + 1, size=n).astype(float)
    if method == "multiplicative":
        return rng.normal(0, scale, size=n)
    raise ValueError(f"Unknown noise method: {method!r}")


def _noise_group_mean(data: pd.DataFrame, col: str, k: int, by: str | None) -> pd.Series:
    """Microaggregation: sort within `by` (or globally), group into k-tuples,
    replace each value with group mean. Returns Series aligned to data.index.
    """
    if k < 2:
        return data[col]

    def _agg(s: pd.Series) -> pd.Series:
        sorted_idx = s.sort_values().index
        result = s.copy()
        for start in range(0, len(sorted_idx), k):
            group = sorted_idx[start:start + k]
            result.loc[group] = s.loc[group].mean()
        return result

    if by is None:
        return _agg(data[col])
    return data.groupby(by, group_keys=False)[col].apply(_agg)


def jitter(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    scale: float | str | pd.Timedelta = "auto",
    distribution: str = "uniform",
    unit_id: str | None = None,
    share: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Small symmetric noise — for numeric or date columns.

    Use for plot-safe perturbation; use `noise` when distribution and scale
    matter for downstream statistics.

    Default scale='auto' computes 0.01 x column_range for numeric columns
    and '1 day' for date columns.
    """
    rng = _resolve_random_state(random_state)
    columns = _validate_columns(data, columns)
    out = data.copy()
    select_mask = _select_share(data, share, unit_id, rng)
    n_total = len(data)

    if not select_mask.any():
        return out

    for col in columns:
        is_date = pd.api.types.is_datetime64_any_dtype(out[col])
        col_scale = _resolve_jitter_scale(out[col], scale, is_date)

        if unit_id is not None:
            draws = _apply_per_unit(
                data, unit_id,
                lambda _u, _s=col_scale, _d=is_date: _draw_jitter_scalar(rng, distribution, _s, _d),
            )
            noise_arr = draws.values
        else:
            noise_arr = _draw_jitter_array(rng, distribution, col_scale, n_total, is_date)

        if is_date:
            applied = np.where(select_mask.values, noise_arr, pd.Timedelta(0))
            out[col] = out[col] + pd.to_timedelta(applied)
        else:
            applied = np.where(select_mask.values, noise_arr, 0.0)
            out[col] = out[col].values + applied

    return out


def _resolve_jitter_scale(series: pd.Series, scale, is_date: bool):
    """Compute the effective scale for jitter, handling 'auto'."""
    if scale == "auto":
        if is_date:
            return pd.Timedelta("1 day")
        rng_ = float(series.max() - series.min())
        if rng_ == 0 or np.isnan(rng_):
            return 1.0
        return 0.01 * rng_
    return pd.Timedelta(scale) if is_date else float(scale)


def _draw_jitter_scalar(rng, distribution, scale, is_date):
    """Draw a single jitter sample (numeric or Timedelta)."""
    if is_date:
        rng_value = rng.uniform(-1, 1) if distribution == "uniform" else rng.normal(0, 1)
        return rng_value * scale
    if distribution == "uniform":
        return rng.uniform(-scale, scale)
    return rng.normal(0, scale)


def _draw_jitter_array(rng, distribution, scale, n, is_date):
    """Draw an array of jitter samples."""
    if is_date:
        u = rng.uniform(-1, 1, size=n) if distribution == "uniform" else rng.normal(0, 1, size=n)
        return np.array([x * scale for x in u])
    if distribution == "uniform":
        return rng.uniform(-scale, scale, size=n)
    return rng.normal(0, scale, size=n)


def winsorize(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    limits: tuple[float | None, float | None] = (0.01, 0.99),
    method: str = "percentile",
    by: str | None = None,
    unit_id: str | None = None,
    share: float = 1.0,
) -> pd.DataFrame:
    """Cap extremes.

    Methods
    -------
    percentile : limits are quantiles, e.g. (0.01, 0.99)
    value      : limits are exact bounds, e.g. (None, 90) for top-code at 90
    gaussian   : limits are SD multipliers; cap at mean ± k·SD
    iqr        : limits are IQR multipliers; cap at Q1 - k·IQR and Q3 + k·IQR
    mad        : limits are MAD multipliers; cap at median ± k·MAD
    """
    columns = _validate_columns(data, columns)
    out = data.copy()
    lo_arg, hi_arg = limits

    def _bounds(s: pd.Series) -> tuple[float | None, float | None]:
        if method == "percentile":
            lo = s.quantile(lo_arg) if lo_arg is not None else None
            hi = s.quantile(hi_arg) if hi_arg is not None else None
            return lo, hi
        if method == "value":
            return lo_arg, hi_arg
        if method == "gaussian":
            m, sd = s.mean(), s.std()
            return (m - lo_arg * sd if lo_arg else None,
                    m + hi_arg * sd if hi_arg else None)
        if method == "iqr":
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            return (q1 - lo_arg * iqr if lo_arg else None,
                    q3 + hi_arg * iqr if hi_arg else None)
        if method == "mad":
            med = s.median()
            mad = (s - med).abs().median()
            return (med - lo_arg * mad if lo_arg else None,
                    med + hi_arg * mad if hi_arg else None)
        raise ValueError(f"Unknown winsorize method: {method!r}")

    for col in columns:
        if by is None:
            lo, hi = _bounds(out[col])
            out[col] = out[col].clip(lower=lo, upper=hi)
        else:
            def _grp(s):
                lo, hi = _bounds(s)
                return s.clip(lower=lo, upper=hi)
            out[col] = out.groupby(by, group_keys=False)[col].apply(_grp)

    return out


def bin(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    bins: int | Sequence[float] = 10,
    method: str = "quantile",
    labels: str | Sequence[str] = "range",
    min_count: int | None = None,
    unit_id: str | None = None,
    share: float = 1.0,
) -> pd.DataFrame:
    """Numeric → discrete intervals.

    Methods
    -------
    quantile     : equal-frequency bins (n bins)
    equal_width  : equal-width bins (n bins between min and max)
    manual       : `bins` is interpreted as explicit edge list

    Labels
    ------
    range    : "10-20" string
    midpoint : numeric midpoint of each interval
    index    : integer index (0, 1, 2, ...)
    list[str]: custom list of labels (length = #bins)

    min_count
    ---------
    If set, sparse bins (count < min_count) are merged into the smaller of
    their adjacent neighbors until all bins meet the threshold.
    """
    columns = _validate_columns(data, columns)
    out = data.copy()

    for col in columns:
        s = out[col]
        if method == "quantile":
            edges = np.unique(s.quantile(np.linspace(0, 1, bins + 1)).values)
        elif method == "equal_width":
            edges = np.linspace(s.min(), s.max(), bins + 1)
        elif method == "manual":
            edges = np.asarray(bins, dtype=float)
        else:
            raise ValueError(f"Unknown bin method: {method!r}")

        cat = pd.cut(s, edges, include_lowest=True, duplicates="drop")

        if min_count is not None:
            cat = _merge_sparse_bins(cat, min_count)

        if labels == "range":
            out[col] = cat.astype(str)
        elif labels == "midpoint":
            mids = {iv: (iv.left + iv.right) / 2 for iv in cat.cat.categories}
            out[col] = cat.map(mids).astype(float)
        elif labels == "index":
            out[col] = cat.cat.codes
        else:
            mapping = dict(zip(cat.cat.categories, labels))
            out[col] = cat.map(mapping)

    return out


def _merge_sparse_bins(cat, min_count: int):
    """Merge bins below min_count into adjacent bins until all bins meet
    the threshold. Greedy: merge each sparse bin into its smaller neighbor first.
    Returns a Series with merged categories.
    """
    s = pd.Series(cat).copy()
    counts = s.value_counts()
    cats = sorted(counts.index, key=lambda iv: iv.left)
    while True:
        sparse = [c for c in cats if counts.get(c, 0) < min_count]
        if not sparse:
            break
        target = sparse[0]
        i = cats.index(target)
        left = cats[i - 1] if i > 0 else None
        right = cats[i + 1] if i < len(cats) - 1 else None
        if left is None and right is None:
            break  # only one bin left
        if left is None:
            neighbor = right
        elif right is None:
            neighbor = left
        else:
            neighbor = left if counts.get(left, 0) <= counts.get(right, 0) else right
        new_iv = pd.Interval(min(target.left, neighbor.left),
                             max(target.right, neighbor.right),
                             closed=target.closed)
        s = s.map(lambda x, t=target, n=neighbor, nv=new_iv: nv if x in (t, n) else x)
        cats = sorted(set(s.dropna().unique()), key=lambda iv: iv.left)
        counts = s.value_counts()
    return pd.Categorical(s, categories=cats, ordered=True)


# ============================================================================
# Date verbs
# ============================================================================


def year(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    bin: int | None = None,
    as_date: bool = False,
    unit_id: str | None = None,
    share: float = 1.0,
) -> pd.DataFrame:
    """Truncate dates to year resolution.

    Default returns integer year. `as_date=True` returns a date floored to
    January 1 of that year. `bin=N` produces N-year period labels like
    "1990-1994".
    """
    columns = _validate_columns(data, columns)
    out = data.copy()
    for col in columns:
        y = pd.to_datetime(out[col]).dt.year
        if bin is None:
            if as_date:
                out[col] = pd.to_datetime(y.astype(str) + "-01-01")
            else:
                out[col] = y.astype(int)
        else:
            floor = (y // bin) * bin
            ceil = floor + bin - 1
            if as_date:
                out[col] = pd.to_datetime(floor.astype(str) + "-01-01")
            else:
                out[col] = floor.astype(str) + "-" + ceil.astype(str)
    return out


def month(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    bin: int | None = None,
    as_date: bool = False,
    unit_id: str | None = None,
    share: float = 1.0,
) -> pd.DataFrame:
    """Truncate dates to month resolution. `bin=3` groups into quarters."""
    columns = _validate_columns(data, columns)
    out = data.copy()
    for col in columns:
        dt = pd.to_datetime(out[col])
        y = dt.dt.year
        m = dt.dt.month
        if bin is not None:
            m = ((m - 1) // bin) * bin + 1
        if as_date:
            out[col] = pd.to_datetime(
                y.astype(str) + "-" + m.astype(str).str.zfill(2) + "-01"
            )
        else:
            out[col] = y.astype(str) + "-" + m.astype(str).str.zfill(2)
    return out


def diff(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    ref="first_per_unit",
    unit: str = "days",
    keep_order: bool = True,
    unit_id: str | None = None,
    share: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Convert dates to numeric diff from a reference.

    ref options
    -----------
    'first_per_unit' (default) : earliest date per unit (requires unit_id)
    'min'                      : minimum date in the column
    'random_per_unit'          : random anchor per unit (requires unit_id)
    column name (str)          : pairwise anchor from another date column
    pd.Timestamp or date string: fixed scalar anchor

    unit : 'days' | 'months' | 'years'

    keep_order=True raises if the result would reorder events within a unit
    (critical for survival-analysis correctness).
    """
    columns = _validate_columns(data, columns)
    out = data.copy()

    if ref in ("first_per_unit", "random_per_unit") and unit_id is None:
        raise ValueError(f"ref={ref!r} requires unit_id to be set")

    rng = _resolve_random_state(random_state)

    for col in columns:
        dt = pd.to_datetime(out[col])
        if ref == "first_per_unit":
            anchor = data.groupby(unit_id)[col].transform("min")
        elif ref == "min":
            anchor = pd.Timestamp(dt.min())
        elif ref == "random_per_unit":
            units = data[unit_id].unique()
            min_date = dt.min()
            max_date = dt.max()
            span_days = max((max_date - min_date).days, 1)
            unit_anchors = {
                u: min_date + pd.Timedelta(days=int(rng.integers(0, span_days + 1)))
                for u in units
            }
            anchor = data[unit_id].map(unit_anchors)
        elif isinstance(ref, str) and ref in data.columns:
            anchor = pd.to_datetime(data[ref])
        elif isinstance(ref, (pd.Timestamp,)):
            anchor = ref
        elif isinstance(ref, str):
            anchor = pd.Timestamp(ref)
        else:
            raise ValueError(f"Unsupported ref: {ref!r}")

        delta = (dt - anchor)
        if isinstance(delta, pd.Series):
            days = delta.dt.days
        else:
            days = pd.Series([delta.days] * len(out), index=out.index)

        if unit == "days":
            result = days.astype(int)
        elif unit == "months":
            result = (days / 30.44).astype(int)
        elif unit == "years":
            result = (days / 365.25).astype(int)
        else:
            raise ValueError(f"Unknown unit: {unit!r}")

        if keep_order and unit_id is not None:
            for pid, grp in data.groupby(unit_id):
                orig_order = dt.loc[grp.index].rank(method="first")
                new_order = result.loc[grp.index].rank(method="first")
                if not (orig_order.values == new_order.values).all():
                    raise ValueError(
                        f"diff would reorder events within unit {pid!r}; "
                        f"keep_order=True"
                    )

        out[col] = result

    return out


# ============================================================================
# Code & category verbs
# ============================================================================


def shorten(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    keep: int = 3,
    sep: str | None = None,
    side: str = "left",
    min_count: int | None = None,
    fallback: str = "*",
    per_value: dict[str, str] | None = None,
    unit_id: str | None = None,
    share: float = 1.0,
) -> pd.DataFrame:
    """Truncate codes (ICD, ZIP, NACE).

    keep      : number of characters to keep
    sep       : if set, truncate at first occurrence of this character
    side      : 'left' = keep prefix; 'right' = keep suffix
    min_count : cascade — if a truncated value appears < min_count times,
                truncate further until it meets the threshold, or replace
                with `fallback` if no further truncation is possible
    per_value : dict mapping a value or "PREFIX*" pattern to an action:
                'keep_full' or 'keep_N' (e.g., 'keep_1' = keep 1 character)
    """
    columns = _validate_columns(data, columns)
    out = data.copy()

    def _truncate(value, keep_n):
        if pd.isna(value):
            return value
        s = str(value)
        if sep is not None and sep in s:
            return s.split(sep)[0] if side == "left" else s.split(sep)[-1]
        return s[:keep_n] if side == "left" else s[-keep_n:]

    for col in columns:
        s = out[col].astype(str)

        if per_value:
            def _apply_rule(v):
                for pattern, action in per_value.items():
                    matches = (v == pattern or
                               (pattern.endswith("*") and v.startswith(pattern[:-1])))
                    if matches:
                        if action == "keep_full":
                            return v
                        if action.startswith("keep_"):
                            n = int(action.split("_")[1])
                            return _truncate(v, n)
                return _truncate(v, keep)
            s = s.map(_apply_rule)
        else:
            s = s.map(lambda v: _truncate(v, keep))

        if min_count is not None:
            current_keep = keep
            while current_keep >= 1:
                counts = s.value_counts()
                rare = counts[counts < min_count].index
                if len(rare) == 0:
                    break
                current_keep -= 1
                if current_keep < 1:
                    s = s.where(~s.isin(rare), fallback)
                    break
                s = s.map(lambda v, _r=rare, _k=current_keep:
                          _truncate(v, _k) if v in _r else v)

        out[col] = s

    return out


def collapse(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    mapping: dict | None = None,
    rare_below: int | None = None,
    keep_top: int | None = None,
    keep_prop: float | None = None,
    other_label: str = "Other",
    by: str | None = None,
    unit_id: str | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Merge categorical levels. Exactly one mode per call.

    Modes
    -----
    mapping={old: new, ...}      : explicit hierarchy
    rare_below=N                 : values appearing < N times → other_label
    keep_top=N                   : keep N most common; rest → other_label
    keep_prop=p                  : keep values with proportion ≥ p; rest → other_label
    """
    columns = _validate_columns(data, columns)
    modes = [mapping is not None, rare_below is not None,
             keep_top is not None, keep_prop is not None]
    if sum(modes) != 1:
        raise ValueError(
            "collapse requires exactly one mode: "
            "mapping, rare_below, keep_top, or keep_prop"
        )

    out = data.copy()

    for col in columns:
        s = out[col]

        if mapping is not None:
            out[col] = s.map(lambda v: mapping.get(v, v))
            continue

        def _apply_threshold(series: pd.Series) -> pd.Series:
            if rare_below is not None:
                counts = series.value_counts()
                keep_set = set(counts[counts >= rare_below].index)
            elif keep_top is not None:
                counts = series.value_counts()
                keep_set = set(counts.head(keep_top).index)
            elif keep_prop is not None:
                props = series.value_counts(normalize=True)
                keep_set = set(props[props >= keep_prop].index)
            else:
                return series
            return series.where(series.isin(keep_set), other_label)

        if by is not None:
            out[col] = out.groupby(by, group_keys=False)[col].apply(_apply_threshold)
        else:
            out[col] = _apply_threshold(s)

    return out


# ============================================================================
# ID verb
# ============================================================================


def pseudonymize(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    method: str = "random",
    salt: str | None = None,
    return_key: bool = True,
    key_path: str | None = None,
    prefix: str = "P",
    random_state: int | np.random.Generator | None = None,
):
    """Replace IDs (random or deterministic hash).

    method='random' : random new IDs, key dict returned (different per run unless seeded)
    method='hash'   : deterministic hash with `salt` (stable across runs that share salt)

    Returns (df, key) when return_key=True (default), otherwise just df.

    key_path persists the key to a JSON file (warning logged about co-location).
    """
    columns = _validate_columns(data, columns)
    out = data.copy()
    keys: dict[str, dict] = {}

    if method == "random":
        rng = _resolve_random_state(random_state)
        for col in columns:
            uniques = list(out[col].unique())
            order = rng.permutation(len(uniques))
            mapping = {
                uniques[i]: f"{prefix}{order[i] + 1:06d}"
                for i in range(len(uniques))
            }
            out[col] = out[col].map(mapping)
            keys[col] = mapping
    elif method == "hash":
        if salt is None:
            warnings.warn(
                "pseudonymize(method='hash') without salt is weak; "
                "provide a salt for production use",
                stacklevel=2,
            )
        salt_bytes = (salt or "").encode("utf-8")
        for col in columns:
            def _h(v, _salt=salt_bytes):
                if pd.isna(v):
                    return v
                h = hashlib.blake2b(str(v).encode("utf-8") + _salt, digest_size=8)
                return prefix + h.hexdigest()
            out[col] = out[col].map(_h)
            keys[col] = {"method": "hash", "salt_provided": salt is not None}
    else:
        raise ValueError(f"Unknown pseudonymize method: {method!r}")

    if key_path is not None:
        warnings.warn(
            f"Persisting pseudonymization key to {key_path}; "
            "store it separately from the data",
            stacklevel=2,
        )
        with open(key_path, "w") as f:
            json.dump(keys, f, indent=2, default=str)

    if return_key:
        return out, keys
    return out


# ============================================================================
# Record-level verbs
# ============================================================================


def insert(
    data: pd.DataFrame,
    *,
    n: int | None = None,
    share: float = 0.01,
    level: str = "row",
    source: str = "resample",
    modify: dict | None = None,
    new_unit_ids: bool = True,
    unit_id: str | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Inject decoy rows or units.

    level='row'  : add N decoy rows
    level='unit' : add N decoy units, each with realistic row-count history
                   (drawn from real units' row-count distribution)

    source='resample'           : draw real rows then optionally modify
    source='sample_per_column'  : draw each column independently (breaks correlations)
    """
    if level not in ("row", "unit"):
        raise ValueError(f"level must be 'row' or 'unit', got {level!r}")
    if level == "unit" and unit_id is None:
        raise ValueError("level='unit' requires unit_id to be set")

    rng = _resolve_random_state(random_state)

    if share > 0.05:
        warnings.warn(f"insert share={share} > 0.05 may distort statistics", stacklevel=2)

    if level == "row":
        n_decoys = n if n is not None else int(round(len(data) * share))
        sample = _generate_decoys(data, n_decoys, source, rng, modify)
        if new_unit_ids and unit_id is not None and unit_id in sample.columns:
            sample[unit_id] = [f"DECOY{i:06d}" for i in range(n_decoys)]
        return pd.concat([data, sample], ignore_index=True)

    # level == "unit"
    n_units = data[unit_id].nunique()
    n_decoy_units = n if n is not None else int(round(n_units * share))
    row_counts = data.groupby(unit_id).size().values
    decoys = []
    for i in range(n_decoy_units):
        rc = int(rng.choice(row_counts))
        sample = _generate_decoys(data, rc, source, rng, modify)
        new_id = f"DECOY{i:06d}"
        sample[unit_id] = new_id
        decoys.append(sample)
    if decoys:
        return pd.concat([data] + decoys, ignore_index=True)
    return data.copy()


def _generate_decoys(
    data: pd.DataFrame,
    n: int,
    source: str,
    rng: np.random.Generator,
    modify: dict | None,
) -> pd.DataFrame:
    """Generate n decoy rows."""
    if source == "resample":
        idx = rng.choice(len(data), size=n, replace=True)
        sample = data.iloc[idx].reset_index(drop=True)
    elif source == "sample_per_column":
        sample = pd.DataFrame({
            c: data[c].sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31))).values
            for c in data.columns
        })
    else:
        raise ValueError(f"Unknown source: {source!r}")

    if modify:
        for col, (op, mag) in modify.items():
            if col not in sample.columns:
                continue
            if op == "noise":
                sample[col] = sample[col] + rng.normal(0, mag, size=n)
            elif op == "shift" and pd.api.types.is_datetime64_any_dtype(sample[col]):
                offsets = rng.integers(-mag, mag + 1, size=n)
                sample[col] = sample[col] + pd.to_timedelta(offsets, unit="D")

    return sample


def eliminate(
    data: pd.DataFrame,
    *,
    where: pd.Series | None = None,
    rare_below: int | None = None,
    share: float | None = None,
    level: str = "row",
    columns: Sequence[str] | None = None,
    replace_with=None,
    unit_id: str | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Drop rows/units or mask cells.

    Modes (exactly one of where, rare_below, share, OR columns-only):
    - where=<bool Series>   : drop matching rows (or units with level='unit')
    - rare_below=N          : mask cells whose value count < N in given columns
    - share=p               : drop random p% of rows (or units with level='unit')
    - columns=[...] only    : mask all cells in those columns to NaN (or replace_with)

    Raises if no mode is given (no silent no-op).
    """
    modes = [where is not None, rare_below is not None, share is not None]
    only_columns = (sum(modes) == 0 and columns is not None)
    if sum(modes) == 0 and not only_columns:
        raise ValueError(
            "eliminate requires a mode arg: where, rare_below, share, or columns"
        )
    if sum(modes) > 1:
        raise ValueError("eliminate accepts exactly one of where, rare_below, or share")
    if level == "unit" and unit_id is None:
        raise ValueError("level='unit' requires unit_id to be set")

    rng = _resolve_random_state(random_state)
    out = data.copy()

    if where is not None:
        if level == "unit":
            units_to_drop = data.loc[where, unit_id].unique()
            return out[~out[unit_id].isin(units_to_drop)].reset_index(drop=True)
        return out[~where].reset_index(drop=True)

    if share is not None:
        if share > 0.05:
            warnings.warn(f"eliminate share={share} > 0.05 may distort statistics", stacklevel=2)
        mask = _select_share(data, share, unit_id if level == "unit" else None, rng)
        return out[~mask].reset_index(drop=True)

    if rare_below is not None:
        cols = _validate_columns(out, columns) if columns else list(out.columns)
        for col in cols:
            counts = out[col].value_counts()
            rare = counts[counts < rare_below].index
            if level == "unit":
                units_with_rare = data.loc[data[col].isin(rare), unit_id].unique()
                mask_rows = out[unit_id].isin(units_with_rare)
                out.loc[mask_rows, col] = replace_with if replace_with is not None else np.nan
            else:
                out.loc[out[col].isin(rare), col] = replace_with if replace_with is not None else np.nan
        return out

    # only_columns mode
    if columns:
        cols = _validate_columns(out, columns)
        for col in cols:
            out[col] = replace_with if replace_with is not None else np.nan
        return out

    return out


def swap(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    method: str = "rank",
    level: str = "row",
    by: str | None = None,
    share: float = 0.05,
    swap_range_pct: float = 0.05,
    transition: dict | None = None,
    unit_id: str | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Exchange values between rows or whole records between units.

    method describes HOW to match: rank | random | shuffle | pram
    level  describes WHAT is swapped: row | unit
    """
    columns = _validate_columns(data, columns)
    if level == "unit" and unit_id is None:
        raise ValueError("level='unit' requires unit_id to be set")

    rng = _resolve_random_state(random_state)
    out = data.copy()

    if method == "shuffle":
        for col in columns:
            if by is None:
                perm = rng.permutation(len(out))
                out[col] = out[col].values[perm]
            else:
                def _shuf(s, _rng=rng):
                    return pd.Series(_rng.permutation(s.values), index=s.index)
                out[col] = out.groupby(by, group_keys=False)[col].apply(_shuf)
        return out

    if method == "pram":
        if transition is None:
            raise ValueError("method='pram' requires transition matrix dict")
        for col in columns:
            out[col] = out[col].map(lambda v, _t=transition, _r=rng: _pram_recode(v, _t, _r))
        return out

    if level == "row":
        # rank or random row-pair swap
        for col in columns:
            n = len(out)
            n_swap = int(round(n * share))
            col_idx = out.columns.get_loc(col)
            if method == "rank":
                order = out[col].rank(method="first").values.argsort()
                pairs_done = 0
                attempts = 0
                while pairs_done < n_swap // 2 and attempts < n_swap * 10:
                    i = int(rng.integers(0, n))
                    window = max(1, int(n * swap_range_pct))
                    j_candidates = order[max(0, i - window):min(n, i + window + 1)]
                    j = int(rng.choice(j_candidates))
                    if j != i:
                        a = out.iloc[i, col_idx]
                        b = out.iloc[j, col_idx]
                        out.iloc[i, col_idx] = b
                        out.iloc[j, col_idx] = a
                        pairs_done += 1
                    attempts += 1
            elif method == "random":
                idx_to_swap = rng.choice(n, size=(n_swap // 2) * 2, replace=False)
                pairs = idx_to_swap.reshape(-1, 2)
                for i, j in pairs:
                    a = out.iloc[i, col_idx]
                    b = out.iloc[j, col_idx]
                    out.iloc[i, col_idx] = b
                    out.iloc[j, col_idx] = a
            else:
                raise ValueError(f"Unknown method: {method!r}")
        return out

    # level == "unit": swap whole records between matched units
    units = data[unit_id].unique()
    n_units = len(units)
    n_swap_units = int(round(n_units * share))

    if method == "random":
        chosen = rng.choice(units, size=(n_swap_units // 2) * 2, replace=False)
        pairs = chosen.reshape(-1, 2)
    elif method == "rank":
        # rank units by the first column's per-unit mean, swap within window
        first_col = columns[0]
        unit_vals = data.groupby(unit_id)[first_col].mean().sort_values()
        ordered = list(unit_vals.index)
        window = max(1, int(n_units * swap_range_pct))
        used: set = set()
        pairs_list: list = []
        for _ in range(n_swap_units // 2):
            available_positions = [k for k in range(n_units) if ordered[k] not in used]
            if not available_positions:
                break
            i_pos = int(rng.choice(available_positions))
            i = ordered[i_pos]
            j_candidates = [ordered[k]
                            for k in range(max(0, i_pos - window), min(n_units, i_pos + window + 1))
                            if ordered[k] != i and ordered[k] not in used]
            if not j_candidates:
                continue
            j = j_candidates[int(rng.integers(0, len(j_candidates)))]
            pairs_list.append((i, j))
            used.add(i)
            used.add(j)
        pairs = np.array(pairs_list) if pairs_list else np.empty((0, 2))
    else:
        raise ValueError(f"Unknown method for level='unit': {method!r}")

    for u1, u2 in pairs:
        mask1 = out[unit_id] == u1
        mask2 = out[unit_id] == u2
        for col in columns:
            v1 = out.loc[mask1, col].iloc[0] if mask1.any() else None
            v2 = out.loc[mask2, col].iloc[0] if mask2.any() else None
            out.loc[mask1, col] = v2
            out.loc[mask2, col] = v1

    return out


def _pram_recode(value, transition: dict, rng: np.random.Generator):
    """PRAM: probabilistic categorical recoding given a transition matrix."""
    if pd.isna(value):
        return value
    row = transition.get(value)
    if row is None:
        return value
    items = list(row.items())
    targets = [k for k, _ in items]
    probs = np.array([v for _, v in items], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(targets, p=probs)


# ============================================================================
# Output verb
# ============================================================================


def suppress(target, **kwargs):
    """Polymorphic output protection. Dispatches on target type.

    For pandas Series/DataFrame: min_n, counts, dominance, p_percent, round,
        ranges, contributions, secondary
    For statsmodels result: redact_intercept, widen_alpha, group_counts
    For plot data ((x, y) tuple): hexbin, bin_histogram, jitter, gridsize,
        bins, min_count
    """
    if isinstance(target, (pd.Series, pd.DataFrame)):
        return _suppress_table(target, **kwargs)
    if hasattr(target, "params") and hasattr(target, "conf_int"):
        return _suppress_regression(target, **kwargs)
    if isinstance(target, tuple) and len(target) == 2:
        return _suppress_plot(target, **kwargs)
    raise NotImplementedError(
        f"suppress does not handle target of type {type(target).__name__}"
    )


def _suppress_table(
    target,
    *,
    min_n: int | None = None,
    counts=None,
    dominance: tuple[int, float] | None = None,
    p_percent: float | None = None,
    round: int | None = None,
    ranges: Sequence[tuple[int, int]] | None = None,
    contributions: dict | None = None,
    secondary: bool = False,
):
    out = target.copy()

    # primary suppression by frequency
    if min_n is not None:
        if counts is None:
            counts = out
        mask = counts < min_n
        out = out.where(~mask, other=np.nan)

    # dominance rule
    if dominance is not None and contributions is not None:
        n, k = dominance
        idx_iter = list(out.index)
        for idx in idx_iter:
            contribs = sorted(contributions.get(idx, []), reverse=True)
            total = sum(contribs) if contribs else 0
            top_n_sum = sum(contribs[:n])
            if total > 0 and top_n_sum / total > k:
                if isinstance(out, pd.Series):
                    out[idx] = np.nan
                else:
                    out.loc[idx] = np.nan

    # p%-rule
    if p_percent is not None and contributions is not None:
        idx_iter = list(out.index)
        for idx in idx_iter:
            contribs = sorted(contributions.get(idx, []), reverse=True)
            if len(contribs) < 3:
                continue
            x1, x2 = contribs[0], contribs[1]
            sum_rest = sum(contribs[2:])
            if sum_rest == 0 or x1 == 0:
                continue
            if sum_rest / x1 < p_percent:
                if isinstance(out, pd.Series):
                    out[idx] = np.nan
                else:
                    out.loc[idx] = np.nan

    # rounding
    if round is not None:
        out = (out / round).round() * round

    # fuzzy ranges
    if ranges is not None:
        def _range_label(v):
            if pd.isna(v):
                return v
            for lo, hi in ranges:
                if lo <= v <= hi:
                    return f"{lo}-{hi}"
            return f">{ranges[-1][1]}"
        if isinstance(out, pd.Series):
            out = out.map(_range_label)
        else:
            out = out.applymap(_range_label)

    if secondary:
        out = _secondary_suppression(out)

    return out


def _secondary_suppression(table):
    """Greedy secondary suppression. For DataFrames: if a row or column has
    exactly one NaN, suppress the smallest remaining value so marginals can't
    recover the suppressed value. For Series: no-op (no marginal structure).
    """
    if isinstance(table, pd.Series):
        return table
    changed = True
    while changed:
        changed = False
        for axis_idx in range(2):
            n = table.shape[axis_idx]
            for i in range(n):
                row = table.iloc[i, :] if axis_idx == 0 else table.iloc[:, i]
                nan_count = row.isna().sum()
                if nan_count == 1:
                    remaining = row.dropna()
                    if len(remaining) == 0:
                        continue
                    smallest = remaining.idxmin()
                    if axis_idx == 0:
                        table.iloc[i, table.columns.get_loc(smallest)] = np.nan
                    else:
                        table.iloc[table.index.get_loc(smallest), i] = np.nan
                    changed = True
    return table


def _suppress_regression(
    result,
    *,
    redact_intercept: int | None = None,
    widen_alpha: float | None = None,
    group_counts: dict | None = None,
):
    """Return a lightweight namespace mimicking the statsmodels API surface
    we need (params, conf_int).
    """
    import types
    raw_params = result.params
    params = raw_params.copy()
    if widen_alpha is not None:
        ci = result.conf_int(alpha=widen_alpha)
    else:
        ci = result.conf_int()
    # statsmodels returns plain ndarrays when fit on raw numpy arrays; convert
    # to pandas so we have a uniform name-based API. Use "const" for the
    # intercept (statsmodels add_constant convention) and x1, x2, ... otherwise.
    if not hasattr(params, "index"):
        names = ["const"] + [f"x{i}" for i in range(1, len(params))]
        params = pd.Series(params, index=names)
    if not hasattr(ci, "loc"):
        ci = pd.DataFrame(ci, index=params.index, columns=[0, 1])

    if redact_intercept is not None and group_counts is not None:
        smallest = min(group_counts.values())
        if smallest < redact_intercept:
            intercept_name = "const" if "const" in params.index else params.index[0]
            params[intercept_name] = np.nan
            ci.loc[intercept_name] = np.nan

    ns = types.SimpleNamespace()
    ns.params = params
    ns.conf_int = lambda: ci
    ns.summary_text = (
        f"Suppressed regression result:\n{params.to_string()}\n\nCI:\n{ci.to_string()}"
    )
    return ns


def _suppress_plot(
    xy,
    *,
    hexbin: bool = False,
    bin_histogram: bool = False,
    gridsize: int = 30,
    bins: int = 20,
    min_count: int = 5,
    jitter: tuple[float, float] | None = None,
):
    x, y = xy
    x = np.asarray(x)
    y = np.asarray(y)
    if hexbin:
        h, xedges, yedges = np.histogram2d(x, y, bins=gridsize)
        h_safe = np.where(h >= min_count, h, 0)
        return {
            "x_centers": (xedges[:-1] + xedges[1:]) / 2,
            "y_centers": (yedges[:-1] + yedges[1:]) / 2,
            "counts": h_safe,
        }
    if bin_histogram:
        h, edges = np.histogram(x, bins=bins)
        h_safe = np.where(h >= min_count, h, 0)
        return {"edges": edges, "counts": h_safe}
    if jitter is not None:
        rng = np.random.default_rng()
        sd_x, sd_y = jitter
        return (x + rng.normal(0, sd_x, size=len(x)),
                y + rng.normal(0, sd_y, size=len(y)))
    raise ValueError("Plot suppress requires one of: hexbin, bin_histogram, jitter")


# ============================================================================
# Risk
# ============================================================================


# ============================================================================
# Meta verbs
# ============================================================================


# ============================================================================
# Profile implementations
# ============================================================================
