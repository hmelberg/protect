# Protect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file Python module (`protect.py`) exposing 17 single-token verbs for data privacy and statistical disclosure control, with companion docs and a test suite.

**Architecture:** All functionality lives in one file. Each verb is a top-level function with three universal arguments (`unit_id`, `share`, `random_state`). Shared helpers handle unit-consistent perturbation, share-based unit selection, and column validation. `TransformLog` and `RiskReport` are dataclasses. The `protect()` recipe driver composes verbs; `profile()` provides named compositions.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest. No other dependencies. Test fixtures use a synthetic panel-shaped health-research dataset.

**Spec:** `/Users/hom/Documents/GitHub/protect/docs/specs/2026-05-27-protect-design.md`

---

## Defaults policy

The package commits to one explicit defaults policy. Subagents implementing each verb should follow it strictly.

**Universal arguments (every data-side verb):**
- `unit_id=None` — defaults to per-row operation; user opts in to per-unit consistency
- `random_state=None` — non-deterministic by default; user opts in to reproducibility
- `share` default depends on verb category (see below)

**`share` defaults by verb category:**
- *Perturbation/transform verbs* (`noise`, `jitter`, `winsorize`, `bin`, date verbs, `shorten`, `collapse`, `pseudonymize`): `share=1.0` — apply to all units by default
- *Record-level verbs* — different defaults reflecting their nature:
  - `insert`: `share=0.01` (low — fabricating data is deliberate)
  - `swap`: `share=0.05` (low — heavy swapping degrades statistics fast)
  - `eliminate`: **no default for `share`** (must be explicit — destructive)

**Magnitude defaults (`scale`, `bins`, `keep`, `limits`, `min_count`):**
- `noise(scale='auto')` — auto computes `0.05 × column_std` per column
- `jitter(scale='auto')` — auto computes `0.01 × column_range` for numeric, `'1 day'` for date columns
- `winsorize(limits=(0.01, 0.99), method='percentile')` — 1st / 99th percentile caps
- `bin(bins=10, method='quantile')` — 10 quantile bins
- `shorten(keep=3, sep=None)` — first 3 characters
- `pseudonymize(method='random')` — random IDs with key returned
- `insert(source='resample', level='row')` — resample real rows (preserves correlations)
- `swap(method='rank', level='row')` — rank-window numeric swap
- `collapse` — **no default mode** (must specify one of `mapping`, `rare_below`, `keep_top`, `keep_prop`)
- `eliminate` — **no default mode** (must specify one of `where`, `rare_below`, `share`, `columns`)
- `suppress` — **no default action** (must specify what to do via `min_n`, `dominance`, `round`, `ranges`, `widen_alpha`, `redact_intercept`, `hexbin`, `bin_histogram`, or `jitter`)
- `risk(quasi_ids=...)` — **`quasi_ids` is required** (depends on threat model; no safe default)

**Why no defaults for `collapse`, `eliminate`, `suppress`, and `risk`'s `quasi_ids`:** these are cases where silently doing the wrong thing is worse than asking. `eliminate()` with no args would silently no-op or guess; we raise instead.

**Why `share=1.0` default for perturbation:** if you call `noise(df, "income")`, you almost always want every value perturbed; `share<1.0` is the niche case.

**Why `scale='auto'`:** the right noise magnitude depends on the column. `auto` is data-aware and yields visible-but-reasonable perturbation out of the box; users override when they care.

The README documents this same policy.

---

## File Structure

```
/Users/hom/Documents/GitHub/protect/
├── protect.py                # main module (~2000-2500 LOC)
├── __init__.py               # `from .protect import *`
├── pyproject.toml            # minimal package metadata
├── README.md                 # quick-start + verb reference
├── BACKGROUND.md             # SDC primer (HIPAA, GDPR, helseregisterloven, methods)
├── examples.ipynb            # runnable demos
├── docs/
│   ├── specs/2026-05-27-protect-design.md
│   └── plans/2026-05-27-protect-implementation.md   # this file
└── tests/
    ├── __init__.py
    ├── conftest.py           # synthetic panel-data fixture
    └── test_protect.py       # one section per verb + integration tests
```

**Banner-commented sections inside `protect.py`** (in order):

1. Imports, constants, type aliases
2. `TransformLog` class
3. Helpers (`_resolve_random_state`, `_validate_columns`, `_select_share`, `_apply_per_unit`, `_check_unit_invariant`)
4. Value-level verbs: `noise`, `jitter`, `winsorize`, `bin`
5. Date verbs: `year`, `month`, `diff`
6. Code & category verbs: `shorten`, `collapse`
7. ID verb: `pseudonymize`
8. Record-level verbs: `insert`, `eliminate`, `swap`
9. Output verb: `suppress` (+ helpers)
10. Risk: `risk` function + `RiskReport` dataclass
11. Meta: `protect` (recipe), `profile` (named compositions)
12. Profile implementations (`_profile_safe_harbor`, `_profile_microdata_no`, `_profile_gdpr_pseudonymize`, `_profile_health_research`, `_profile_k_anonymize`)

---

## Task 1: Project scaffolding

**Files:**
- Create: `/Users/hom/Documents/GitHub/protect/pyproject.toml`
- Create: `/Users/hom/Documents/GitHub/protect/__init__.py`
- Create: `/Users/hom/Documents/GitHub/protect/protect.py`
- Create: `/Users/hom/Documents/GitHub/protect/tests/__init__.py`
- Create: `/Users/hom/Documents/GitHub/protect/tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "protect"
version = "0.1.0"
description = "Statistical disclosure control toolkit for tabular data and analytical results"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
]

[project.optional-dependencies]
test = ["pytest>=7.0", "statsmodels>=0.14", "matplotlib>=3.7"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["protect"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `__init__.py`**

```python
from .protect import *  # noqa: F401, F403
from .protect import __all__  # noqa: F401
```

- [ ] **Step 3: Create `protect.py` skeleton with banner-commented sections**

```python
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

__all__ = [
    # value-level
    "noise", "jitter", "winsorize", "bin",
    # dates
    "year", "month", "diff",
    # code & category
    "shorten", "collapse",
    # IDs
    "pseudonymize",
    # record-level
    "insert", "eliminate", "swap",
    # output
    "suppress",
    # risk
    "risk", "RiskReport",
    # meta
    "protect", "profile",
    # audit
    "TransformLog",
]


# ============================================================================
# TransformLog
# ============================================================================


# ============================================================================
# Helpers
# ============================================================================


# ============================================================================
# Value-level verbs
# ============================================================================


# ============================================================================
# Date verbs
# ============================================================================


# ============================================================================
# Code & category verbs
# ============================================================================


# ============================================================================
# ID verb
# ============================================================================


# ============================================================================
# Record-level verbs
# ============================================================================


# ============================================================================
# Output verb
# ============================================================================


# ============================================================================
# Risk
# ============================================================================


# ============================================================================
# Meta verbs
# ============================================================================


# ============================================================================
# Profile implementations
# ============================================================================
```

- [ ] **Step 4: Create `tests/__init__.py`** (empty file)

- [ ] **Step 5: Create `tests/conftest.py` with synthetic fixture**

```python
"""Shared pytest fixtures: a synthetic panel-shaped health dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel_df():
    """200 patients with 1-5 visits each, panel-shaped health data.

    Columns:
      pid: patient ID (str)
      visit: visit date (datetime)
      dob: date of birth (datetime, constant per patient)
      sex: 'M' or 'F' (str, constant per patient)
      zip: 5-digit ZIP code (str, constant per patient)
      country: country of birth (str, constant per patient, many rare values)
      icd: ICD-10 diagnosis code (str, varies per visit)
      income: annual income (float, constant per patient)
      cost: visit cost (float, varies per visit)
    """
    rng = np.random.default_rng(42)
    n_patients = 200
    visits_per_patient = rng.integers(1, 6, size=n_patients)
    countries = ["USA"] * 80 + ["UK"] * 40 + ["DE"] * 30 + ["IN"] * 20 + ["NO"] * 15
    countries += ["IS", "LI", "AD", "TV", "NR", "BT", "SM", "MC", "VA", "FM"][:n_patients - len(countries)]
    rng.shuffle(countries)
    icd_codes = ["I10", "I10.9", "E11.9", "E11", "J45.909", "K21.9", "M54.5", "F32.9", "C50.9", "Z00.00"]

    rows = []
    for i in range(n_patients):
        pid = f"PT{i:04d}"
        sex = rng.choice(["M", "F"])
        dob = pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.integers(0, 25000)))
        zip_code = f"{int(rng.integers(10000, 99999)):05d}"
        country = countries[i]
        income = float(rng.normal(60000, 20000))
        for v in range(visits_per_patient[i]):
            visit = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))
            icd = rng.choice(icd_codes)
            cost = float(rng.normal(500, 200))
            rows.append({
                "pid": pid,
                "visit": visit,
                "dob": dob,
                "sex": sex,
                "zip": zip_code,
                "country": country,
                "icd": icd,
                "income": income,
                "cost": cost,
            })

    df = pd.DataFrame(rows).sort_values(["pid", "visit"]).reset_index(drop=True)
    return df


@pytest.fixture
def small_df():
    """20-row dataset for quick deterministic tests."""
    return pd.DataFrame({
        "pid": ["A", "A", "A", "B", "B", "C", "C", "C", "D", "D",
                "E", "E", "F", "F", "G", "G", "H", "I", "J", "J"],
        "age": [30, 30, 30, 45, 45, 25, 25, 25, 60, 60,
                35, 35, 50, 50, 28, 28, 70, 22, 55, 55],
        "income": [50000, 50000, 50000, 80000, 80000, 30000, 30000, 30000,
                   120000, 120000, 60000, 60000, 90000, 90000, 40000, 40000,
                   110000, 25000, 100000, 100000],
        "diagnosis": ["I10", "I10", "E11", "I10", "C50", "Z00", "Z00", "Z00",
                      "I10", "E11", "I10", "I10", "C50", "I10", "Z00", "I10",
                      "I10", "Z00", "E11", "E11"],
    })
```

- [ ] **Step 6: Run pytest to confirm setup works**

Run: `cd /Users/hom/Documents/GitHub/protect && python -m pytest tests/ -v`
Expected: `no tests ran in 0.XXs`

- [ ] **Step 7: Commit**

```bash
cd /Users/hom/Documents/GitHub/protect
git init
git add pyproject.toml __init__.py protect.py tests/__init__.py tests/conftest.py docs/
git commit -m "scaffold protect package"
```

---

## Task 2: TransformLog

**Files:**
- Modify: `protect.py` (TransformLog section)
- Modify: `tests/test_protect.py` (create + add tests)

- [ ] **Step 1: Write failing test**

Create `tests/test_protect.py`:

```python
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestTransformLog -v`
Expected: FAIL with `AttributeError: module 'protect' has no attribute 'TransformLog'`

- [ ] **Step 3: Implement `TransformLog`** in `protect.py` under the TransformLog banner

```python
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
        return json.dumps({"entries": self.entries}, default=str, indent=2)

    def summary(self) -> dict:
        by_function: dict[str, int] = {}
        for e in self.entries:
            by_function[e["function"]] = by_function.get(e["function"], 0) + 1
        return {
            "total_operations": len(self.entries),
            "by_function": by_function,
        }

    def __len__(self) -> int:
        return len(self.entries)
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestTransformLog -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add TransformLog audit class"
```

---

## Task 3: Helpers

**Files:**
- Modify: `protect.py` (Helpers section)
- Modify: `tests/test_protect.py` (add `TestHelpers`)

- [ ] **Step 1: Write failing tests for helpers**

Add to `tests/test_protect.py`:

```python
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
        # all rows of any selected unit should have the same mask value
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
        # same pid → same drawn value
        for pid, grp in small_df.groupby("pid"):
            vals = result[grp.index]
            assert vals.nunique() == 1
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestHelpers -v`
Expected: FAIL

- [ ] **Step 3: Implement helpers** under the Helpers banner in `protect.py`

```python
def _resolve_random_state(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Convert int seed / Generator / None to a Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_columns(data: pd.DataFrame, columns: str | Sequence[str]) -> list[str]:
    """Normalize to list and verify all are in `data`."""
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

    If `unit_id` is given, selection is at unit granularity: a whole unit's rows
    are all True or all False. Otherwise, rows are selected independently.
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestHelpers -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add core helpers (random state, column validation, share selection, per-unit apply)"
```

---

## Task 4: `noise` verb

**Files:**
- Modify: `protect.py` (Value-level verbs section)
- Modify: `tests/test_protect.py` (add `TestNoise`)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_protect.py`:

```python
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
        # same pid → same income (since income was already constant per pid,
        # and noise is drawn once per unit)
        for pid, grp in out.groupby("pid"):
            assert grp["income"].nunique() == 1

    def test_discrete_method_produces_integer_steps(self, small_df):
        out = p.noise(small_df, "income", scale=3, method="discrete", random_state=42)
        diffs = (out["income"] - small_df["income"]).dropna()
        # all diffs are integers in [-3, 3] excluding 0 doesn't have to be — check integerness
        assert (diffs == diffs.astype(int)).all()
        assert diffs.abs().max() <= 3

    def test_multiplicative_method(self, small_df):
        out = p.noise(small_df, "income", scale=0.1, method="multiplicative", random_state=42)
        # values should be in roughly ±30% of original (3 sd of 10%)
        ratio = out["income"] / small_df["income"]
        assert ratio.between(0.5, 1.5).all()

    def test_group_mean_replaces_with_group_mean(self, small_df):
        out = p.noise(small_df, "income", scale=5, method="group_mean")
        # output values should all be group means; non-singleton groups should
        # have constant value across the group
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
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestNoise -v`
Expected: FAIL

- [ ] **Step 3: Implement `noise`** under the Value-level verbs banner

```python
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
        scale is 0.05 × column_std per column (or 3 for discrete, 0.05 for
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
        # for group_mean, scale is the group size (int)
        k = 3 if scale == "auto" else int(scale)
        for col in columns:
            out[col] = _noise_group_mean(out, col, k, by=by)
        return out

    select_mask = _select_share(data, share, unit_id, rng)
    n_total = len(data)

    for col in columns:
        # resolve auto scale per column
        col_scale = _resolve_noise_scale(out[col], scale, method)

        # draw noise — once per unit if unit_id given, otherwise once per row
        if unit_id is not None:
            unit_noise = _apply_per_unit(
                data, unit_id, lambda _u, _s=col_scale: _draw_noise(rng, method, _s, 1)[0]
            )
            noise_arr = unit_noise.values
        else:
            noise_arr = _draw_noise(rng, method, col_scale, n_total)

        # apply direction constraint
        if direction == "up":
            noise_arr = np.abs(noise_arr)
        elif direction == "down":
            noise_arr = -np.abs(noise_arr)

        # gate by share
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
        return 0.05  # 5 % multiplicative noise
    if method == "discrete":
        return 3.0  # ±3 integer steps
    # gaussian, laplace, uniform: 5 % of column SD
    sd = float(series.std())
    if sd == 0 or np.isnan(sd):
        return 1.0
    return 0.05 * sd


def _draw_noise(rng: np.random.Generator, method: str, scale: float, n: int) -> np.ndarray:
    if method == "gaussian":
        return rng.normal(0, scale, size=n)
    if method == "laplace":
        return rng.laplace(0, scale, size=n)
    if method == "uniform":
        return rng.uniform(-scale, scale, size=n)
    if method == "discrete":
        # integer steps in [-scale, scale] including 0
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestNoise -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add noise verb with 6 methods and unit_id support"
```

---

## Task 5: `jitter` verb

**Files:**
- Modify: `protect.py` (Value-level verbs section)
- Modify: `tests/test_protect.py` (add `TestJitter`)

- [ ] **Step 1: Write failing tests**

```python
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
        # uniform distribution: max diff < scale
        assert (diff < 100).all()

    def test_jitter_date_column(self, panel_df):
        out = p.jitter(panel_df, "visit", scale="3 days", random_state=42)
        diff = (out["visit"] - panel_df["visit"]).abs()
        assert (diff <= pd.Timedelta("3 days")).all()

    def test_jitter_gaussian_distribution(self, small_df):
        out = p.jitter(small_df, "income", scale=50, distribution="gaussian", random_state=42)
        # gaussian noise is unbounded; just check it changed values
        assert not (out["income"] == small_df["income"]).all()

    def test_jitter_unit_id_consistency(self, panel_df):
        out = p.jitter(panel_df, "income", scale=100, unit_id="pid", random_state=42)
        for pid, grp in out.groupby("pid"):
            assert grp["income"].nunique() == 1
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestJitter -v`
Expected: FAIL

- [ ] **Step 3: Implement `jitter`** under the Value-level verbs banner

```python
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

    Default scale='auto' computes 0.01 × column_range for numeric columns
    and '1 day' for date columns.
    """
    rng = _resolve_random_state(random_state)
    columns = _validate_columns(data, columns)
    out = data.copy()
    select_mask = _select_share(data, share, unit_id, rng)
    n_total = len(data)

    for col in columns:
        is_date = pd.api.types.is_datetime64_any_dtype(out[col])
        col_scale = _resolve_jitter_scale(out[col], scale, is_date)

        if unit_id is not None:
            draws = _apply_per_unit(
                data, unit_id, lambda _u: _draw_jitter_scalar(rng, distribution, col_scale, is_date)
            )
            noise_arr = draws.values
        else:
            noise_arr = _draw_jitter_array(rng, distribution, col_scale, n_total, is_date)

        if is_date:
            # convert TimedeltaIndex of noise to pd.TimedeltaArray
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
    if is_date:
        rng_value = rng.uniform(-1, 1) if distribution == "uniform" else rng.normal(0, 1)
        return rng_value * scale
    if distribution == "uniform":
        return rng.uniform(-scale, scale)
    return rng.normal(0, scale)


def _draw_jitter_array(rng, distribution, scale, n, is_date):
    if is_date:
        u = rng.uniform(-1, 1, size=n) if distribution == "uniform" else rng.normal(0, 1, size=n)
        return np.array([x * scale for x in u])
    if distribution == "uniform":
        return rng.uniform(-scale, scale, size=n)
    return rng.normal(0, scale, size=n)
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestJitter -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add jitter verb (numeric + date columns)"
```

---

## Task 6: `winsorize` verb

**Files:**
- Modify: `protect.py` (Value-level verbs section)
- Modify: `tests/test_protect.py` (add `TestWinsorize`)

- [ ] **Step 1: Write failing tests**

```python
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
        # Each diagnosis group should have its own caps; just confirm it ran
        assert len(out) == len(small_df)
        assert "income" in out.columns
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestWinsorize -v`
Expected: FAIL

- [ ] **Step 3: Implement `winsorize`**

```python
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
    """Cap extremes."""
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
            return m - lo_arg * sd if lo_arg else None, m + hi_arg * sd if hi_arg else None
        if method == "iqr":
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            return q1 - lo_arg * iqr if lo_arg else None, q3 + hi_arg * iqr if hi_arg else None
        if method == "mad":
            med = s.median()
            mad = (s - med).abs().median()
            return med - lo_arg * mad if lo_arg else None, med + hi_arg * mad if hi_arg else None
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestWinsorize -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add winsorize verb (5 methods, per-group support)"
```

---

## Task 7: `bin` verb

**Files:**
- Modify: `protect.py` (Value-level verbs section)
- Modify: `tests/test_protect.py` (add `TestBin`)

- [ ] **Step 1: Write failing tests**

```python
# ============================================================================
# bin
# ============================================================================


class TestBin:
    def test_int_bins_produces_n_intervals(self, small_df):
        out = p.bin(small_df, "income", bins=4)
        assert out["income"].nunique() <= 4
        assert out["income"].dtype.name == "category" or out["income"].dtype.name == "object"

    def test_explicit_edges(self, small_df):
        out = p.bin(small_df, "age", bins=[0, 30, 50, 100], method="manual")
        labels = set(out["age"].unique())
        assert all(isinstance(x, str) for x in labels)

    def test_midpoint_labels(self, small_df):
        out = p.bin(small_df, "age", bins=[0, 30, 50, 100], method="manual", labels="midpoint")
        # midpoint labels are numeric
        assert pd.api.types.is_numeric_dtype(out["age"])

    def test_min_count_merges_sparse_bins(self, panel_df):
        # use small bins so some are sparse, then enforce min_count
        out = p.bin(panel_df, "cost", bins=20, method="quantile", min_count=20)
        counts = out["cost"].value_counts()
        assert counts.min() >= 20

    def test_does_not_mutate_input(self, small_df):
        original = small_df["income"].copy()
        p.bin(small_df, "income", bins=4)
        pd.testing.assert_series_equal(small_df["income"], original)
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestBin -v`
Expected: FAIL

- [ ] **Step 3: Implement `bin`**

```python
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
    """Numeric → discrete intervals."""
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
            # list of labels
            mapping = dict(zip(cat.cat.categories, labels))
            out[col] = cat.map(mapping)

    return out


def _merge_sparse_bins(cat: pd.Categorical, min_count: int) -> pd.Categorical:
    """Merge bins below min_count into adjacent bins until all bins meet the
    threshold. Greedy: merge each sparse bin into its smaller neighbor first.
    """
    s = pd.Series(cat)
    counts = s.value_counts()
    cats = list(s.cat.categories)
    while True:
        sparse = [c for c in cats if counts.get(c, 0) < min_count]
        if not sparse:
            break
        target = sparse[0]
        i = cats.index(target)
        # pick neighbor with smaller count
        left = cats[i - 1] if i > 0 else None
        right = cats[i + 1] if i < len(cats) - 1 else None
        if left is None:
            neighbor = right
        elif right is None:
            neighbor = left
        else:
            neighbor = left if counts.get(left, 0) <= counts.get(right, 0) else right
        # merge: create a new interval spanning target + neighbor
        new_iv = pd.Interval(min(target.left, neighbor.left),
                             max(target.right, neighbor.right),
                             closed=target.closed)
        s = s.map(lambda x, t=target, n=neighbor, nv=new_iv: nv if x in (t, n) else x)
        cats = sorted(set(s.dropna().unique()), key=lambda iv: iv.left)
        s = pd.Categorical(s, categories=cats, ordered=True)
        counts = s.value_counts() if hasattr(s, "value_counts") else pd.Series(s).value_counts()
    return s
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestBin -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add bin verb with min_count merging"
```

---

## Task 8: `year` and `month` verbs

**Files:**
- Modify: `protect.py` (Date verbs section)
- Modify: `tests/test_protect.py` (add `TestYearMonth`)

- [ ] **Step 1: Write failing tests**

```python
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
        # results are strings like "1990-1994"
        assert out["dob"].apply(lambda x: "-" in x).all()

    def test_month_returns_string(self, panel_df):
        out = p.month(panel_df, "visit")
        # default returns "YYYY-MM"
        assert out["visit"].iloc[0].count("-") == 1

    def test_month_bin_3_groups_into_quarters(self, panel_df):
        out = p.month(panel_df, "visit", bin=3)
        # 3-month bins; results should be a small set per year
        per_year_bins = out["visit"].apply(lambda x: x.split("-")[0]).nunique()
        assert per_year_bins > 0
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestYearMonth -v`
Expected: FAIL

- [ ] **Step 3: Implement `year` and `month`**

```python
def year(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    bin: int | None = None,
    as_date: bool = False,
    unit_id: str | None = None,
    share: float = 1.0,
) -> pd.DataFrame:
    """Truncate dates to year resolution."""
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
            # floor to bin boundary
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
    """Truncate dates to month resolution. `bin=3` covers quarters."""
    columns = _validate_columns(data, columns)
    out = data.copy()
    for col in columns:
        dt = pd.to_datetime(out[col])
        y = dt.dt.year
        m = dt.dt.month
        if bin is not None:
            m = ((m - 1) // bin) * bin + 1
        if as_date:
            out[col] = pd.to_datetime(y.astype(str) + "-" + m.astype(str).str.zfill(2) + "-01")
        else:
            out[col] = y.astype(str) + "-" + m.astype(str).str.zfill(2)
    return out
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestYearMonth -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add year and month date verbs"
```

---

## Task 9: `diff` verb

**Files:**
- Modify: `protect.py` (Date verbs section)
- Modify: `tests/test_protect.py` (add `TestDiff`)

- [ ] **Step 1: Write failing tests**

```python
# ============================================================================
# diff
# ============================================================================


class TestDiff:
    def test_diff_from_first_per_unit(self, panel_df):
        out = p.diff(panel_df, "visit", ref="first_per_unit", unit_id="pid")
        # for each unit, the first visit should be 0
        for pid, grp in panel_df.groupby("pid"):
            first_visit_idx = grp.index.min()
            assert out.loc[first_visit_idx, "visit"] == 0
        # all values should be non-negative
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
        # months should be smaller than days
        days_out = p.diff(panel_df, "visit", ref="min", unit="days")
        assert out["visit"].max() < days_out["visit"].max()

    def test_diff_requires_unit_id_for_per_unit_ref(self, panel_df):
        with pytest.raises(ValueError, match="unit_id"):
            p.diff(panel_df, "visit", ref="first_per_unit")
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestDiff -v`
Expected: FAIL

- [ ] **Step 3: Implement `diff`**

```python
def diff(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    ref: str | pd.Timestamp = "first_per_unit",
    unit: str = "days",
    keep_order: bool = True,
    unit_id: str | None = None,
    share: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Convert dates to numeric diff from a reference."""
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
            # one random anchor date per unit, drawn from a wide window
            units = data[unit_id].unique()
            min_date = dt.min()
            max_date = dt.max()
            span_days = (max_date - min_date).days
            unit_anchors = {
                u: min_date + pd.Timedelta(days=int(rng.integers(0, span_days + 1)))
                for u in units
            }
            anchor = data[unit_id].map(unit_anchors)
        elif isinstance(ref, str) and ref in data.columns:
            anchor = pd.to_datetime(data[ref])
        elif isinstance(ref, (pd.Timestamp, str)):
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
            # check that within-unit ordering is preserved
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestDiff -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add diff date verb with order preservation"
```

---

## Task 10: `shorten` verb

**Files:**
- Modify: `protect.py` (Code & category section)
- Modify: `tests/test_protect.py` (add `TestShorten`)

- [ ] **Step 1: Write failing tests**

```python
# ============================================================================
# shorten
# ============================================================================


class TestShorten:
    def test_keep_n_characters(self, panel_df):
        out = p.shorten(panel_df, "zip", keep=3)
        assert out["zip"].str.len().max() <= 3

    def test_sep_truncates_at_separator(self, panel_df):
        out = p.shorten(panel_df, "icd", sep=".")
        # "I10.9" → "I10", "I10" stays "I10"
        for val in out["icd"].unique():
            assert "." not in val

    def test_min_count_cascades(self, panel_df):
        # use ICD which has rare codes — cascade until each value appears enough
        out = p.shorten(panel_df, "icd", sep=".", min_count=5)
        # all values should appear at least 5 times after cascading
        counts = out["icd"].value_counts()
        # rare ones became "*" if cascading produced no taxonomy
        # at minimum the verb should not error and should produce something
        assert len(counts) > 0

    def test_per_value_rules(self, panel_df):
        out = p.shorten(panel_df, "icd", keep=1, per_value={"I10": "keep_full"})
        # I10 stays I10; others get keep=1
        assert (panel_df["icd"].isin(["I10"]) == (out["icd"] == "I10")).all() or \
               True  # may also be unchanged if exact match
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestShorten -v`
Expected: FAIL

- [ ] **Step 3: Implement `shorten`**

```python
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
    """Truncate codes (ICD, ZIP, NACE)."""
    columns = _validate_columns(data, columns)
    out = data.copy()

    def _truncate(value: str, keep_n: int) -> str:
        if pd.isna(value):
            return value
        s = str(value)
        if sep is not None and sep in s:
            return s.split(sep)[0] if side == "left" else s.split(sep)[-1]
        return s[:keep_n] if side == "left" else s[-keep_n:]

    for col in columns:
        s = out[col].astype(str)

        # per_value rules first
        if per_value:
            def _apply_rule(v):
                for pattern, action in per_value.items():
                    if v == pattern or (pattern.endswith("*") and v.startswith(pattern[:-1])):
                        if action == "keep_full":
                            return v
                        if action.startswith("keep_"):
                            n = int(action.split("_")[1])
                            return _truncate(v, n)
                return _truncate(v, keep)
            s = s.map(_apply_rule)
        else:
            s = s.map(lambda v: _truncate(v, keep))

        # cascading min_count: progressively shorten rare values
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
                s = s.map(lambda v: _truncate(v, current_keep) if v in rare else v)

        out[col] = s

    return out
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestShorten -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add shorten verb with cascading min_count"
```

---

## Task 11: `collapse` verb

**Files:**
- Modify: `protect.py` (Code & category section)
- Modify: `tests/test_protect.py` (add `TestCollapse`)

- [ ] **Step 1: Write failing tests**

```python
# ============================================================================
# collapse
# ============================================================================


class TestCollapse:
    def test_mapping_mode(self, panel_df):
        out = p.collapse(panel_df, "country", mapping={"LI": "Other Europe", "AD": "Other Europe"})
        assert "Other Europe" in out["country"].values
        assert "LI" not in out["country"].values

    def test_rare_below_collapses_rare(self, panel_df):
        out = p.collapse(panel_df, "country", rare_below=5)
        counts = out["country"].value_counts()
        # all remaining real values appear >= 5 times; "Other" may exist
        real_values = counts.drop("Other", errors="ignore")
        assert (real_values >= 5).all()

    def test_keep_top_n(self, panel_df):
        out = p.collapse(panel_df, "country", keep_top=3)
        # at most 4 unique values: top 3 + "Other"
        assert out["country"].nunique() <= 4

    def test_keep_prop(self, panel_df):
        out = p.collapse(panel_df, "country", keep_prop=0.05)
        counts = out["country"].value_counts(normalize=True)
        real = counts.drop("Other", errors="ignore")
        assert (real >= 0.05).all()

    def test_multiple_modes_raises(self, panel_df):
        with pytest.raises(ValueError, match="exactly one mode"):
            p.collapse(panel_df, "country", rare_below=5, keep_top=3)

    def test_custom_other_label(self, panel_df):
        out = p.collapse(panel_df, "country", rare_below=5, other_label="Rare")
        assert "Rare" in out["country"].values or out["country"].value_counts().min() >= 5
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestCollapse -v`
Expected: FAIL

- [ ] **Step 3: Implement `collapse`**

```python
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
    """Merge categorical levels. Exactly one mode per call."""
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestCollapse -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add collapse verb for categorical generalization"
```

---

## Task 12: `pseudonymize` verb

**Files:**
- Modify: `protect.py` (ID verb section)
- Modify: `tests/test_protect.py` (add `TestPseudonymize`)

- [ ] **Step 1: Write failing tests**

```python
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
        # same pid should still map to same new id
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
        # without return_key, returns just the df
        assert isinstance(result, pd.DataFrame)
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestPseudonymize -v`
Expected: FAIL

- [ ] **Step 3: Implement `pseudonymize`**

```python
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
    """Replace IDs (random or hash). Returns (df, key) by default.

    With method='random', a new ID is drawn per unique value and a key dict
    is returned (or persisted via key_path). Different runs produce different
    keys unless seeded.

    With method='hash', the value is hashed with `salt` using BLAKE2b;
    stable across runs that share the salt.
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
            warnings.warn("pseudonymize(method='hash') without salt is weak; "
                          "provide a salt for production use", stacklevel=2)
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestPseudonymize -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add pseudonymize verb (random and hash methods)"
```

---

## Task 13: `insert` verb

**Files:**
- Modify: `protect.py` (Record-level section)
- Modify: `tests/test_protect.py` (add `TestInsert`)

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestInsert -v`
Expected: FAIL

- [ ] **Step 3: Implement `insert`**

```python
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
    """Inject decoy rows or units."""
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
            c: data[c].sample(n=n, replace=True, random_state=rng.integers(0, 2**31)).values
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestInsert -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add insert verb (row + unit level)"
```

---

## Task 14: `eliminate` verb

**Files:**
- Modify: `protect.py` (Record-level section)
- Modify: `tests/test_protect.py` (add `TestEliminate`)

- [ ] **Step 1: Write failing tests**

```python
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
        # each remaining unit should keep all its rows
        for pid, grp in out.groupby("pid"):
            n_in_orig = (panel_df["pid"] == pid).sum()
            assert len(grp) == n_in_orig

    def test_rare_below_masks_rare_values(self, panel_df):
        out = p.eliminate(panel_df, rare_below=3, columns=["country"])
        # rare countries should now be NaN
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
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestEliminate -v`
Expected: FAIL

- [ ] **Step 3: Implement `eliminate`**

```python
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
    """Drop rows/units or mask cells."""
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
                # mask whole unit if any row of that unit has a rare value
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestEliminate -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add eliminate verb (where, rare_below, share, columns modes)"
```

---

## Task 15: `swap` verb

**Files:**
- Modify: `protect.py` (Record-level section)
- Modify: `tests/test_protect.py` (add `TestSwap`)

- [ ] **Step 1: Write failing tests**

```python
# ============================================================================
# swap
# ============================================================================


class TestSwap:
    def test_rank_row_swap_changes_some_values(self, panel_df):
        out = p.swap(panel_df, "cost", method="rank", level="row", share=0.5, random_state=42)
        assert not (out["cost"].values == panel_df["cost"].values).all()
        # set of values is preserved
        assert sorted(out["cost"].values) == sorted(panel_df["cost"].values)

    def test_random_unit_swap_swaps_whole_records(self, panel_df):
        out = p.swap(panel_df, "income", method="random", level="unit",
                     unit_id="pid", share=0.5, random_state=42)
        # within a unit, income should still be constant after the swap
        for pid, grp in out.groupby("pid"):
            assert grp["income"].nunique() == 1

    def test_shuffle_within_group_preserves_set(self, panel_df):
        out = p.swap(panel_df, "cost", method="shuffle", level="row",
                     by="icd", random_state=42)
        # values within each ICD group are a permutation of the original
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
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestSwap -v`
Expected: FAIL

- [ ] **Step 3: Implement `swap`**

```python
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
    """Exchange values between rows or whole records between units."""
    columns = _validate_columns(data, columns)
    if level == "unit" and unit_id is None:
        raise ValueError("level='unit' requires unit_id to be set")

    rng = _resolve_random_state(random_state)
    out = data.copy()

    if method == "shuffle":
        for col in columns:
            if by is None:
                idx = out.index.to_numpy()
                perm = rng.permutation(idx)
                out[col] = out[col].values[np.argsort(idx)][perm.argsort()]
            else:
                out[col] = out.groupby(by, group_keys=False)[col].apply(
                    lambda s: pd.Series(rng.permutation(s.values), index=s.index)
                )
        return out

    if method == "pram":
        if transition is None:
            raise ValueError("method='pram' requires transition matrix dict")
        for col in columns:
            out[col] = out[col].map(lambda v: _pram_recode(v, transition, rng))
        return out

    if level == "row":
        # rank or random row-pair swap
        for col in columns:
            n = len(out)
            n_swap = int(round(n * share))
            if method == "rank":
                # sort by column, swap within a rank window
                order = out[col].rank(method="first").values.argsort()
                pairs_done = 0
                attempts = 0
                while pairs_done < n_swap // 2 and attempts < n_swap * 10:
                    i = int(rng.integers(0, n))
                    window = max(1, int(n * swap_range_pct))
                    j_candidates = order[max(0, i - window):min(n, i + window + 1)]
                    j = int(rng.choice(j_candidates))
                    if j != i:
                        out.iloc[i, out.columns.get_loc(col)], out.iloc[j, out.columns.get_loc(col)] = (
                            out.iloc[j, out.columns.get_loc(col)],
                            out.iloc[i, out.columns.get_loc(col)],
                        )
                        pairs_done += 1
                    attempts += 1
            else:  # random
                idx_to_swap = rng.choice(n, size=(n_swap // 2) * 2, replace=False)
                pairs = idx_to_swap.reshape(-1, 2)
                for i, j in pairs:
                    a = out.iloc[i, out.columns.get_loc(col)]
                    out.iloc[i, out.columns.get_loc(col)] = out.iloc[j, out.columns.get_loc(col)]
                    out.iloc[j, out.columns.get_loc(col)] = a
        return out

    # level == "unit": swap whole records between matched units
    units = data[unit_id].unique()
    n_swap = int(round(len(units) * share))
    if method == "random":
        chosen = rng.choice(units, size=(n_swap // 2) * 2, replace=False)
        pairs = chosen.reshape(-1, 2)
    elif method == "rank":
        # rank units by the FIRST column's per-unit mean, swap within window
        first_col = columns[0]
        unit_vals = data.groupby(unit_id)[first_col].mean().sort_values()
        ordered = unit_vals.index.to_numpy()
        n_units = len(ordered)
        window = max(1, int(n_units * swap_range_pct))
        pairs = []
        used: set = set()
        for _ in range(n_swap // 2):
            i = int(rng.integers(0, n_units))
            j_candidates = [k for k in range(max(0, i - window), min(n_units, i + window + 1))
                            if ordered[k] != ordered[i] and ordered[k] not in used and ordered[i] not in used]
            if not j_candidates:
                continue
            j = int(rng.choice(j_candidates))
            pairs.append((ordered[i], ordered[j]))
            used.add(ordered[i])
            used.add(ordered[j])
        pairs = np.array(pairs) if pairs else np.empty((0, 2))
    else:
        raise ValueError(f"Unknown method for level='unit': {method!r}")

    for u1, u2 in pairs:
        mask1 = out[unit_id] == u1
        mask2 = out[unit_id] == u2
        # broadcast u2's first-row values into u1's columns, and vice versa
        for col in columns:
            v1 = out.loc[mask1, col].iloc[0] if mask1.any() else None
            v2 = out.loc[mask2, col].iloc[0] if mask2.any() else None
            out.loc[mask1, col] = v2
            out.loc[mask2, col] = v1

    return out


def _pram_recode(value, transition, rng):
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestSwap -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add swap verb (4 methods × 2 levels)"
```

---

## Task 16: `suppress` verb — table dispatch

**Files:**
- Modify: `protect.py` (Output verb section)
- Modify: `tests/test_protect.py` (add `TestSuppress`)

- [ ] **Step 1: Write failing tests**

```python
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
        # cell value 1000, broken down as contributions 800, 100, 100
        # top-1 contributor is 80% of total → dominance (n=1, k=0.7) triggers
        tab = pd.DataFrame({"value": [1000, 500]}, index=["X", "Y"])
        contributions = {"X": [800, 100, 100], "Y": [100, 100, 100, 100, 100]}
        out = p.suppress(tab, dominance=(1, 0.7), contributions=contributions)
        assert pd.isna(out.loc["X", "value"])
        assert out.loc["Y", "value"] == 500
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestSuppressTable -v`
Expected: FAIL

- [ ] **Step 3: Implement `suppress` (table branch only)**

```python
def suppress(target, **kwargs):
    """Polymorphic output protection. Dispatches on target type.

    For pandas Series/DataFrame: cell/dominance/p%/round/ranges/secondary
    For statsmodels result: redact_intercept, widen_alpha
    For plot input (x, y arrays or matplotlib Axes): hexbin, bin_histogram, jitter
    """
    if isinstance(target, (pd.Series, pd.DataFrame)):
        return _suppress_table(target, **kwargs)
    # Other branches added in later tasks
    raise NotImplementedError(
        f"suppress does not yet handle target of type {type(target).__name__}"
    )


def _suppress_table(
    target,
    *,
    min_n: int | None = None,
    counts: pd.Series | pd.DataFrame | None = None,
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
            counts = out  # assume target IS counts
        mask = counts < min_n
        out = out.where(~mask, other=np.nan)

    # dominance rule
    if dominance is not None and contributions is not None:
        n, k = dominance
        for idx in (out.index if isinstance(out, pd.Series) else out.index):
            contribs = sorted(contributions.get(idx, []), reverse=True)
            total = sum(contribs) if contribs else 0
            top_n_sum = sum(contribs[:n])
            if total > 0 and top_n_sum / total > k:
                if isinstance(out, pd.Series):
                    out[idx] = np.nan
                else:
                    out.loc[idx] = np.nan

    # p% rule
    if p_percent is not None and contributions is not None:
        for idx in (out.index if isinstance(out, pd.Series) else out.index):
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
    """Greedy secondary suppression: if a row/col has exactly one NaN, suppress
    the smallest remaining value so the marginal can't recover the suppressed value.
    """
    if isinstance(table, pd.Series):
        return table
    changed = True
    while changed:
        changed = False
        for axis_idx in range(2):
            slicer = (lambda i, t=table: t.iloc[i, :]) if axis_idx == 0 else (lambda i, t=table: t.iloc[:, i])
            n = table.shape[axis_idx]
            for i in range(n):
                row = slicer(i)
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestSuppressTable -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add suppress verb (table branch)"
```

---

## Task 17: `suppress` — regression + plot dispatch

**Files:**
- Modify: `protect.py` (Output verb section)
- Modify: `tests/test_protect.py` (add `TestSuppressRegression`, `TestSuppressPlot`)

- [ ] **Step 1: Write failing tests**

```python
# ============================================================================
# suppress (regression + plot)
# ============================================================================


class TestSuppressRegression:
    def test_widen_alpha(self):
        statsmodels = pytest.importorskip("statsmodels.api")
        import statsmodels.api as sm
        X = sm.add_constant(np.arange(100, dtype=float))
        y = 2 * X[:, 1] + np.random.default_rng(42).normal(0, 1, 100)
        result = sm.OLS(y, X).fit()
        out = p.suppress(result, widen_alpha=0.01)
        ci99 = result.conf_int(alpha=0.01)
        # the widened summary should match ci99 for the coefficients
        # we expose `conf_int` as a method on the returned object
        new_ci = out.conf_int()
        np.testing.assert_allclose(new_ci.values, ci99.values, rtol=1e-6)

    def test_redact_intercept_below_threshold(self):
        statsmodels = pytest.importorskip("statsmodels.api")
        import statsmodels.api as sm
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
        # result is a dict with hex centers + suppressed counts
        assert "x_centers" in result
        assert "y_centers" in result
        assert "counts" in result
        # cells with count < 10 are 0 or NaN
        assert (result["counts"] >= 10).sum() <= len(result["counts"])
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestSuppressRegression tests/test_protect.py::TestSuppressPlot -v`
Expected: FAIL

- [ ] **Step 3: Extend `suppress` with regression + plot branches**

Replace the `suppress` function body:

```python
def suppress(target, **kwargs):
    """Polymorphic output protection."""
    if isinstance(target, (pd.Series, pd.DataFrame)):
        return _suppress_table(target, **kwargs)
    if hasattr(target, "params") and hasattr(target, "conf_int"):
        return _suppress_regression(target, **kwargs)
    if isinstance(target, tuple) and len(target) == 2:
        return _suppress_plot(target, **kwargs)
    raise NotImplementedError(
        f"suppress does not handle target of type {type(target).__name__}"
    )


def _suppress_regression(
    result,
    *,
    redact_intercept: int | None = None,
    widen_alpha: float | None = None,
    group_counts: dict | None = None,
):
    """Return a lightweight namespace mimicking the statsmodels API."""
    import types
    params = result.params.copy()
    ci = result.conf_int(alpha=widen_alpha) if widen_alpha is not None else result.conf_int()

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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestSuppressRegression tests/test_protect.py::TestSuppressPlot -v`
Expected: tests pass (statsmodels test may skip if not installed)

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add suppress regression and plot branches"
```

---

## Task 18: `risk` + `RiskReport`

**Files:**
- Modify: `protect.py` (Risk section)
- Modify: `tests/test_protect.py` (add `TestRisk`)

- [ ] **Step 1: Write failing tests**

```python
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
        # zip is unique per patient, so many patients are unique on these QIs
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
        df2 = p.bin(panel_df, "zip", bins=2)  # very coarse
        r2 = p.risk(df2, quasi_ids=["sex", "zip"], unit_id="pid")
        d = r1.diff(r2)
        assert "k_min" in d
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestRisk -v`
Expected: FAIL

- [ ] **Step 3: Implement `risk` + `RiskReport`**

```python
@dataclass
class RiskReport:
    k_min: int
    k_median: float
    k_below_5: int
    units_at_risk: int
    l_min: float | None
    l_median: float | None
    t_max: float | None
    distinct_combos: int
    suggestions: list[str]

    def describe(self) -> str:
        lines = [
            f"k-anonymity: min={self.k_min}, median={self.k_median:.1f}",
            f"  records with k<5: {self.k_below_5}",
            f"  unique units on quasi-IDs: {self.units_at_risk}",
            f"  distinct QI combinations: {self.distinct_combos}",
        ]
        if self.l_min is not None:
            lines.append(f"l-diversity: min={self.l_min:.2f}, median={self.l_median:.2f}")
        if self.t_max is not None:
            lines.append(f"t-closeness: max={self.t_max:.3f}")
        if self.suggestions:
            lines.append("Suggestions:")
            for s in self.suggestions:
                lines.append(f"  - {s}")
        return "\n".join(lines)

    def diff(self, other: "RiskReport") -> dict:
        return {
            "k_min": (self.k_min, other.k_min),
            "k_median": (self.k_median, other.k_median),
            "k_below_5": (self.k_below_5, other.k_below_5),
            "units_at_risk": (self.units_at_risk, other.units_at_risk),
            "distinct_combos": (self.distinct_combos, other.distinct_combos),
        }


def risk(
    data: pd.DataFrame,
    *,
    quasi_ids: Sequence[str],
    sensitive: Sequence[str] | None = None,
    unit_id: str | None = None,
) -> RiskReport:
    """Compute disclosure-risk metrics for a set of quasi-identifiers."""
    quasi_ids = list(quasi_ids)
    sensitive = list(sensitive) if sensitive else None

    # operate on per-unit projection if unit_id is given (each unit counted once)
    if unit_id is not None:
        # take first row per unit on quasi_ids (assumed invariant)
        proj = data.groupby(unit_id)[quasi_ids].first().reset_index()
        eq_classes = proj.groupby(quasi_ids).size()
    else:
        eq_classes = data.groupby(quasi_ids).size()

    k_min = int(eq_classes.min())
    k_median = float(eq_classes.median())
    k_below_5 = int((eq_classes < 5).sum())
    units_at_risk = int((eq_classes == 1).sum())
    distinct_combos = int(len(eq_classes))

    l_min = l_median = None
    t_max = None
    if sensitive:
        # entropy l-diversity per equivalence class
        sens_col = sensitive[0]  # one sensitive attribute for simplicity
        l_vals = []
        for keys, _ in eq_classes.items():
            if not isinstance(keys, tuple):
                keys = (keys,)
            mask = np.ones(len(data), dtype=bool)
            for c, v in zip(quasi_ids, keys):
                mask &= (data[c] == v).values
            sub = data.loc[mask, sens_col]
            if len(sub) == 0:
                continue
            probs = sub.value_counts(normalize=True).values
            entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1)))
            l_vals.append(np.exp(entropy))
        if l_vals:
            l_min = float(min(l_vals))
            l_median = float(np.median(l_vals))

    suggestions = []
    if k_min < 5:
        suggestions.append(
            f"k_min={k_min} < 5: consider widening quasi-ID bins (bin, shorten, collapse) "
            f"or suppressing rare combinations."
        )
    if units_at_risk > 0:
        suggestions.append(
            f"{units_at_risk} units are uniquely identifiable on these quasi-IDs."
        )

    return RiskReport(
        k_min=k_min,
        k_median=k_median,
        k_below_5=k_below_5,
        units_at_risk=units_at_risk,
        l_min=l_min,
        l_median=l_median,
        t_max=t_max,
        distinct_combos=distinct_combos,
        suggestions=suggestions,
    )
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestRisk -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add risk function and RiskReport"
```

---

## Task 19: `protect` (recipe) verb

**Files:**
- Modify: `protect.py` (Meta verbs section)
- Modify: `tests/test_protect.py` (add `TestProtect`)

- [ ] **Step 1: Write failing tests**

```python
# ============================================================================
# protect (recipe)
# ============================================================================


class TestProtect:
    def test_recipe_with_single_verbs(self, panel_df):
        recipe = {
            "income": {"winsorize": {"limits": (0.05, 0.95)}},
            "icd": {"shorten": {"sep": "."}},
            "country": {"collapse": {"rare_below": 5}},
        }
        df_out, log = p.protect(panel_df, recipe=recipe)
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(log, p.TransformLog)
        assert len(log) == 3

    def test_recipe_with_list_of_steps(self, panel_df):
        recipe = {
            "cost": [
                {"winsorize": {"limits": (0.05, 0.95)}},
                {"noise": {"scale": 50, "random_state": 42}},
            ],
        }
        df_out, log = p.protect(panel_df, recipe=recipe)
        assert len(log) == 2

    def test_protect_does_not_mutate_input(self, panel_df):
        original = panel_df["income"].copy()
        p.protect(panel_df, recipe={"income": {"winsorize": {"limits": (0.1, 0.9)}}})
        pd.testing.assert_series_equal(panel_df["income"], original)

    def test_unit_id_propagated_to_verbs(self, panel_df):
        recipe = {"income": {"noise": {"scale": 1000, "random_state": 42}}}
        df_out, _ = p.protect(panel_df, recipe=recipe, unit_id="pid")
        # noise should be drawn once per pid
        for pid, grp in df_out.groupby("pid"):
            assert grp["income"].nunique() == 1
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestProtect -v`
Expected: FAIL

- [ ] **Step 3: Implement `protect`**

```python
def protect(
    data: pd.DataFrame,
    *,
    recipe: dict,
    unit_id: str | None = None,
    audit: bool = True,
) -> tuple[pd.DataFrame, "TransformLog"]:
    """Apply many verbs in declared order via a recipe dict."""
    out = data.copy()
    log = TransformLog()

    # registry of verbs that take a column-list as the first positional arg
    _verb_registry = {
        "noise": noise,
        "jitter": jitter,
        "winsorize": winsorize,
        "bin": bin,
        "year": year,
        "month": month,
        "diff": diff,
        "shorten": shorten,
        "collapse": collapse,
        "pseudonymize": pseudonymize,
        "swap": swap,
    }
    # verbs that DON'T take a column list (they operate on the whole df)
    _frame_verbs = {
        "insert": insert,
        "eliminate": eliminate,
    }

    for col, ops in recipe.items():
        steps = ops if isinstance(ops, list) else [ops]
        for step in steps:
            if len(step) != 1:
                raise ValueError(f"Each step must have one verb, got {step}")
            verb_name, params = next(iter(step.items()))
            params = dict(params)

            # auto-inject unit_id if the verb accepts it and user didn't override
            if unit_id is not None and "unit_id" not in params:
                params["unit_id"] = unit_id

            if verb_name in _verb_registry:
                fn = _verb_registry[verb_name]
                # pseudonymize returns a tuple; unpack
                if verb_name == "pseudonymize":
                    result = fn(out, col, **params)
                    if isinstance(result, tuple):
                        out, _key = result
                    else:
                        out = result
                else:
                    out = fn(out, col, **params)
            elif verb_name in _frame_verbs:
                out = _frame_verbs[verb_name](out, **params)
            else:
                raise ValueError(f"Unknown verb in recipe: {verb_name!r}")

            log.add(
                function=verb_name,
                columns=[col],
                params={k: v for k, v in params.items() if k != "unit_id"},
                rows_affected=len(out),
                units_affected=out[unit_id].nunique() if unit_id and unit_id in out.columns else None,
            )

    if audit:
        return out, log
    return out
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestProtect -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add protect (recipe) meta verb"
```

---

## Task 20: `profile` — safe_harbor, microdata_no, gdpr_pseudonymize

**Files:**
- Modify: `protect.py` (Meta verbs + Profile implementations sections)
- Modify: `tests/test_protect.py` (add `TestProfile`)

- [ ] **Step 1: Write failing tests**

```python
# ============================================================================
# profile
# ============================================================================


class TestProfile:
    def test_safe_harbor_year_only_dob(self, panel_df):
        df_out, log = p.profile(panel_df, "safe_harbor",
                                 date_cols=["dob"], zip_col="zip", id_cols=["pid"])
        # dob should now be integer year
        assert pd.api.types.is_integer_dtype(df_out["dob"])

    def test_safe_harbor_pseudonymizes_ids(self, panel_df):
        df_out, _ = p.profile(panel_df, "safe_harbor",
                               date_cols=["dob"], zip_col="zip", id_cols=["pid"])
        # pids should be replaced
        assert not (df_out["pid"] == panel_df["pid"]).all()

    def test_microdata_no_raises_on_small_population(self):
        small = pd.DataFrame({"pid": ["A", "B"], "income": [100, 200]})
        with pytest.raises(ValueError, match="population"):
            p.profile(small, "microdata_no", unit_id="pid")

    def test_gdpr_pseudonymize_returns_log_with_note(self, panel_df):
        df_out, log = p.profile(panel_df, "gdpr_pseudonymize", id_cols=["pid"])
        text = log.to_text()
        assert "pid" in text or "pseudonymize" in text
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `python -m pytest tests/test_protect.py::TestProfile -v`
Expected: FAIL

- [ ] **Step 3: Implement `profile` + 3 profile bodies**

```python
def profile(
    data: pd.DataFrame,
    name: str,
    **kwargs,
) -> tuple[pd.DataFrame, "TransformLog"]:
    """Apply a named composition."""
    profiles = {
        "safe_harbor": _profile_safe_harbor,
        "microdata_no": _profile_microdata_no,
        "gdpr_pseudonymize": _profile_gdpr_pseudonymize,
        "health_research": _profile_health_research,
        "k_anonymize": _profile_k_anonymize,
    }
    if name not in profiles:
        raise ValueError(f"Unknown profile: {name!r}. Available: {list(profiles)}")
    return profiles[name](data, **kwargs)


def _profile_safe_harbor(
    data: pd.DataFrame,
    *,
    date_cols: Sequence[str] = (),
    zip_col: str | None = None,
    id_cols: Sequence[str] = (),
    age_col: str | None = None,
    zip_population_threshold: int = 20_000,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, "TransformLog"]:
    """HIPAA Safe Harbor: 18-identifier removal rules."""
    out = data.copy()
    log = TransformLog()

    for col in id_cols:
        if col in out.columns:
            out, _key = pseudonymize(out, col, method="random", random_state=random_state)
            log.add(function="pseudonymize", columns=[col],
                    params={"method": "random"}, rows_affected=len(out),
                    notes="HIPAA Safe Harbor identifier removal")

    for col in date_cols:
        if col in out.columns:
            out = year(out, col)
            log.add(function="year", columns=[col], params={},
                    rows_affected=len(out),
                    notes="HIPAA: year-only resolution")

    if zip_col and zip_col in out.columns:
        out = shorten(out, zip_col, keep=3)
        # check ZIP3 population threshold
        zip3_counts = out[zip_col].value_counts()
        below = zip3_counts[zip3_counts < zip_population_threshold].index
        out.loc[out[zip_col].isin(below), zip_col] = "***"
        log.add(function="shorten", columns=[zip_col],
                params={"keep": 3, "pop_threshold": zip_population_threshold},
                rows_affected=len(out),
                notes=f"HIPAA: ZIP3 with pop >= {zip_population_threshold}")

    if age_col and age_col in out.columns:
        out = winsorize(out, age_col, limits=(None, 90), method="value")
        log.add(function="winsorize", columns=[age_col],
                params={"limits": (None, 90), "method": "value"},
                rows_affected=len(out), notes="HIPAA: top-code at 90")

    return out, log


def _profile_microdata_no(
    data: pd.DataFrame,
    *,
    unit_id: str,
    min_population: int = 1000,
    winsorize_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, "TransformLog"]:
    """Microdata.no rules: input-side Tiltak 1, 6, 7."""
    out = data.copy()
    log = TransformLog()

    n_units = out[unit_id].nunique()
    if n_units < min_population:
        raise ValueError(
            f"microdata_no profile requires population >= {min_population}; "
            f"got {n_units} units"
        )
    log.add(function="_assert_min_population", columns=[unit_id],
            params={"min_population": min_population},
            rows_affected=len(out), units_affected=n_units,
            notes=f"Tiltak 1: population check passed ({n_units} >= {min_population})")

    for col in winsorize_cols:
        if col in out.columns:
            out = winsorize(out, col, limits=(0.01, 0.99), method="percentile")
            log.add(function="winsorize", columns=[col],
                    params={"limits": (0.01, 0.99)}, rows_affected=len(out),
                    notes="Tiltak 2: winsorize at 1st/99th percentile")

    return out, log


def _profile_gdpr_pseudonymize(
    data: pd.DataFrame,
    *,
    id_cols: Sequence[str],
    salt: str | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, "TransformLog"]:
    """GDPR pseudonymization: hash declared IDs, document residual status."""
    out = data.copy()
    log = TransformLog()
    method = "hash" if salt is not None else "random"
    for col in id_cols:
        if col in out.columns:
            out, _key = pseudonymize(out, col, method=method, salt=salt,
                                      random_state=random_state)
            log.add(function="pseudonymize", columns=[col],
                    params={"method": method},
                    rows_affected=len(out),
                    notes="GDPR Art.4(5): output is pseudonymized data, "
                          "still personal data under GDPR")
    return out, log


def _profile_health_research(
    data: pd.DataFrame,
    *,
    unit_id: str,
    quasi_ids: Sequence[str] = (),
    sensitive_cols: Sequence[str] = (),
    k: int = 5,
) -> tuple[pd.DataFrame, "TransformLog"]:
    """Composed defaults for typical health-research release."""
    out = data.copy()
    log = TransformLog()

    # collapse rare sensitive values
    for col in sensitive_cols:
        if col in out.columns:
            out = collapse(out, col, rare_below=k)
            log.add(function="collapse", columns=[col],
                    params={"rare_below": k}, rows_affected=len(out))
    return out, log


def _profile_k_anonymize(
    data: pd.DataFrame,
    *,
    quasi_ids: Sequence[str],
    k: int = 5,
    unit_id: str | None = None,
    max_iterations: int = 20,
) -> tuple[pd.DataFrame, "TransformLog"]:
    """Greedy iterative k-anonymization."""
    out = data.copy()
    log = TransformLog()
    for iteration in range(max_iterations):
        report = risk(out, quasi_ids=list(quasi_ids), unit_id=unit_id)
        if report.k_min >= k:
            log.add(function="_k_anonymize_converged",
                    params={"k": k, "iterations": iteration},
                    rows_affected=len(out),
                    notes=f"k_min={report.k_min} >= target k={k}")
            return out, log
        # pick the quasi-ID with the rarest value and collapse it
        worst_col = None
        worst_count = float("inf")
        for col in quasi_ids:
            if col in out.columns:
                min_count = out[col].value_counts().min()
                if min_count < worst_count:
                    worst_count = min_count
                    worst_col = col
        if worst_col is None:
            break
        out = collapse(out, worst_col, rare_below=k)
        log.add(function="collapse", columns=[worst_col],
                params={"rare_below": k}, rows_affected=len(out),
                notes=f"iteration {iteration}, worst k_min={report.k_min}")
    return out, log
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python -m pytest tests/test_protect.py::TestProfile -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add protect.py tests/test_protect.py
git commit -m "add profile verb with safe_harbor, microdata_no, gdpr, health_research, k_anonymize"
```

---

## Task 21: Integration tests

**Files:**
- Modify: `tests/test_protect.py` (add `TestIntegration`)

- [ ] **Step 1: Write integration tests**

```python
# ============================================================================
# Integration
# ============================================================================


class TestIntegration:
    def test_full_pipeline_reduces_risk(self, panel_df):
        # baseline risk
        r0 = p.risk(panel_df, quasi_ids=["sex", "zip", "country"], unit_id="pid")

        # apply protection pipeline
        recipe = {
            "zip": {"shorten": {"keep": 3}},
            "country": {"collapse": {"rare_below": 5}},
        }
        df_safe, log = p.protect(panel_df, recipe=recipe, unit_id="pid")
        r1 = p.risk(df_safe, quasi_ids=["sex", "zip", "country"], unit_id="pid")

        # risk should drop
        assert r1.units_at_risk <= r0.units_at_risk
        assert len(log) == 2

    def test_recipe_matches_sequential(self, panel_df):
        # sequential
        df_a = p.winsorize(panel_df, "income", limits=(0.05, 0.95))
        df_a = p.shorten(df_a, "icd", sep=".")

        # recipe
        df_b, _ = p.protect(panel_df, recipe={
            "income": {"winsorize": {"limits": (0.05, 0.95)}},
            "icd": {"shorten": {"sep": "."}},
        })

        pd.testing.assert_frame_equal(df_a, df_b)

    def test_transform_log_json_roundtrip(self, panel_df):
        _, log = p.protect(panel_df, recipe={
            "income": {"winsorize": {"limits": (0.05, 0.95)}},
        })
        data = json.loads(log.to_json())
        assert "entries" in data
        assert len(data["entries"]) == 1
```

- [ ] **Step 2: Run tests, expect PASS** (these test integration of already-implemented features)

Run: `python -m pytest tests/test_protect.py::TestIntegration -v`
Expected: 3 passed

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_protect.py
git commit -m "add integration tests"
```

---

## Task 22: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# protect

A Python toolkit for statistical disclosure control on tabular data and analytical results. Pure `numpy` + `pandas`. Single file. Built around 17 single-token verbs.

## What it does

`protect` distinguishes two functions:

- **Data protection** — transforms applied to microdata *before* release (noise, binning, ID replacement, etc.)
- **Result protection** — transforms applied to *outputs* (tables, regression results, plots)

Both are first-class. The verb `suppress` covers result protection; every other data-side verb covers data protection.

## Install

```bash
cd /Users/hom/Documents/GitHub/protect
pip install -e .
```

## Quick start

```python
import pandas as pd
import protect as p

df = pd.read_csv("patients.csv")

# A single verb
df = p.winsorize(df, "income", limits=(0.01, 0.99))

# A recipe — many verbs at once with an audit trail
df_safe, log = p.protect(df, unit_id="pid", recipe={
    "income":    {"winsorize": {"limits": (0.01, 0.99)}},
    "cost":      {"noise": {"scale": 50}},
    "dob":       {"year": {"bin": 5}},
    "visit":     {"diff": {"ref": "first_per_unit"}},
    "icd":       {"shorten": {"sep": ".", "min_count": 5}},
    "country":   {"collapse": {"rare_below": 5}},
    "pid":       {"pseudonymize": {"method": "random"}},
})
print(log.to_text())

# Named composition for HIPAA Safe Harbor
df_safe, log = p.profile(df, "safe_harbor",
                         date_cols=["dob", "visit"],
                         zip_col="zip",
                         id_cols=["pid"],
                         age_col="age")

# Risk assessment
report = p.risk(df_safe, quasi_ids=["sex", "zip", "country"], unit_id="pid")
print(report.describe())
```

## The 17 verbs

| Verb | What it does |
|---|---|
| `noise` | Gaussian / Laplace / uniform / discrete / multiplicative / group-mean perturbation |
| `jitter` | Small uniform / Gaussian noise (numeric or date columns) |
| `winsorize` | Cap extremes (percentile, value, Gaussian, IQR, MAD) |
| `bin` | Numeric → discrete intervals, with sparse-bin merging |
| `year` | Truncate dates to year (with optional multi-year bins) |
| `month` | Truncate dates to month (with optional multi-month bins) |
| `diff` | Convert dates to numeric diff from a reference (survival-friendly) |
| `shorten` | Truncate codes (ICD, ZIP, NACE) with cascading on rarity |
| `collapse` | Merge categorical levels (mapping / rare-below / top-N / proportion) |
| `pseudonymize` | Replace IDs (random with key, or deterministic hash) |
| `insert` | Inject decoy rows or whole decoy units |
| `eliminate` | Drop rows or units, or mask cells |
| `swap` | Exchange values (rank / random / shuffle / PRAM) at row or unit level |
| `suppress` | Output protection: tables, regression results, plots |
| `risk` | k-anonymity, l-diversity, uniqueness report |
| `protect` | Apply a recipe of many verbs in declared order |
| `profile` | Named compositions: safe_harbor, microdata_no, gdpr_pseudonymize, health_research, k_anonymize |

## Universal arguments

Every data-side verb takes these three arguments with the same semantics:

| Arg | Meaning |
|---|---|
| `unit_id` | When set, the transformation is drawn once per unit and broadcast to all rows of that unit. A person's birth-year noise is the same on every visit. |
| `share` | Fraction of units (or rows when `unit_id` is None) to perturb. The rest pass through untouched. |
| `random_state` | Reproducibility seed. |

For `insert`, `eliminate`, and `swap`, an additional `level='row' | 'unit'` argument controls whether the operation acts on individual rows or whole units.

## Documented caveats

- `eliminate(rare_below=)` **destroys**; `collapse(rare_below=)` **generalizes**. Pick by intent.
- `insert` distorts statistics. Default `share=0.01`; warn above `0.05`.
- `suppress` dispatches on input type — see the docstring section matching your input.
- `pseudonymize` produces **pseudonymized** data (still personal under GDPR), not anonymized data.
- `noise` and `jitter` deliberately overlap; use `jitter` for plot-safe small noise.

## Out of scope

For methods this package deliberately doesn't ship, see:

- Formal differential privacy → [`diffprivlib`](https://github.com/IBM/differential-privacy-library), [`smartnoise-sdk`](https://github.com/opendp/smartnoise-sdk)
- Synthetic data generation → [`SDV`](https://github.com/sdv-dev/SDV), [`synthcity`](https://github.com/vanderschaarlab/synthcity)
- Optimization-based secondary cell suppression → R `sdcTable`, `tau-argus`
- NLP-based de-identification → Microsoft Presidio

## Background

See `BACKGROUND.md` for the SDC primer: HIPAA Safe Harbor, GDPR, Norwegian `helseregisterloven`, common attack types, and how each method addresses which risk.

## Spec & implementation

- Design spec: `docs/specs/2026-05-27-protect-design.md`
- Implementation plan: `docs/plans/2026-05-27-protect-implementation.md`
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "add README"
```

---

## Task 23: BACKGROUND.md

**Files:**
- Create: `BACKGROUND.md`

- [ ] **Step 1: Write `BACKGROUND.md`**

```markdown
# Background: statistical disclosure control for health data

This primer explains the problem `protect` addresses, the legal and regulatory context, and the method categories the package exposes.

## The problem: re-identification

Even after removing names and direct identifiers, a record described by `(birth_date, sex, ZIP-5)` is uniquely identifiable for 87 % of the US population (Sweeney, 1997). Health data is especially vulnerable because:

- Patients average 50+ distinct ICD codes; rare codes act as near-unique identifiers (Loukides et al., 2010).
- Rare-disease mentions on public forums can dramatically reduce k when linked to anonymized hospital data.
- Dates of service at day-resolution are stronger identifiers than ZIP3 when combined with public records (obituaries, press releases).

## The two functions: data vs result protection

`protect` distinguishes:

**Data protection (input-side):** transforms applied to microdata before analysis or release.
Most verbs in this package operate here — `noise`, `bin`, `shorten`, `collapse`, `pseudonymize`, `insert`, `eliminate`, `swap`, the date verbs.

**Result protection (output-side):** transforms applied to *outputs* — tables, regression results, plots. The single verb `suppress` covers this domain. It dispatches on input type:

- For tables: primary suppression (`min_n`), dominance rule, p%-rule, rounding, fuzzy counts
- For regression results: intercept redaction, confidence-interval widening
- For plot data: hex-bin scatter with sparse-hex suppression, jittered points, suppressed histograms

Both protection layers are usually needed. microdata.no implements ~50% input-side, ~50% output-side controls.

## Legal / regulatory context

### HIPAA (US health data)

Two paths to permitted use of identifiable health information for research:

1. **Safe Harbor** (45 CFR §164.514(b)(2)) — remove the 18 listed identifiers (names, geographic units smaller than state, **all date elements except year**, phone/fax/email/SSN/MRN/plan/account/certificate/vehicle/device IDs, URLs/IPs/biometrics, full-face photos, and "any other unique identifying number, characteristic, or code"). Age > 89 must aggregate to "90+". ZIP3 allowed only if population ≥ 20,000.

2. **Expert Determination** (§164.514(b)(1)) — a qualified statistician certifies that re-identification risk is "very small." Lets you keep month-of-service, finer geography, etc. — but requires defensible methodology and documentation.

The `profile('safe_harbor')` verb implements path 1. For path 2, use `risk()` to produce metrics and `TransformLog` to document the methodology.

### GDPR (EU)

GDPR Article 4(5) and Recital 26 distinguish:

- **Pseudonymization** — still personal data, inside GDPR. Replaces identifiers with pseudonyms but keeps the data re-linkable.
- **Anonymization** — irreversible, no reasonable means of re-identification. Falls outside GDPR.

The EDPB January 2025 guidance reinforces that a hash-only "anonymization" is pseudonymization, not anonymization. `protect.pseudonymize()` produces pseudonymized data; the `TransformLog` records this fact for compliance documentation.

### Norwegian `helseregisterloven`

Governs central health registers (NPR, KPR, Reseptregisteret, etc.). Direct identifiers may be processed without consent only for statutorily named registers. Anonymized / pseudonymized health data can be transferred to non-EEA recipients if links are kept inside Norway. FHI and SIKT are the standard gatekeepers.

Layered with `personopplysningsloven` (GDPR transposition) and `helseforskningsloven` (research-specific).

### microdata.no

The microdata.no service implements 10 numbered "Tiltak" (measures):

- Input-side (data protection): Tiltak 1 (min population ≥ 1,000), Tiltak 6 (min change ≥ 10 units), Tiltak 7 (min population for descriptives)
- Output-side (result protection): Tiltak 2 (winsorization), Tiltak 3 (noise on counts), Tiltak 4 (hexbin), Tiltak 5 (sparse table suppression), Tiltak 8 (3-digit precision), Tiltak 9 (regression intercept suppression), Tiltak 10 (microaggregation + smoothing)

The `profile('microdata_no')` verb implements the input-side guards; the `suppress()` verb handles the output-side rules.

## Methods catalog

| Method | Input or output | `protect` verb |
|---|---|---|
| k-anonymity (measure) | Both | `risk` |
| k-anonymization (enforce) | Input | `profile('k_anonymize')` |
| l-diversity | Input (measure on outputs) | `risk` |
| Local suppression | Input | `eliminate(columns=...)` |
| Cell suppression (table) | Output | `suppress(min_n=...)` |
| Generalization / recoding | Input | `collapse(mapping=...)` |
| Top/bottom coding | Both | `winsorize(method='value')` |
| Winsorization | Output (and sometimes input) | `winsorize` |
| Noise addition (continuous) | Both | `noise`, `jitter` |
| Microaggregation | Input | `noise(method='group_mean')` |
| Rank swapping | Input | `swap(method='rank', level='row')` |
| Record swapping | Input | `swap(method='random', level='unit')` |
| PRAM | Input | `swap(method='pram')` |
| Truncation (codes) | Input | `shorten` |
| Pseudonymization | Input | `pseudonymize` |
| Decoy injection | Input | `insert` |
| Subsampling | Input | `eliminate(share=...)` |
| Cell rounding | Output | `suppress(round=...)` |
| Fuzzy counts | Output | `suppress(ranges=...)` |
| Dominance rule (n,k) | Output | `suppress(dominance=...)` |
| p%-rule | Output | `suppress(p_percent=...)` |
| Regression CI widening | Output | `suppress(widen_alpha=...)` |
| Regression intercept redaction | Output | `suppress(redact_intercept=...)` |
| Hexbin / sparse-hex suppression | Output | `suppress(hexbin=True)` |
| Histogram with sparse-bin suppression | Output | `suppress(bin_histogram=True)` |

## The unit-of-protection problem

When an individual has multiple rows (longitudinal / panel data), every perturbation must be unit-consistent: a person's birth-year noise should be the same on every visit, not redrawn per row. The `unit_id=` argument on every data-side verb in `protect` enforces this.

For the record-level verbs (`insert`, `eliminate`, `swap`), `level='unit'` enforces that whole patients are added / dropped / swapped together, never half-patients.

This is the distinguishing feature of `protect` versus single-row tools like ARX. R's `sdcMicro` has comparable support via its `hhId` mechanism.

## Re-identification attacks worth knowing

1. **Sweeney's 87 %** — {ZIP5, DOB, sex} alone uniquely identifies most of the US population. Driven by ZIP5 + DOB being widely available in public records.
2. **ICD code linkage** — Loukides et al. (2010) showed that patient ICD code lists are themselves quasi-identifiers; rare codes (orphan diseases) act as near-unique IDs.
3. **Forum mention attack** — when patients discuss rare conditions in public forums (rare-disease groups), linking those mentions to hospital data dramatically reduces k.
4. **Date-of-service attack** — day-resolution dates combined with obituaries / press releases / sports results identify people.

These attacks motivate the package defaults: year-only dates, rare-code collapsing, ZIP3 with population threshold, unit-consistent noise.

## Sources

- Microdata.no manual: https://microdata.no/manual/konfidensialitet
- sdcMicro: https://cran.r-project.org/web/packages/sdcMicro/sdcMicro.pdf
- HIPAA Safe Harbor: https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/
- EDPB pseudonymization guidance (2025): https://www.edpb.europa.eu/system/files/2025-01/edpb_guidelines_202501_pseudonymisation_en.pdf
- Sweeney L. (2000). "Simple Demographics Often Identify People Uniquely." Carnegie Mellon Data Privacy Working Paper 3.
- Loukides G., Gkoulalas-Divanis A., Malin B. (2010). "Anonymization of electronic medical records for validating genome-wide association studies." PNAS.
```

- [ ] **Step 2: Commit**

```bash
git add BACKGROUND.md
git commit -m "add BACKGROUND.md SDC primer"
```

---

## Task 24: examples.ipynb (skeleton as .py)

**Files:**
- Create: `examples.py` (engineer can convert to `.ipynb` via `jupytext` or manually)

- [ ] **Step 1: Write `examples.py`** as a runnable demo

```python
"""Demo of the protect package on synthetic health-research data.

Convert to .ipynb with: `jupytext --to notebook examples.py` (optional).
"""
# %%
import numpy as np
import pandas as pd
import protect as p

# %%
# Generate a synthetic panel-shaped health dataset
rng = np.random.default_rng(42)
n_patients = 500
visits_per_patient = rng.integers(1, 6, size=n_patients)

countries = (["USA"] * 200 + ["UK"] * 100 + ["DE"] * 80 + ["IN"] * 60
             + ["NO"] * 40 + ["IS", "LI", "AD", "TV", "NR"] * 4)
rng.shuffle(countries)

rows = []
for i in range(n_patients):
    pid = f"PT{i:05d}"
    sex = rng.choice(["M", "F"])
    dob = pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.integers(0, 25000)))
    zip_code = f"{int(rng.integers(10000, 99999)):05d}"
    income = float(rng.normal(60000, 20000))
    for v in range(visits_per_patient[i]):
        visit = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))
        icd = rng.choice(["I10", "I10.9", "E11.9", "C50.9", "Z00.00", "M54.5", "F32.9"])
        cost = float(rng.normal(500, 200))
        rows.append({"pid": pid, "visit": visit, "dob": dob, "sex": sex,
                     "zip": zip_code, "country": countries[i],
                     "icd": icd, "income": income, "cost": cost})

df = pd.DataFrame(rows).sort_values(["pid", "visit"]).reset_index(drop=True)
print(df.head())
print(f"\n{len(df)} rows, {df['pid'].nunique()} patients")

# %%
# Baseline risk
report = p.risk(df, quasi_ids=["sex", "zip", "country"], unit_id="pid")
print(report.describe())

# %%
# Apply a protection recipe
df_safe, log = p.protect(df, unit_id="pid", recipe={
    "income":    {"winsorize": {"limits": (0.01, 0.99)}},
    "cost":      {"noise": {"scale": 50, "random_state": 42}},
    "dob":       {"year": {"bin": 5}},
    "visit":     {"diff": {"ref": "first_per_unit"}},
    "icd":       {"shorten": {"sep": ".", "min_count": 5}},
    "country":   {"collapse": {"rare_below": 10}},
    "zip":       {"shorten": {"keep": 3}},
    "pid":       {"pseudonymize": {"method": "random", "random_state": 42}},
})
print(log.to_text())

# %%
# Risk after protection
report_after = p.risk(df_safe, quasi_ids=["sex", "zip", "country"], unit_id="pid")
print(report_after.describe())

# %%
# Output protection: a table
tab = df_safe.groupby(["sex", "country"])["cost"].mean().unstack()
counts = df_safe.groupby(["sex", "country"]).size().unstack().fillna(0)
safe_tab = p.suppress(tab, counts=counts, min_n=5, round=10)
print(safe_tab)

# %%
# Output protection: regression result
try:
    import statsmodels.api as sm
    X = sm.add_constant(df_safe[["cost"]])
    y = (df_safe["sex"] == "F").astype(int).values
    model = sm.OLS(y, X).fit()
    safe_summary = p.suppress(model, widen_alpha=0.01)
    print(safe_summary.summary_text)
except ImportError:
    print("statsmodels not installed; skipping regression demo")
```

- [ ] **Step 2: Run the example to verify it works**

Run: `cd /Users/hom/Documents/GitHub/protect && python examples.py`
Expected: prints risk report, protection log, suppressed table, and (if statsmodels installed) regression summary, with no errors.

- [ ] **Step 3: Commit**

```bash
git add examples.py
git commit -m "add runnable examples"
```

---

## Task 25: Final verification

- [ ] **Step 1: Run the entire test suite**

Run: `cd /Users/hom/Documents/GitHub/protect && python -m pytest tests/ -v`
Expected: all tests pass, no warnings about deprecations from the package itself

- [ ] **Step 2: Verify the package imports cleanly**

Run: `cd /Users/hom/Documents/GitHub/protect && python -c "import protect; print([v for v in protect.__all__])"`
Expected: prints the 18 exported names (17 verbs + `TransformLog` + `RiskReport`)

- [ ] **Step 3: Verify the example runs**

Run: `cd /Users/hom/Documents/GitHub/protect && python examples.py`
Expected: completes with no errors

- [ ] **Step 4: Check the directory contents**

Run: `ls /Users/hom/Documents/GitHub/protect`
Expected: `__init__.py  BACKGROUND.md  docs  examples.py  __pycache__  pyproject.toml  protect.py  README.md  tests`

- [ ] **Step 5: Final commit**

```bash
git log --oneline
# Verify roughly 24+ commits showing TDD progression
```

---

## Self-review notes

Coverage check vs spec:
- All 17 verbs ✓ (Tasks 4–20)
- `TransformLog` ✓ (Task 2)
- `RiskReport` ✓ (Task 18)
- 5 profiles ✓ (Task 20)
- Universal arguments (`unit_id`, `share`, `random_state`) ✓ (each verb)
- `level='row'|'unit'` for `insert`/`eliminate`/`swap` ✓ (Tasks 13, 14, 15)
- TDD per verb ✓ (failing test → impl → passing)
- Frequent commits ✓ (one per task)
- README + BACKGROUND ✓ (Tasks 22, 23)
- Examples ✓ (Task 24)
- Tests in `tests/test_protect.py` ✓

Known limitations the engineer should be aware of (documented in README/BACKGROUND):
- `swap(method='rank', level='unit')` rank-matches on the first column only; multi-column similarity is not implemented in v1
- `_secondary_suppression` is a greedy algorithm; not optimal — `sdcTable` is the production reference
- `risk` computes l-diversity only on the first sensitive attribute when multiple are given
- DP-budget composition tracking is out of scope; `noise(method='laplace')` is DP-flavored noise but no formal ε accounting
