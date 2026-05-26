# `protect` — Design Spec

**Date:** 2026-05-27
**Status:** Draft for user approval

## 1. Goal

A single-file Python module exposing a small, intuitive set of single-word verbs for reducing
re-identification risk in tabular data and statistical results. The package distinguishes two
distinct functions:

- **Data protection** — transforms applied to microdata *before* analysis (noise, binning, ID
  replacement, etc.)
- **Result protection** — transforms applied to *outputs* of analysis (tables, regression
  results, plots)

Both are first-class. The verb `suppress` covers result protection; every other data-side verb
covers data protection.

## 2. Scope

### In scope (v1)
- Single Python file (`protect.py`) at `~/Documents/GitHub/protect/`
- Pure dependencies: `numpy`, `pandas` only
- 17 single-token verbs (listed below)
- Named profiles for HIPAA, microdata.no, GDPR pseudonymization, health-research, k-anonymization
- `TransformLog` class for audit-trail documentation
- `RiskReport` dataclass with metrics + recommendations
- Test suite covering each verb and integration paths

### Out of scope (v1) — pointers in `README.md`
- Formal differential privacy mechanisms → `diffprivlib`, `smartnoise`
- Synthetic data generation → `SDV`, `synthcity`
- Optimization-based secondary cell suppression → R `sdcTable`, `tau-argus`
- NLP-based de-identification of free text → Microsoft Presidio
- Full ARX-style k-anonymization solvers (a greedy version IS in scope via `profile('k_anonymize')`)

## 3. Layout

```
/Users/hom/Documents/GitHub/protect/
├── protect.py            # the module (~2000 LOC estimated)
├── __init__.py           # `from .protect import *`
├── README.md             # quick-start, verb reference, usage examples
├── BACKGROUND.md         # SDC primer: HIPAA, GDPR, helseregisterloven, attack types,
│                         # the data-vs-result distinction, method catalog
├── examples.ipynb        # runnable demo on synthetic health data
├── docs/specs/           # design docs (this file)
└── tests/
    └── test_protect.py   # pytest, one section per verb plus integration tests
```

Inside `protect.py`, banner-commented sections in this order:

1. Imports, constants, type aliases
2. `TransformLog` class
3. Helpers (`_apply_per_unit`, `_validate_columns`, `_resolve_random_state`, …)
4. Data verbs — value-level: `noise`, `jitter`, `winsorize`, `bin`, `year`, `month`, `diff`,
   `shorten`, `collapse`
5. Data verbs — ID and record-level: `pseudonymize`, `insert`, `eliminate`, `swap`
6. Output verb: `suppress`
7. Risk: `risk` + `RiskReport`
8. Aggregate / meta: `protect`, `profile`
9. Profile implementations (`_profile_safe_harbor`, `_profile_microdata_no`, …)

## 4. Verbs

### 4.1 Universal arguments

Every data-side verb takes these three arguments with consistent semantics:

| Arg | Type | Meaning |
|---|---|---|
| `unit_id=None` | `str \| None` | When set, the transformation is drawn **once per unit** and broadcast to all rows of that unit. A person's birth-year noise is the same on every visit. When `None`, applied per-row independently. |
| `share=1.0` | `float ∈ [0, 1]` | Fraction of units (or rows, if `unit_id is None`) to apply the transform to. The rest pass through untouched. |
| `random_state=None` | `int \| None` | Reproducibility seed. |

Functions return a new DataFrame (no mutation). `audit=True` returns `(df, TransformLog)`.

### 4.2 Perturbation

#### `noise(data, columns, *, scale, method='gaussian', share=1.0, direction='both', clip=None, by=None, unit_id=None, random_state=None)`

General noise. Replaces `add_noise`.

- `method` ∈ `{'gaussian', 'laplace', 'uniform', 'discrete', 'multiplicative', 'group_mean'}`
  - `'gaussian'`: normal noise with SD = `scale`
  - `'laplace'`: Laplace noise (DP-flavored; no budget tracking)
  - `'uniform'`: uniform on ±`scale`
  - `'discrete'`: integer steps; `scale` is max step magnitude
  - `'multiplicative'`: value × (1 + ε), ε ~ N(0, scale²)
  - `'group_mean'`: microaggregation — sort within `by` (or unsorted if no `by`), group into
    k-tuples where `k = scale` (note: for this method only, `scale` is an integer group size,
    not a noise magnitude), replace each value with the group mean. Output has zero
    within-group variance.
- `direction` ∈ `{'both', 'up', 'down'}` for asymmetric noise (e.g., counts stay non-negative)
- `clip=(lo, hi)`: post-noise clipping to keep values plausible
- `by=`: group context for `group_mean` and group-aware scaling

#### `jitter(data, columns, *, scale, distribution='uniform', unit_id=None, share=1.0, random_state=None)`

Small, plot-friendly noise. Convenience wrapper distinct from `noise` by intent:

- Numeric columns: `scale` is a number
- Date columns: `scale` is a string like `'3 days'` or a `pd.Timedelta`
- `distribution` ∈ `{'uniform', 'gaussian'}`

When in doubt: `jitter` for plot-safe small symmetric noise; `noise` when distribution and
scale matter for statistics.

### 4.3 Bounding & binning

#### `winsorize(data, columns, *, limits=(0.01, 0.99), method='percentile', by=None, unit_id=None, share=1.0)`

Caps extremes. `method` ∈ `{'percentile', 'value', 'gaussian', 'iqr', 'mad'}`.

- `limits=(0.01, 0.99)` with `method='percentile'`: clip at the 1st and 99th percentile
- `limits=(None, 90)` with `method='value'`: top-code at value 90 (used for HIPAA age 90+)
- `method='gaussian'`: mean ± k·SD where k comes from `limits` (fold)
- `method='iqr'`: Q1 − k·IQR, Q3 + k·IQR
- `method='mad'`: median ± k·MAD
- `by=`: per-group caps (cap age within sex, etc.)

#### `bin(data, columns, *, bins=10, method='quantile', labels='range', min_count=None, unit_id=None, share=1.0)`

Numeric → discrete intervals.

- `bins=10` or `bins=[0, 18, 30, 50, 65, 120]`
- `method` ∈ `{'quantile', 'equal_width', 'manual'}`
- `labels` ∈ `{'range', 'midpoint', 'index', list_of_strings}`
- `min_count=N`: merge sparse bins into neighbors until every bin has ≥ N members (SDC-specific
  feature not in pandas / scipy)

### 4.4 Dates

#### `year(data, columns, *, bin=None, as_date=False, unit_id=None, share=1.0)`

Truncate dates to year resolution.

- `bin=5`: 5-year bins ("1990-1994")
- `as_date=False` (default): return integer year (`1990`)
- `as_date=True`: return date floored to year-start (`1990-01-01`)

#### `month(data, columns, *, bin=None, as_date=False, unit_id=None, share=1.0)`

Truncate dates to month resolution.

- `bin=3` covers quarters
- `as_date=False`: returns `"1990-03"` string
- `as_date=True`: returns date floored to month-start

#### `diff(data, columns, *, ref='first_per_unit', unit='days', keep_order=True, unit_id=None, share=1.0, random_state=None)`

Convert dates to numeric diff from a reference.

- `ref` ∈ `{'first_per_unit', 'min', 'random_per_unit', column_name, scalar_date}`
- `unit` ∈ `{'days', 'months', 'years'}`
- `keep_order=True`: raises if `share` or implicit noise would reorder events within a unit
  (critical for survival-analysis correctness)

### 4.5 Code & category generalization

#### `shorten(data, columns, *, keep=3, sep=None, side='left', min_count=None, fallback='*', per_value=None, unit_id=None, share=1.0)`

Truncate codes (ICD, ZIP, NACE).

- `keep=3`: keep first 3 characters
- `sep='.'`: truncate at separator (ICD `H25.11` → `H25`)
- `side` ∈ `{'left', 'right'}`: which prefix to keep
- `min_count=N`: cascading — truncate further if frequency < N
- `per_value={'C50.*': 'keep_full', 'Z*': 'keep_1'}`: value-specific rules

#### `collapse(data, columns, *, mapping=None, rare_below=None, keep_top=None, keep_prop=None, other_label='Other', by=None, unit_id=None, random_state=None)`

Merge categorical levels. Exactly one mode per call (raises if multiple mode args given):

- `mapping={'Liechtenstein': 'Other Europe', ...}`: explicit hierarchy
- `rare_below=N`: merge values with count < N into `other_label`
- `keep_top=N`: keep top-N most common; rest → `other_label`
- `keep_prop=p`: keep values with share ≥ p; rest → `other_label`
- `by=`: apply thresholds within group

### 4.6 IDs

#### `pseudonymize(data, columns, *, method='random', salt=None, return_key=True, key_path=None, prefix='P', random_state=None)`

Replace IDs.

- `method='random'`: random new IDs, key returned as dict (different every run unless seeded)
- `method='hash'`: deterministic hash with `salt` (stable across runs sharing the salt)
- `return_key=True` returns `(df, key_dict)`
- `key_path=` persists key to file (warning logged: don't store next to the data)
- `prefix='P'`: human-readable prefix for random IDs (`P000001`, `P000002`)

### 4.7 Record-level

All three verbs in this section take an explicit **`level='row' | 'unit'`** argument that
selects granularity:

- **`level='row'`** — the operation acts on individual rows.
- **`level='unit'`** — the operation acts on whole units (all rows of one unit are dropped,
  fabricated, or swapped together). Requires `unit_id=` to be set; raises a clear error
  otherwise.

Default is `level='row'`. The explicit argument replaces the implicit "unit_id presence
toggles granularity" behavior of earlier drafts.

#### `insert(data, *, n=None, share=0.01, level='row', source='resample', modify=None, new_unit_ids=True, unit_id=None, random_state=None)`

Inject decoy data.

- `share=0.01`: 1 % decoys (default opt-in low)
- `level='row'`: insert N decoy **rows**, each a standalone fake observation
- `level='unit'`: insert N decoy **units**, each with a row-count drawn from the real
  units' row-count distribution (realistic panel histories). Requires `unit_id=`.
- `source` ∈ `{'resample', 'sample_per_column'}` — resample preserves correlations,
  sample breaks them
- `modify={'birth_year': ('noise', 2), 'visit_date': ('shift', 30)}`: post-resample modifications
- `new_unit_ids=True`: decoys get fresh unit IDs (forced True when `level='unit'`)
- Decoys **are not marked** in the output (a marker defeats the protection); the
  `TransformLog` records count and source rows for the data owner's internal records
- Warns if `share > 0.05`

#### `eliminate(data, *, where=None, rare_below=None, share=None, level='row', columns=None, replace_with=None, unit_id=None, random_state=None)`

Drop rows/units or mask cells. **Raises if no mode arg given** (no silent no-op).
**Exactly one of `where`, `rare_below`, or `share` per call** (raises if multiple given);
`columns` may accompany `rare_below` to scope cell masking.

Modes:
- `where=<boolean Series>`: drop matching rows. With `level='unit'`, any unit with at least
  one matching row is dropped entirely.
- `rare_below=N` + `columns=[...]`: mask cells whose value occurs < N times in the given
  columns. With `level='unit'`, all rows of a unit whose value is rare get the same masking.
- `share=p`: drop a random fraction. `level='row'` drops random rows; `level='unit'` drops
  random whole units (requires `unit_id=`).
- `columns=[...]` without `where`/`rare_below`/`share`: mask all cells in those columns to
  NaN (or `replace_with`). `level` has no effect in this mode.

Emits a warning if `share > 0.05` or if the call would drop more than 5 % of rows.

#### `swap(data, columns, *, method='rank', level='row', by=None, share=0.05, swap_range_pct=0.05, transition=None, unit_id=None, random_state=None)`

Exchange values. `method=` describes *how to match* swap partners; `level=` describes
*what is swapped*. The two arguments are orthogonal.

`method` ∈ `{'rank', 'random', 'shuffle', 'pram'}`:
- `'rank'`: match within a rank window of `swap_range_pct` (numeric)
- `'random'`: match uniformly at random
- `'shuffle'`: permute values within `by` group (no explicit pair matching)
- `'pram'`: categorical — probabilistic recoding via `transition=` matrix (always row-level)

`level` ∈ `{'row', 'unit'}`:
- `'row'`: swap individual cell values between matched rows
- `'unit'`: swap whole records — all rows of unit A take unit B's values and vice versa.
  Requires `unit_id=`.

Common combinations:
- `swap(method='rank', level='row')` — classic numeric rank swap
- `swap(method='random', level='unit', unit_id='pid')` — pure record swap between random units
- `swap(method='rank', level='unit', unit_id='pid')` — record swap between rank-similar units
  (matched on the first column in `columns`)
- `swap(method='shuffle', level='row', by='hospital')` — within-hospital value permutation

Note: `method='record'` from earlier drafts is removed; it's now expressed as
`level='unit'` combined with any `method`.

### 4.8 Output protection

#### `suppress(target, *, **kwargs)`

Polymorphic, dispatches on `target` type:

**Table (pandas Series / DataFrame):**
- `min_n=5`: cells with count < n → NaN/`'*'`
- `counts=`: separate counts table if `target` is means/ratios
- `dominance=(n, k)`: (n, k)-rule — suppress when top n contributors > k of total
- `p_percent=p`: p%-rule
- `round=base`: random / controlled rounding
- `ranges=[(1,4),(5,9),...]`: fuzzy counts ("5-9")
- `secondary=False`: greedy secondary suppression on marginals when True

**Regression result (statsmodels):**
- `redact_intercept=k` + `group_counts=`: redact intercept if smallest cell < k
- `widen_alpha=alpha`: return summary with widened confidence intervals
- Returns a new summary object, not a model

**Plot data ((x, y) arrays or matplotlib axis):**
- `hexbin=True, gridsize=30, min_count=5`: hex-bin scatter with sparse-hex suppression
- `bin_histogram=True, bins=20, min_count=5`: histogram with sparse-bin suppression
- `jitter=(sd, sd)`: coordinate jitter

### 4.9 Risk

#### `risk(data, *, quasi_ids, sensitive=None, unit_id=None) -> RiskReport`

```python
@dataclass
class RiskReport:
    k_min: int
    k_median: float
    k_below_5: int          # records with k < 5
    units_at_risk: int      # unique units uniquely identifiable on quasi_ids
    l_min: float | None     # entropy l-diversity, when sensitive given
    l_median: float | None
    t_max: float | None     # t-closeness EMD, when sensitive given
    distinct_combos: int
    suggestions: list[str]  # heuristic next-step suggestions

    def describe(self) -> str
    def diff(self, other: 'RiskReport') -> dict
```

### 4.10 Meta

#### `protect(data, *, recipe, unit_id=None, audit=True) -> tuple[pd.DataFrame, TransformLog]`

Apply many verbs in declared order via a recipe dict:

```python
recipe = {
    "income":     {"winsorize": {"limits": (0.01, 0.05)}},
    "salary":     {"noise": {"scale": 0.05, "share": 0.5}},
    "dob":        {"year": {"bin": 5}},
    "visit_date": {"diff": {"ref": "first_per_unit"}},
    "diagnosis":  {"collapse": {"rare_below": 5}},
    "country":    {"collapse": {"keep_top": 10}},
    "icd":        {"shorten": {"sep": ".", "min_count": 5}},
    "patient_id": {"pseudonymize": {"method": "random"}},
}
df_safe, log = protect(df, recipe=recipe, unit_id="patient_id")
```

A column may take multiple steps via list form:
```python
"income": [{"winsorize": {"limits": (0.01, 0.99)}},
           {"noise": {"scale": 0.02}}]
```

#### `profile(data, name, *, **kwargs) -> tuple[pd.DataFrame, TransformLog]`

Named compositions:

- `'safe_harbor'(date_cols, zip_col, id_cols, age_col)`: HIPAA 18-identifier rules
- `'microdata_no'(unit_id)`: asserts min population ≥ 1000, min change ≥ 10; defaults aligned
  with microdata.no Tiltak 1, 6, 7 (input) — output Tiltak (2, 3, 4, 5, 8, 9, 10) live in
  `suppress`
- `'gdpr_pseudonymize'(id_cols, salt=None)`: hashes IDs, audit-logs "still personal data
  under GDPR"
- `'health_research'(unit_id, quasi_ids, sensitive_cols=None, k=5)`: composed defaults for
  typical register-data release
- `'k_anonymize'(quasi_ids, k=5, unit_id=None)`: iterative generalization — calls `bin`,
  `shorten`, `collapse` on quasi_ids until `risk(...).k_min >= k`

## 5. `TransformLog`

```python
class TransformLog:
    entries: list[dict]   # {timestamp, function, columns, params, rows_affected, units_affected, notes}

    def add(self, **fields): ...
    def to_text(self) -> str: ...
    def to_json(self) -> str: ...
    def summary(self) -> dict: ...
```

Captures every operation. The audit artifact for HIPAA Expert Determination, GDPR
documentation, and microdata.no method reporting.

## 6. Key invariants

1. **No mutation** — every verb returns a new DataFrame; input is never modified.
2. **Reproducibility** — `random_state=` honored on every verb that uses randomness.
3. **Unit consistency** — when `unit_id=` is set, perturbations are drawn once per unit and
   broadcast to every row of that unit. Warns if a column declared person-invariant varies
   within `unit_id`.
4. **Order preservation** — `diff(keep_order=True)` raises rather than silently reordering
   survival events within a unit.
5. **Explicit modes** — `collapse` requires exactly one mode per call.
6. **No silent destruction** — `eliminate` without args raises.
7. **Audit by default** — `profile` and `protect` return `(df, TransformLog)`.

## 7. Testing

For every verb:
- Shape/dtype of basic call
- `unit_id` consistency: same unit → same transform across all its rows
- `share=0.0` is a no-op
- `share=1.0` modifies every unit
- `random_state` reproducibility
- Edge cases: empty df, all-NaN column, single unit, single row

Integration tests:
- Each profile on a synthetic health dataset (created in `tests/conftest.py`)
- Recipe via `protect()` matches sequential calls
- `risk()` before/after a pipeline shows reduction
- `TransformLog` round-trips through `to_json` / from_json

## 8. Documented caveats (in `README.md`)

1. `eliminate(rare_below)` **destroys**; `collapse(rare_below)` **generalizes**. Pick by intent.
2. `insert` distorts statistics. Default `share=0.01`; warn above `0.05`.
3. `suppress` dispatches on input type — read the docstring section matching your input.
4. `pseudonymize` produces **pseudonymized** data (still personal under GDPR), not
   anonymized data.
5. `noise` and `jitter` deliberately overlap; pick by intent.
6. `bin(min_count=)` and `collapse(rare_below=)` both protect by enforcing minimum counts —
   the first on bins, the second on categorical levels.

## 9. Open questions deferred to v2

- Iterative k-anonymization with full lattice search (v1 ships a greedy version)
- DP budget accountant
- Synthetic data hand-off interfaces to SDV / synthcity
- ARX-style generalization hierarchies as named constants

## 10. Decisions log (from brainstorming)

- Single file `protect.py`, no submodules (`__init__.py` re-exports)
- Pure deps: `numpy` + `pandas`
- 17 single-token verbs (no underscores, no namespaces)
- `noise` over `add_noise` (single token)
- `redate` rejected in favor of flat `year` / `month` / `diff`
- `coarsen` and `aggregate` rejected; functionality split into `bin`,
  `noise(method='group_mean')`, and `year`/`month`/`diff`
- `pseudonymize` over `scramble` (legal-precision term, works on any column)
- `collapse` for categorical generalization (new verb)
- `quarter` and `shift` excluded from v1 (express as `month(bin=3)` and 1-line pandas ops)
- `insert`, `eliminate`, `swap` take an explicit `level='row' | 'unit'` argument; `level='unit'`
  requires `unit_id=`. Replaces the earlier implicit "unit_id presence toggles granularity"
  behavior. `swap`'s `method='record'` removed in favor of `level='unit'`.
- Min-population assertions baked into `profile('microdata_no')` and `protect(min_population=)`,
  not a standalone verb
- `recommend()` heuristic suggester deferred to v2
- All data-side verbs share `unit_id`, `share`, `random_state`
