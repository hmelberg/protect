# protect

A Python toolkit for statistical disclosure control on tabular data and analytical results. Pure `numpy` + `pandas`. Single file. Built around 18 single-token verbs.

**Try it now without installing:** [hmelberg.github.io/protect/playground.html](https://hmelberg.github.io/protect/playground.html) — runs the full package in your browser via Pyodide.

## What it does

`protect` distinguishes two functions:

- **Data protection** — transforms applied to microdata *before* release (noise, binning, ID replacement, etc.)
- **Result protection** — transforms applied to *outputs* (tables, regression results, plots)

Both are first-class. The verb `suppress` covers result protection; every other data-side verb covers data protection.

## Install

Install directly from GitHub:

```bash
pip install git+https://github.com/hmelberg/protect.git
```

Pin to a specific commit or tag for reproducibility:

```bash
pip install git+https://github.com/hmelberg/protect.git@main
pip install "git+https://github.com/hmelberg/protect.git@v0.1.0"   # once a tag exists
```

Or clone and install editable for development:

```bash
git clone https://github.com/hmelberg/protect.git
cd protect
pip install -e ".[test]"
pytest    # 112 tests
```

Requires Python 3.10+. Dependencies: `numpy>=1.24`, `pandas>=2.0`. Optional extras for the test suite (`pytest`, `statsmodels`, `matplotlib`) install via `[test]`.

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

## The 18 verbs

| Verb | What it does |
|---|---|
| `noise` | Gaussian / Laplace / uniform / discrete / multiplicative / group-mean perturbation |
| `jitter` | Small uniform / Gaussian noise (numeric or date columns) |
| `winsorize` | Cap extremes (percentile, value, Gaussian, IQR, MAD) |
| `bin` | Numeric → discrete intervals, with sparse-bin merging |
| `coarsen` | Snap values to a coarser resolution (numeric → multiple of N; date → period boundary) |
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

## Defaults policy

Sensible defaults so most calls take only a column name:

- `noise(scale='auto')` — 0.05 × column SD per column
- `jitter(scale='auto')` — 0.01 × column range; `'1 day'` for date columns
- `winsorize(limits=(0.01, 0.99))` — 1st/99th percentile caps
- `bin(bins=10, method='quantile')` — 10 equal-frequency bins
- `shorten(keep=3)` — first 3 characters
- `pseudonymize(method='random')` — random IDs with key returned

Some verbs intentionally have **no default mode** — you must specify what you want:
- `coarsen(to=...)` — no default (must specify the resolution)
- `collapse` — pick `mapping`, `rare_below`, `keep_top`, or `keep_prop`
- `eliminate` — pick `where`, `rare_below`, `share`, or `columns`
- `suppress` — pick the kind of output protection you need
- `risk` requires `quasi_ids` (depends on your threat model)

## Documented caveats

- `eliminate(rare_below=)` **destroys**; `collapse(rare_below=)` **generalizes**. Pick by intent.
- `insert` distorts statistics. Default `share=0.01`; warns above `0.05`.
- `suppress` dispatches on input type — see the docstring section matching your input.
- `pseudonymize` produces **pseudonymized** data (still personal under GDPR), not anonymized data.
- `noise` and `jitter` deliberately overlap; use `jitter` for plot-safe small noise.
- `coarsen` returns the same dtype (numeric → numeric, date → date). For categorical/string code coarsening (e.g. ICD chapter), use `shorten`. For numeric → categorical labels, use `bin`.

## Out of scope

For methods this package deliberately doesn't ship, see:

- Formal differential privacy → [`diffprivlib`](https://github.com/IBM/differential-privacy-library), [`smartnoise-sdk`](https://github.com/opendp/smartnoise-sdk)
- Synthetic data generation → [`SDV`](https://github.com/sdv-dev/SDV), [`synthcity`](https://github.com/vanderschaarlab/synthcity)
- Optimization-based secondary cell suppression → R `sdcTable`, `tau-argus`
- NLP-based de-identification → Microsoft Presidio

## Background

See `BACKGROUND.md` for the SDC primer: HIPAA Safe Harbor, GDPR, Norwegian `helseregisterloven`, common attack types, and how each method addresses which risk.

## Playground

Try the verbs against a sample 200-patient panel dataset, entirely in your browser:

- **Hosted:** [hmelberg.github.io/protect/playground.html](https://hmelberg.github.io/protect/playground.html)
- **Local:** clone the repo and open `playground.html` directly (no server needed)

Click any verb in the sidebar to drop a working example into the editor; `Ctrl/⌘+Enter` runs. First load takes ~10–20 s while Pyodide + numpy + pandas download (~20 MB, cached after).

## Spec & implementation

- Design spec: [`docs/specs/2026-05-27-protect-design.md`](docs/specs/2026-05-27-protect-design.md)
- Implementation plan: [`docs/plans/2026-05-27-protect-implementation.md`](docs/plans/2026-05-27-protect-implementation.md)

## License

[MIT](LICENSE) © 2026 Hans Melberg.

---

Built by [Hans Melberg](https://github.com/hmelberg) for register-data and health-research disclosure protection. Contributions and issue reports welcome.
