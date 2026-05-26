"""Demo of the protect package on synthetic health-research data.

Run as: python examples.py
(Or convert to .ipynb with `jupytext --to notebook examples.py`)
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
