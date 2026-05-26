"""Shared pytest fixtures: a synthetic panel-shaped health dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel_df():
    """200 patients with 1-5 visits each, panel-shaped health data."""
    rng = np.random.default_rng(42)
    n_patients = 200
    visits_per_patient = rng.integers(1, 6, size=n_patients)
    countries = ["USA"] * 80 + ["UK"] * 40 + ["DE"] * 30 + ["IN"] * 20 + ["NO"] * 15
    rare = ["IS", "LI", "AD", "TV", "NR", "BT", "SM", "MC", "VA", "FM"]
    # Pad with rare codes (cycling if needed) so length matches n_patients.
    deficit = n_patients - len(countries)
    if deficit > 0:
        countries += [rare[i % len(rare)] for i in range(deficit)]
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
                "pid": pid, "visit": visit, "dob": dob, "sex": sex,
                "zip": zip_code, "country": country, "icd": icd,
                "income": income, "cost": cost,
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
