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

# Group A exports only TransformLog. Verbs (noise, jitter, winsorize, bin,
# year, month, diff, shorten, collapse, pseudonymize, insert, eliminate, swap,
# suppress, risk, RiskReport, protect, profile) are forthcoming in later groups.
__all__ = [
    "TransformLog",
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
