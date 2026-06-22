"""Results: query, compare, and export benchmark output."""
from __future__ import annotations

import json


class BenchmarkResults:
    """Holds long-format rows: model, task, dataset, metric, value."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return bool(self.rows)

    def to_dataframe(self):
        """Wide comparison: one row per model, one column per metric."""
        import pandas as pd

        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame(self.rows)
        df["column"] = df["task"] + ":" + df["dataset"] + "/" + df["metric"]
        wide = df.pivot_table(index="model", columns="column", values="value")
        wide.columns.name = None
        return wide

    def to_long_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def to_table(self, float_format: str = "%.4f") -> str:
        df = self.to_dataframe()
        if df.empty:
            return "(no results)"
        return df.to_string(float_format=lambda x: float_format % x)

    def best_model(self, metric: str, task: str | None = None) -> str | None:
        """Model with the highest value for ``metric`` (optionally per task)."""
        candidates = [
            r
            for r in self.rows
            if r["metric"] == metric and (task is None or r["task"] == task)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["value"])["model"]

    def ranking(self, metric: str, task: str | None = None) -> list[tuple[str, float]]:
        """All models ranked by ``metric``, best first."""
        agg: dict[str, list[float]] = {}
        for r in self.rows:
            if r["metric"] == metric and (task is None or r["task"] == task):
                agg.setdefault(r["model"], []).append(r["value"])
        means = {m: sum(v) / len(v) for m, v in agg.items()}
        return sorted(means.items(), key=lambda kv: kv[1], reverse=True)

    def to_csv(self, path: str) -> None:
        self.to_long_dataframe().to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.rows, fh, indent=2)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BenchmarkResults({len(self.rows)} rows)"
