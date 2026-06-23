"""Results: query, compare, and export benchmark output."""
from __future__ import annotations

import json


#: Performance metrics emitted by the runner (speed/cost), kept separate
#: from quality metrics so the comparison table stays readable.
PERF_METRICS = frozenset(
    {"eval_seconds", "encode_seconds", "texts_encoded", "texts_per_sec"}
)


class BenchmarkResults:
    """Holds long-format rows: model, task, dataset, metric, value."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return bool(self.rows)

    def to_dataframe(self, include_perf: bool = False, include_std: bool = False):
        """Wide comparison: one row per model, one column per metric.

        Performance metrics (speed/cost) and ``*_std`` spread metrics are
        excluded by default so the table shows only quality means. Pass
        ``include_perf=True`` / ``include_std=True`` to keep them, or use
        :meth:`performance` and :meth:`to_table(std=True)` for dedicated views.
        """
        import pandas as pd

        rows = self.rows
        if not include_perf:
            rows = [r for r in rows if r["metric"] not in PERF_METRICS]
        if not include_std:
            rows = [r for r in rows if not r["metric"].endswith("_std")]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["column"] = df["task"] + ":" + df["dataset"] + "/" + df["metric"]
        wide = df.pivot_table(index="model", columns="column", values="value")
        wide.columns.name = None
        return wide

    def _std_dataframe(self):
        """Wide std table, columns named like the mean columns (``_std`` stripped)."""
        import pandas as pd

        rows = [r for r in self.rows if r["metric"].endswith("_std")]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["column"] = (
            df["task"] + ":" + df["dataset"] + "/" + df["metric"].str[:-4]
        )
        wide = df.pivot_table(index="model", columns="column", values="value")
        wide.columns.name = None
        return wide

    def performance(self):
        """Wide speed/cost view: one row per model, one column per perf metric.

        Aggregated across tasks (encode time and texts summed, throughput
        recomputed) so you see total cost per model at a glance.
        """
        import pandas as pd

        rows = [r for r in self.rows if r["metric"] in PERF_METRICS]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        wide = df.pivot_table(
            index="model", columns="metric", values="value", aggfunc="sum"
        )
        # throughput must be recomputed from the summed totals, not summed
        if "encode_seconds" in wide and "texts_encoded" in wide:
            secs = wide["encode_seconds"].replace(0, float("nan"))
            wide["texts_per_sec"] = (wide["texts_encoded"] / secs).fillna(0.0)
        wide.columns.name = None
        return wide

    def to_long_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def to_table(self, float_format: str = "%.4f", std: bool = False) -> str:
        df = self.to_dataframe()
        if df.empty:
            return "(no results)"
        if not std:
            return df.to_string(float_format=lambda x: float_format % x)

        # Annotate each cell as "mean ± std" where a std is available.
        import math

        sdf = self._std_dataframe()
        cells = {}
        for col in df.columns:
            values = []
            for model in df.index:
                mean = df.loc[model, col]
                s = sdf.loc[model, col] if col in sdf.columns else None
                if mean is None or (isinstance(mean, float) and math.isnan(mean)):
                    values.append("")
                elif s is None or (isinstance(s, float) and math.isnan(s)):
                    values.append(float_format % mean)
                else:
                    values.append(f"{float_format % mean} ± {float_format % s}")
            cells[col] = values
        import pandas as pd

        return pd.DataFrame(cells, index=df.index).to_string()

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
