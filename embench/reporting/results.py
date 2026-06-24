"""Results: query, compare, and export benchmark output."""
from __future__ import annotations

import json


#: Performance metrics emitted by the runner (speed/cost), kept separate
#: from quality metrics so the comparison table stays readable.
PERF_METRICS = frozenset(
    {"eval_seconds", "encode_seconds", "texts_encoded", "texts_per_sec"}
)


class BenchmarkResults:
    """Holds long-format rows: model, task, dataset, metric, value.

    ``samples`` optionally holds the per-sample (e.g. per-query) values behind
    each scalar, keyed by ``(model, task, dataset, metric)``. They power
    :meth:`significance` and :meth:`win_tie_loss`.
    """

    def __init__(self, rows: list[dict], samples: dict | None = None):
        self.rows = rows
        self.samples = samples or {}

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

    def aggregate_ranking(
        self, metrics=None, task: str | None = None
    ) -> list[tuple[str, float]]:
        """Rank models by their mean across several quality metrics, best first.

        A single overall score per model -- handy when no one metric decides it.
        ``metrics`` defaults to every quality metric present (performance and
        ``*_std`` spread columns excluded). Pass a list to average only those,
        e.g. ``["ndcg@10", "recall@10"]``. Restrict to one task with ``task``.

        Note this is an unweighted mean of metrics that may live on different
        scales; it is a convenience summary, not a substitute for inspecting the
        per-metric table (or :meth:`significance`).
        """
        wanted = set(metrics) if metrics is not None else None
        agg: dict[str, list[float]] = {}
        for r in self.rows:
            metric = r["metric"]
            if metric in PERF_METRICS or metric.endswith("_std"):
                continue
            if wanted is not None and metric not in wanted:
                continue
            if task is not None and r["task"] != task:
                continue
            agg.setdefault(r["model"], []).append(r["value"])
        means = {m: sum(v) / len(v) for m, v in agg.items() if v}
        return sorted(means.items(), key=lambda kv: kv[1], reverse=True)

    def _samples_for(self, metric: str, task: str | None):
        """Per-model aligned per-sample vectors for ``metric`` (or None if absent).

        Concatenates samples across datasets/tasks in a fixed key order so the
        vectors stay aligned query-for-query across models (a paired test needs
        identical ordering on both sides).
        """
        keys = sorted(
            k
            for k in self.samples
            if k[3] == metric and (task is None or k[1] == task)
        )
        if not keys:
            return None
        # group keys by (task, dataset, metric), collect the set of models
        slots = sorted({(k[1], k[2], k[3]) for k in keys})
        models = sorted({k[0] for k in keys})
        per_model: dict[str, list] = {m: [] for m in models}
        for slot in slots:
            for m in models:
                vals = self.samples.get((m, slot[0], slot[1], slot[2]))
                if vals is None:  # this model lacks this slot -> not comparable
                    return None
                per_model[m].extend(vals)
        return per_model

    def significance(
        self,
        metric: str,
        task: str | None = None,
        max_p: float = 0.05,
        n_permutations: int = 10000,
        seed: int = 0,
    ):
        """Pairwise significance of model differences on ``metric``.

        For every pair of models, runs a paired randomization test on the
        per-query scores and reports the mean difference and its p-value -- i.e.
        whether one model *really* beats another on your data or it is noise.
        Requires per-sample data (currently retrieval metrics like ``ndcg@10``).

        Returns a tidy ``DataFrame`` (one row per unordered model pair) with
        columns ``model_a, model_b, mean_a, mean_b, delta, p_value, significant``
        where ``delta = mean_a - mean_b`` and ``significant`` is ``p < max_p``.
        """
        import numpy as np
        import pandas as pd

        from ..utils import paired_permutation_test

        per_model = self._samples_for(metric, task)
        if per_model is None:
            raise ValueError(
                f"no per-sample data for metric {metric!r}"
                + (f" in task {task!r}" if task else "")
                + ". Significance testing needs per-query metrics (e.g. retrieval)."
            )

        models = list(per_model)
        vectors = {m: np.asarray(per_model[m], dtype=float) for m in models}
        means = {m: float(vectors[m].mean()) for m in models}
        rows = []
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                p = paired_permutation_test(
                    vectors[a], vectors[b], n_permutations=n_permutations, seed=seed
                )
                rows.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "mean_a": means[a],
                        "mean_b": means[b],
                        "delta": means[a] - means[b],
                        "p_value": p,
                        "significant": p < max_p,
                    }
                )
        return pd.DataFrame(rows)

    def win_tie_loss(
        self,
        metric: str,
        task: str | None = None,
        max_p: float = 0.05,
        n_permutations: int = 10000,
        seed: int = 0,
    ):
        """Per-model win/tie/loss record on ``metric`` against every other model.

        A *win* is a higher mean that is statistically significant (``p <
        max_p``); a *loss* is a significant lower mean; everything else is a
        *tie*. Returns a ``DataFrame`` indexed by model with columns
        ``score, wins, ties, losses``, sorted by wins then score (best first) --
        the summary view ranx prints, built on :meth:`significance`.
        """
        import pandas as pd

        pairs = self.significance(
            metric, task=task, max_p=max_p, n_permutations=n_permutations, seed=seed
        )
        models = sorted(set(pairs["model_a"]) | set(pairs["model_b"]))
        record = {m: {"wins": 0, "ties": 0, "losses": 0} for m in models}
        score = {}
        for _, row in pairs.iterrows():
            a, b = row["model_a"], row["model_b"]
            score[a], score[b] = row["mean_a"], row["mean_b"]
            if not row["significant"] or row["delta"] == 0:
                record[a]["ties"] += 1
                record[b]["ties"] += 1
            elif row["delta"] > 0:
                record[a]["wins"] += 1
                record[b]["losses"] += 1
            else:
                record[a]["losses"] += 1
                record[b]["wins"] += 1
        out = pd.DataFrame(
            [
                {"model": m, "score": score.get(m, float("nan")), **record[m]}
                for m in models
            ]
        ).set_index("model")
        return out.sort_values(["wins", "score"], ascending=False)

    def to_csv(self, path: str) -> None:
        self.to_long_dataframe().to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.rows, fh, indent=2)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BenchmarkResults({len(self.rows)} rows)"
