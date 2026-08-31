#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re
from typing import Any, Iterable, TypedDict

import data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from scipy.stats import mannwhitneyu  # type: ignore

OUT = Path("analysis/out")
OUT.mkdir(exist_ok=True)
METRICS = ("result_duration", "result_score")
BETTER_HIGHER = {"result_duration": False, "result_score": True}
ALPHA = 0.05
P_VALUE_EVIDENCE_MAX = 4.0


def numbers(values: Iterable[Any]) -> np.ndarray:
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return s.to_numpy(dtype=float)


def a12(left: np.ndarray, right: np.ndarray, higher_is_better: bool) -> float:
    if not higher_is_better:
        left, right = -left, -right
    d = left[:, None] - right[None, :]
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)


def effect_label(value: float) -> str:
    d = abs(value - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def mw_pair(left: Iterable[Any], right: Iterable[Any], metric: str):
    x, y = numbers(left), numbers(right)
    if not len(x) or not len(y):
        return None
    u, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(u), float(p), a12(x, y, BETTER_HIGHER[metric])


def extract_base(exp) -> dict[str, dict[str, dict[str, list[float]]]]:
    out: dict[str, dict[str, dict[str, list[float]]]] = {
        m: {"ours": {}, "base": {}} for m in METRICS
    }
    for name, rows in (("ours", exp.ours), ("base", exp.base)):
        for row in rows:
            sample = row.get("sample")
            if sample is None:
                continue
            sample = str(sample)
            for metric in METRICS:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                out[metric][name].setdefault(sample, []).append(value)
    return out


def extract_multi(exp) -> dict[str, dict[str, dict[str, list[float]]]]:
    out: dict[str, dict[str, dict[str, list[float]]]] = {m: {} for m in METRICS}
    for rows in exp.split_data:
        if not rows:
            continue
        method = str(rows[0].get("experiment", "N/A")).strip()
        for row in rows:
            sample = row.get("sample")
            if sample is None:
                continue
            sample = str(sample)
            for metric in METRICS:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                out[metric].setdefault(method, {}).setdefault(sample, []).append(value)
    return out


def compare_per_sample(
    exp_name: str, data_: dict, left: str, right: str
) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        lb = data_.get(metric, {}).get(left, {})
        rb = data_.get(metric, {}).get(right, {})
        for sample in sorted(set(lb) & set(rb)):
            result = mw_pair(lb[sample], rb[sample], metric)
            if result is None:
                continue
            u, p, value = result
            rows.append(
                {
                    "metric": metric.removeprefix("result_"),
                    "sample": sample,
                    "left": left,
                    "right": right,
                    "n_left": len(lb[sample]),
                    "n_right": len(rb[sample]),
                    "U": u,
                    "p-value": p,
                    "A12": value,
                    "effect": effect_label(value),
                }
            )
    return pd.DataFrame(rows)


class ModelInfo(TypedDict):
    name: str
    family: str
    coder: bool
    size: float


def model_parts(name: str) -> ModelInfo | None:
    m = re.fullmatch(r"(?P<family>[^:]+):(?P<variant>[^:]+)", str(name).strip().lower())
    if not m:
        return None
    size = re.search(r"([0-9]+(?:\.[0-9]+)?)b", m.group("variant"))
    if not size:
        return None
    family = re.sub(r"-coder$", "", m.group("family"))
    return {
        "name": str(name).strip(),
        "family": family,
        "coder": "coder" in m.group("family"),
        "size": float(size.group(1)),
    }


def model_pairs(methods: Iterable[str]):
    info: dict[str, ModelInfo] = {
        method: parts
        for method in methods
        if (parts := model_parts(method)) is not None
    }

    coder_pairs: list[tuple[str, str]] = []
    for coder_name, coder in info.items():
        if not coder["coder"]:
            continue
        normal = next(
            (
                n
                for n, x in info.items()
                if x["family"] == coder["family"]
                and not x["coder"]
                and x["size"] == coder["size"]
            ),
            None,
        )
        if normal:
            coder_pairs.append((coder_name, normal))

    # Compare weights within a family and within the same coder/non-coder variant.
    groups: dict[tuple[str, bool], list[tuple[float, str]]] = {}
    for name, item in info.items():
        groups.setdefault((item["family"], item["coder"]), []).append(
            (item["size"], name)
        )
    weight_pairs: list[tuple[str, str]] = []
    for variants in groups.values():
        variants.sort()
        weight_pairs.extend((a[1], b[1]) for a, b in combinations(variants, 2))
    return coder_pairs, weight_pairs


def prompt_methods(methods: Iterable[str]):
    methods = set(map(str, methods))
    default = next((m for m in methods if m == "DefaultPrompt"), None)
    uncovered = next((m for m in methods if m == "UncoveredTargetsPrompt"), None)
    uncompressed = next(
        (m for m in methods if m in {"NoCompressionPrompt", "UncompressedPrompt"}), None
    )
    return default, uncovered, uncompressed


def descriptive(data_: dict, experiment: str) -> pd.DataFrame:
    rows = []
    for metric, methods in data_.items():
        for method, samples in methods.items():
            for sample, values in samples.items():
                x = numbers(values)
                rows.append(
                    {
                        "experiment": experiment,
                        "metric": metric.removeprefix("result_"),
                        "method": method,
                        "sample": sample,
                        "n": len(x),
                        "mean": np.mean(x) if len(x) else np.nan,
                        "median": np.median(x) if len(x) else np.nan,
                        "std": np.std(x, ddof=1) if len(x) > 1 else np.nan,
                        "q1": np.percentile(x, 25) if len(x) else np.nan,
                        "q3": np.percentile(x, 75) if len(x) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def latex(
    df: pd.DataFrame, caption: str | None = None, label: str | None = None
) -> str:
    if df.empty:
        return ""
    x = df.copy()
    x = x.rename(
        columns={
            "metric": "Metric",
            "sample": "Sample",
            "left": "A",
            "right": "B",
            "n_left": "$n_A$",
            "n_right": "$n_B$",
            "U": "$U$",
            "p-value": "$p$",
            "A12": r"$\hat{A}_{12}$",
            "effect": "Effect",
        }
    )
    cols = [
        "Metric",
        "Sample",
        "A",
        "B",
        "$p$",
        r"$\hat{A}_{12}$",
        "Effect",
    ]
    x = x[[c for c in cols if c in x.columns]]
    for column in ("Metric", "Sample", "A", "B", "Effect"):
        if column in x:
            x[column] = x[column].map(
                lambda value: re.sub(
                    r"Model\s+experiment\s*\((?P<name>[^)]*)\)", r"\1", str(value)
                )
                .replace("\\", r"\textbackslash{}")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("_", r"\_")
                .replace("#", r"\#")
            )
    table = x.to_latex(
        index=False,
        escape=False,
        longtable=True,
        caption=caption,
        label=label,
        float_format=lambda v: f"{v:.4f}",
    )
    # Use a compact layout so wide tables fit within the page.
    table = table.replace(
        "\\begin{longtable}",
        "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}",
        1,
    )
    return table


def a12_heatmap(df: pd.DataFrame, name: str, metric: str):
    d = df[df.metric == metric].copy()
    if d.empty:
        return
    d["comparison"] = d.left + " vs " + d.right
    p = d.pivot(index="sample", columns="comparison", values="A12")
    fig, ax = plt.subplots(
        figsize=(max(8, 1.8 * len(p.columns)), max(4.5, 0.35 * len(p)))
    )
    im = ax.imshow(p.values, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(p.columns)), p.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(p.index)), p.index)
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Sample")
    ax.set_title(f"{name}: {metric} — Vargha-Delaney $\\hat{{A}}_{{12}}$")
    fig.colorbar(im, ax=ax, label=r"$\hat{A}_{12}$")
    if p.size <= 150:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if pd.notna(p.iat[i, j]):
                    ax.text(
                        j, i, f"{p.iat[i, j]:.2f}", ha="center", va="center", fontsize=8
                    )
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_{metric}_a12.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pvalue_heatmap(df: pd.DataFrame, name: str, metric: str):
    d = df[df.metric == metric].copy()
    if d.empty:
        return
    d["comparison"] = d.left + " vs " + d.right
    p = d.pivot(index="sample", columns="comparison", values="p-value")
    z = -np.log10(p.clip(lower=np.finfo(float).tiny))
    fig, ax = plt.subplots(
        figsize=(max(8, 1.8 * len(p.columns)), max(4.5, 0.35 * len(p)))
    )
    im = ax.imshow(
        z.values,
        aspect="auto",
        vmin=0.0,
        vmax=P_VALUE_EVIDENCE_MAX,
    )
    ax.set_xticks(range(len(p.columns)), p.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(p.index)), p.index)
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Sample")
    ax.set_title(f"{name}: {metric} — statistical evidence")
    fig.colorbar(im, ax=ax, label=r"$-\log_{10}(p)$")
    if p.size <= 150:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if pd.notna(p.iat[i, j]):
                    ax.text(
                        j, i, f"{p.iat[i, j]:.2g}", ha="center", va="center", fontsize=8
                    )
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_{metric}_pvalues.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evidence_heatmap(df: pd.DataFrame, name: str, metric: str):
    """Show the effect size and its p-value evidence in one figure."""
    d = df[df.metric == metric].copy()
    if d.empty:
        return
    d["comparison"] = d.left + " vs " + d.right
    a12 = d.pivot(index="sample", columns="comparison", values="A12")
    p = d.pivot(index="sample", columns="comparison", values="p-value").reindex(
        index=a12.index, columns=a12.columns
    )
    evidence = -np.log10(p.clip(lower=np.finfo(float).tiny))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(12, 3.6 * len(a12.columns)), max(4.5, 0.35 * len(a12))),
        sharey=True,
    )
    panels = [
        (axes[0], a12, r"Vargha-Delaney $\hat{A}_{12}$", r"$\hat{A}_{12}$", 0.0, 1.0),
        (
            axes[1],
            evidence,
            "Statistical evidence",
            r"$-\log_{10}(p)$",
            0.0,
            P_VALUE_EVIDENCE_MAX,
        ),
    ]
    for ax, values, title, colorbar_label, vmin, vmax in panels:
        image = ax.imshow(values.values, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(
            range(len(values.columns)), values.columns, rotation=45, ha="right"
        )
        ax.set_xlabel("Comparison")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label=colorbar_label)
        if values.size <= 150:
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if pd.notna(values.iat[i, j]):
                        text = (
                            f"{a12.iat[i, j]:.2f}"
                            if values is a12
                            else f"{p.iat[i, j]:.2g}"
                        )
                        ax.text(j, i, text, ha="center", va="center", fontsize=8)
    axes[0].set_yticks(range(len(a12.index)), a12.index)
    axes[0].set_ylabel("Sample")
    fig.suptitle(f"{name}: {metric} — statistical evidence")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_{metric}_evidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def distribution_plot(desc: pd.DataFrame, experiment: str, metric: str, filename: str):
    d = desc[(desc.experiment == experiment) & (desc.metric == metric)]
    if d.empty:
        return
    methods = list(d.method.drop_duplicates())
    values = [d.loc[d.method == m, "median"].dropna().to_numpy() for m in methods]
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(methods)), 6))
    ax.boxplot(values, tick_labels=methods)
    ax.set_title(f"{experiment}: {metric} — sample-level medians")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Median over stochastic runs")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run():
    normalized = {}
    desc_all = []
    comparisons = {}

    for exp in data.EXPERIMENTS:
        name = str(exp.file_name)
        print(f"Experiment: {name}")
        if isinstance(exp, data.ExperimentDataWithBase):
            d = extract_base(exp)
            comparisons["classic_vs_ours"] = compare_per_sample(name, d, "ours", "base")
        elif isinstance(exp, data.ExperimentData):
            d = extract_multi(exp)
        else:
            continue
        normalized[name] = d
        desc_all.append(descriptive(d, name))

    # Models: coder vs non-coder, then different weights in the same family.
    if "model_results.csv" in normalized:
        d = normalized["model_results.csv"]
        methods = {m for metric in METRICS for m in d.get(metric, {})}
        coder_pairs, weight_pairs = model_pairs(methods)
        comparisons["coder_vs_non_coder"] = (
            pd.concat(
                [
                    compare_per_sample("model_results.csv", d, a, b)
                    for a, b in coder_pairs
                ],
                ignore_index=True,
            )
            if coder_pairs
            else pd.DataFrame()
        )
        comparisons["model_weight"] = (
            pd.concat(
                [
                    compare_per_sample("model_results.csv", d, a, b)
                    for a, b in weight_pairs
                ],
                ignore_index=True,
            )
            if weight_pairs
            else pd.DataFrame()
        )

    # Prompts: compare every pair of available prompt methods.
    if "prompt_results.csv" in normalized:
        d = normalized["prompt_results.csv"]
        methods = {m for metric in METRICS for m in d.get(metric, {})}
        prompt_frames = [
            compare_per_sample("prompt_results.csv", d, left, right)
            for left, right in combinations(sorted(methods), 2)
        ]
        comparisons["prompt"] = (
            pd.concat(prompt_frames, ignore_index=True)
            if prompt_frames
            else pd.DataFrame()
        )

    desc = pd.concat(desc_all, ignore_index=True) if desc_all else pd.DataFrame()
    desc.to_csv(OUT / "descriptive_statistics.csv", index=False)

    for name, df in comparisons.items():
        if df.empty:
            print(f"No results for {name}")
            continue
        df.to_csv(OUT / f"{name}.csv", index=False)
        (OUT / f"{name}.tex").write_text(
            latex(
                df,
                caption=f"All statistical comparisons: {name.replace('_', ' ')}",
                label=f"tab:{name}",
            ),
            encoding="utf-8",
        )
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))
        print("\nLaTeX table:")
        print(latex(df))
        for metric in ("duration", "score"):
            evidence_heatmap(df, name, metric)

    # The distributions are the main benchmark-wide graphical summaries.
    for filename, stem in (
        ("classic_results.csv", "classic"),
        ("model_results.csv", "models"),
        ("prompt_results.csv", "prompts"),
    ):
        for metric in ("duration", "score"):
            distribution_plot(
                desc, filename, metric, f"{stem}_{metric}_distribution.png"
            )

    print("\nStatistical conventions:")
    print("- Metrics: duration and score only; result_success is excluded.")
    print(
        "- Mann-Whitney U, two-sided, alpha=0.05, calculated independently for each sample."
    )
    print("- Vargha-Delaney A12 is computed from the same two samples.")
    print(
        "- For duration, values are oriented so A12 > 0.5 means that A (the left configuration) is faster."
    )
    print("- No pooling of runs from different samples for inferential tests.")
    print(
        "- No Holm correction: results are reported as pre-specified, per-sample comparisons and interpreted alongside the graphical cross-sample distributions."
    )


if __name__ == "__main__":
    run()
