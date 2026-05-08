#!/usr/bin/env python3
"""
Create benchmark figures from llama.cpp vs MLC-LLM Vulkan results.

Input files are hardcoded below.

Expected input files:
  - results/results.csv
  - results/summary.csv
  - results/raw_results.jsonl
  - results/run_metadata.json

Output figures:
  - results/figures/ttft_bar_mean.png
  - results/figures/tps_bar_mean.png
  - results/figures/ttft_boxplot.png
  - results/figures/tps_boxplot.png
  - results/figures/ttft_runs_line.png
  - results/figures/tps_runs_line.png
  - results/figures/ttft_vs_tps_scatter.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_CSV = Path("results/results.csv")
SUMMARY_CSV = Path("results/summary.csv")
RAW_JSONL = Path("results/raw_results.jsonl")
METADATA_JSON = Path("results/run_metadata.json")
FIGURE_DIR = Path("results/figures")


BACKEND_ORDER = ["llama.cpp-vulkan", "mlc-llm-vulkan"]
MODEL_ORDER = [
    "tinyllama-1.1b-chat-q4",
    "qwen2.5-0.5b-instruct-q4",
]


def load_inputs():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing required file: {RESULTS_CSV}")

    results = pd.read_csv(RESULTS_CSV)

    summary = None
    if SUMMARY_CSV.exists():
        summary = pd.read_csv(SUMMARY_CSV)

    metadata = None
    if METADATA_JSON.exists():
        with METADATA_JSON.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    raw_count = 0
    if RAW_JSONL.exists():
        with RAW_JSONL.open("r", encoding="utf-8") as f:
            raw_count = sum(1 for _ in f)

    return results, summary, metadata, raw_count


def clean_results(df):
    df = df.copy()
    df = df[df["status"] == "ok"]

    df["ttft_s"] = pd.to_numeric(df["ttft_s"], errors="coerce")
    df["tps"] = pd.to_numeric(df["tps"], errors="coerce")
    df["run"] = pd.to_numeric(df["run"], errors="coerce")

    df = df.dropna(subset=["ttft_s", "tps", "run", "backend", "model_label"])

    df["backend"] = pd.Categorical(
        df["backend"],
        categories=BACKEND_ORDER,
        ordered=True,
    )
    df["model_label"] = pd.Categorical(
        df["model_label"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df = df.sort_values(["model_label", "backend", "run"])
    return df


def grouped_summary(df):
    return (
        df.groupby(["model_label", "backend"], observed=True)
        .agg(
            ttft_mean_s=("ttft_s", "mean"),
            ttft_stdev_s=("ttft_s", "std"),
            tps_mean=("tps", "mean"),
            tps_stdev=("tps", "std"),
            successful_runs=("run", "count"),
        )
        .reset_index()
    )


def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_bar(summary_df, metric_mean, metric_stdev, ylabel, title, output_path):
    models = [m for m in MODEL_ORDER if m in summary_df["model_label"].astype(str).unique()]
    backends = [b for b in BACKEND_ORDER if b in summary_df["backend"].astype(str).unique()]

    x = range(len(models))
    width = 0.36

    plt.figure(figsize=(9, 5))

    for idx, backend in enumerate(backends):
        values = []
        errors = []

        for model in models:
            row = summary_df[
                (summary_df["model_label"].astype(str) == model)
                & (summary_df["backend"].astype(str) == backend)
            ]

            if row.empty:
                values.append(float("nan"))
                errors.append(0.0)
            else:
                values.append(float(row.iloc[0][metric_mean]))
                errors.append(float(row.iloc[0][metric_stdev]))

        positions = [i + (idx - 0.5) * width for i in x]

        plt.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=4,
            label=backend,
        )

    plt.xticks(list(x), models, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)


def plot_metric_boxplot(df, metric, ylabel, title, output_path):
    labels = []
    data = []

    for model in MODEL_ORDER:
        for backend in BACKEND_ORDER:
            subset = df[
                (df["model_label"].astype(str) == model)
                & (df["backend"].astype(str) == backend)
            ]

            if not subset.empty:
                labels.append(f"{model}\n{backend}")
                data.append(subset[metric].tolist())

    plt.figure(figsize=(11, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.xticks(rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)


def plot_metric_runs_line(df, metric, ylabel, title, output_path):
    plt.figure(figsize=(10, 5))

    for model in MODEL_ORDER:
        for backend in BACKEND_ORDER:
            subset = df[
                (df["model_label"].astype(str) == model)
                & (df["backend"].astype(str) == backend)
            ].sort_values("run")

            if subset.empty:
                continue

            label = f"{model} | {backend}"
            plt.plot(subset["run"], subset[metric], marker="o", label=label)

    plt.xlabel("Run")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(df["run"].unique()))
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    save_current_figure(output_path)


def plot_ttft_vs_tps(df, output_path):
    plt.figure(figsize=(8, 5))

    markers = {
        "tinyllama-1.1b-chat-q4": "o",
        "qwen2.5-0.5b-instruct-q4": "s",
    }

    for model in MODEL_ORDER:
        for backend in BACKEND_ORDER:
            subset = df[
                (df["model_label"].astype(str) == model)
                & (df["backend"].astype(str) == backend)
            ]

            if subset.empty:
                continue

            plt.scatter(
                subset["ttft_s"],
                subset["tps"],
                marker=markers.get(model, "o"),
                label=f"{model} | {backend}",
                alpha=0.8,
            )

    plt.xlabel("TTFT (s)")
    plt.ylabel("TPS")
    plt.title("TTFT vs TPS by Backend and Model")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    save_current_figure(output_path)


def print_report(df, summary_df, metadata, raw_count):
    print("Loaded benchmark data")
    print(f"  results rows: {len(df)}")
    print(f"  raw jsonl rows: {raw_count}")
    print(f"  figure directory: {FIGURE_DIR}")

    if metadata:
        print(f"  prompt: {metadata.get('prompt')}")
        print(f"  runs per model: {metadata.get('runs_per_model')}")
        print(f"  max_tokens: {metadata.get('max_tokens')}")
        print(f"  temperature: {metadata.get('temperature')}")

    print("\nComputed summary:")
    print(
        summary_df[
            [
                "backend",
                "model_label",
                "successful_runs",
                "ttft_mean_s",
                "ttft_stdev_s",
                "tps_mean",
                "tps_stdev",
            ]
        ].to_string(index=False)
    )


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results, provided_summary, metadata, raw_count = load_inputs()
    results = clean_results(results)

    if provided_summary is not None:
        summary_df = provided_summary.copy()
    else:
        summary_df = grouped_summary(results)

    plot_metric_bar(
        summary_df=summary_df,
        metric_mean="ttft_mean_s",
        metric_stdev="ttft_stdev_s",
        ylabel="TTFT (s)",
        title="Mean Time to First Token by Model and Backend",
        output_path=FIGURE_DIR / "ttft_bar_mean.png",
    )

    plot_metric_bar(
        summary_df=summary_df,
        metric_mean="tps_mean",
        metric_stdev="tps_stdev",
        ylabel="Tokens per Second",
        title="Mean Throughput by Model and Backend",
        output_path=FIGURE_DIR / "tps_bar_mean.png",
    )

    plot_metric_boxplot(
        df=results,
        metric="ttft_s",
        ylabel="TTFT (s)",
        title="TTFT Distribution Across Runs",
        output_path=FIGURE_DIR / "ttft_boxplot.png",
    )

    plot_metric_boxplot(
        df=results,
        metric="tps",
        ylabel="Tokens per Second",
        title="TPS Distribution Across Runs",
        output_path=FIGURE_DIR / "tps_boxplot.png",
    )

    plot_metric_runs_line(
        df=results,
        metric="ttft_s",
        ylabel="TTFT (s)",
        title="TTFT Across Repeated Runs",
        output_path=FIGURE_DIR / "ttft_runs_line.png",
    )

    plot_metric_runs_line(
        df=results,
        metric="tps",
        ylabel="Tokens per Second",
        title="TPS Across Repeated Runs",
        output_path=FIGURE_DIR / "tps_runs_line.png",
    )

    plot_ttft_vs_tps(
        df=results,
        output_path=FIGURE_DIR / "ttft_vs_tps_scatter.png",
    )

    print_report(results, summary_df, metadata, raw_count)

    print("\nWrote figures:")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
