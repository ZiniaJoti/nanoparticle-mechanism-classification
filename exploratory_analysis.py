"""
exploratory_analysis.py
=======================
Exploratory data analysis: class distribution, year distribution,
abstract length statistics, uncertain sample count.

Generates and saves a multi-panel PNG of plots.

Usage:
    python exploratory_analysis.py
    python exploratory_analysis.py --input data/labeled/abstracts_labeled.csv
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_eda(df: pd.DataFrame, out_path: str = config.EDA_PLOT_FILE) -> None:
    """
    Generate and save EDA plots.

    Args:
        df:       DataFrame (labeled or processed).
        out_path: Path to save the output PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Nanoparticle–Membrane Interaction: Dataset EDA", fontsize=16, fontweight="bold")

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#999999"]

    # ── 1. LLM Label Distribution ────────────────────
    ax = axes[0, 0]
    label_col = "llm_label" if "llm_label" in df.columns else "final_label"
    label_counts = df[label_col].value_counts()
    bars = ax.bar(label_counts.index, label_counts.values, color=COLORS[:len(label_counts)])
    ax.set_title("LLM Label Distribution (incl. uncertain)", fontsize=12)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    # ── 2. Final Label Distribution (4 classes only) ─
    ax = axes[0, 1]
    if "final_label" in df.columns:
        final_counts = df[df["final_label"].isin(config.LABEL_NAMES)]["final_label"].value_counts()
    else:
        final_counts = df[df[label_col].isin(config.LABEL_NAMES)][label_col].value_counts()
    bars = ax.bar(final_counts.index, final_counts.values, color=COLORS[:4])
    ax.set_title("Final Label Distribution (4 classes)", fontsize=12)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    # ── 3. Uncertain vs Certain ──────────────────────
    ax = axes[0, 2]
    if "uncertain" in df.columns:
        uncertain_counts = df["uncertain"].value_counts()
        labels_pie = ["Certain", "Uncertain"]
        values_pie = [
            uncertain_counts.get(False, 0),
            uncertain_counts.get(True,  0),
        ]
    else:
        n_uncertain = (df[label_col] == "uncertain").sum()
        values_pie  = [len(df) - n_uncertain, n_uncertain]
        labels_pie  = ["Certain", "Uncertain"]
    ax.pie(values_pie, labels=labels_pie, autopct="%1.1f%%",
           colors=["#55A868", "#C44E52"], startangle=90)
    ax.set_title("Uncertain vs Certain Abstracts", fontsize=12)

    # ── 4. Publication Year Distribution ────────────
    ax = axes[1, 0]
    if "year" in df.columns:
        year_series = pd.to_numeric(df["year"], errors="coerce").dropna()
        year_series = year_series[(year_series >= 1990) & (year_series <= 2030)]
        year_counts = year_series.astype(int).value_counts().sort_index()
        ax.bar(year_counts.index, year_counts.values, color="#4C72B0", width=0.8)
        ax.set_title("Publication Year Distribution", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "Year column not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Publication Year Distribution", fontsize=12)

    # ── 5. Abstract Length Distribution ─────────────
    ax = axes[1, 1]
    if "abstract" in df.columns:
        lengths = df["abstract"].dropna().str.split().str.len()
        ax.hist(lengths, bins=40, color="#8172B2", edgecolor="white", linewidth=0.5)
        ax.axvline(lengths.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {lengths.mean():.0f}")
        ax.axvline(lengths.median(), color="orange", linestyle="--", linewidth=1.5,
                   label=f"Median: {lengths.median():.0f}")
        ax.set_title("Abstract Length (words)", fontsize=12)
        ax.set_xlabel("Word count")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Abstract column not available", ha="center", va="center",
                transform=ax.transAxes)

    # ── 6. LLM Confidence Distribution ──────────────
    ax = axes[1, 2]
    if "llm_confidence" in df.columns:
        conf = pd.to_numeric(df["llm_confidence"], errors="coerce").dropna()
        ax.hist(conf, bins=20, color="#DD8452", edgecolor="white", linewidth=0.5)
        ax.axvline(config.UNCERTAIN_CONFIDENCE_THRESHOLD, color="red", linestyle="--",
                   linewidth=1.5, label=f"Threshold: {config.UNCERTAIN_CONFIDENCE_THRESHOLD}")
        ax.set_title("LLM Confidence Score Distribution", fontsize=12)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "llm_confidence column not available", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"EDA plots saved → {out_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print a text summary of the dataset to the logger."""
    logger.info("=" * 50)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total rows         : {len(df)}")

    label_col = "final_label" if "final_label" in df.columns else "llm_label"
    if label_col in df.columns:
        logger.info(f"\n{label_col} distribution:")
        for label, count in df[label_col].value_counts().items():
            pct = count / len(df) * 100
            logger.info(f"  {label:<15}: {count:>5}  ({pct:.1f}%)")

    if "year" in df.columns:
        year_series = pd.to_numeric(df["year"], errors="coerce").dropna()
        if len(year_series):
            logger.info(f"\nYear range         : {int(year_series.min())} – {int(year_series.max())}")

    if "abstract" in df.columns:
        lengths = df["abstract"].dropna().str.split().str.len()
        logger.info(f"Avg abstract length: {lengths.mean():.0f} words")
        logger.info(f"Min/Max length     : {lengths.min()} / {lengths.max()} words")

    if "uncertain" in df.columns:
        n_uncertain = df["uncertain"].sum()
        logger.info(f"\nUncertain samples  : {n_uncertain} ({n_uncertain/len(df)*100:.1f}%)")

    logger.info("=" * 50)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Exploratory analysis of labeled PubMed abstracts."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=config.LABELED_CSV,
        help="Path to labeled CSV (or processed CSV).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.EDA_PLOT_FILE,
        help="Output PNG path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    print_summary(df)
    run_eda(df, out_path=args.output)


if __name__ == "__main__":
    main()
