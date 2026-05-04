"""
exploratory_data_analysis.py
=======================
Exploratory data analysis: class distribution, year distribution,
abstract length statistics, uncertain sample count.

Generates and saves:
  1. A combined multi-panel PNG
  2. Individual plot PNGs for each EDA panel

Usage:
    python exploratory_data_analysis.py
    python exploratory_data_analysis.py --input data/labeled/abstracts_labeled.csv
    python exploratory_data_analysis.py --input data/processed/clean_labeled.csv
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#999999"]


def get_label_col(df: pd.DataFrame) -> str:
    if "llm_label" in df.columns:
        return "llm_label"
    if "final_label" in df.columns:
        return "final_label"
    raise ValueError("No label column found. Expected 'llm_label' or 'final_label'.")


def plot_label_distribution(df: pd.DataFrame, ax=None, save_path=None):
    label_col = get_label_col(df)
    label_counts = df[label_col].value_counts()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None

    bars = ax.bar(label_counts.index, label_counts.values, color=COLORS[:len(label_counts)])
    ax.set_title("LLM Label Distribution (incl. uncertain)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if save_path and fig is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_final_label_distribution(df: pd.DataFrame, ax=None, save_path=None):
    label_col = get_label_col(df)

    if "final_label" in df.columns:
        final_counts = df[df["final_label"].isin(config.LABEL_NAMES)]["final_label"].value_counts()
    else:
        final_counts = df[df[label_col].isin(config.LABEL_NAMES)][label_col].value_counts()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None

    bars = ax.bar(final_counts.index, final_counts.values, color=COLORS[:len(final_counts)])
    ax.set_title("Final Label Distribution (4 classes)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if save_path and fig is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_uncertain_pie(df: pd.DataFrame, ax=None, save_path=None):
    label_col = get_label_col(df)

    if "uncertain" in df.columns:
        uncertain_bool = df["uncertain"].astype(str).str.lower().isin(["true", "1", "yes"])
        n_uncertain = uncertain_bool.sum()
    else:
        n_uncertain = (df[label_col].astype(str).str.lower() == "uncertain").sum()

    values_pie = [len(df) - n_uncertain, n_uncertain]
    labels_pie = ["Certain", "Uncertain"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = None

    ax.pie(
        values_pie,
        labels=labels_pie,
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"],
        startangle=90,
    )
    ax.set_title("Uncertain vs Certain Abstracts", fontsize=12, fontweight="bold")

    if save_path and fig is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_year_distribution(df: pd.DataFrame, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = None

    if "year" in df.columns:
        year_series = pd.to_numeric(df["year"], errors="coerce").dropna()
        year_series = year_series[(year_series >= 1990) & (year_series <= 2030)]
        year_counts = year_series.astype(int).value_counts().sort_index()

        ax.bar(year_counts.index, year_counts.values, color="#4C72B0", width=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "Year column not available", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Publication Year Distribution", fontsize=12, fontweight="bold")

    if save_path and fig is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_abstract_length(df: pd.DataFrame, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = None

    if "abstract" in df.columns:
        lengths = df["abstract"].dropna().astype(str).str.split().str.len()
        ax.hist(lengths, bins=40, color="#8172B2", edgecolor="white", linewidth=0.5)
        ax.axvline(lengths.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {lengths.mean():.0f}")
        ax.axvline(lengths.median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median: {lengths.median():.0f}")
        ax.set_xlabel("Word count")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Abstract column not available", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Abstract Length (words)", fontsize=12, fontweight="bold")

    if save_path and fig is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_confidence_distribution(df: pd.DataFrame, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = None

    if "llm_confidence" in df.columns:
        conf = pd.to_numeric(df["llm_confidence"], errors="coerce").dropna()
        ax.hist(conf, bins=20, color="#DD8452", edgecolor="white", linewidth=0.5)
        ax.axvline(
            config.UNCERTAIN_CONFIDENCE_THRESHOLD,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold: {config.UNCERTAIN_CONFIDENCE_THRESHOLD}",
        )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "llm_confidence column not available", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("LLM Confidence Score Distribution", fontsize=12, fontweight="bold")

    if save_path and fig is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def run_eda(
    df: pd.DataFrame,
    out_path: str = config.EDA_PLOT_FILE,
    output_dir_individual: str = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if output_dir_individual is None:
        output_dir_individual = os.path.join(os.path.dirname(out_path), "eda_individual_plots")
    os.makedirs(output_dir_individual, exist_ok=True)

    # Combined plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Nanoparticle-Membrane Interaction: Dataset EDA", fontsize=16, fontweight="bold")

    plot_label_distribution(df, ax=axes[0, 0])
    plot_final_label_distribution(df, ax=axes[0, 1])
    plot_uncertain_pie(df, ax=axes[0, 2])
    plot_year_distribution(df, ax=axes[1, 0])
    plot_abstract_length(df, ax=axes[1, 1])
    plot_confidence_distribution(df, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Combined EDA plot saved -> {out_path}")

    # Individual plots
    plot_label_distribution(
        df,
        save_path=os.path.join(output_dir_individual, "01_label_distribution.png"),
    )
    plot_final_label_distribution(
        df,
        save_path=os.path.join(output_dir_individual, "02_final_label_distribution.png"),
    )
    plot_uncertain_pie(
        df,
        save_path=os.path.join(output_dir_individual, "03_uncertain_vs_certain.png"),
    )
    plot_year_distribution(
        df,
        save_path=os.path.join(output_dir_individual, "04_year_distribution.png"),
    )
    plot_abstract_length(
        df,
        save_path=os.path.join(output_dir_individual, "05_abstract_length_distribution.png"),
    )
    plot_confidence_distribution(
        df,
        save_path=os.path.join(output_dir_individual, "06_confidence_distribution.png"),
    )

    logger.info(f"Individual EDA plots saved -> {output_dir_individual}")


def print_summary(df: pd.DataFrame) -> None:
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
            logger.info(f"\nYear range         : {int(year_series.min())} - {int(year_series.max())}")

    if "abstract" in df.columns:
        lengths = df["abstract"].dropna().astype(str).str.split().str.len()
        logger.info(f"Avg abstract length: {lengths.mean():.0f} words")
        logger.info(f"Min/Max length     : {lengths.min()} / {lengths.max()} words")

    if "uncertain" in df.columns:
        uncertain_bool = df["uncertain"].astype(str).str.lower().isin(["true", "1", "yes"])
        n_uncertain = uncertain_bool.sum()
        logger.info(f"\nUncertain samples  : {n_uncertain} ({n_uncertain / len(df) * 100:.1f}%)")

    logger.info("=" * 50)


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
        help="Output combined PNG path.",
    )
    parser.add_argument(
        "--output_dir_individual",
        type=str,
        default=None,
        help="Directory to save individual EDA plots.",
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
    run_eda(
        df,
        out_path=args.output,
        output_dir_individual=args.output_dir_individual,
    )


if __name__ == "__main__":
    main()