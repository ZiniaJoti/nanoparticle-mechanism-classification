"""
manual_review_prepare.py
========================
Exports uncertain or low-confidence examples to a CSV for human review.

The output CSV adds a 'final_label' column (initially empty) which the
researcher fills in manually. After review, run preprocess_dataset.py to
merge corrections back into the main dataset.

Usage:
    python manual_review_prepare.py
    python manual_review_prepare.py --input data/labeled/abstracts_labeled.csv
    python manual_review_prepare.py --confidence_threshold 0.80
"""

import argparse
import logging
import os

import pandas as pd

import config

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "manual_review.log")),
    ],
)
logger = logging.getLogger(__name__)


def prepare_manual_review( 
    df: pd.DataFrame,
    confidence_threshold: float,
    out_path: str,
) -> pd.DataFrame:
    """
    Select uncertain or low-confidence examples and export for manual annotation.

    Args:
        df:                   Full labeled DataFrame.
        confidence_threshold: Rows with llm_confidence below this are flagged.
        out_path:             Path to save the review CSV.

    Returns:
        DataFrame of rows selected for review.
    """
    is_uncertain     = df["llm_label"].astype(str).str.lower() == "uncertain"
    is_low_conf      = pd.to_numeric(df["llm_confidence"], errors="coerce") < confidence_threshold

    review_mask = is_uncertain | is_low_conf
    review_df   = df[review_mask].copy()

    logger.info(f"Total abstracts             : {len(df)}")
    logger.info(f"Flagged as uncertain        : {is_uncertain.sum()}")
    logger.info(f"Low confidence (<{confidence_threshold:.2f})     : {is_low_conf.sum()}")
    logger.info(f"Total for review (unique)   : {len(review_df)}")

    # Add final_label column (blank — to be filled manually)
    review_df["final_label"] = ""

    # Select and order columns for readability
    cols_to_keep = [
        "pmid", "title", "abstract", "journal", "year",
        "llm_label", "llm_confidence", "llm_reason",
        "uncertain", "final_label",
    ]
    # Only keep columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in review_df.columns]
    review_df    = review_df[cols_to_keep]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    review_df.to_csv(out_path, index=False)
    logger.info(f"Manual review CSV saved → {out_path}")

    return review_df


def merge_reviewed_labels(
    labeled_df: pd.DataFrame,
    reviewed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge human-reviewed final_labels back into the main dataset.

    For rows where the reviewer filled in 'final_label', that value is used.
    For all other rows, 'final_label' falls back to 'llm_label'.

    Args:
        labeled_df:  Full labeled DataFrame (from LABELED_CSV).
        reviewed_df: Reviewed DataFrame with 'final_label' filled in.

    Returns:
        Updated DataFrame with 'final_label' column for all rows.
    """
    # Build a mapping pmid → final_label from the reviewed sheet
    reviewed_map = (
        reviewed_df[reviewed_df["final_label"].notna() & (reviewed_df["final_label"] != "")]
        .set_index("pmid")["final_label"]
        .to_dict()
    )

    if "final_label" not in labeled_df.columns:
        labeled_df["final_label"] = None

    # Apply corrections
    labeled_df["pmid"] = labeled_df["pmid"].astype(str)
    labeled_df["final_label"] = labeled_df.apply(
        lambda row: reviewed_map.get(str(row["pmid"]), row.get("final_label")),
        axis=1,
    )

    # Fallback: rows with no final_label get the llm_label (if it's not uncertain)
    fallback_mask = labeled_df["final_label"].isna() | (labeled_df["final_label"] == "")
    labeled_df.loc[fallback_mask, "final_label"] = labeled_df.loc[fallback_mask, "llm_label"].apply(
        lambda x: x if x in config.LABEL_NAMES else None
    )

    n_reviewed  = labeled_df["final_label"].notna().sum()
    n_remaining = labeled_df["final_label"].isna().sum()
    logger.info(f"final_label set for {n_reviewed} rows | Missing: {n_remaining}")

    return labeled_df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare uncertain abstracts for manual review."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=config.LABELED_CSV,
        help="Path to labeled abstracts CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.MANUAL_REVIEW_CSV,
        help="Output path for manual review CSV.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=config.UNCERTAIN_CONFIDENCE_THRESHOLD,
        help="Confidence threshold below which rows are flagged for review.",
    )
    parser.add_argument(
        "--merge",
        type=str,
        default=None,
        help="Path to a completed manual review CSV to merge back. "
             "If provided, merges reviewed labels into --input and resaves it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} labeled abstracts from {args.input}")

    if args.merge:
        if not os.path.exists(args.merge):
            logger.error(f"Reviewed CSV not found: {args.merge}")
            return
        reviewed = pd.read_csv(args.merge)
        logger.info(f"Merging {len(reviewed)} reviewed rows from {args.merge}")
        df = merge_reviewed_labels(df, reviewed)
        df.to_csv(args.input, index=False)
        logger.info(f"Updated labeled CSV saved → {args.input}")
    else:
        prepare_manual_review(df, args.confidence_threshold, args.output)
        logger.info(
            "\nInstructions:\n"
            f"  1. Open {args.output} in Excel or Google Sheets\n"
            "  2. Fill in the 'final_label' column for each row\n"
            "  3. Use one of: wrapping, adhesion, penetration, endocytosis, or leave blank to discard\n"
            f"  4. Re-run with --merge {args.output} to incorporate your corrections"
        )


if __name__ == "__main__":
    main()
