"""
preprocess_dataset.py
=====================
Cleans and preprocesses the labeled dataset, then creates stratified
train/validation/test splits.

Steps:
  1. Load labeled CSV
  2. Remove duplicates (by PMID and by abstract text)
  3. Remove rows with empty abstracts
  4. Keep only rows with a valid final_label
  5. Normalize text (whitespace, encoding)
  6. Create stratified train / val / test splits
  7. Save processed CSVs + full clean CSV

Usage:
    python preprocess_dataset.py
    python preprocess_dataset.py --input data/labeled/final_dataset.csv
    python preprocess_dataset.py --train 0.70 --val 0.15 --test 0.15
"""

import argparse
import logging
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "preprocess.log")),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Normalize a text string:
      - Decode to unicode NFC form
      - Replace special whitespace with regular spaces
      - Collapse multiple spaces/newlines
      - Strip leading/trailing whitespace

    Args:
        text: Raw text string.

    Returns:
        Normalized string.
    """
    if not isinstance(text, str):
        return ""
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Replace various whitespace characters with a single space
    text = re.sub(r"[\r\n\t\xa0\u200b\u00ad]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the DataFrame.

    Args:
        df: Raw labeled DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    original_size = len(df)
    logger.info(f"Starting cleaning. Rows: {original_size}")

    # ── 1. Normalize text columns ─────────────────
    df["abstract"] = df["abstract"].apply(normalize_text)
    df["title"]    = df["title"].apply(normalize_text)

    # ── 2. Drop rows with empty abstracts ─────────
    df = df[df["abstract"].str.len() > 50].copy()
    logger.info(f"After dropping short/empty abstracts: {len(df)}")

    # ── 3. Keep only valid final_labels ───────────
    if "final_label" not in df.columns:
        # Fall back to llm_label if final_label was never created
        df["final_label"] = df["llm_label"]
    
    df = df[df["final_label"].isin(config.LABEL_NAMES)].copy()
    logger.info(f"After filtering to valid labels {config.LABEL_NAMES}: {len(df)}")

    # ── 4. Drop duplicate PMIDs ───────────────────
    if "pmid" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["pmid"], keep="first").copy()
        logger.info(f"Dropped {before - len(df)} duplicate PMIDs")

    # ── 5. Drop duplicate abstracts ───────────────
    before = len(df)
    df = df.drop_duplicates(subset=["abstract"], keep="first").copy()
    logger.info(f"Dropped {before - len(df)} duplicate abstracts")

    # ── 6. Add label_id column ────────────────────
    df["label_id"] = df["final_label"].map(config.LABEL2ID)

    # ── 7. Reset index ────────────────────────────
    df = df.reset_index(drop=True)

    logger.info(f"Cleaning complete. {original_size} → {len(df)} rows retained.")
    logger.info(f"Label distribution after cleaning: {df['llm_label'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio:   float = config.VAL_RATIO,
    test_ratio:  float = config.TEST_RATIO,
    seed:        int   = config.RANDOM_SEED,
) -> tuple:
    """
    Create stratified train / validation / test splits.

    Args:
        df:          Cleaned DataFrame with 'final_label' column.
        train_ratio: Fraction of data for training.
        val_ratio:   Fraction for validation.
        test_ratio:  Fraction for test.
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    logger.info(
        f"Splitting: train={train_ratio:.0%} | val={val_ratio:.0%} | test={test_ratio:.0%} "
        f"| seed={seed}"
    )

    # Step 1: split off test set
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=df["final_label"],
    )

    # Step 2: split remaining into val and test
    test_fraction_of_temp = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_fraction_of_temp,
        random_state=seed,
        stratify=temp_df["final_label"],
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Log per-class distribution
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_df["final_label"].value_counts().to_dict()
        logger.info(f"  {split_name} distribution: {counts}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ─────────────────────────────────────────────
# SAVE SPLITS
# ─────────────────────────────────────────────
def save_splits(
    all_df:   pd.DataFrame,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Save all four DataFrames to the processed data directory."""
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    all_df.to_csv(config.PROCESSED_ALL,   index=False)
    train_df.to_csv(config.PROCESSED_TRAIN, index=False)
    val_df.to_csv(config.PROCESSED_VAL,   index=False)
    test_df.to_csv(config.PROCESSED_TEST,  index=False)

    logger.info(f"Saved: {config.PROCESSED_ALL}   ({len(all_df)} rows)")
    logger.info(f"Saved: {config.PROCESSED_TRAIN} ({len(train_df)} rows)")
    logger.info(f"Saved: {config.PROCESSED_VAL}   ({len(val_df)} rows)")
    logger.info(f"Saved: {config.PROCESSED_TEST}  ({len(test_df)} rows)")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and split the labeled abstract dataset."
    )
    parser.add_argument("--input",  type=str, default=config.LABELED_CSV,
                        help="Input labeled CSV path.")
    parser.add_argument("--train",  type=float, default=config.TRAIN_RATIO,
                        help="Train split ratio.")
    parser.add_argument("--val",    type=float, default=config.VAL_RATIO,
                        help="Validation split ratio.")
    parser.add_argument("--test",   type=float, default=config.TEST_RATIO,
                        help="Test split ratio.")
    parser.add_argument("--seed",   type=int,   default=config.RANDOM_SEED,
                        help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    # Clean
    clean_df = clean_dataframe(df)
    
    # Split
    train_df, val_df, test_df = split_dataset(
        clean_df,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )

    # Save
    save_splits(clean_df, train_df, val_df, test_df)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
