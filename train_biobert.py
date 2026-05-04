"""
train_biobert.py
================
Fine-tunes BioBERT on the nanoparticle-membrane interaction
classification task (4 classes).

Outputs:
  - Best model checkpoint in models/biobert_finetuned/
  - Metrics JSON in results/biobert_metrics.json
  - Confusion matrix PNG in results/biobert_confusion_matrix.png
  - Per-epoch training curve PNG in results/biobert_training_curve.png

Usage:
    python train_biobert.py
    python train_biobert.py --epochs 5 --batch_size 16 --lr 2e-5
    python train_biobert.py --no_cuda
"""

import argparse
import logging
import os

import config
from train_bert import train

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "train_biobert.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BioBERT for abstract classification."
    )
    parser.add_argument("--model_name", type=str, default=config.BIOBERT_MODEL_NAME)
    parser.add_argument("--save_dir", type=str, default=config.BIOBERT_SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA.")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("BioBERT TRAINING")
    logger.info(f"Model         : {args.model_name}")
    logger.info("=" * 60)

    metrics = train(
        model_name=args.model_name,
        save_dir=args.save_dir,
        metrics_file=config.BIOBERT_METRICS_FILE,
        cm_file=config.BIOBERT_CONFUSION_MATRIX,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        seed=args.seed,
        use_cuda=not args.no_cuda,
    )

    logger.info("BioBERT training complete.")
    logger.info(f"  Test Accuracy : {metrics['test_accuracy']:.4f}")
    logger.info(f"  Test Macro F1 : {metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()