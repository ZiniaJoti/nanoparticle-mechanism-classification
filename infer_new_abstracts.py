"""
infer_new_abstracts.py
======================
Load a fine-tuned BERT or SciBERT model and predict interaction mechanism
labels for new, unseen PubMed abstracts.

Returns the predicted label, class probabilities, and confidence.

Usage:
    # Predict a single abstract via CLI
    python infer_new_abstracts.py \
        --model_dir models/scibert_finetuned \
        --text "Nanoparticles of 50 nm diameter were observed to undergo 
                clathrin-mediated endocytosis in HeLa cells."

    # Predict from a CSV file
    python infer_new_abstracts.py \
        --model_dir models/bert_finetuned \
        --input data/raw/new_abstracts.csv \
        --output results/predictions.csv
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
class AbstractClassifier:
    """
    Wraps a fine-tuned transformer for easy single or batch inference.

    Args:
        model_dir:  Path to the saved model directory.
        max_length: Tokenizer max length (default from config).
        device:     'cpu', 'cuda', or None (auto-detect).
    """

    def __init__(
        self,
        model_dir: str,
        max_length: int = config.MAX_LENGTH,
        device: str = None,
    ):
        if not os.path.isdir(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.max_length = max_length

        logger.info(f"Loading model from {model_dir} on {self.device}…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def predict(self, texts: list) -> list:
        """
        Predict labels and probabilities for a list of abstract texts.

        Args:
            texts: List of strings (raw abstract text).

        Returns:
            List of dicts, each with:
              - predicted_label (str)
              - predicted_id    (int)
              - confidence      (float)
              - probabilities   (dict {label: prob})
        """
        if isinstance(texts, str):
            texts = [texts]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids      = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        token_type_ids = encodings.get("token_type_ids")

        with torch.no_grad():
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids.to(self.device)
            outputs = self.model(**kwargs)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for prob_row in probs:
            pred_id    = int(np.argmax(prob_row))
            pred_label = config.ID2LABEL[pred_id]
            confidence = float(prob_row[pred_id])
            prob_dict  = {
                config.ID2LABEL[i]: round(float(p), 4)
                for i, p in enumerate(prob_row)
            }
            results.append({
                "predicted_label": pred_label,
                "predicted_id":    pred_id,
                "confidence":      round(confidence, 4),
                "probabilities":   prob_dict,
            })

        return results

    def predict_one(self, text: str) -> dict:
        """Convenience wrapper for a single abstract."""
        return self.predict([text])[0]


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on new abstracts using a fine-tuned model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=config.SCIBERT_SAVE_DIR,
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single abstract text to classify (for quick testing).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="CSV file with an 'abstract' column to classify in batch.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(config.RESULTS_DIR, "predictions.csv"),
        help="Output CSV path for batch predictions.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=config.MAX_LENGTH,
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cpu" if args.no_cuda else None
    clf    = AbstractClassifier(
        model_dir=args.model_dir,
        max_length=args.max_length,
        device=device,
    )

    # ── Single text mode ──────────────────────────
    if args.text:
        result = clf.predict_one(args.text)
        print("\n" + "=" * 55)
        print("PREDICTION RESULT")
        print("=" * 55)
        print(f"Predicted label : {result['predicted_label']}")
        print(f"Confidence      : {result['confidence']:.4f}")
        print("Probabilities:")
        for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {label:<15} {prob:.4f}  {bar}")
        print("=" * 55)
        return

    # ── Batch CSV mode ────────────────────────────
    if args.input:
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return

        df = pd.read_csv(args.input)
        if "abstract" not in df.columns:
            logger.error("Input CSV must have an 'abstract' column.")
            return

        logger.info(f"Predicting {len(df)} abstracts…")
        if "title" in df.columns:
            texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
        else:
            texts = df["abstract"].fillna("").tolist()
        results = clf.predict(texts)

        df["predicted_label"] = [r["predicted_label"] for r in results]
        df["confidence"]      = [r["confidence"]      for r in results]
        for label in config.LABEL_NAMES:
            df[f"prob_{label}"] = [r["probabilities"].get(label, 0.0) for r in results]

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info(f"Predictions saved → {args.output}")

        # Summary
        logger.info("Prediction distribution:")
        for label, count in df["predicted_label"].value_counts().items():
            logger.info(f"  {label:<15}: {count}")
        return

    logger.warning("No input provided. Use --text or --input. Run with --help for usage.")


if __name__ == "__main__":
    main()
