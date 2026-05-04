"""
train_bert.py
=============
Fine-tunes bert-base-uncased on the nanoparticle–membrane interaction
classification task (4 classes).

Outputs:
  - Best model checkpoint in models/bert_finetuned/
  - Metrics JSON in results/bert_metrics.json
  - Confusion matrix PNG in results/bert_confusion_matrix.png
  - Per-epoch training curve PNG in results/bert_training_curve.png

Usage:
    python train_bert.py
    python train_bert.py --epochs 5 --batch_size 16 --lr 2e-5
    python train_bert.py --no_cuda
"""

import argparse
import json
import logging
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import config

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "train_bert.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class AbstractDataset(Dataset):
    """PyTorch Dataset for tokenized PubMed abstracts."""

    def __init__(
        self,
        texts:     list,
        labels:    list,
        tokenizer,
        max_length: int = config.MAX_LENGTH,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, dataloader, device) -> dict:
    """
    Run inference on a DataLoader and return all metrics.

    Returns:
        Dict with accuracy, macro_f1, per-class metrics, preds, labels.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            labels         = batch["labels"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids.to(device)

            outputs = model(**kwargs)
            preds   = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        target_names=config.LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "report":    report,
        "preds":     all_preds,
        "labels":    all_labels,
    }


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    save_path: str,
    model_name: str = "BERT",
) -> None:
    """Plot and save a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Normalized (row %)"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(config.LABEL_NAMES)))
        ax.set_yticks(range(len(config.LABEL_NAMES)))
        ax.set_xticklabels(config.LABEL_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(config.LABEL_NAMES)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)
        for i in range(len(config.LABEL_NAMES)):
            for j in range(len(config.LABEL_NAMES)):
                val = data[i, j]
                text = f"{val:{fmt}}"
                color = "white" if val > data.max() * 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────
# TRAINING CURVE PLOT
# ─────────────────────────────────────────────
def plot_training_curve(history: dict, save_path: str, model_name: str = "BERT") -> None:
    """Plot and save train loss + val macro F1 over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} Training Curves", fontsize=13, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_macro_f1"], "o-", color="orange", label="Val Macro F1")
    ax2.set_title("Validation Macro F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curve saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(
    model_name:   str   = config.BERT_MODEL_NAME,
    save_dir:     str   = config.BERT_SAVE_DIR,
    metrics_file: str   = config.BERT_METRICS_FILE,
    cm_file:      str   = config.BERT_CONFUSION_MATRIX,
    epochs:       int   = config.NUM_EPOCHS,
    batch_size:   int   = config.BATCH_SIZE,
    lr:           float = config.LEARNING_RATE,
    max_length:   int   = config.MAX_LENGTH,
    weight_decay: float = config.WEIGHT_DECAY,
    warmup_ratio: float = config.WARMUP_RATIO,
    seed:         int   = config.RANDOM_SEED,
    use_cuda:     bool  = True,
) -> dict:
    """
    Fine-tune a BERT-type model for 4-class abstract classification.

    Args:
        model_name:   Hugging Face model identifier.
        save_dir:     Directory to save the best checkpoint.
        metrics_file: JSON file to save evaluation metrics.
        cm_file:      PNG file to save the confusion matrix.
        (all other args: standard training hyperparameters)

    Returns:
        Dict of test-set metrics.
    """
    set_seed(seed)
    display_name = os.path.basename(model_name.rstrip("/"))
    logger.info(f"{'='*60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"Epochs={epochs} | BatchSize={batch_size} | LR={lr} | MaxLen={max_length}")
    logger.info(f"{'='*60}")

    # ── Device ───────────────────────────────────
    device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Load data ────────────────────────────────
    train_df = pd.read_csv(config.PROCESSED_TRAIN)
    val_df   = pd.read_csv(config.PROCESSED_VAL)
    test_df  = pd.read_csv(config.PROCESSED_TEST)
    
    classes = np.array(sorted(train_df["label_id"].unique()))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label_id"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    logger.info(f"Class weights: {class_weights}")

    #train_texts  = train_df["abstract"].tolist()
    train_texts = (train_df["title"].fillna("") + " " + train_df["abstract"].fillna("")).tolist()
    val_texts   = (val_df["title"].fillna("") + " " + val_df["abstract"].fillna("")).tolist()
    test_texts  = (test_df["title"].fillna("") + " " + test_df["abstract"].fillna("")).tolist()
    train_labels = train_df["label_id"].tolist()
    #val_texts    = val_df["abstract"].tolist()
    val_labels   = val_df["label_id"].tolist()
    #0test_texts   = test_df["abstract"].tolist()
    test_labels  = test_df["label_id"].tolist()

    logger.info(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # ── Tokenizer ────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded.")

    train_dataset = AbstractDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset   = AbstractDataset(val_texts,   val_labels,   tokenizer, max_length)
    test_dataset  = AbstractDataset(test_texts,  test_labels,  tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ── Model ────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    model.to(device)
    logger.info("Model loaded.")

    # ── Weighted loss for imbalanced classes ────────────────────
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # ── Optimizer & Scheduler ────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps   = len(train_loader) * epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ────────────────────────────
    best_val_f1    = -1.0
    best_epoch     = 0
    history        = {"train_loss": [], "val_macro_f1": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids.to(device)

            outputs = model(**model_inputs)

            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (step + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        val_res  = evaluate(model, val_loader, device)

        history["train_loss"].append(avg_loss)
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["val_accuracy"].append(val_res["accuracy"])

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} "
            f"| Val Acc: {val_res['accuracy']:.4f} | Val Macro F1: {val_res['macro_f1']:.4f}"
        )

        # Save best model
        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            best_epoch  = epoch
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info(f"  ✓ New best model saved (Val Macro F1 = {best_val_f1:.4f}) → {save_dir}")

    logger.info(f"Training complete. Best epoch: {best_epoch} | Best Val Macro F1: {best_val_f1:.4f}")

    # ── Test evaluation ──────────────────────────
    logger.info("Loading best model for test evaluation…")
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    model.to(device)
    test_res = evaluate(model, test_loader, device)

    logger.info(f"TEST RESULTS — {display_name}")
    logger.info(f"  Accuracy : {test_res['accuracy']:.4f}")
    logger.info(f"  Macro F1 : {test_res['macro_f1']:.4f}")
    logger.info("\nPer-class report:")
    for label in config.LABEL_NAMES:
        r = test_res["report"][label]
        logger.info(
            f"  {label:<15} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}"
        )

    # ── Save metrics ─────────────────────────────
    metrics = {
        "model":          model_name,
        "best_epoch":     best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy":  test_res["accuracy"],
        "test_macro_f1":  test_res["macro_f1"],
        "test_macro_precision": precision_score(
            test_res["labels"], test_res["preds"], average="macro", zero_division=0
        ),
        "test_macro_recall": recall_score(
            test_res["labels"], test_res["preds"], average="macro", zero_division=0
        ),
        "per_class": {
            label: test_res["report"][label]
            for label in config.LABEL_NAMES
        },
        "training_history": history,
    }
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_file}")

    # ── Plots ─────────────────────────────────────
    plot_confusion_matrix(
        test_res["labels"], test_res["preds"],
        save_path=cm_file,
        model_name=display_name,
    )
    curve_path = cm_file.replace("confusion_matrix", "training_curve")
    plot_training_curve(history, save_path=curve_path, model_name=display_name)

    return metrics


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for abstract classification.")
    parser.add_argument("--model_name",  type=str, default=config.BERT_MODEL_NAME)
    parser.add_argument("--save_dir",    type=str, default=config.BERT_SAVE_DIR)
    parser.add_argument("--epochs",      type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max_length",  type=int,   default=config.MAX_LENGTH)
    parser.add_argument("--seed",        type=int,   default=config.RANDOM_SEED)
    parser.add_argument("--no_cuda",     action="store_true", help="Disable CUDA.")
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        model_name=args.model_name,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        use_cuda=not args.no_cuda,
    )


if __name__ == "__main__":
    main()
