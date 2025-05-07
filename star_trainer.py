"""
star_trainer.py — Class‑based STAR sentence classifier
=====================================================

Train a Hugging Face sequence‑classification model that labels interview sentences as
Situation, Task, Action, or Result (STAR).  The script is fully configurable from the
command line, logs metrics and plots, and saves the best checkpoint for inference.

Major fixes in this version
---------------------------
* **labels key**: Datasets now expose the field name `labels` (plural) so the model
  receives it and can compute the loss.  This solves the *"model did not return a
  loss"* runtime error.
* **Typed configuration** via a `Config` dataclass instead of loose argparse values.
* Clearer logging and error handling throughout.
"""

from __future__ import annotations

import os
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------------------


@dataclass
class Config:
    csv_file: str
    model_name: str = "bert-base-uncased"
    output_dir: str = "star_output"
    epochs: int = 5
    lr: float = 5e-5
    batch_size: int = 16
    max_length: int = 128
    seed: int = 42

    @staticmethod
    def from_args(arg_list: List[str] | None = None) -> "Config":
        parser = argparse.ArgumentParser(
            description="Train a STAR-method sentence classifier"
        )
        parser.add_argument(
            "--csv_file",
            type=str,
            required=True,
            help="Path to CSV with columns: sentence,label",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="bert-base-uncased",
            help="Any HF checkpoint e.g. roberta-base",
        )
        parser.add_argument("--output_dir", type=str, default="star_output")
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_length", type=int, default=128)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(arg_list)
        return Config(**vars(args))


# --------------------------------------------------------------------------------------
# Trainer Class
# --------------------------------------------------------------------------------------


class StarTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        torch.manual_seed(self.cfg.seed)

        # 1) Load and split data
        logger.info("Loading data …")
        df = pd.read_csv(self.cfg.csv_file)
        if not {"sentence", "label"}.issubset(df.columns):
            raise ValueError("CSV must contain 'sentence' and 'label' columns")
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=self.cfg.seed, stratify=df["label"]
        )

        # 2) Encode labels
        self.label_encoder = LabelEncoder().fit(train_df["label"])
        self.label_names = list(self.label_encoder.classes_)
        print(self.label_names)

        # 3) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

        # 4) Build datasets
        self.train_ds = self._to_dataset(train_df)
        self.val_ds = self._to_dataset(val_df)

        # 5) Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=len(self.label_names)
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

        # 6) Hugging Face Trainer
        logger.info("Building Hugging Face Trainer …")
        training_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.lr,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.cfg.seed,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            tokenizer=self.tokenizer,  # FutureWarning until HF v5
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics,
        )

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def _to_dataset(self, df: pd.DataFrame) -> Dataset:
        """Convert a pandas DataFrame to a HF Dataset with tokenization."""
        ds = Dataset.from_pandas(df)

        def tokenize_fn(batch):
            toks = self.tokenizer(
                batch["sentence"],
                truncation=True,
                padding="max_length",
                max_length=self.cfg.max_length,
            )
            # IMPORTANT: field must be **labels** for HF models to produce loss
            toks["labels"] = self.label_encoder.transform(batch["label"])
            return toks

        return ds.map(tokenize_fn, batched=True, remove_columns=["sentence", "label"])

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        mcc = matthews_corrcoef(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mcc": mcc,
            "kappa": kappa,
        }

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------
    def train(self):
        logger.info("Training …")
        self.trainer.train()
        self._plot_training_curves()
        self._final_evaluation()
        self._save_best_model()

    # ------------------------------------------------------------------
    # Post‑training helpers
    # ------------------------------------------------------------------
    def _plot_training_curves(self):
        history = self.trainer.state.log_history
        train_loss = [
            x["loss"] for x in history if "loss" in x and "eval_loss" not in x
        ]
        eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
        eval_acc = [x["eval_accuracy"] for x in history if "eval_accuracy" in x]
        epochs = list(range(1, len(eval_loss) + 1))
        logger.info(f"Train loss {train_loss}")
        logger.info(f"Validation loss {eval_loss}")

        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, eval_loss, label="Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")
        plt.savefig(os.path.join(self.cfg.output_dir, "loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, eval_acc, label="Eval Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.savefig(os.path.join(self.cfg.output_dir, "accuracy_curve.png"))
        plt.close()

    def _final_evaluation(self):
        logger.info("Evaluating best checkpoint …")
        preds_output = self.trainer.predict(self.val_ds)
        y_pred = np.argmax(preds_output.predictions, axis=1)
        y_true = preds_output.label_ids

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=self.label_names,
            yticklabels=self.label_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "confusion_matrix.png"))
        plt.close()

        # Metrics table
        prec, rec, f1, supp = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        df = pd.DataFrame(
            {
                "Class": self.label_names,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Support": supp,
            }
        )
        overall = {
            "Class": "Overall",
            "Precision": np.average(prec, weights=supp),
            "Recall": np.average(rec, weights=supp),
            "F1": np.average(f1, weights=supp),
            "Support": supp.sum(),
        }
        df = df._append(overall, ignore_index=True)
        logger.info("\n" + df.to_markdown(index=False))

    def _save_best_model(self):
        best_path = os.path.join(self.cfg.output_dir, "best_model")
        self.trainer.save_model(best_path)
        self.tokenizer.save_pretrained(best_path)
        logger.info(f"Best model saved to {best_path}")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main():
    cfg = Config.from_args()
    logger.info(f"Configuration: {asdict(cfg)}")
    trainer = StarTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
