"""
Fine-tune BERT on SST-2 (binary sentiment) via HuggingFace Trainer.

Usage:
    python -m src.train
    python -m src.train --fp16 --batch_size 64 --epochs 3
"""

import argparse
import logging
import os
import random
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
import evaluate as hf_evaluate

from src.config import TrainingConfig
from src.model import BertSentimentClassifier, get_tokenizer, export_onnx
from src.dataset import load_sst2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_acc_metric = hf_evaluate.load("accuracy")
_f1_metric = hf_evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = _acc_metric.compute(predictions=preds, references=labels)
    f1 = _f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {**acc, **f1}


def train(cfg: TrainingConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    tokenizer = get_tokenizer(cfg.model_name)
    train_ds, val_ds = load_sst2(tokenizer, cfg.max_length)

    model = BertSentimentClassifier(cfg.model_name, cfg.num_labels)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=int(cfg.warmup_ratio * (len(train_ds) // cfg.batch_size) * cfg.num_epochs),
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model.bert,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training...")
    trainer.train()

    best_path = os.path.join(cfg.output_dir, "best_model")
    model.bert.save_pretrained(best_path)
    tokenizer.save_pretrained(best_path)
    logger.info(f"Best model saved to {best_path}")

    eval_results = trainer.evaluate()
    logger.info(f"Final validation results: {eval_results}")

    logger.info("Exporting to ONNX...")
    export_onnx(model, cfg.onnx_path, cfg.max_length, cfg.onnx_opset)

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()

    cfg = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        output_dir=args.output_dir,
    )
    results = train(cfg)
    print(f"\nTraining complete. Results: {results}")
