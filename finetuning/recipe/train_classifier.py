#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune BEiT or ViT on a large image classification dataset using torchvision transforms.
Adapted from Kaggle-style fine-tuning for high efficiency.
Author: Jean Petit BIKIM
"""

import argparse
import json
import os
import time
import random
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_dataset, DatasetDict
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.metrics import f1_score

# -------------------------- Utility Functions --------------------------
def seed_everything(seed: int = 42):
    """Ensure deterministic behavior for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_splits(data_dir: str):
    """Return paths to train and validation directories if they exist."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    return (train_dir, val_dir) if os.path.isdir(val_dir) else (train_dir, None)


# -------------------------- Metrics --------------------------
def compute_metrics(eval_pred):
    """Compute multiple metrics for classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


# -------------------------- Main Training Function --------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BEiT or ViT using torchvision transforms")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing train/ and optionally val/")
    parser.add_argument("--model_name", type=str, default="microsoft/beit-base-patch16-384")
    parser.add_argument("--output_dir", type=str, default="./model_saved")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./cached_dataset", help="Where to store processed dataset")

    args = parser.parse_args()

    seed_everything(args.seed)

    # Load dataset
    print("======= Loading dataset =======")
    train_dir, val_dir = infer_splits(args.data_dir)
    if val_dir:
        ds = DatasetDict({
            "train": load_dataset("imagefolder", data_dir=train_dir, split="train"),
            "validation": load_dataset("imagefolder", data_dir=val_dir, split="train"),
        })
    else:
        full = load_dataset("imagefolder", data_dir=train_dir, split="train")
        ds_split = full.train_test_split(test_size=args.val_ratio, seed=args.seed)
        ds = DatasetDict({"train": ds_split["train"], "validation": ds_split["test"]})

    print(f"✅ Dataset loaded: {len(ds['train'])} train / {len(ds['validation'])} val")

    # Extract labels
    labels = ds["train"].features["label"].names
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}

    # ----------------- Processor & torchvision transforms -----------------
    feature_extractor = AutoImageProcessor.from_pretrained(args.model_name)
    image_size = feature_extractor.size["height"] if isinstance(feature_extractor.size, dict) else feature_extractor.size

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose([
        RandomResizedCrop(image_size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    val_transforms = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        normalize,
    ])

    def preprocess_train(batch):
        batch["pixel_values"] = [train_transforms(img.convert("RGB")) for img in batch["image"]]
        return batch

    def preprocess_val(batch):
        batch["pixel_values"] = [val_transforms(img.convert("RGB")) for img in batch["image"]]
        return batch

    from datasets import load_from_disk

    # Check if cached dataset exists
    if os.path.exists(args.cache_dir):
        print(f"Loading preprocessed dataset from {args.cache_dir} ...")
        ds = load_from_disk(args.cache_dir)
    else:
        print("Applying image transforms ...")
        ds["train"] = ds["train"].map(preprocess_train, batched=True, num_proc=4)
        ds["validation"] = ds["validation"].map(preprocess_val, batched=True, num_proc=4)
        ds["train"].set_format(type="torch", columns=["pixel_values", "label"])
        ds["validation"].set_format(type="torch", columns=["pixel_values", "label"])

        print(f"Saving preprocessed dataset to {args.cache_dir}")
        ds.save_to_disk(args.cache_dir)

    # ----------------- Model -----------------
    print(f"🚀 Loading model: {args.model_name}")
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ----------------- Collate Function -----------------
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # ----------------- Training Arguments -----------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=50,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=None,
    )

    # ----------------- Trainer -----------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # ----------------- Training -----------------
    print("🧩 Starting training ...")
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    print(f"✅ Training completed in {duration / 3600:.2f} hours")

    # Save model
    trainer.save_model()
    trainer.save_metrics("train", trainer.state.log_history[-1])
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    print("📊 Final Evaluation:", metrics)


if __name__ == "__main__":
    main()
