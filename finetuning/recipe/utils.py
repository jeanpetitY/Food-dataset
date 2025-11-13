#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune BEiT or ViT on a balanced image classification dataset.
Each subfolder under /train and /val represents a class.
Author: 
"""

import argparse
import json
import os
from dataclasses import dataclass
import time
from typing import Any, Dict

import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
import evaluate
from sklearn.metrics import f1_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from datasets import Image as DatasetImage


# -------------------------- Utility Functions --------------------------
def seed_everything(seed: int = 42):
    """Ensure deterministic behavior for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def infer_splits(data_dir: str):
    """Return paths to train and validation directories if they exist."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    return (train_dir, val_dir) if os.path.isdir(val_dir) else (train_dir, None)

def load_imagefolder_with_progress(data_dir: str, split: str = "train"):
    """Custom loader with tqdm progress bar."""
    print(f"Loading dataset from {data_dir} ...")

    # Count total image files
    total_files = sum(
        len(files)
        for _, _, files in os.walk(data_dir)
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
    )

    # Display progress while building the dataset
    dataset = load_dataset("imagefolder", data_dir=data_dir, split=split)
    print("Dataset metadata loaded, verifying samples...")

    for _ in tqdm(range(total_files), desc=f"Scanning {split}", ncols=80):
        pass  # Just to simulate progress bar — loading is done internally

    print(f"✅ Finished loading {split} dataset ({len(dataset)} samples)")
    return dataset


# -------------------------- Metrics --------------------------
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 score for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


# -------------------------- Main Training Function --------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BEiT or ViT on a folder dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing train/ and optionally val/")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224",
                        help="Model name (e.g., google/vit-base-patch16-224 or microsoft/beit-base-patch16-224-pt22k-ft22k)")
    parser.add_argument("--output_dir", type=str, default="./model_saved")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Used if no val folder is provided")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision (AMP) if supported")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze all layers except the classification head")
    args = parser.parse_args()

    # Set random seed
    seed_everything(args.seed)

    # Identify dataset splits
    train_dir, val_dir = infer_splits(args.data_dir)
    print(train_dir, val_dir)

    # Load dataset from image folders
    print("=======Load Dataset=============================")
    if val_dir is not None:
        ds = DatasetDict({
            "train": load_imagefolder_with_progress(train_dir, "train"),
            "validation": load_imagefolder_with_progress(val_dir, "train"),
        })
    else:
        full = load_imagefolder_with_progress(train_dir, "train")
        ds_split = full.train_test_split(test_size=args.val_ratio, seed=args.seed)
        ds = DatasetDict({"train": ds_split["train"], "validation": ds_split["test"]})
    
    

    ds["train"] = ds["train"].cast_column("image", DatasetImage(decode=True))
    ds["validation"] = ds["validation"].cast_column("image", DatasetImage(decode=True))
    
    print("Dataset columns:", ds["train"].column_names)



    # Extract label mapping
    labels = ds["train"].features["label"].names
    id2label = dict(enumerate(labels))
    label2id = {l: i for i, l in enumerate(labels)}
    
    # Automatically pick the correct image size if not specified
    if "beit" in args.model_name:
        default_size = 384 if "384" in args.model_name else 224
    elif "vit" in args.model_name:
        default_size = 224
    else:
        default_size = args.image_size  # fallback

    # Load processor (resize, normalize, etc.)
    print("=========== Load Processor====================")
    if "beit" in args.model_name:
        processor = AutoImageProcessor.from_pretrained(
            args.model_name,
            size={"height": default_size, "width": default_size}
        )
    else:
        processor = AutoImageProcessor.from_pretrained(
            args.model_name,
            size={"shortest_edge": default_size}
    )


    # Apply transforms (no augmentation)
    print("===============Transform dataset===================")
    print(f"Train sample keys before transform: {ds['train'][0].keys()}")
    def apply_image_transform(examples):
        images = [
            x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x).convert("RGB")
            for x in examples["image"]
        ]
        inputs = processor(images=images, return_tensors="pt")
        examples["pixel_values"] = inputs["pixel_values"]
        return {"pixel_values": inputs["pixel_values"], "labels": examples["label"]}

    # Apply transformation once to all splits
    ds["train"] = ds["train"].map(apply_image_transform, batched=True)
    ds["validation"] = ds["validation"].map(apply_image_transform, batched=True)

    # Keep only necessary columns
    ds["train"].set_format(type="torch", columns=["pixel_values", "labels"])
    ds["validation"].set_format(type="torch", columns=["pixel_values", "labels"])
    


    # Load pre-trained model
    print(f"============Load model: {args.model_name} =============")
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Optionally freeze the backbone
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name or "head" in name
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation,
        report_to="none",
        dataloader_num_workers=0,
    )

    # Data collator (for batching)
    collator = DefaultDataCollator()
    print("Sample keys:", ds["train"][0].keys())


    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=processor,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("========= Start training===========")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time
    print("=========End training==============")
    print(f"🕒 Total training time: {training_time / 60:.2f} minutes ({training_time / 3600:.2f} hours)")

    # Save model and processor
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=4, ensure_ascii=False)

    # Evaluate final performance
    metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training completed successfully.")
    print("📊 Final evaluation metrics:", metrics)

model_beit = "microsoft/beit-base-patch16-384"
model_vit = "google/vit-base-patch16-224"

if __name__ == "__main__":
    main()
