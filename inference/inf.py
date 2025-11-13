#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a fine-tuned BEiT or ViT model on a test dataset.
Computes Accuracy and F1-score (macro).
"""

import os
import json
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer
import evaluate
from sklearn.metrics import f1_score

# -------------------- Configuration --------------------
# model_dir_inf = "prithivMLmods/Food-101-93M"  
# model_dir_inf = "prithivMLmods/Food-101-93M"  
# model_dir_inf = "prithivMLmods/Food-101-93M"  
model_dir_inf = "prithivMLmods/Food-101-93M"  
test_dir = "dataset/merged_dataset/test"  # Path to your test folder
batch_size = 16

# -------------------- Load Model and Processor --------------------
print("📦 Loading model and processor...")
model = AutoModelForImageClassification.from_pretrained(model_dir_inf)
processor = AutoImageProcessor.from_pretrained(model_dir_inf)

# -------------------- Load Test Dataset --------------------
print("🧩 Loading test dataset...")
ds_test = load_dataset("imagefolder", data_dir=test_dir, split="train")
print(f"✅ Loaded {len(ds_test)} test samples.")
print("Classes:", ds_test.features["label"].names[:10], "...")

# -------------------- Preprocessing Function --------------------
def transform_test(examples):
    """Applies preprocessing on test images."""
    images = [img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img in examples["image"]]
    inputs = processor(images=images, return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"], "labels": examples["label"]}

# Apply transform lazily
ds_test.set_transform(transform_test)

# -------------------- Evaluation Setup --------------------
print("🧮 Preparing evaluation metrics...")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Compute accuracy and F1 macro."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}

# -------------------- Initialize Trainer --------------------
trainer = Trainer(
    model=model,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=None,
)

# -------------------- Run Evaluation --------------------
print("🚀 Running evaluation on test set...")
results = trainer.evaluate(eval_dataset=ds_test, metric_key_prefix="test")
print("\n📊 Evaluation Results:")
print(json.dumps(results, indent=4))

# -------------------- Save Results --------------------
os.makedirs(model_dir_inf, exist_ok=True)
with open(os.path.join(f"results/{model_dir_inf}", "test_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)
print("✅ Results saved to:", os.path.join(model_dir_inf, "test_metrics.json"))
