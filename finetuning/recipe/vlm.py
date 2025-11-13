#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning a Vision-Language Model (Qwen2.5-VL) with preprocessing & caching similar to BEiT/ViT setup.
Author:
"""

import argparse
import os
import json
import random
import gc
import numpy as np
from PIL import Image
from datasets import load_from_disk, Dataset
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
)
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
import torch


# -------------------------- Utility --------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_food_dataset(file_path):
    """Load dataset JSON into HuggingFace format."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted = []
    for item in data:
        answer = (
            f"This food is **{item['label']}**.\n"
            f"Here are its main nutrients:\n- " + "\n- ".join(item["components"])
        )
        formatted.append({
            "image": item["image"],
            "instruction": item["instruction"],
            "answer": answer
        })
    return Dataset.from_list(formatted)


# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with torchvision-style preprocessing")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./qwen2.5-vl-food")
    parser.add_argument("--cache_dir", type=str, default="./cached_vlm")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)

    args = parser.parse_args()
    seed_everything(args.seed)

    # ---------------- Load model & processor ----------------
    print(f"🚀 Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # ---------------- Preprocessing setup ----------------
    image_size = 350
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transforms = Compose([
        RandomResizedCrop(image_size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    eval_transforms = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        normalize,
    ])

    def preprocess_example(example, transform):
        """Apply image + text preprocessing using torchvision & Qwen processor."""
        image = Image.open(example["image"]).convert("RGB")
        image = transform(image)

        inputs = processor(
            text=example["instruction"],
            images=image,
            return_tensors="pt",
            padding=True
        )

        inputs["labels"] = processor.tokenizer(
            example["answer"],
            return_tensors="pt",
            padding=True
        ).input_ids

        del image
        gc.collect()
        return {k: v.squeeze(0) for k, v in inputs.items()}

    # ---------------- Dataset Loading ----------------
    if os.path.exists(args.cache_dir):
        print(f"📦 Loading cached dataset from {args.cache_dir}")
        dataset = load_from_disk(args.cache_dir)
    else:
        print("📂 Loading and preprocessing dataset...")
        dataset = load_food_dataset(args.data_path)
        dataset = dataset.map(lambda x: preprocess_example(x, train_transforms), num_proc=4)
        print("💾 Saving preprocessed dataset...")
        dataset.save_to_disk(args.cache_dir)

    # Reduce for debug
    if args.max_train_samples:
        dataset = dataset.select(range(args.max_train_samples))

    # ---------------- Training ----------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=args.fp16,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
    )

    print("🏋️ Fine-tuning started ...")
    trainer.train()
    print("✅ Fine-tuning complete!")

    print("💾 Saving model ...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"✅ Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
