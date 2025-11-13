#!/bin/zsh

#SBATCH --job-name=qwen
#SBATCH --output=loutput_qween.log
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:a3090:1
#SBATCH --mem=128G
# SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --time=48:00:00


# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

# python finetuning/recipe/train_classifier.py \
#   --model_name google/vit-base-patch16-224 \
#   --data_dir dataset/merged_dataset \
#   --output_dir ./model_saved/vit-food-224 \
#   --epochs 1 \
#   --batch_size 32 \
#   --weight_decay 0.01 \
#   --fp16 \
#   --warmup_ratio 0.05 \

python finetuning/recipe/vlm.py  \
  --data_path dataset/multimodal/merged/train_final.json \
  --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir ./model_saved/qwen-food-vl \
  --epochs 1 \
  --batch_size 32 \
  --lr 1e-5 \
  --fp16 \
  --gradient_accumulation 4