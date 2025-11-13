#!/bin/zsh

#SBATCH --job-name=import_usda_data
#SBATCH --output=output_beit_falcon_inf_except_merged.log
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:a3090:1
#SBATCH --ntasks=1

# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

python inference/rag.py