#!/bin/zsh

#SBATCH --job-name=import_usda_data
#SBATCH --output=output.log
#SBATCH --ntasks=1
#SBATCH --partition=p_48G

# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

python import_data.py
