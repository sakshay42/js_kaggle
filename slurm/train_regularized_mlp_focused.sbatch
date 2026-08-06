#!/bin/bash
#SBATCH --job-name=mlp_focus
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --output=logs/mlp_focus_%j.out
#SBATCH --error=logs/mlp_focus_%j.err

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs models

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet "numpy<2" pandas pyarrow scikit-learn torch tqdm joblib psutil

python scripts/train_regularized_mlp_focused.py
