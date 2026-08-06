#!/bin/bash
#SBATCH --job-name=roll_lgbm
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --output=logs/roll_lgbm_%j.out
#SBATCH --error=logs/roll_lgbm_%j.err

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs models

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy pandas scipy scikit-learn pyarrow lightgbm tqdm joblib psutil

python scripts/train_rolling_lgbm.py
