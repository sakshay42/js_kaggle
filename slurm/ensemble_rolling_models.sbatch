#!/bin/bash
#SBATCH --job-name=roll_ens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/roll_ens_%j.out
#SBATCH --error=logs/roll_ens_%j.err

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs models

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy pandas pyarrow xgboost scikit-learn tqdm joblib psutil

python scripts/ensemble_rolling_models.py
