#!/bin/bash
#SBATCH --job-name=ens_focus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ens_focus_%j.out
#SBATCH --error=logs/ens_focus_%j.err

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs models

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet "numpy<2" pandas pyarrow xgboost scikit-learn tqdm joblib psutil

python scripts/ensemble_focused_mlp_xgb.py
