#!/bin/bash
#SBATCH --job-name=lgbm_top20
#SBATCH --partition=gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:nvidia_geforce_gtx_1080:1
#SBATCH --output=logs/lgbm_top20_%j.out
#SBATCH --error=logs/lgbm_top20_%j.err

cd /users/s/a/sakshay/js_kaggle
mkdir -p logs models

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy pandas scipy scikit-learn pyarrow lightgbm tqdm joblib psutil

python scripts/lgbm_top20.py
