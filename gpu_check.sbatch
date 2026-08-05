#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --partition=gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:nvidia_geforce_gtx_1080:1
#SBATCH --output=logs/gpu_check_%j.out
#SBATCH --error=logs/gpu_check_%j.err

cd /users/s/a/sakshay/js_kaggle
mkdir -p logs

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install --quiet numpy pandas scipy scikit-learn pyarrow lightgbm xgboost tqdm joblib psutil optuna torchmetrics pytorch-lightning einops

python scripts/gpu_check.py
