#!/bin/bash
#SBATCH --job-name=clean_p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/clean_p8_%j.out
#SBATCH --error=logs/clean_p8_%j.err

cd /users/s/a/sakshay/js_kaggle
mkdir -p logs

module load python
python -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy pandas pyarrow

python scripts/clean_p8.py
