#!/bin/bash
#SBATCH --job-name=validate_p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/validate_p8_%j.out
#SBATCH --error=logs/validate_p8_%j.err

cd /users/s/a/sakshay/js_kaggle
mkdir -p logs

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet pandas pyarrow

python scripts/validate_p8.py
