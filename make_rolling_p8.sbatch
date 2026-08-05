#!/bin/bash
#SBATCH --job-name=make_roll
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/make_roll_%j.out
#SBATCH --error=logs/make_roll_%j.err

cd /users/s/a/sakshay/js_kaggle
mkdir -p logs

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet pandas pyarrow

python scripts/make_rolling_p8.py
