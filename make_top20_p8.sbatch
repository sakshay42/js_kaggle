#!/bin/bash
#SBATCH --job-name=make_top20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/make_top20_%j.out
#SBATCH --error=logs/make_top20_%j.err

cd /users/s/a/sakshay/js_kaggle
mkdir -p logs

module load python
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet pyarrow

python scripts/make_top20_p8.py
