#!/bin/zsh
#SBATCH --job-name=lyft-dataset
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=2
#SBATCH -G 0 -c 1
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

mkdir /scratch/forecasting/

cd /scratch/forecasting
export KAGGLE_CONFIG_DIR=/home2/vikrant.dewangan/

kaggle competitions download -c lyft-motion-prediction-autonomous-vehicles

