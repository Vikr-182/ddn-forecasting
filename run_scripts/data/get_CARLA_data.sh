#!/bin/zsh
#SBATCH --job-name=dataset
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=2
#SBATCH -G 0 -c 1
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

mkdir /scratch/forecasting/

cd /scratch/forecasting

gdown --id 1dwt9_EvXB1a6ihlMVMyYx0Bw0mN27SLy

scp CARLA_challenge_autopilot.tar.gz ada:/share1/vikrant.dewangan/dataset/CARLA/

