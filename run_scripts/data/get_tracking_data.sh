#!/bin/zsh
#SBATCH --job-name=track-dataset
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=1
#SBATCH -G 0 -c 1
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

mkdir /scratch/forecasting/

cd /scratch/forecasting

wget https://s3.amazonaws.com/argoai-argoverse/tracking_sample_v1.1.tar.gz
