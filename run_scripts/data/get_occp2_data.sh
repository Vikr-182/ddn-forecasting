#!/bin/zsh
#SBATCH --job-name=2occp
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=2
#SBATCH -G 0 -c 1
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

mkdir /scratch/occp

cd /scratch/occp

gdown --id 1QqLPpDRf6atvpyHQP_6xvJx1ayOkNjjO
