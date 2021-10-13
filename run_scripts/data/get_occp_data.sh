#!/bin/zsh
#SBATCH --job-name=occp
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=2
#SBATCH -G 0 -c 1
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

mkdir /scratch/occp

cd /scratch/occp

gdown --id 1U1nG0r8H-5x9M8NviT_SmzdA8-r3Z5_o
