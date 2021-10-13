#!/bin/zsh
#SBATCH --job-name=dataset
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=2
#SBATCH -G 0 -c 2
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

mkdir /scratch/forecasting/

cd /scratch/forecasting


wget https://s3.amazonaws.com/argoai-argoverse/rectified_stereo_images_v1.1.tar.gz

wget https://s3.amazonaws.com/argoai-argoverse/disparity_maps_v1.1.tar.gz
