#!/bin/zsh
#SBATCH --job-name=prp-lanegcn
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=2
#SBATCH -G 1 -c 5
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

mkdir /scratch/forecasting/

cd /scratch/forecasting

# step2: download Argoverse Motion Forecasting **v1.1** 
# train + val + test
# wget https://s3.amazonaws.com/argoai-argoverse/forecasting_train_v1.1.tar.gz
# wget https://s3.amazonaws.com/argoai-argoverse/forecasting_val_v1.1.tar.gz
# wget https://s3.amazonaws.com/argoai-argoverse/forecasting_test_v1.1.tar.gz
scp ada:/share1/vikrant.dewangan/forecasting_train_v1.1.tar.gz ./
scp ada:/share1/vikrant.dewangan/forecasting_val_v1.1.tar.gz ./
scp ada:/share1/vikrant.dewangan/forecasting_test_v1.1.tar.gz ./

tar xvf forecasting_train_v1.1.tar.gz
tar xvf forecasting_val_v1.1.tar.gz
tar xvf forecasting_test_v1.1.tar.gz


scp ada:/share1/vikrant.dewangan/train_crs_dist6_angle90.p ./
scp ada:/share1/vikrant.dewangan/val_crs_dist6_angle90.p ./
scp ada:/share1/vikrant.dewangan/test_test.p ./


cd ~/forecasting/LaneGCN
python preprocess_data.py
