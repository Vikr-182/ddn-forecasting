#!/bin/zsh
#SBATCH --job-name=prp-WIMP
#SBATCH --time=3-23:00:00
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
#
mkdir data
mkdir prep
cd data
scp ada:/share1/vikrant.dewangan/dataset/argoverse/forecasting_train_v1.1.tar.gz ./
scp ada:/share1/vikrant.dewangan/dataset/argoverse/forecasting_val_v1.1.tar.gz ./
scp ada:/share1/vikrant.dewangan/dataset/argoverse/forecasting_test_v1.1.tar.gz ./

tar xvf forecasting_train_v1.1.tar.gz
tar xvf forecasting_val_v1.1.tar.gz
tar xvf forecasting_test_v1.1.tar.gz

mv test_obs test

mv train/data/* train/
mv test/data/* test/
mv val/data/* val/

rm -rf train/data train/data val/data train/Argoverse-Terms_of_Use.txt test/Argoverse-Terms_of_Use.txt val/Argoverse-Terms_of_Use.txt 

cd /home2/vikrant.dewangan/forecasting/WIMP/
python scripts/run_preprocess.py --dataroot /scratch/forecasting/data \
--mode val --save-dir /scratch/forecasting/prep --social-features \
--map-features --xy-features --normalize --extra-map-features \
--compute-all --generate-candidate-centerlines 4

