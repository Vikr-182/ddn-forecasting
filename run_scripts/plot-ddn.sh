#!/bin/zsh
#SBATCH --job-name=train-ddn
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=5
#SBATCH --output /home2/vikrant.dewangan/_out/2_train_ddn.log
#SBATCH -G 0 -c 5
#SBATCH -w gnode33
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

#module load cuda/11.0
#module load cudnn/7-cuda-11.0

DATASET_DST="/scratch/forecasting"

mkdir -p $DATASET_DST
cd $DATASET_DST
scp ada:/share1/vikrant.dewangan/dataset/argoverse/val_data.npy   ./
#scp ada:/share1/vikrant.dewangan/dataset/argoverse/forecasting_val_v1.1.tar.gz ./

#wget https://s3.amazonaws.com/argoai-argoverse/forecasting_train_v1.1.tar.gz
#wget https://s3.amazonaws.com/argoai-argoverse/forecasting_val_v1.1.tar.gz

#tar -zxf forecasting_train_v1.1.tar.gz
#tar -zxvf forecasting_val_v1.1.tar.gz

echo "Done copying"
#conda activate forecasting
echo "Done conda"
echo "LALA" >> lala.txt
#cd ~/forecasting/sengming/argo-forecasting-competition
cd ~/forecasting/ddn-forecasting
python3 plot_traj.py --test_dir /scratch/forecasting/val_data.npy --network LSTMEP --model_path  checkpoints/final_with.ckpt --test True
echo "Done Training"
