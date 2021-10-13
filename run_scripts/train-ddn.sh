#!/bin/zsh
#SBATCH --job-name=train-ddn
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=5
#SBATCH --output 2_train_ddn.log
#SBATCH -G 0 -c 5
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

#module load cuda/11.0
#module load cudnn/7-cuda-11.0

DATASET_FILE1="/share1/vikrant.dewangan/dataset/argoverse/val_test_data.npy"
DATASET_FILE2="/share1/vikrant.dewangan/dataset/argoverse/val_data.npy"
DATASET_FILE3="/share1/vikrant.dewangan/dataset/argoverse/val_offsets.npy"
CHECKPOINT1="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map_old/LSTM_rollout1.pth.tar"
CHECKPOINT10="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map_old/LSTM_rollout10.pth.tar"
CHECKPOINT30="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map_old/LSTM_rollout30.pth.tar"
DATASET_DST="/scratch/forecasting"

function setup {
    mkdir -p $DATASET_DST
    echo "Inside setup"

    scp ada:$DATASET_FILE1 $DATASET_DST
    scp ada:$DATASET_FILE2 $DATASET_DST
    scp ada:$DATASET_FILE3 $DATASET_DST
    cd $DATASET_DST
}

echo "Copying"
#[ -d $DATASET_DST  ] || setup
rm -rf /scratch/forecasting
setup

echo "Done copying"
#conda activate forecasting
echo "Done conda"
echo "LALA" >> lala.txt
#cd ~/forecasting/sengming/argo-forecasting-competition
cd ~/forecasting/ddn-forecasting
python3 LSTMv2.py --train_dir /scratch/forecasting/val_data.npy --test_dir /scratch/forecasting/val_data.npy --network LSTMEP --lr 0.0001 --val_offsets_dir /scratch/forecasting/val_offsets.npy --end_epoch 1000 
echo "Done Training"
