#!/bin/zsh
#SBATCH --job-name=train-lstm
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=15
#SBATCH -G 2 -c 20
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

#module load cuda/11.0
#module load cudnn/7-cuda-11.0

DATASET_FILE1="/share1/vikrant.dewangan/dataset/argoverse/npy/"
DATASET_FILE2="/share1/vikrant.dewangan/dataset/argoverse/forecasting_features_val.pkl"
DATASET_FILE3="/share1/vikrant.dewangan/dataset/argoverse/forecasting_features_test.pkl"
CHECKPOINT1="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map/LSTM_rollout1.pth.tar"
CHECKPOINT10="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map/LSTM_rollout10.pth.tar"
CHECKPOINT30="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map/LSTM_rollout30.pth.tar"
DATASET_DST="/scratch/forecasting"

function setup {
    mkdir -p $DATASET_DST
    echo "Inside setup"

    scp -r ada:$DATASET_FILE1 $DATASET_DST
    scp ada:$CHECKPOINT1 $DATASET_DST
    scp ada:$CHECKPOINT10 $DATASET_DST
    scp ada:$CHECKPOINT30 $DATASET_DST
    cd $DATASET_DST
    mv npy/* ./
}

echo "Copying"
#[ -d $DATASET_DST  ] || setup
setup

echo "Done copying"
#conda activate forecasting
echo "Done conda"
echo "LALA" >> lala.txt
cd ~/forecasting/ddn-forecasting
python me_train.py
echo "Done Training"

