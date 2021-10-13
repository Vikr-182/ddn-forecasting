#!/bin/zsh
#SBATCH --job-name=test-lstm
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=1
#SBATCH -G 0 -c 1
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

#module load cuda/11.0
#module load cudnn/7-cuda-11.0

DATASET_FILE1="/share1/vikrant.dewangan/dataset/argoverse/forecasting_features_train.pkl"
DATASET_FILE2="/share1/vikrant.dewangan/dataset/argoverse/forecasting_features_val_top1000.pkl"
DATASET_FILE3="/share1/vikrant.dewangan/dataset/argoverse/forecasting_features_test.pkl"
CHECKPOINT1="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map_old/LSTM_rollout1.pth.tar"
CHECKPOINT10="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map_old/LSTM_rollout10.pth.tar"
CHECKPOINT30="/share1/vikrant.dewangan/checkpts/LSTM/saved_models/lstm_map/LSTM_rollout30.pth.tar"
DATASET_DST="/scratch/forecasting"

function setup {
    mkdir -p $DATASET_DST
    echo "Inside setup"

    scp ada:$DATASET_FILE1 $DATASET_DST
    scp ada:$DATASET_FILE2 $DATASET_DST
    scp ada:$DATASET_FILE3 $DATASET_DST
    scp ada:$CHECKPOINT1 $DATASET_DST
    scp ada:$CHECKPOINT10 $DATASET_DST
    scp ada:$CHECKPOINT30 $DATASET_DST
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
cd ~/forecasting/argoverse-forecasting
#python lstm_train_test.py --train_features /scratch/forecasting/forecasting_features_train.pkl --val_features /scratch/forecasting/forecasting_features_val.pkl --use_map --end_epoch=5000
python3 lstm_train_test.py --traj_save_path ~/file_val.pkl --test --test_features /scratch/forecasting/forecasting_features_val_top1000.pkl --model_path /scratch/forecasting/LSTM_rollout30.pth.tar
echo "Done Training"

