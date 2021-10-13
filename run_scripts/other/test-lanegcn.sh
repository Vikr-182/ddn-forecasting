#!/bin/zsh
#SBATCH --job-name=test-lanegcn
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=15
#SBATCH -G 3 -c 20
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

#module load cuda/11.0
#module load cudnn/7-cuda-11.0

DATASET_FILE1="/share1/vikrant.dewangan/dataset/argoverse/forecasting_train_v1.1.tar.gz"
DATASET_FILE2="/share1/vikrant.dewangan/dataset/argoverse/forecasting_val_v1.1.tar.gz"
DATASET_FILE3="/share1/vikrant.dewangan/dataset/argoverse/forecasting_test_v1.1.tar.gz"
CHECKPOINT1="/share1/vikrant.dewangan/dataset/argoverse/val_crs_dist6_angle90.p"
CHECKPOINT10="/share1/vikrant.dewangan/dataset/argoverse/train_crs_dist6_angle90.p"
CHECKPOINT30="/share1/vikrant.dewangan/dataset/argoverse/test_test.p"
CHECKPOINT_ACT="/share1/vikrant.dewangan/checkpts/LaneGCN/36.000.ckpt"
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
    scp ada:$CHECKPOINT_ACT $DATASET_DST
    cd $DATASET_DST
    wget http://yun.sfo2.digitaloceanspaces.com/public/lanegcn/36.000.ckpt
    tar xvf forecasting_train_v1.1.tar.gz
    tar xvf forecasting_val_v1.1.tar.gz
    tar xvf forecasting_test_v1.1.tar.gz
}

echo "Copying"
#[ -d $DATASET_DST  ] || setup
rm -rf /scratch/forecasting
setup

echo "Done copying"
#conda activate forecasting
echo "Done conda"
echo "LALA" >> lala.txt
cd ~/forecasting/LaneGCN
#python lstm_train_test.py --train_features /scratch/forecasting/forecasting_features_train.pkl --val_features /scratch/forecasting/forecasting_features_val.pkl --use_map --end_epoch=5000
python test.py -m lanegcn --weight=/scratch/forecasting/36.000.ckpt --split=test
echo "Done Training"

