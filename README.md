#DDN Forecasting

## Installation
```
git clone https://github.com/Vikr-182/ddn-forecasting
```

## Dataset
Find the top 1000sq in `data/` folder.

## Training
```
python3 script_train.py --train_dir <path/to/train/dataset> --test_dir <path/to/val/dir> --lr 0.0005 --traj_save_path <path/to/saved/results> --end_epoch 10
```
If using the MLP model, make sure to flatten using the `--flatten` option -
```
python3 script_train.py --train_dir <path/to/train/dataset> --test_dir <path/to/val/dir> --lr 0.0005 --traj_save_path <path/to/saved/results> --flatten
```

## Test
```
python3 script_train.py --test --train_dir <path/to/train/dataset> --test_dir <path/to/test/dir> --traj_save_path <path/to/saved/results>
```
