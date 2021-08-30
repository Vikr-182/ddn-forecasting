#DDN Forecasting

## Installation
```
git clone https://github.com/Vikr-182/ddn-forecasting
```

## Dataset
Find the top 1000sq in `data/` folder.

## Training
```
python3 script_train.py --train_dir <path/to/train/dataset> --test_dir <path/to/val/dir> --lr 0.0005 --traj_save_path <path/to/saved/results>
```

## Test
```
python3 script_train.py --test --train_dir <path/to/train/dataset> --test_dir <path/to/test/dir> --traj_save_path <path/to/saved/results>
```
