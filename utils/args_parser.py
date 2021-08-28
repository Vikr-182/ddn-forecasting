import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size",
            type=int,
            default=20,
            help="Train batch size")
    parser.add_argument("--model_path",
            required=False,
            type=str,
            help="path to the saved model")
    parser.add_argument("--obs_len",
            default=20,
            type=int,
            help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
            default=30,
            type=int,
            help="Prediction Horizon")
    parser.add_argument(
            "--train_dir",
            default="/datasets/argoverse/val_data.npy",
            type=str,
            help="path to the file which has train data.",
            )
    parser.add_argument(
            "--train_centerlines_dir",
            default="/datasets/argoverse/val_centerlines.npy",
            type=str,
            help="path to the file which has train centerlines.",
            )
    parser.add_argument(
            "--val_offsets_dir",
            default="/datasets/argoverse/val_offsets.npy",
            type=str,
            help="path to the file which has val offsets.",
            )
    parser.add_argument(
            "--val_dir",
            default="/datasets/argoverse/val_data.npy",
            type=str,
            help="path to the file which has val data.",
            )
    parser.add_argument(
            "--val_centerlines_dir",
            default="/datasets/argoverse/val_centerlines.npy",
            type=str,
            help="path to the file which has val centerlines.",
            )
    parser.add_argument(
            "--test_dir",
            default="/datasets/argoverse/val_test_data.npy",
            type=str,
            help="path to the file which has test data.",
            )
    parser.add_argument(
            "--test_centerlines_dir",
            default="/datasets/argoverse/val_test_centerlines.npy",
            type=str,
            help="path to the file which has test centerlines.",
            )
    parser.add_argument(
            "--joblib_batch_size",
            default=100,
            type=int,
            help="Batch size for parallel computation",
            )
    parser.add_argument("--shuffle",
            action="store_true",
            help="Shuffle data")
    parser.add_argument("--flatten",
            action="store_true",
            help="Flatten data")
    parser.add_argument("--include_centerline",
            action="store_true",
            help="Include centerline")
    parser.add_argument("--test",
            action="store_false",
            help="If true, only run the inference")
    parser.add_argument("--num",
            type=int,
            default=30,
            help="Num")
    parser.add_argument("--val_batch_size",
            type=int,
            default=20,
            help="Val batch size")
    parser.add_argument("--end_epoch",
            type=int,
            default=10,
            help="Last epoch")
    parser.add_argument("--num_waypoints",
            type=int,
            default=2,
            help="Number of waypoints")
    parser.add_argument("--lr",
            type=float,
            default=0.0005,
            help="Learning rate")
    parser.add_argument("--num_elems",
            type=float,
            default=15,
            help="Number of points in centerline")
    parser.add_argument(
            "--traj_save_path",
            default="./results/",
            type=str,
            help=
            "path to the pickle file where forecasted trajectories will be saved.",
            )
    return parser.parse_args()
