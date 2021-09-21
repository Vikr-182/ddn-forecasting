import pickle
import pandas as pd
import math
import argparse
import numpy as np
from typing import Dict, List, Tuple
# from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt

def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde


def get_displacement_errors_and_miss_rate(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
) -> Tuple[float, float, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate

    Returns:
        mean_min_ade: Min ade averaged over all the samples
        mean_min_fde: Min fde averaged over all the samples
        miss_rate: Mean number of misses

    """
    min_ade = []
    min_fde = []
    n_misses = []
    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        for j in range(0, min(max_guesses, len(forecasted_trajectories[k]))):
            fde = get_fde(forecasted_trajectories[k][j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                curr_min_fde = fde
                curr_min_ade = get_ade(forecasted_trajectories[k][j][:horizon], v[:horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)
    mean_min_ade = sum(min_ade) / len(min_ade)
    mean_min_fde = sum(min_fde) / len(min_fde)
    miss_rate = sum(n_misses) / len(n_misses)
    return mean_min_ade, mean_min_fde, miss_rate

def visualize_predictions(pred: np.ndarray, gt: np.ndarray, obs: np.ndarray, ind: str):
    """

    pred:   Predicted trajectory, shape: (pred_len x 2)
    gt:     Ground truth trajectory, shape: (pred_len x 2)
    obs:    Observed trajectory, shape: (obs_len x 2)
    """
    print(pred.shape)
    print(gt.shape)
    print(obs.shape)


    # plt.plot([i[0] for i in pred],[i[1] for i in pred],'r')
    # plt.plot([i[0] for i in gt],[i[1] for i in gt],'g')

    color_obs = "#ECA154"
    color_pred = "blue"
    color_gt = "#d33e4c"

    plt.plot(
        obs[ :, 0],
        obs[ :, 1],
        color=color_obs,
        label="Observed",
        alpha=1,
        linewidth=3,
        zorder=15,
    )
    plt.plot(
        obs[ -1, 0],
        obs[ -1, 1],
        "o",
        color=color_obs,
        alpha=1,
        linewidth=3,
        zorder=15,
        markersize=9,
    )
    plt.plot(
        pred[ :, 0],
        pred[ :, 1],
        color=color_pred,
        label="Predicted",
        alpha=1,
        linewidth=3,
        zorder=15,
    )
    plt.plot(
        pred[ -1, 0],
        pred[ -1, 1],
        "o",
        color=color_pred,
        alpha=1,
        linewidth=3,
        zorder=15,
        markersize=9,
    )    
    plt.plot(
        gt[ :, 0],
        gt[ :, 1],
        color=color_gt,
        label="Target",
        alpha=1,
        linewidth=3,
        zorder=20,
    )
    plt.plot(
        gt[ -1, 0],
        gt[ -1, 1],
        "o",
        color=color_gt,
        alpha=1,
        linewidth=3,
        zorder=20,
        markersize=9,
    )
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.savefig(ind);
    plt.show()

    return None
