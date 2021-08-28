import copy
import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from argoverse.data_loading.stereo_dataloader import ArgoverseStereoDataLoader
#from argoverse.evaluation.stereo.eval import StereoEvaluator
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.camera_stats import RECTIFIED_STEREO_CAMERA_LIST

STEREO_FRONT_LEFT_RECT = RECTIFIED_STEREO_CAMERA_LIST[0]
STEREO_FRONT_RIGHT_RECT = RECTIFIED_STEREO_CAMERA_LIST[1]


# Path to the dataset (please change accordingly).
data_dir = "/media/vikrant/New Volume/dataset/argoverse/stereo"
data_dir = "/media/vikrant/New Volume/dataset/argoverse/stereo" 

#/train/rectified/08a8b7f0-c317-3bdb-b3dc-b7c9b6d033e2/stereo_front_left_rect/"

# Choosing the data split: train, val, or test (note that we do not provide ground truth for the test set).
split_name = "train"

# Choosing a specific log id. For example, 273c1883-673a-36bf-b124-88311b1a80be.
log_id = "08a8b7f0-c317-3bdb-b3dc-b7c9b6d033e2"

# Choosing an index to select a specific stereo image pair. You can always modify this to loop over all data.
idx = 0

# Creating the Argoverse Stereo data loader.
stereo_data_loader = ArgoverseStereoDataLoader(data_dir, split_name)

# Loading the left rectified stereo image paths for the chosen log.
left_stereo_img_fpaths = stereo_data_loader.get_ordered_log_stereo_image_fpaths(
    log_id=log_id,
    camera_name=STEREO_FRONT_LEFT_RECT,
)

# Loading the right rectified stereo image paths for the chosen log.
right_stereo_img_fpaths = stereo_data_loader.get_ordered_log_stereo_image_fpaths(
    log_id=log_id,
    camera_name=STEREO_FRONT_RIGHT_RECT,
)

# Loading the disparity map paths for the chosen log.
disparity_map_fpaths = stereo_data_loader.get_ordered_log_disparity_map_fpaths(
    log_id=log_id,
    disparity_name="stereo_front_left_rect_disparity",
)

# Loading the disparity map paths for foreground objects for the chosen log.
disparity_obj_map_fpaths = stereo_data_loader.get_ordered_log_disparity_map_fpaths(
    log_id=log_id,
    disparity_name="stereo_front_left_rect_disparity",
)

print(left_stereo_img_fpaths )
# Loading the rectified stereo images.
stereo_front_left_rect_image = stereo_data_loader.get_rectified_stereo_image(left_stereo_img_fpaths[idx])
stereo_front_right_rect_image = stereo_data_loader.get_rectified_stereo_image(right_stereo_img_fpaths[idx])

# Loading the ground-truth disparity maps.
stereo_front_left_rect_disparity = stereo_data_loader.get_disparity_map(disparity_map_fpaths[idx])

# Loading the ground-truth disparity maps for foreground objects only.
stereo_front_left_rect_objects_disparity = stereo_data_loader.get_disparity_map(disparity_obj_map_fpaths[idx])

# Dilating the disparity maps for a better visualization.
'''
stereo_front_left_rect_disparity_dil = cv2.dilate(
    stereo_front_left_rect_disparity,
    kernel=np.ones((2, 2), np.uint8),
    iterations=7,
)

stereo_front_left_rect_objects_disparity_dil = cv2.dilate(
    stereo_front_left_rect_objects_disparity,
    kernel=np.ones((2, 2), np.uint8),
    iterations=7,
)

plt.figure(figsize=(9, 9))
plt.subplot(2, 2, 1)
plt.title("Rectified left stereo image")
plt.imshow(stereo_front_left_rect_image)
plt.axis("off")
plt.subplot(2, 2, 2)
plt.title("Rectified right stereo image")
plt.imshow(stereo_front_right_rect_image)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.title("Left disparity map")
plt.imshow(
    stereo_front_left_rect_disparity_dil,
    cmap="nipy_spectral",
    vmin=0,
    vmax=192,
    interpolation="none",
)
plt.axis("off")
plt.subplot(2, 2, 4)
plt.title("Left object disparity map")
plt.imshow(
    stereo_front_left_rect_objects_disparity_dil,
    cmap="nipy_spectral",
    vmin=0,
    vmax=192,
    interpolation="none",
)
'''
#plt.axis("off")
#plt.tight_layout()
plt.plot([1,2], [3,4])
plt.show()
