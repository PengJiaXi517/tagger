from utils.viz_utils.map import draw_map
from utils.viz_utils.obstacles import draw_obstacle, draw_trajectory
import cv2
import os
from utils.opencv_viewer import View
from base import LabelScene
import numpy as np

def draw_scene(scene: LabelScene, save_dir, save_sub_dir, pickle_name):
    ego_states = scene.obstacles[-9]['future_trajectory']['future_states']

    ego_car_view = View(
        2000,
        1000,
        (100, 50),
        (
            ego_states[0]["x"] + 20 * np.cos(ego_states[0]["theta"]),
            ego_states[0]["y"] + 20 * np.sin(ego_states[0]["theta"]),
        ),
        -ego_states[0]["theta"],
    )

    frame = ego_car_view.build_frame()

    draw_map(
        scene.percepmap,
        ego_car_view,
        frame
    )

    draw_trajectory(scene.obstacles, ego_car_view, frame)
    draw_obstacle(scene.obstacles, ego_car_view, frame)

    os.makedirs(os.path.join(save_dir, save_sub_dir), exist_ok=True)

    cv2.imwrite(
        os.path.join(save_dir, save_sub_dir, f"{pickle_name}.jpg"),
        frame,
    )