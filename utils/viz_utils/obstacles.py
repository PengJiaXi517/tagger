from typing import Dict

import numpy as np

from utils.opencv_viewer import View


def draw_obstacle(obstacles: Dict, view: View, frame: np.ndarray):
    for id, obstacle in obstacles.items():
        latest_feature = obstacle["features"]
        length = latest_feature["length"]
        width = latest_feature["width"]
        x = latest_feature["history_states"][-1]["x"]
        y = latest_feature["history_states"][-1]["y"]
        theta = latest_feature["history_states"][-1]["theta"]
        if id == -9:
            view.draw_rotated_rectangle(
                frame, (x, y), (length, width), theta, color=(0, 0, 255), thickness=2
            )
        else:
            view.draw_rotated_rectangle(
                frame, (x, y), (length, width), theta, color=(255, 0, 0), thickness=2
            )


def draw_ego_path(
    ego_path_info: Dict, view: View, frame: np.ndarray, max_length: int = 34
):
    future_path = np.array(ego_path_info["future_path"][:max_length])
    in_junction_id = ego_path_info["in_junction_id"]
    view.draw_polyline(
        frame,
        future_path[:, 0],
        future_path[:, 1],
        color=(76, 144, 255),
        thickness=3,
    )

    for i in range(len(future_path)):
        if in_junction_id[i] is not None:
            view.draw_circle(
                frame,
                (future_path[i, 0], future_path[i, 1]),
                1.0,
                color=(0, 255, 255),
            )
