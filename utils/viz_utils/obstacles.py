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
                frame, (x, y), (length, width), theta, color=(0, 0, 255), thickness=3
            )
        else:
            view.draw_rotated_rectangle(
                frame, (x, y), (length, width), theta, color=(255, 0, 0), thickness=3
            )
            view.draw_text(frame, str(id), (x, y))


def draw_trajectory(scene_obstacles: Dict, view: View, frame: np.ndarray, ego_color=(100, 100, 255), obs_color=(255, 100, 100), thickness=2):
    """
    绘制障碍物未来轨迹
    参数：
        scene_obstacles: 场景障碍物字典
        view: View对象 包含坐标系转换方法
        frame: 图像帧
        ego_color: 自车轨迹颜色 BGR格式
        obs_color: 他车轨迹颜色
        thickness: 线条粗细
    """
    for obs_id, obs_info in scene_obstacles.items():
        # 获取轨迹数据
        future_states = obs_info.get('future_trajectory', {}).get('future_states', [])
        if len(future_states) < 2:
            continue
        
        # 设置颜色
        color = ego_color if obs_id == -9 else obs_color
        
        # 转换轨迹点
        points = []
        for state in future_states:
            points.append((state["x"], state["y"]))
        
        # 绘制连续线段
        for i in range(1, len(points)):
            view.draw_line(frame, points[i-1], points[i], color, thickness)


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
