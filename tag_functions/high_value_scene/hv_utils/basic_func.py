from typing import Dict, List, Tuple
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon
from base import TagData
from collections import defaultdict

# 判断是不是一直静止
def is_static(obstacle):
    future_states = obstacle["future_trajectory"]["future_states"]
    if len(future_states) < 2:
        return obstacle["features"]["is_still"]
    pt_start = Point([future_states[0]["x"], future_states[0]["y"]])
    pt_end = Point([future_states[-1]["x"], future_states[-1]["y"]])
    if pt_start.distance(pt_end) < 0.2:
        return True
    return False


# 判断是不是一直在动
def is_moving(obstacle):
    future_states = obstacle["future_trajectory"]["future_states"]
    if len(future_states) < 1:
        return not obstacle["features"]["is_still"]
    for state in future_states:
        if np.linalg.norm([state["vx"], state["vy"]]) < 0.5:
            return False
    return True


def get_bbox(
    center_x: float,
    center_y: float,
    heading: float,
    length: float,
    width: float,
):
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    dx1 = cos_heading * length / 2
    dy1 = sin_heading * length / 2
    dx2 = sin_heading * width / 2
    dy2 = -cos_heading * width / 2
    corner1 = (center_x + dx1 + dx2, center_y + dy1 + dy2)
    corner2 = (center_x + dx1 - dx2, center_y + dy1 - dy2)
    corner3 = (center_x - dx1 - dx2, center_y - dy1 - dy2)
    corner4 = (center_x - dx1 + dx2, center_y - dy1 + dy2)
    bbox = [corner1, corner2, corner3, corner4]
    return bbox


def get_ego_polygon(future_path, idx, obstacle):
    x, y = future_path[idx]
    if idx == 0:
        next_x, next_y = future_path[1]
        heading = math.atan2(next_y - y, next_x - x)
    else:
        last_x, last_y = future_path[idx - 1]
        heading = math.atan2(y - last_y, x - last_x)
    veh_length = obstacle["features"]["length"]
    veh_width = obstacle["features"]["width"]
    veh_bbox = get_bbox(x, y, heading, veh_length, veh_width)
    veh_polygon = Polygon(
        [veh_bbox[0], veh_bbox[1], veh_bbox[2], veh_bbox[3], veh_bbox[0]]
    )
    return veh_polygon


def get_obs_future_polygon(obstacles):
    obs_future_polygon = defaultdict(dict)
    for obs_id, obs in obstacles.items():
        if obs_id == -9:
            continue
        obs_future_states = obs["future_trajectory"]["future_states"]
        length = obs["features"]["length"]
        width = obs["features"]["width"]
        for obs_state in obs_future_states:
            obs_bbox = get_bbox(
                obs_state["x"],
                obs_state["y"],
                obs_state["theta"],
                length,
                width,
            )
            obs_polygon = Polygon(
                [
                    obs_bbox[0],
                    obs_bbox[1],
                    obs_bbox[2],
                    obs_bbox[3],
                    obs_bbox[0],
                ]
            )
            obs_future_polygon[obs_state["timestamp"]][obs_id] = obs_polygon
    return obs_future_polygon


def get_curvature(ego_path_info):
    def normal_angle(theta):
        while theta >= np.pi:
            theta -= 2 * np.pi
        while theta <= -np.pi:
            theta += 2 * np.pi
        return theta

    applyall = np.vectorize(normal_angle)

    path_points = np.array(ego_path_info.future_path)
    diff_points = path_points[1:] - path_points[:-1]
    theta = np.arctan2(diff_points[:, 1], diff_points[:, 0])
    theta_diff = theta[1:] - theta[:-1]
    length = np.linalg.norm(diff_points[:-1], axis=-1)
    theta_diff = applyall(theta_diff)
    curvature = theta_diff / length
    turn_type = np.sign(theta_diff)

    curvature = np.insert(curvature, 0, [curvature[0], curvature[0]], axis=0)
    turn_type = np.insert(turn_type, 0, [turn_type[0], turn_type[0]], axis=0)
    return curvature, turn_type


def valid_check(data: TagData) -> bool:
    ego_path_info = data.label_scene.ego_path_info
    if len(ego_path_info.future_path) < 20:
        return False
    curvature, _ = get_curvature(ego_path_info)
    if np.abs(curvature).max() > 1.57:
        return False
    return True


def get_sl(path, path_point):
    proj_s = path.project(path_point)
    if 0 < proj_s < path.length:
        proj_l = path.distance(path_point)
        coords = np.array(path.coords)
        point_coords = np.array(path_point.coords[0])
        distances = np.linalg.norm(coords - point_coords, axis=1)
        seg_idx = min(np.argmin(distances).item(), len(path.coords) - 2)
        if (
            np.cross(
                np.array(path.coords[seg_idx + 1])
                - np.array(path.coords[seg_idx]),
                np.array(path_point.coords[0]) - np.array(path.coords[seg_idx]),
            )
            < 0
        ):
            proj_l = -proj_l
        return proj_s, proj_l
    return None, None
