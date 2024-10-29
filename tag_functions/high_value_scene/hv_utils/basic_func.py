from typing import Dict, List, Tuple, Set
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon
from base import TagData
from collections import defaultdict
from tag_functions.high_value_scene.hv_utils.tag_type import LcPATHTag

# 判断障碍物是不是一直静止
def is_obstacle_always_static(obstacle: Dict) -> bool:
    future_states = obstacle["future_trajectory"]["future_states"]
    if len(future_states) < 2:
        return obstacle["features"]["is_still"]

    start_point = Point([future_states[0]["x"], future_states[0]["y"]])
    end_point = Point([future_states[-1]["x"], future_states[-1]["y"]])

    return start_point.distance(end_point) < 0.3


# 判断自车是不是一直在动
def is_ego_vehicle_always_moving(ego_obstacle: Dict) -> bool:
    future_states = ego_obstacle["future_trajectory"]["future_states"]
    if len(future_states) < 1:
        return not ego_obstacle["features"]["is_still"]

    for state in future_states:
        if np.linalg.norm([state["vx"], state["vy"]]) < 0.5:
            return False

    return True


def build_obstacle_bbox(
    center_x: float,
    center_y: float,
    heading: float,
    length: float,
    width: float,
) -> List[Tuple[float, float]]:
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


def build_ego_vehicle_polygon(
    future_path: List[Tuple[float, float]], idx: int, ego_obstacle: Dict
) -> Polygon:
    x, y = future_path[idx]
    if idx == 0:
        next_x, next_y = future_path[1]
        heading = math.atan2(next_y - y, next_x - x)
    else:
        last_x, last_y = future_path[idx - 1]
        heading = math.atan2(y - last_y, x - last_x)

    veh_length = ego_obstacle["features"]["length"]
    veh_width = ego_obstacle["features"]["width"]
    veh_bbox = build_obstacle_bbox(x, y, heading, veh_length, veh_width)
    veh_polygon = Polygon(
        [veh_bbox[0], veh_bbox[1], veh_bbox[2], veh_bbox[3], veh_bbox[0]]
    )

    return veh_polygon


# 计算每个障碍物 在未来每个时刻对应的polygon
def build_obstacle_future_state_polygons(obstacles: Dict) -> Dict:
    obstacle_future_polygons = defaultdict(dict)
    for obs_id, obs in obstacles.items():
        if obs_id == -9:
            continue
        obs_future_states = obs["future_trajectory"]["future_states"]
        length = obs["features"]["length"]
        width = obs["features"]["width"]
        for obs_state in obs_future_states:
            obs_bbox = build_obstacle_bbox(
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
            obstacle_future_polygons[obs_state["timestamp"]][
                obs_id
            ] = obs_polygon
    return obstacle_future_polygons


def calculate_future_path_curvature_and_turn_type(
    future_path: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    def normal_angle(theta):
        while theta >= np.pi:
            theta -= 2 * np.pi
        while theta <= -np.pi:
            theta += 2 * np.pi
        return theta

    applyall = np.vectorize(normal_angle)

    path_points = np.array(future_path)
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


def future_path_validity_check(data: TagData) -> bool:
    ego_path_info = data.label_scene.ego_path_info
    if len(ego_path_info.future_path) < 20:
        return False

    curvature, _ = calculate_future_path_curvature_and_turn_type(
        ego_path_info.future_path
    )

    if np.abs(curvature).max() > 1.57:
        return False

    return True


def xy_to_sl(line_string: LineString, point: Point) -> Tuple[float, float]:
    proj_s = line_string.project(point)
    if 0 < proj_s < line_string.length:
        proj_l = line_string.distance(point)

        line_coords = np.array(line_string.coords)
        point_coord = np.array(point.coords[0])
        point_to_line_distances = np.linalg.norm(
            line_coords - point_coord, axis=1
        )
        seg_idx = min(
            np.argmin(point_to_line_distances).item(),
            len(line_string.coords) - 2,
        )

        if (
            np.cross(
                np.array(line_string.coords[seg_idx + 1])
                - np.array(line_string.coords[seg_idx]),
                point_coord - np.array(line_string.coords[seg_idx]),
            )
            < 0
        ):
            proj_l = -proj_l

        return proj_s, proj_l

    return None, None


def get_lane_change_direction(lc_path_tag: LcPATHTag) -> Tuple[LineString, int]:
    lane_change_direction = -1
    min_start_pose_l = 1e6
    for tag in lc_path_tag:
        if abs(tag.start_pose_l) < min_start_pose_l:
            min_start_pose_l = abs(tag.start_pose_l)
            lane_change_direction = tag.lane_change_direction

    return lane_change_direction if min_start_pose_l < 50 else -1


def build_linestring_from_lane_seq_ids(
    lane_map: Dict, lane_seq_ids: List[int]
) -> LineString:
    lane_seq_polyline = []

    for lane_id in lane_seq_ids:
        lane = lane_map.get(lane_id, None)
        if lane is not None:
            lane_seq_polyline.extend(lane["polyline"])

    return LineString(lane_seq_polyline) if len(lane_seq_polyline) > 2 else None


def distance_point_to_linestring_list(
    point: Point, linestring_list: List[LineString]
) -> float:
    distances = [point.distance(linestring) for linestring in linestring_list]
    return min(distances)
