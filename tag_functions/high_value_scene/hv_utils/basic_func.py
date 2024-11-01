from typing import Dict, List, Tuple, Set
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon
from base import TagData
from collections import defaultdict


def is_obstacle_always_static(obstacle: Dict) -> bool:
    future_states = obstacle["future_trajectory"]["future_states"]
    if len(future_states) < 2:
        return obstacle["features"]["is_still"]

    start_point = Point([future_states[0]["x"], future_states[0]["y"]])
    end_point = Point([future_states[-1]["x"], future_states[-1]["y"]])

    return start_point.distance(end_point) < 0.3


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

    if len(future_path) < 2:
        heading = 0.0
    elif idx == 0:
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

    if len(future_path) <= 2:
        ret = np.zeros(len(future_path))
        return ret, ret

    path_points = np.array(future_path)
    diff_points = path_points[1:] - path_points[:-1]
    theta = np.arctan2(diff_points[:, 1], diff_points[:, 0])
    theta_diff = theta[1:] - theta[:-1]
    length = np.linalg.norm(diff_points[:-1], axis=-1)
    theta_diff = applyall(theta_diff)
    curvature = theta_diff / length
    turn_type = np.sign(theta_diff)

    curvature = np.concatenate([[curvature[0]], curvature, [curvature[-1]]])
    turn_type = np.concatenate([[turn_type[0]], turn_type, [turn_type[-1]]])

    return curvature, turn_type


def project_point_to_linestring_in_sl_coordinate(
    line_string: LineString, point: Point
) -> Tuple[float, float]:
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


def judge_lane_change_direction(
    future_path_points_sl_coordinate_projected_to_condition: List[
        Tuple[float, float, Point]
    ]
) -> int:
    if len(future_path_points_sl_coordinate_projected_to_condition) == 0:
        return -1

    _, proj_l, _ = future_path_points_sl_coordinate_projected_to_condition[0]

    if proj_l is None:
        return -1

    lane_change_direction = 1 if proj_l > 0 else 0

    return lane_change_direction


def find_nearest_condition_linestring(
    data: TagData,
    condition_start_lane_seq_ids: List[List[int]],
    condition_end_lane_seq_ids: List[List[int]],
) -> List[LineString]:
    future_path = data.label_scene.ego_path_info.future_path
    lane_map = data.label_scene.percepmap.lane_map

    min_lateral_dist = np.inf
    nearest_condition_linestring = None

    for start_ids, end_ids in zip(
        condition_start_lane_seq_ids, condition_end_lane_seq_ids
    ):
        condition_linestring = [
            linestring
            for linestring in [
                build_linestring_from_lane_seq_ids(lane_map, start_ids),
                build_linestring_from_lane_seq_ids(lane_map, end_ids),
            ]
            if linestring is not None
        ]

        sum_proj_l = 0
        for point in future_path:
            path_point = Point(point)
            for linestring in condition_linestring:
                proj_s = linestring.project(path_point)
                if 0 < proj_s < linestring.length:
                    sum_proj_l += linestring.distance(path_point)
                    break

        if 0 < sum_proj_l < min_lateral_dist:
            min_lateral_dist = sum_proj_l
            nearest_condition_linestring = condition_linestring

    return nearest_condition_linestring
