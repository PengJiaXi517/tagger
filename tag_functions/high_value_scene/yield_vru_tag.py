from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from base import TagData

# from registry import TAG_FUNCTIONS
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    valid_check,
)


@dataclass(repr=False)
class YieldVRUTag:
    def __init__(self) -> None:
        self.is_yield_vru: bool = False

    def as_dict(self):
        return {
            "is_yield_vru": self.is_yield_vru,
        }


def detect_braking_and_stop(ego_future_states):
    speeds = []
    is_braking = False
    stop_point = None
    stop_idx = None
    for idx, state in enumerate(ego_future_states):
        speed = np.linalg.norm([state["vx"], state["vy"]])
        speeds.append(speed)
        if speed < 0.2:
            if len(speeds) < 3 or abs(speeds[-1] - speeds[0]) < 0.3:
                return False, None, None
            speed_diff = [
                speeds[i] - speeds[i + 1] for i in range(len(speeds) - 1)
            ]
            is_braking = (
                sum(1 for v in speed_diff if v >= 0) / len(speed_diff)
            ) > 0.7
            stop_point = Point([state["x"], state["y"]])
            stop_idx = idx
            break
    return is_braking, stop_point, stop_idx


def check_intersection(
    obstacles,
    ego_future_states,
    future_path_linestring,
    stop_point,
    stop_idx,
    time_window_bef_stop,
):
    for idx, obs in obstacles.items():
        if idx == -9 or (
            obs["features"]["type"] != "PEDESTRIAN"
            and obs["features"]["type"] != "BICYCLE"
        ):
            continue
        obs_future_traj = [
            (state["x"], state["y"])
            for state in obs["future_trajectory"]["future_states"]
            if state["timestamp"]
            > ego_future_states[stop_idx]["timestamp"] - time_window_bef_stop
        ]
        if len(obs_future_traj) < 2:
            continue

        obs_polyline = LineString(obs_future_traj)
        intersection_pt = obs_polyline.intersection(future_path_linestring)
        if (
            not intersection_pt.is_empty
            and stop_point.distance(intersection_pt) < 5
        ):
            return True

    return False


# 判断是否在礼让vru而减速
# @TAG_FUNCTIONS.register()
def yield_vru_tag(data: TagData, params: Dict) -> Dict:
    yield_vru_tag = YieldVRUTag()
    if not valid_check(data):
        return yield_vru_tag

    obstacles = data.label_scene.obstacles
    future_path_linestring = (
        data.label_scene.ego_path_info.future_path_linestring
    )
    ego_future_states = obstacles[-9]["future_trajectory"]["future_states"]

    # 判断是否有减速行为
    is_braking, stop_point, stop_idx = detect_braking_and_stop(
        ego_future_states
    )
    if not is_braking or stop_point is None:
        return yield_vru_tag

    # 判断刹停点前方是否有车辆
    filter = ObstacleFilter()
    if filter.vehicle_in_front(obstacles, stop_point, stop_idx):
        return yield_vru_tag

    # 判断vru future_states与ego future_path是否相交，计算交点与刹停点的距离
    if check_intersection(
        obstacles,
        ego_future_states,
        future_path_linestring,
        stop_point,
        stop_idx,
        params.time_window_bef_stop,
    ):
        yield_vru_tag.is_yield_vru = True

    return yield_vru_tag
