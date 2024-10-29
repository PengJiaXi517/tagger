from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from base import TagData
from tag_functions.high_value_scene.hv_utils.tag_type import (
    YieldVRUTag,
)
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)


class YieldVruTagHelper:
    def __init__(self) -> None:
        pass

    def is_ego_vehicle_braking(
        self, ego_future_states: List[Dict]
    ) -> Tuple[bool, Point, int]:
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

    def check_intersection_of_future_path_and_obs_future_traj(
        self,
        obstacles: Dict,
        ego_future_states: List[Dict],
        future_path_linestring: LineString,
        stop_point: Point,
        stop_idx: int,
        time_window_bef_stop: float,
    ) -> bool:
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
                > ego_future_states[stop_idx]["timestamp"]
                - time_window_bef_stop
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


def label_yield_vru_tag(data: TagData, params: Dict) -> YieldVRUTag:
    yield_vru_tag = YieldVRUTag()
    obstacles = data.label_scene.obstacles
    future_path_linestring = (
        data.label_scene.ego_path_info.future_path_linestring
    )
    ego_future_states = obstacles[-9]["future_trajectory"]["future_states"]
    yield_vru_tag_helper = YieldVruTagHelper()

    # 判断是否有减速行为，并记录刹停点以及刹停点时刻在future states中的索引
    (
        is_braking,
        stop_point,
        stop_idx,
    ) = yield_vru_tag_helper.is_ego_vehicle_braking(ego_future_states)

    if not is_braking or stop_point is None:
        return yield_vru_tag

    # 判断刹停点前方是否有车辆
    obs_filter = ObstacleFilter()
    if obs_filter.has_vehicle_in_front(obstacles, stop_point, stop_idx):
        return yield_vru_tag

    # 判断vru的未来轨迹与自车的未来轨迹是否相交，并计算交点与刹停点的距离
    if yield_vru_tag_helper.check_intersection_of_future_path_and_obs_future_traj(
        obstacles,
        ego_future_states,
        future_path_linestring,
        stop_point,
        stop_idx,
        params.time_window_bef_stop,
    ):
        yield_vru_tag.is_yield_vru = True

    return yield_vru_tag
