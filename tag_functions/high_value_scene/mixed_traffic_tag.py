from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from shapely.geometry import LineString, Point, Polygon
from base import TagData

# from registry import TAG_FUNCTIONS
from tag_functions.high_value_scene.hv_utils.collision_detector import (
    CollisionDetector,
)
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    valid_check,
    is_moving,
    get_obs_future_polygon,
    get_bbox,
)


@dataclass(repr=False)
class MixedTrafficTag:
    def __init__(self) -> None:
        self.is_mixed_traffic: bool = False
        self.future_mixed_traffic = [[False, False] for i in range(80)]

    def as_dict(self):
        return {
            "is_mixed_traffic": self.is_mixed_traffic,
        }


def label_mixed_traffic_tag(obstacles, moving_obs, future_obs_polygon, params):
    mixed_traffic_tag = MixedTrafficTag()
    detector = CollisionDetector(params)
    ego_obs = obstacles[-9]
    ego_future_states = ego_obs["future_trajectory"]["future_states"]
    ego_length = ego_obs["features"]["length"]
    ego_width = ego_obs["features"]["width"]

    for idx, ego_state in enumerate(ego_future_states):
        ts_us = ego_state["timestamp"]
        ego_bbox = get_bbox(
            ego_state["x"],
            ego_state["y"],
            ego_state["theta"],
            ego_length,
            ego_width,
        )
        ego_polygon = Polygon(
            [ego_bbox[0], ego_bbox[1], ego_bbox[2], ego_bbox[3], ego_bbox[0]]
        )
        collision_info = detector.check_collision_moving_obs(
            ego_polygon, moving_obs, future_obs_polygon[ts_us]
        )
        mixed_traffic_tag.future_mixed_traffic[idx][0] |= collision_info[
            "has_moving_obs_left"
        ]
        mixed_traffic_tag.future_mixed_traffic[idx][1] |= collision_info[
            "has_moving_obs_right"
        ]

    mixed_traffic_tag.is_mixed_traffic = any(
        [any(obj) for obj in mixed_traffic_tag.future_mixed_traffic]
    )
    return mixed_traffic_tag


# 判断8s内是否与动目标距离过近
# @TAG_FUNCTIONS.register()
def mixed_traffic_tag(data: TagData, params: Dict) -> Dict:
    mixed_traffic_tag = MixedTrafficTag()
    obstacles = data.label_scene.obstacles

    if not valid_check(data) or not is_moving(obstacles[-9]):
        return mixed_traffic_tag

    # 筛选出动态障碍物
    filter = ObstacleFilter()
    moving_obs = filter.get_moving_obs(obstacles)

    # 提前计算动态障碍物的未来状态的polygon
    future_obs_polygon = get_obs_future_polygon(moving_obs)

    # 判断未来时刻是否与动态障碍物距离过近
    return label_mixed_traffic_tag(
        obstacles, moving_obs, future_obs_polygon, params
    )
