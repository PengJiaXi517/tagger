from typing import Dict, List, Tuple
from shapely.geometry import Polygon
from base import TagData
from tag_functions.high_value_scene.hv_utils.tag_type import (
    InteractWithMovingObsTag,
)
from tag_functions.high_value_scene.hv_utils.collision_detector import (
    CollisionDetector,
)
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    is_ego_vehicle_always_moving,
    build_obstacle_future_state_polygons,
    build_obstacle_bbox,
)


class InteractWithMovingObsTagHelper:
    def __init__(self) -> None:
        pass

    def check_distance_to_moving_obs_for_future_states(
        self,
        params: Dict,
        ego_obstacle: Dict,
        moving_obs: Dict,
        obstacle_future_state_polygons: Dict,
    ) -> InteractWithMovingObsTag:
        interact_with_moving_obs_tag = InteractWithMovingObsTag()
        collision_detector = CollisionDetector(params)
        ego_future_states = ego_obstacle["future_trajectory"]["future_states"]
        ego_length = ego_obstacle["features"]["length"]
        ego_width = ego_obstacle["features"]["width"]

        for idx, ego_state in enumerate(ego_future_states):
            ts_us = ego_state["timestamp"]
            ego_bbox = build_obstacle_bbox(
                ego_state["x"],
                ego_state["y"],
                ego_state["theta"],
                ego_length,
                ego_width,
            )
            ego_polygon = Polygon(
                [
                    ego_bbox[0],
                    ego_bbox[1],
                    ego_bbox[2],
                    ego_bbox[3],
                    ego_bbox[0],
                ]
            )
            collision_info = collision_detector.check_distance_to_moving_obs(
                ego_polygon, moving_obs, obstacle_future_state_polygons[ts_us]
            )
            interact_with_moving_obs_tag.future_interaction_with_moving_obs[
                idx
            ][0] |= collision_info["has_moving_obs_left"]
            interact_with_moving_obs_tag.future_interaction_with_moving_obs[
                idx
            ][1] |= collision_info["has_moving_obs_right"]

        interact_with_moving_obs_tag.is_interact_with_moving_obs = any(
            [
                any(obj)
                for obj in interact_with_moving_obs_tag.future_interaction_with_moving_obs
            ]
        )

        return interact_with_moving_obs_tag


def label_interact_with_moving_obs_tag(
    data: TagData, params: Dict
) -> InteractWithMovingObsTag:
    interact_with_moving_obs_tag = InteractWithMovingObsTag()
    obstacles = data.label_scene.obstacles

    # 判断自车是否在移动
    if not is_ego_vehicle_always_moving(obstacles[-9]):
        return interact_with_moving_obs_tag

    # 筛选出动态障碍物
    obs_filter = ObstacleFilter()
    moving_obs = obs_filter.find_moving_obstacles(obstacles)

    # 提前计算动态障碍物的未来状态的polygon
    obstacle_future_state_polygons = build_obstacle_future_state_polygons(
        moving_obs
    )

    # 判断未来时刻自车是否与动态障碍物距离过近
    interact_with_moving_obs_tag_helper = InteractWithMovingObsTagHelper()
    return interact_with_moving_obs_tag_helper.check_distance_to_moving_obs_for_future_states(
        params, obstacles[-9], moving_obs, obstacle_future_state_polygons
    )
