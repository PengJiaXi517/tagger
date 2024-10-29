from typing import Dict, List
from base import TagData, EgoPathInfo
from tag_functions.high_value_scene.hv_utils.tag_type import (
    NarrowRoadTag,
)
from tag_functions.high_value_scene.hv_utils.collision_detector import (
    CollisionDetector,
)
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    is_ego_vehicle_always_moving,
    build_ego_vehicle_polygon,
)


class NarrowRoadTagHelper:
    def __init__(self):
        pass

    def fill_narrow_road_tag(self, narrow_road_tag, collision_info, idx):
        narrow_road_tag.future_narrow_road_states[idx][0] |= collision_info[
            "has_static_obs_left_strict"
        ]
        narrow_road_tag.future_narrow_road_states[idx][1] |= collision_info[
            "has_static_obs_right_strict"
        ]
        narrow_road_tag.future_narrow_road_states_loose_threshold[idx][
            0
        ] |= collision_info["has_static_obs_left_loose"]
        narrow_road_tag.future_narrow_road_states_loose_threshold[idx][
            1
        ] |= collision_info["has_static_obs_left_loose"]

    def check_future_path_distance_to_curb_and_static_obs(
        self,
        params: Dict,
        ego_path_info: EgoPathInfo,
        obstacles: Dict,
        static_obs: Dict,
        id_polygon: Dict,
        curbs_linestring: Dict,
        curbs_lat_decision: Dict,
    ) -> NarrowRoadTag:
        narrow_road_tag = NarrowRoadTag()
        collision_detector = CollisionDetector(params)
        for idx, (x, y) in enumerate(ego_path_info.future_path):
            veh_polygon = build_ego_vehicle_polygon(
                ego_path_info.future_path, idx, obstacles[-9]
            )
            collision_info = collision_detector.check_distance_to_curb(
                veh_polygon, curbs_linestring, curbs_lat_decision
            )
            self.fill_narrow_road_tag(narrow_road_tag, collision_info, idx)

            if (
                collision_info["has_static_obs_left_strict"]
                and collision_info["has_static_obs_right_strict"]
            ):
                continue

            collision_info = collision_detector.check_distance_to_static_obs(
                veh_polygon, static_obs, id_polygon
            )
            self.fill_narrow_road_tag(narrow_road_tag, collision_info, idx)

        narrow_road_tag.is_narrow_road = any(
            [all(obj) for obj in narrow_road_tag.future_narrow_road_states]
        )
        return narrow_road_tag


def label_narrow_road_tag(data: TagData, params: Dict) -> NarrowRoadTag:
    ego_path_info = data.label_scene.ego_path_info
    obstacles = data.label_scene.obstacles
    curb_decision = data.label_scene.label_res["curb_label"].get(
        "decision", None
    )
    obs_filter = ObstacleFilter()
    narrow_road_tag_helper = NarrowRoadTagHelper()

    # 判断自车是否在移动
    if curb_decision is None or not is_ego_vehicle_always_moving(obstacles[-9]):
        return NarrowRoadTag()

    # 筛选出静态障碍物，并提前计算其polygon
    static_obs, id_polygon = obs_filter.build_static_obstacle_polygons(
        obstacles
    )

    # 过滤l绝对值大的curb，并提前计算curbs的linestring
    curbs_linestring = obs_filter.build_curbs_linestring(curb_decision)

    # 判断future_path中每一个点与障碍物/curb的距离是否过近
    return narrow_road_tag_helper.check_future_path_distance_to_curb_and_static_obs(
        params,
        ego_path_info,
        obstacles,
        static_obs,
        id_polygon,
        curbs_linestring,
        curb_decision["interactive_lat_type"],
    )
