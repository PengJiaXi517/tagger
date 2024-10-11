from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from base import TagData
from registry import TAG_FUNCTIONS
from tag_functions.high_value_scene.hv_utils.collision_detector import (
    CollisionDetector,
)
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    valid_check,
    is_moving,
    get_curvature,
    get_ego_polygon,
)


@dataclass(repr=False)
class NarrowRoadTag:
    def __init__(self) -> None:
        self.is_narrow_road: bool = False
        self.future_narrow_road = [[False, False] for i in range(100)]
        self.future_narrow_road_relax = [[False, False] for i in range(100)]

    def as_dict(self):
        return {
            "is_narrow_road": self.is_narrow_road,
            "future_narrow_road": self.future_narrow_road,
        }


@dataclass(repr=False)
class JunctionBypassTag:
    def __init__(self) -> None:
        self.is_junction_bypass: bool = False
        self.future_junction_bypass = [False for i in range(100)]

    def as_dict(self):
        return {
            "is_junction_bypass": self.is_junction_bypass,
            "future_junction_bypass": self.future_junction_bypass,
        }


@dataclass(repr=False)
class NearStaticObjTag:
    narrow_road_tag: NarrowRoadTag = None
    junction_bypass_tag: JunctionBypassTag = None

    def as_dict(self):
        return {
            "narrow_road_tag": self.narrow_road_tag.as_dict(),
            "junction_bypass_tag": self.junction_bypass_tag.as_dict(),
        }


def fill_narrow_road_tag(narrow_road_tag, collision_info, idx):
    narrow_road_tag.future_narrow_road[idx][0] |= collision_info["left_strict"]
    narrow_road_tag.future_narrow_road[idx][1] |= collision_info["right_strict"]
    narrow_road_tag.future_narrow_road_relax[idx][0] |= collision_info[
        "left_relax"
    ]
    narrow_road_tag.future_narrow_road_relax[idx][1] |= collision_info[
        "right_relax"
    ]


def label_narrow_road_tag(data: TagData, params: Dict) -> NarrowRoadTag:
    narrow_road_tag = NarrowRoadTag()
    if not valid_check(data):
        return narrow_road_tag

    ego_path_info = data.label_scene.ego_path_info
    obstacles = data.label_scene.obstacles
    curb_vec = data.label_scene.label_res["curb_label"]["decision"]["vec"]
    curb_src = data.label_scene.label_res["curb_label"]["decision"]["src_point"]
    curb_end = curb_src + curb_vec
    curbs = [(curb_src[i], curb_end[i]) for i in range(len(curb_src))]
    curbs_lat_decision = data.label_scene.label_res["curb_label"]["decision"][
        "interactive_lat_type"
    ]
    curbs_l = data.label_scene.label_res["curb_label"]["decision"][
        "obs_l"
    ]
    detector = CollisionDetector(params)
    filter = ObstacleFilter(params)

    if not is_moving(obstacles[-9]):
        return narrow_road_tag

    # 筛选出静态障碍物，并提前计算其polygon
    static_obs, id_polygon = filter.get_static_obs_polygon(obstacles)

    # 过滤l绝对值大的curb，并提前计算curbs的linestring
    curbs_linestring = filter.get_curbs_linestring(curbs, curbs_l)

    for idx, (x, y) in enumerate(ego_path_info.future_path):
        veh_polygon = get_ego_polygon(
            ego_path_info.future_path, idx, obstacles[-9]
        )
        collision_info = detector.check_collision_curb(
            veh_polygon, curbs_linestring, curbs_lat_decision
        )
        fill_narrow_road_tag(narrow_road_tag, collision_info, idx)

        if collision_info["left_strict"] and collision_info["right_strict"]:
            continue

        collision_info = detector.check_collision_obs(
            veh_polygon, static_obs, id_polygon
        )
        fill_narrow_road_tag(narrow_road_tag, collision_info, idx)

    tmp = [all(obj) for obj in narrow_road_tag.future_narrow_road]
    narrow_road_tag.is_narrow_road = any(tmp)
    return narrow_road_tag


def label_junction_bypass_tag(
    data: TagData, params: Dict, narrow_road_tag: NarrowRoadTag
) -> JunctionBypassTag:
    junction_bypass_tag = JunctionBypassTag()
    if not valid_check(data):
        return junction_bypass_tag
    ego_path_info = data.label_scene.ego_path_info
    in_junction_id = ego_path_info.in_junction_id

    junction_scene = any(obj is not None for obj in in_junction_id)
    if not junction_scene:
        return junction_bypass_tag

    curvature, turn_type = get_curvature(ego_path_info)

    for idx, (x, y) in enumerate(ego_path_info.future_path):
        if abs(curvature[idx]) < params.curvature_th:
            continue
        if in_junction_id[idx] is None:
            continue
        if (
            turn_type[idx] > 0
            and narrow_road_tag.future_narrow_road_relax[idx][0]
        ):
            junction_bypass_tag.future_junction_bypass[idx] = True
        if (
            turn_type[idx] < 0
            and narrow_road_tag.future_narrow_road_relax[idx][1]
        ):
            junction_bypass_tag.future_junction_bypass[idx] = True

    junction_bypass_tag.is_junction_bypass = any(
        junction_bypass_tag.future_junction_bypass
    )
    return junction_bypass_tag


@TAG_FUNCTIONS.register()
def near_static_obj_tag(data: TagData, params: Dict) -> Dict:
    near_static_obj_tag = NearStaticObjTag()
    # 判断窄路通行
    near_static_obj_tag.narrow_road_tag = label_narrow_road_tag(data, params)
    # 判断路口内绕障
    near_static_obj_tag.junction_bypass_tag = label_junction_bypass_tag(
        data, params, near_static_obj_tag.narrow_road_tag
    )

    return near_static_obj_tag.as_dict()
