from dataclasses import dataclass, field
from typing import Dict, List
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
        }


@dataclass(repr=False)
class JunctionBypassTag:
    def __init__(self) -> None:
        self.is_junction_bypass: bool = False
        self.future_junction_bypass = [False for i in range(100)]

    def as_dict(self):
        return {
            "is_junction_bypass": self.is_junction_bypass,
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
    narrow_road_tag.future_narrow_road[idx][0] |= collision_info[
        "has_static_obs_left_strict"
    ]
    narrow_road_tag.future_narrow_road[idx][1] |= collision_info[
        "has_static_obs_right_strict"
    ]
    narrow_road_tag.future_narrow_road_relax[idx][0] |= collision_info[
        "has_static_obs_left_relax"
    ]
    narrow_road_tag.future_narrow_road_relax[idx][1] |= collision_info[
        "has_static_obs_left_relax"
    ]


def check_future_path_collision(
    ego_path_info,
    obstacles,
    static_obs,
    id_polygon,
    curbs_linestring,
    curbs_lat_decision,
    params,
):
    narrow_road_tag = NarrowRoadTag()
    detector = CollisionDetector(params)
    for idx, (x, y) in enumerate(ego_path_info.future_path):
        veh_polygon = get_ego_polygon(
            ego_path_info.future_path, idx, obstacles[-9]
        )
        collision_info = detector.check_collision_curb(
            veh_polygon, curbs_linestring, curbs_lat_decision
        )
        fill_narrow_road_tag(narrow_road_tag, collision_info, idx)

        if (
            collision_info["has_static_obs_left_strict"]
            and collision_info["has_static_obs_right_strict"]
        ):
            continue

        collision_info = detector.check_collision_static_obs(
            veh_polygon, static_obs, id_polygon
        )
        fill_narrow_road_tag(narrow_road_tag, collision_info, idx)

    narrow_road_tag.is_narrow_road = any(
        [all(obj) for obj in narrow_road_tag.future_narrow_road]
    )
    return narrow_road_tag


def check_future_path_bypass(
    ego_path_info, curvature, turn_type, narrow_road_tag, params
):
    junction_bypass_tag = JunctionBypassTag()
    in_junction_id = ego_path_info.in_junction_id
    for idx, (x, y) in enumerate(ego_path_info.future_path):
        if (
            in_junction_id[idx] is None
            or abs(curvature[idx]) < params.curvature_th
        ):
            continue
        if (
            turn_type[idx] > 0
            and narrow_road_tag.future_narrow_road_relax[idx][0]
        ) or (
            turn_type[idx] < 0
            and narrow_road_tag.future_narrow_road_relax[idx][1]
        ):
            junction_bypass_tag.future_junction_bypass[idx] = True

    junction_bypass_tag.is_junction_bypass = any(
        junction_bypass_tag.future_junction_bypass
    )
    return junction_bypass_tag


def label_narrow_road_tag(data: TagData, params: Dict) -> NarrowRoadTag:
    ego_path_info = data.label_scene.ego_path_info
    obstacles = data.label_scene.obstacles
    curb_decision = data.label_scene.label_res["curb_label"]["decision"]
    curbs_lat_decision = curb_decision["interactive_lat_type"]
    filter = ObstacleFilter()

    if not valid_check(data) or not is_moving(obstacles[-9]):
        return NarrowRoadTag()

    # 筛选出静态障碍物，并提前计算其polygon
    static_obs, id_polygon = filter.get_static_obs_polygon(obstacles)

    # 过滤l绝对值大的curb，并提前计算curbs的linestring
    curbs_linestring = filter.get_curbs_linestring(curb_decision)

    # 判断future_path中每一个点的碰撞情况
    return check_future_path_collision(
        ego_path_info,
        obstacles,
        static_obs,
        id_polygon,
        curbs_linestring,
        curbs_lat_decision,
        params,
    )


def label_junction_bypass_tag(
    data: TagData, params: Dict, narrow_road_tag: NarrowRoadTag
) -> JunctionBypassTag:
    ego_path_info = data.label_scene.ego_path_info
    in_junction_id = ego_path_info.in_junction_id

    junction_scene = any(obj is not None for obj in in_junction_id)

    if not valid_check(data) or not junction_scene:
        return JunctionBypassTag()

    curvature, turn_type = get_curvature(ego_path_info)

    # 判断future_path中每一个点的路口内绕行情况
    return check_future_path_bypass(
        ego_path_info, curvature, turn_type, narrow_road_tag, params
    )


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
