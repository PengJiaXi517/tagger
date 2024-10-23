from dataclasses import dataclass, field
from typing import Dict, List
from base import TagData
from tag_functions.high_value_scene.narrow_road_tag import *
from tag_functions.high_value_scene.hv_utils.basic_func import (
    valid_check,
    get_curvature,
)


@dataclass(repr=False)
class JunctionBypassTag:
    def __init__(self) -> None:
        self.is_junction_bypass: bool = False
        self.future_junction_bypass = [False for i in range(100)]

    def as_dict(self):
        return {
            "is_junction_bypass": self.is_junction_bypass,
        }


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


def junction_bypass_tag(
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
