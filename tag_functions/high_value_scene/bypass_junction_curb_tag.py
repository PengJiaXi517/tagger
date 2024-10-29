from typing import Dict, List
import numpy as np
from base import TagData, EgoPathInfo
from tag_functions.high_value_scene.hv_utils.tag_type import (
    BypassJunctionCurbTag,
    NarrowRoadTag,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    calculate_future_path_curvature_and_turn_type,
)


class BypassJunctionCurbTagHelper:
    def __init__(self) -> None:
        pass

    def check_future_path_bypass_static_object_in_junction(
        self,
        params: Dict,
        ego_path_info: EgoPathInfo,
        curvature: np.ndarray,
        turn_type: np.ndarray,
        narrow_road_tag: NarrowRoadTag,
    ) -> BypassJunctionCurbTag:
        bypass_junction_curb_tag = BypassJunctionCurbTag()
        in_junction_id = ego_path_info.in_junction_id

        for idx, (x, y) in enumerate(ego_path_info.future_path):
            if (
                in_junction_id[idx] is None
                or abs(curvature[idx]) < params.large_curvature_threshold
            ):
                continue
            if (
                turn_type[idx] > 0
                and narrow_road_tag.future_narrow_road_states_loose_threshold[
                    idx
                ][0]
            ) or (
                turn_type[idx] < 0
                and narrow_road_tag.future_narrow_road_states_loose_threshold[
                    idx
                ][1]
            ):
                bypass_junction_curb_tag.future_bypass_junction_curb[idx] = True

        bypass_junction_curb_tag.is_bypass_junction_curb = any(
            bypass_junction_curb_tag.future_bypass_junction_curb
        )

        return bypass_junction_curb_tag


def label_bypass_junction_curb_tag(
    data: TagData, params: Dict, narrow_road_tag: NarrowRoadTag
) -> BypassJunctionCurbTag:
    ego_path_info = data.label_scene.ego_path_info

    # 判断是否过路口
    if not any(obj is not None for obj in ego_path_info.in_junction_id):
        return BypassJunctionCurbTag()

    # 计算future path每个点的曲率和转弯类型
    curvature, turn_type = calculate_future_path_curvature_and_turn_type(
        ego_path_info.future_path
    )

    # 判断future path的每个点是否有绕行路口内静态障碍物或curb的行为
    bypass_junction_curb_tag_helper = BypassJunctionCurbTagHelper()
    return bypass_junction_curb_tag_helper.check_future_path_bypass_static_object_in_junction(
        params, ego_path_info, curvature, turn_type, narrow_road_tag
    )
