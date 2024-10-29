from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from base import TagData
from tag_functions.high_value_scene.hv_utils.basic_func import (
    build_linestring_from_lane_seq_ids,
    distance_point_to_linestring_list,
    xy_to_sl,
)
from tag_functions.high_value_scene.hv_utils.tag_type import (
    FuturePathTag,
    FuturePATHType,
    RightTurnOnlyTag,
)


class RightTurnOnlyTagHelper:
    def __init__(self) -> None:
        self.ego_to_junction_dist_threshold: float = 50.0
        self.future_path_to_junction_dist_threshold: float = 8.0
        self.sum_path_curvature_threshold: float = -0.2

    def is_ego_vehicle_near_junction(
        self,
        future_path_linestring: LineString,
        junction_map: Dict,
        current_lanes: List[int],
    ) -> bool:
        if future_path_linestring.length < 3:
            return False

        # 计算离自车最近的路口，记录其id与距离
        ego_point = Point(future_path_linestring.coords[0])
        nearest_junction = min(
            (
                (id, Polygon(junction["polygon"]).distance(ego_point))
                for id, junction in junction_map.items()
                if junction["type"] == "IN_ROAD"
            ),
            key=lambda x: x[1],
            default=(None, float("inf")),
        )

        # 距离小于阈值，则认为自车不在路口附近
        if (
            nearest_junction[1] >= self.ego_to_junction_dist_threshold
            or nearest_junction[0] is None
        ):
            return False

        if not junction_map[nearest_junction[0]]["pd_intersection"]:
            return False

        # future path与路口的距离需要小于阈值
        if (
            future_path_linestring.distance(
                Polygon(junction_map[nearest_junction[0]]["polygon"])
            )
            > self.future_path_to_junction_dist_threshold
        ):
            return False

        # 判断自车是否在路口的出口车道上，在的话需要筛掉
        exit_junction_lanes = set(
            lane_id
            for group in junction_map[nearest_junction[0]]["exit_groups"]
            for lane_id in group
        )

        if len(exit_junction_lanes) == 0 or any(
            current_lane in exit_junction_lanes
            for current_lane in current_lanes
        ):
            return False

        return True

    def get_target_lane_seq_linestring(
        self,
        lane_map: Dict,
        future_path: List[Tuple[float, float]],
        current_lane_seqs: List[List[int]],
        future_path_tag: FuturePathTag,
    ) -> List[LineString]:
        target_linestring = (
            future_path_tag.condition_res_tag.nearest_condition_linestring
        )
        condition_lane_is_in_left = False

        for linestring in (
            target_linestring if target_linestring is not None else []
        ):
            _, proj_l = xy_to_sl(linestring, Point(future_path[0]))
            if proj_l < -1.75:
                condition_lane_is_in_left = True
                break

        # 如果目标车道在左边，则认为是出右转专用道后的变道行为，用current_lane_seqs
        if condition_lane_is_in_left:
            if len(current_lane_seqs) > 0:
                target_linestring = [
                    build_linestring_from_lane_seq_ids(
                        lane_map, current_lane_seqs[0]
                    )
                ]
            else:
                target_linestring = None

        return target_linestring

    def is_right_turn_only_scene(
        self, data: TagData, future_path_tag: FuturePathTag
    ) -> bool:
        if future_path_tag.path_type not in [
            FuturePATHType.CRUISE,
            FuturePATHType.LANE_CHANGE,
            FuturePATHType.UNKNOWN,
        ]:
            return False

        if (
            future_path_tag.basic_path_tag.sum_path_curvature
            > self.sum_path_curvature_threshold
        ):
            return False

        in_junction_id = data.label_scene.ego_path_info.in_junction_id
        future_path_linestring = (
            data.label_scene.ego_path_info.future_path_linestring
        )
        junction_map = data.label_scene.percepmap.junction_map
        current_lanes = data.label_scene.ego_obs_lane_seq_info.current_lanes

        if any(obj is not None for obj in in_junction_id):
            return False

        if not self.is_ego_vehicle_near_junction(
            future_path_linestring, junction_map, current_lanes
        ):
            return False

        return True


def label_right_turn_only_tag(
    data: TagData, params: Dict, future_path_tag: FuturePathTag
) -> RightTurnOnlyTag:
    right_turn_only_tag = RightTurnOnlyTag()
    right_turn_only_tag_helper = RightTurnOnlyTagHelper()
    lane_map = data.label_scene.percepmap.lane_map
    future_path = data.label_scene.ego_path_info.future_path
    current_lane_seqs = data.label_scene.ego_obs_lane_seq_info.current_lane_seqs

    # 首先，判断是否为右转专用道场景
    if right_turn_only_tag_helper.is_right_turn_only_scene(
        data, future_path_tag
    ):
        right_turn_only_tag.is_right_turn_only = True

        # 获取目标lane seq的linestring, 用于计算与future path point的距离，从而得到变道前的valid len
        target_lane_seq_linestring = (
            right_turn_only_tag_helper.get_target_lane_seq_linestring(
                lane_map, future_path, current_lane_seqs, future_path_tag
            )
        )

        # 计算future path的有效长度(变道前的那段路)
        if target_lane_seq_linestring is not None:
            valid_len = 0
            for idx, point in enumerate(future_path):
                if (
                    distance_point_to_linestring_list(
                        Point(point), target_lane_seq_linestring
                    )
                    <= 0.5
                ):
                    valid_len = idx

            right_turn_only_tag.right_turn_only_valid_path_len = max(
                valid_len - 5, 0
            )

    return right_turn_only_tag
