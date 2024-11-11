from typing import Dict, List, Tuple
from shapely.geometry import LineString, Point, Polygon
from base import TagData
from tag_functions.high_value_scene.common.basic_info import BasicInfo
from tag_functions.high_value_scene.common.tag_type import (
    FuturePathTag,
    FuturePATHType,
    RightTurnOnlyTag,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    build_linestring_from_lane_seq_ids,
    distance_point_to_linestring_list,
)


class RightTurnOnlyTagHelper:
    def __init__(
        self,
        ego_to_junction_dist_threshold: float,
        future_path_to_junction_dist_threshold: float,
        sum_path_curvature_threshold: float,
    ) -> None:
        self.ego_to_junction_dist_threshold: float = (
            ego_to_junction_dist_threshold
        )
        self.future_path_to_junction_dist_threshold: float = (
            future_path_to_junction_dist_threshold
        )
        self.sum_path_curvature_threshold: float = sum_path_curvature_threshold

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
        current_lane_seqs: List[List[int]],
        basic_info: BasicInfo,
    ) -> List[LineString]:
        target_linestring = basic_info.nearest_condition_linestring

        if (
            len(
                basic_info.future_path_points_sl_coordinate_projected_to_condition
            )
            == 0
        ):
            return target_linestring

        (
            _,
            proj_l,
            _,
        ) = basic_info.future_path_points_sl_coordinate_projected_to_condition[
            0
        ]

        condition_lane_is_in_left = (
            True if (proj_l is not None and proj_l < -1.75) else False
        )

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
        self,
        data: TagData,
        future_path_tag: FuturePathTag,
        basic_info: BasicInfo,
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

        future_path_linestring = (
            data.label_scene.ego_path_info.future_path_linestring
        )
        junction_map = data.label_scene.percepmap.junction_map
        current_lanes = data.label_scene.ego_obs_lane_seq_info.current_lanes

        if basic_info.is_cross_junction:
            return False

        if not self.is_ego_vehicle_near_junction(
            future_path_linestring, junction_map, current_lanes
        ):
            return False

        return True


def label_right_turn_only_tag(
    data: TagData, basic_info: BasicInfo, future_path_tag: FuturePathTag
) -> RightTurnOnlyTag:
    right_turn_only_tag = RightTurnOnlyTag()
    right_turn_only_tag_helper = RightTurnOnlyTagHelper(
        ego_to_junction_dist_threshold=50.0,
        future_path_to_junction_dist_threshold=8.0,
        sum_path_curvature_threshold=-0.2,
    )
    lane_map = data.label_scene.percepmap.lane_map
    future_path = data.label_scene.ego_path_info.future_path
    current_lane_seqs = data.label_scene.ego_obs_lane_seq_info.current_lane_seqs

    # 首先，判断是否为右转专用道场景
    if right_turn_only_tag_helper.is_right_turn_only_scene(
        data, future_path_tag, basic_info
    ):
        right_turn_only_tag.is_right_turn_only = True

        # 获取目标lane seq的linestring, 用于计算与future path point的距离，从而得到变道前的valid len
        target_lane_seq_linestring = (
            right_turn_only_tag_helper.get_target_lane_seq_linestring(
                lane_map, current_lane_seqs, basic_info
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
