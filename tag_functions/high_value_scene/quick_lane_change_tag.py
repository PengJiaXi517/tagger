from typing import Dict, List, Tuple, Union
from shapely.geometry import LineString, Point
from base import TagData
from tag_functions.high_value_scene.hv_utils.tag_type import (
    NarrowRoadTag,
    InteractWithMovingObsTag,
    QuickLaneChangeTag,
    FuturePathTag,
    FuturePATHType,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    get_lane_change_direction,
    distance_point_to_linestring_list,
)


class QuickLaneChangeTagHelper:
    def __init__(self) -> None:
        self.consider_index_num: int = 6
        self.delta_dist_threshold_low: float = -0.015
        self.delta_dist_threshold_medium: float = -0.03
        self.delta_dist_threshold_high: float = -0.08

    def check_obstacles_before_lane_change(
        self,
        interact_with_moving_obs_tag: InteractWithMovingObsTag,
        narrow_road_tag: NarrowRoadTag,
        corr_frame_idx: List[Union[None, int]],
        lane_change_begin_index: int,
        lane_change_direction: int,
    ) -> Tuple[int, int]:
        # 障碍物距离信息
        future_narrow_road_states = narrow_road_tag.future_narrow_road_states
        future_interaction_with_moving_obs = (
            interact_with_moving_obs_tag.future_interaction_with_moving_obs
        )

        # 变道方向一侧的障碍物距离信息
        static_obstacle_collision_info = [
            obj[lane_change_direction] for obj in future_narrow_road_states
        ]
        moving_obstacle_collision_info = [
            obj[lane_change_direction]
            for obj in future_interaction_with_moving_obs
        ]

        min_obstacle_index = -1  # 离障碍物距离过近的点的最小index(future path上的index)
        max_obstacle_index = -1  # 离障碍物距离过近的点的最大index(future path上的index)
        for i in range(lane_change_begin_index):
            future_time_index = corr_frame_idx[i]
            if static_obstacle_collision_info[i] or (
                future_time_index < len(future_interaction_with_moving_obs)
                and moving_obstacle_collision_info[future_time_index]
            ):
                max_obstacle_index = i
                if min_obstacle_index == -1:
                    min_obstacle_index = i

        return min_obstacle_index, max_obstacle_index

    def judge_lane_change_begin_index(
        self,
        future_path: List[Tuple[float, float]],
        condition_linestring: List[LineString],
    ) -> int:
        distance_to_condition_linestring = [
            distance_point_to_linestring_list(
                Point(point), condition_linestring
            )
            for point in future_path
        ]

        distance_diffs = [
            distance_to_condition_linestring[i]
            - distance_to_condition_linestring[i - 1]
            for i in range(1, len(distance_to_condition_linestring))
        ]

        lane_change_begin_index = -1
        for i in range(len(distance_diffs) - self.consider_index_num):
            if distance_diffs[i] > self.delta_dist_threshold_low:
                continue

            future_distance_diffs = distance_diffs[
                i : i + self.consider_index_num
            ]

            if distance_diffs[i] < self.delta_dist_threshold_high or (
                sum(
                    1
                    for d in future_distance_diffs
                    if d <= self.delta_dist_threshold_medium
                )
                > 0.8 * self.consider_index_num
            ):
                lane_change_begin_index = i
                break

        return lane_change_begin_index


def label_quick_lane_change_tag(
    data: TagData,
    params: Dict,
    interact_with_moving_obs_tag: InteractWithMovingObsTag,
    narrow_road_tag: NarrowRoadTag,
    future_path_tag: FuturePathTag,
) -> QuickLaneChangeTag:
    quick_lane_chanege_tag = QuickLaneChangeTag()
    quick_lane_chanege_tag_helper = QuickLaneChangeTagHelper()
    future_path = data.label_scene.ego_path_info.future_path

    # 判断是否为变道场景
    if future_path_tag.path_type not in [
        FuturePATHType.LANE_CHANGE,
    ]:
        return quick_lane_chanege_tag

    quick_lane_chanege_tag.is_lane_change = True

    # 获取变道方向 0: left; 1: right
    lane_change_direction = get_lane_change_direction(
        future_path_tag.lc_path_tag
    )

    # 获取离future path横向距离最近的condtion_lane_seq linestring
    condition_linestring = (
        future_path_tag.condition_res_tag.nearest_condition_linestring
    )

    if lane_change_direction == -1 or condition_linestring is None:
        return quick_lane_chanege_tag

    # 计算开始变道的位置在future path中的index
    lane_change_begin_index = (
        quick_lane_chanege_tag_helper.judge_lane_change_begin_index(
            future_path, condition_linestring
        )
    )
    if lane_change_begin_index == -1:
        return quick_lane_chanege_tag

    quick_lane_chanege_tag.lane_change_begin_index = lane_change_begin_index

    # 判断自车与开始变道位置之间是否有障碍物(导致当前时刻无法变道)
    (
        min_obstacle_index,
        max_obstacle_index,
    ) = quick_lane_chanege_tag_helper.check_obstacles_before_lane_change(
        interact_with_moving_obs_tag,
        narrow_road_tag,
        data.label_scene.ego_path_info.corr_frame_idx,
        lane_change_begin_index,
        lane_change_direction,
    )

    # 判断是否为合理的变道样本(变道急 或 有因变道缓)
    if min_obstacle_index == -1 or max_obstacle_index == -1:
        if lane_change_begin_index <= 1:
            quick_lane_chanege_tag.is_quick_lane_change = True
    elif min_obstacle_index < 10 and (
        lane_change_begin_index - max_obstacle_index < 10
    ):
        quick_lane_chanege_tag.is_quick_lane_change = True

    return quick_lane_chanege_tag
