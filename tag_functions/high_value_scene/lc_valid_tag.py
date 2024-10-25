from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from base import TagData
from tag_functions.high_value_scene.narrow_road_tag import *
from tag_functions.high_value_scene.mixed_traffic_tag import *
from tag_functions.future_path_tag import *
from tag_functions.high_value_scene.hv_utils.basic_func import (
    get_sl,
)


@dataclass(repr=False)
class LcValidTag:
    def __init__(self) -> None:
        self.is_lane_change: bool = False
        self.is_lc_valid: bool = False
        self.lane_change_begin_index: int = -1

    def as_dict(self):
        return {
            "is_lane_change": self.is_lane_change,
            "is_lc_valid": self.is_lc_valid,
            "lane_change_begin_index": self.lane_change_begin_index,
        }


def check_obstacles_before_lane_change(
    mixed_traffic_tag,
    narrow_road_tag,
    corr_frame_idx,
    lane_change_begin_index,
    lane_change_direction,
):
    future_narrow_road = narrow_road_tag.future_narrow_road
    future_mixed_traffic = mixed_traffic_tag.future_mixed_traffic

    static_obstacle_collision_info = [
        obj[lane_change_direction] for obj in future_narrow_road
    ]
    moving_obstacle_collision_info = [
        obj[lane_change_direction] for obj in future_mixed_traffic
    ]

    min_obstacle_index = -1
    max_obstacle_index = -1
    for i in range(lane_change_begin_index):
        future_time_index = corr_frame_idx[i]
        if static_obstacle_collision_info[i] or (
            future_time_index < len(future_mixed_traffic)
            and moving_obstacle_collision_info[future_time_index]
        ):
            max_obstacle_index = i
            if min_obstacle_index == -1:
                min_obstacle_index = i

    return min_obstacle_index, max_obstacle_index


def get_labeled_lane_seq_linestring(lane_map, future_path_tag):
    lc_path_tag = future_path_tag.lc_path_tag
    labeled_lane_seq = None
    min_start_pose_l = 1e6
    for tag in lc_path_tag:
        if abs(tag.start_pose_l) < min_start_pose_l:
            min_start_pose_l = abs(tag.start_pose_l)
            labeled_lane_seq = tag.labeled_lane_seq

    return LineString(
        [
            point
            for lane_id in labeled_lane_seq
            for point in lane_map[lane_id]["polyline"]
        ]
    )


def judge_lane_change_direction(future_path, labeled_lane_seq_linestring):
    ego_point = Point(future_path[0])

    _, proj_l = get_sl(labeled_lane_seq_linestring, ego_point)
    
    if proj_l is None:
        return -1
    
    lane_change_direction = 1 if proj_l > 0 else 0

    return lane_change_direction


def judge_lane_change_begin_index(future_path, labeled_lane_seq_linestring):
    distance_to_target_lane_seq = [
        labeled_lane_seq_linestring.distance(Point(point))
        for point in future_path
    ]

    distance_diffs = [
        distance_to_target_lane_seq[i] - distance_to_target_lane_seq[i - 1]
        for i in range(1, len(distance_to_target_lane_seq))
    ]

    consider_len = 8
    lane_change_begin_index = -1

    for i in range(len(distance_diffs) - consider_len):
        if distance_diffs[i] > -0.015:
            continue

        future_distance_diffs = distance_diffs[i : i + consider_len]

        # if distance_to_target_lane_seq[i] - distance_to_target_lane_seq[0] < -0.5:

        if distance_diffs[i] < -0.08 or (
            sum(1 for d in future_distance_diffs if d <= -0.025)
            > 0.5 * consider_len
        ):
            lane_change_begin_index = i
            break

    return lane_change_begin_index


def lc_valid_tag(
    data: TagData,
    mixed_traffic_tag: MixedTrafficTag,
    narrow_road_tag: NarrowRoadTag,
    future_path_tag: FuturePathTag,
    params: Dict,
) -> Dict:
    lc_valid_tag = LcValidTag()
    future_path = data.label_scene.ego_path_info.future_path
    lane_map = data.label_scene.percepmap.lane_map

    if future_path_tag.path_type not in [
        FuturePATHType.LANE_CHANGE,
    ]:
        return lc_valid_tag

    lc_valid_tag.is_lane_change = True

    labeled_lane_seq_linestring = get_labeled_lane_seq_linestring(
        lane_map, future_path_tag
    )

    lane_change_begin_index = judge_lane_change_begin_index(
        future_path, labeled_lane_seq_linestring
    )

    if lane_change_begin_index == -1:
        return lc_valid_tag

    lc_valid_tag.lane_change_begin_index = lane_change_begin_index

    lane_change_direction = judge_lane_change_direction(
        future_path, labeled_lane_seq_linestring
    )
    if lane_change_direction == -1:
        return lc_valid_tag

    (
        min_obstacle_index,
        max_obstacle_index,
    ) = check_obstacles_before_lane_change(
        mixed_traffic_tag,
        narrow_road_tag,
        data.label_scene.ego_path_info.corr_frame_idx,
        lane_change_begin_index,
        lane_change_direction,
    )

    if (
        min_obstacle_index == -1 or max_obstacle_index == -1
    ):
        if lane_change_begin_index <= 1:
            lc_valid_tag.is_lc_valid = True
    elif min_obstacle_index < 10 and (
        lane_change_begin_index - max_obstacle_index < 10
    ):
        lc_valid_tag.is_lc_valid = True

    return lc_valid_tag
