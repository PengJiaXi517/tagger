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

    def as_dict(self):
        return {
            "is_lane_change": self.is_lane_change,
            "is_lc_valid": self.is_lc_valid,
        }


def has_lateral_obstacle(
    mixed_traffic_tag, narrow_road_tag, corr_frame_idx, lc_await_len, lc_dir
):
    future_narrow_road = narrow_road_tag.future_narrow_road
    future_mixed_traffic = mixed_traffic_tag.future_mixed_traffic

    consider_obs_static = [obj[lc_dir] for obj in future_narrow_road]
    consider_obs_moving = [obj[lc_dir] for obj in future_mixed_traffic]

    obs_idx = 0
    for i in range(int(lc_await_len)):
        future_time_idx = corr_frame_idx[i]
        if consider_obs_static[i] or (
            future_time_idx < len(future_mixed_traffic)
            and consider_obs_moving[future_time_idx]
        ):
            obs_idx = i

    return obs_idx


def judge_lc_direction(lane_map, future_path, future_path_tag):
    labeled_lane_seq = future_path_tag.lc_path_tag[0].labeled_lane_seq
    lane_seq_linestring = LineString(
        [
            point
            for lane_id in labeled_lane_seq
            for point in lane_map[lane_id]["polyline"]
        ]
    )
    ego_point = Point(future_path[0])
    _, proj_l = get_sl(lane_seq_linestring, ego_point)
    return 1 if proj_l > 0 else 0


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

    arrive_length = min(
        [tag.arrive_length for tag in future_path_tag.lc_path_tag]
    )
    lc_await_len = future_path_tag.basic_path_tag.valid_path_len - arrive_length
    if lc_await_len <= 0:
        return lc_valid_tag

    lc_dir = judge_lc_direction(lane_map, future_path, future_path_tag)

    obs_idx = has_lateral_obstacle(
        mixed_traffic_tag,
        narrow_road_tag,
        data.label_scene.ego_path_info.corr_frame_idx,
        lc_await_len,
        lc_dir,
    )

    lc_valid_tag.is_lc_valid = lc_await_len - obs_idx < 20

    return lc_valid_tag
