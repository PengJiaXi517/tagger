from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from shapely.geometry import LineString, Point, Polygon
from base import TagData
import numpy as np
from tag_functions.high_value_scene.narrow_road_tag import *
from tag_functions.high_value_scene.mixed_traffic_tag import *
from tag_functions.high_value_scene.hv_utils.basic_func import (
    get_sl,
)


@dataclass(repr=False)
class LateralDeviationTag:
    def __init__(self, future_path) -> None:
        self.corrected_path = [[coord for coord in pt] for pt in future_path]

    def as_dict(self):
        return {
            "corrected_path": self.corrected_path,
        }


def is_near_obstacle(
    mixed_traffic_tag, narrow_road_tag, corr_frame_idx, future_path_idx, is_left
):
    range = 15
    future_time_idx = corr_frame_idx[future_path_idx]
    future_narrow_road = narrow_road_tag.future_narrow_road
    future_mixed_traffic = mixed_traffic_tag.future_mixed_traffic
    consider_obs_static = future_narrow_road[
        max(0, future_path_idx - range) : min(
            len(future_narrow_road), future_path_idx + range
        )
    ]
    consider_obs_moving = future_mixed_traffic[
        max(0, future_time_idx - range) : min(
            len(future_mixed_traffic), future_time_idx + range
        )
    ]

    if any(obj[is_left] for obj in consider_obs_static):
        return True
    if any(obj[is_left] for obj in consider_obs_moving):
        return True

    return False


def lane_seq_linestring(lane_map, seq_lane_ids, start_ind, end_ind):
    candidate_linestring: List[LineString] = []
    for ind in [start_ind, end_ind]:
        if ind != -1:
            linestring = []
            for lane_id in seq_lane_ids[ind]:
                lane = lane_map.get(lane_id, None)
                if lane is not None:
                    linestring += lane["polyline"]
            if len(linestring) > 2:
                candidate_linestring.append(LineString(linestring))
    return candidate_linestring


def get_nearest_lane_seq(
    future_path, lane_map, lane_seq_pair, seq_lane_ids, valid_ind
):
    min_lat_dist = 1e6
    nearest_linestring = None
    for ind in valid_ind:
        start_ind, end_ind, _ = lane_seq_pair[ind]
        candidate_linestring = lane_seq_linestring(
            lane_map, seq_lane_ids, start_ind, end_ind
        )
        sum_proj_l = 0
        for point in future_path:
            path_point = Point(point)
            for cand_linestring in candidate_linestring:
                proj_s, proj_l = get_sl(cand_linestring, path_point)
                if proj_s is not None or proj_l is not None:
                    sum_proj_l += abs(proj_l)
                    break
        if 0 < sum_proj_l < 1000 and sum_proj_l < min_lat_dist:
            min_lat_dist = sum_proj_l
            nearest_linestring = candidate_linestring
    return nearest_linestring


def lane_adsorb_considering_obs(
    future_path,
    corr_frame_idx,
    nearest_lane_seq,
    mixed_traffic_tag,
    narrow_road_tag,
):
    corrected_path = [[coord for coord in pt] for pt in future_path]
    for idx, point in enumerate(future_path):
        path_point = Point(point)
        for linestring in nearest_lane_seq:
            proj_s, proj_l = get_sl(linestring, path_point)
            if proj_s is None or abs(proj_l) > 1:
                continue

            is_lat_deviation = (
                proj_l > 0
                and is_near_obstacle(
                    mixed_traffic_tag,
                    narrow_road_tag,
                    corr_frame_idx,
                    idx,
                    1,
                )
            ) or (
                proj_l < 0
                and is_near_obstacle(
                    mixed_traffic_tag,
                    narrow_road_tag,
                    corr_frame_idx,
                    idx,
                    0,
                )
            )

            if not is_lat_deviation:
                corr_point = linestring.interpolate(proj_s)
                vec = np.array(
                    [corr_point.x - path_point.x, corr_point.y - path_point.y]
                )
                ratio = (1 - (abs(proj_l) - 0.25) / 0.75) ** 2
                corrected_path[idx] = (
                    [corr_point.x, corr_point.y]
                    if abs(proj_l) < 0.25
                    else [point[0] + vec[0] * ratio, point[1] + vec[1] * ratio]
                )
            break
    return corrected_path


def lateral_deviation_tag(
    data: TagData,
    mixed_traffic_tag: MixedTrafficTag,
    narrow_road_tag: NarrowRoadTag,
    params: Dict,
) -> Dict:
    future_path = data.label_scene.ego_path_info.future_path
    corr_frame_idx = data.label_scene.ego_path_info.corr_frame_idx
    lane_seq_pair = data.condition_res.lane_seq_pair
    seq_lane_ids = data.condition_res.seq_lane_ids
    lane_map = data.label_scene.percepmap.lane_map
    lateral_deviation_tag = LateralDeviationTag(future_path)

    valid_ind = [
        i
        for i in range(len(lane_seq_pair))
        if lane_seq_pair[i][0] != -1 or lane_seq_pair[i][1] != -1
    ]
    if len(valid_ind) == 0:
        return lateral_deviation_tag

    nearest_lane_seq = get_nearest_lane_seq(
        future_path, lane_map, lane_seq_pair, seq_lane_ids, valid_ind
    )
    if nearest_lane_seq is None:
        return lateral_deviation_tag

    lateral_deviation_tag.corrected_path = lane_adsorb_considering_obs(
        future_path,
        corr_frame_idx,
        nearest_lane_seq,
        mixed_traffic_tag,
        narrow_road_tag,
    )
    return lateral_deviation_tag
