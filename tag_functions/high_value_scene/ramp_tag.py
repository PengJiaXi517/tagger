from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from base import TagData
from registry import TAG_FUNCTIONS
from tag_functions.high_value_scene.hv_utils.ramp_judge import RampJudge


@dataclass(repr=False)
class RampTag:
    def __init__(self) -> None:
        self.is_enter_ramp: bool = False
        self.is_exit_ramp: bool = False

    def as_dict(self):
        return {
            "is_enter_ramp": self.is_enter_ramp,
            "is_exit_ramp": self.is_exit_ramp,
        }


# 判断汇入汇出匝道
@TAG_FUNCTIONS.register()
def ramp_tag(data: TagData, params: Dict) -> Dict:
    ramp_tag = RampTag()
    obstacles = data.label_scene.obstacles
    lane_map = data.label_scene.percepmap.lane_map
    current_lanes = data.label_scene.ego_obs_lane_seq_info.current_lanes
    current_lane_seqs = data.label_scene.ego_obs_lane_seq_info.current_lane_seqs
    nearby_lane_seqs = data.label_scene.ego_obs_lane_seq_info.nearby_lane_seqs
    ramp_judge = RampJudge()

    if not ramp_judge.pre_check(lane_map, current_lanes, current_lane_seqs):
        return ramp_tag.as_dict()

    ego_x = obstacles[-9]["features"]["history_states"][-1]["x"]
    ego_y = obstacles[-9]["features"]["history_states"][-1]["y"]
    ego_point = Point([ego_x, ego_y])

    has_fork = False
    # 判断是否为进匝道: 一分成二的情况
    pred_ids = lane_map[current_lanes[0]]["predecessor_id"]
    if len(pred_ids) == 1:
        succ_ids = lane_map[pred_ids[0]]["successor_id"]
        if len(succ_ids) == 2 and current_lanes[0] in succ_ids:
            has_fork = True
            adjacent_lane_id = (
                succ_ids[0] if succ_ids[0] != current_lanes[0] else succ_ids[1]
            )
            if ramp_judge.judge_enter_ramp(
                lane_map, current_lanes[0], adjacent_lane_id, ego_point
            ):
                ramp_tag.is_enter_ramp = True
                return ramp_tag.as_dict()

    # 判断是否为进匝道: lane无分叉点的情况
    if not has_fork:
        nearest_lane_id = ramp_judge.get_nearest_lane_id(
            lane_map, nearby_lane_seqs, ego_point, current_lane_seqs
        )
        if nearest_lane_id is not None:
            if ramp_judge.judge_enter_ramp(
                lane_map, current_lanes[0], nearest_lane_id, ego_point
            ):
                ramp_tag.is_enter_ramp = True
                return ramp_tag.as_dict()

    has_merge = False
    # 判断是否为出匝道: 二合成一的情况
    succ_ids = lane_map[current_lanes[0]]["successor_id"]
    if len(succ_ids) == 1:
        pred_ids = lane_map[succ_ids[0]]["predecessor_id"]
        if len(pred_ids) == 2 and current_lanes[0] in pred_ids:
            has_merge = True
            adjacent_lane_id = (
                pred_ids[0] if pred_ids[0] != current_lanes[0] else pred_ids[1]
            )
            if ramp_judge.judge_exit_ramp(
                lane_map, current_lanes[0], adjacent_lane_id, ego_point
            ):
                ramp_tag.is_exit_ramp = True
                return ramp_tag.as_dict()

    # 判断是否为出匝道: lane无分叉点的情况
    if not has_merge:
        if len(current_lane_seqs) == 1 and len(current_lane_seqs[0]) == 2:
            # 出口位置的大致坐标
            target_point = Point(
                lane_map[current_lane_seqs[0][0]]["polyline"][-1]
            )
            # 找到出口处的最近车道
            nearest_lane_id = ramp_judge.get_nearest_lane_id(
                lane_map,
                data.condition_res.seq_lane_ids_raw,
                target_point,
                current_lane_seqs,
            )
            if nearest_lane_id is not None:
                if ramp_judge.judge_exit_ramp(
                    lane_map, current_lanes[0], nearest_lane_id, ego_point
                ):
                    ramp_tag.is_exit_ramp = True
                    return ramp_tag.as_dict()

    return ramp_tag.as_dict()
