from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from base import TagData
from registry import TAG_FUNCTIONS
from tag_functions.high_value_scene.hv_utils.ramp_judge import RampJudge
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    valid_check,
)


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
    ramp_judge = RampJudge()
    obs_filter = ObstacleFilter()

    lane_map = data.label_scene.percepmap.lane_map
    current_lanes = data.label_scene.ego_obs_lane_seq_info.current_lanes
    current_lane_seqs = data.label_scene.ego_obs_lane_seq_info.current_lane_seqs
    ego_path_info = data.label_scene.ego_path_info
    curb_decision = data.label_scene.label_res["curb_label"].get(
        "decision", None
    )
    obstacles = data.label_scene.obstacles

    ego_point = obs_filter.get_ego_point(obstacles)

    if (
        curb_decision is None
        or not valid_check(data)
        or not ramp_judge.pre_check(lane_map, current_lanes, current_lane_seqs)
    ):
        return ramp_tag.as_dict()

    # 判断进匝道, 情况一: cruise场景下lane一分为二，且有curb隔开
    # 情况二: lc场景下，当前车道和目标车道间有curb隔开
    if ramp_judge.enter_ramp_cruise(
        lane_map, current_lanes, curb_decision
    ) or ramp_judge.enter_ramp_lane_change(
        lane_map, current_lanes, curb_decision, current_lane_seqs, ego_path_info
    ):
        ramp_tag.is_enter_ramp = True
    # 判断出匝道: lane二合成一，且分叉的两条lane满足距离阈值
    elif ramp_judge.exit_ramp(lane_map, current_lanes, ego_point):
        ramp_tag.is_exit_ramp = True

    return ramp_tag.as_dict()
