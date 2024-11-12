from typing import Dict, List, Tuple
from base import TagData
from tag_functions.high_value_scene.common.tag_type import (
    RampTag,
)
from tag_functions.high_value_scene.hv_utils.ramp_tag_helper import (
    RampTagHelper,
)
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)
from tag_functions.high_value_scene.common.basic_info import (
    BasicInfo,
)


def label_ramp_tag(data: TagData, basic_info: BasicInfo) -> RampTag:
    ramp_tag = RampTag()
    ramp_tag_helper = RampTagHelper(
        enter_fork_consider_len=30,
        exit_fork_consider_len=50,
        large_dist_th=10.0,
        large_dist_num_th=3,
        curb_roi_s_min=-5.0,
        curb_roi_s_max=100.0,
        curb_roi_l_max=5.0,
    )
    obs_filter = ObstacleFilter(
        filter_obs_max_l=5.0, front_vehicle_rel_x=10.0, front_vehicle_rel_y=0.5
    )

    lane_map = data.label_scene.percepmap.lane_map
    current_lanes = data.label_scene.ego_obs_lane_seq_info.current_lanes
    current_lane_seqs = data.label_scene.ego_obs_lane_seq_info.current_lane_seqs
    curb_decision = data.label_scene.label_res["curb_label"].get(
        "decision", None
    )
    obstacles = data.label_scene.obstacles
    ego_point = obs_filter.get_ego_point(obstacles[-9])

    if basic_info.is_cross_junction:
        return ramp_tag

    if curb_decision is None or not ramp_tag_helper.lane_seq_validity_check(
        lane_map, current_lanes, current_lane_seqs
    ):
        return ramp_tag

    # 判断进匝道, cruise场景下lane一分为二，且有curb隔开
    if ramp_tag_helper.enter_ramp_cruise(
        lane_map, current_lanes, curb_decision, basic_info
    ):
        ramp_tag.is_enter_ramp = True

    # 判断出匝道: lane二合成一，且分叉的两条lane满足距离阈值
    elif ramp_tag_helper.exit_ramp(lane_map, current_lanes, ego_point):
        ramp_tag.is_exit_ramp = True

    return ramp_tag
