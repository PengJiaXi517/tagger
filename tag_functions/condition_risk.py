from dataclasses import dataclass, field
from typing import Dict

from base import TagData
from registry import TAG_FUNCTIONS


@dataclass(repr=False)
class ConditionRiskTag:
    ego_car_not_in_obstacles: bool = False
    no_condition_lane: bool = False  # == 0
    multi_condition_lane: bool = False  # >= 3
    condition_num: int = 0
    cond_no_entry_b_not_in_junction: bool = (
        False  # wo entry cond and adc not in junction
    )
    cond_entry_and_exit_b_not_pass_junction: bool = False
    num_pass_junction_w_cond_entry_and_exit: int = -1

    def as_dict(self):
        return {
            "condition_risk": {
                "ego_car_not_in_obstacles": self.ego_car_not_in_obstacles,
                "no_condition_lane": self.no_condition_lane,
                "multi_condition_lane": self.multi_condition_lane,
                "condition_num": self.condition_num,
                "cond_no_entry_b_not_in_junction": self.cond_no_entry_b_not_in_junction,
                "cond_entry_and_exit_b_not_pass_junction": self.cond_entry_and_exit_b_not_pass_junction,
                "num_pass_junction_w_cond_entry_and_exit": self.num_pass_junction_w_cond_entry_and_exit,
            }
        }


@TAG_FUNCTIONS.register()
def condition_risk_check(data: TagData, params: Dict) -> Dict:

    condition_risk_tag = ConditionRiskTag()

    # Check no condition case:
    condition_risk_tag.no_condition_lane = (
        True if len(data.condition_res.lane_seq_pair) == 0 else False
    )

    # Check multi_condition_lane
    condition_risk_tag.multi_condition_lane = (
        True if len(data.condition_res.lane_seq_pair) >= 3 else False
    )

    # Fill condition num
    condition_risk_tag.condition_num = len(data.condition_res.lane_seq_pair)

    # Check if cond exit but no entry w not in junction
    if any(
        [
            (s_ind == -1 and e_ind != -1)
            for s_ind, e_ind, _ in data.condition_res.lane_seq_pair
        ]
    ):
        if data.label_scene.ego_path_info.in_junction_id[0] is None:
            condition_risk_tag.cond_no_entry_b_not_in_junction = True

    # Check if cond entry and exit
    if any(
        [
            (s_ind != -1 and e_ind != -1)
            for s_ind, e_ind, _ in data.condition_res.lane_seq_pair
        ]
    ):
        junction_info = data.label_scene.obstacles[-9].get(
            "junction_info", dict(in_junction=False, exit_lanes=[])
        )

        if len(junction_info["exit_lanes"]) == 0:
            condition_risk_tag.cond_entry_and_exit_b_not_pass_junction = True
        else:
            in_junction_id = junction_info["junction_id"]
            count_in_junction = 0
            for (
                curr_path_in_junction_id
            ) in data.label_scene.ego_path_info.in_junction_id:
                if (
                    curr_path_in_junction_id is not None
                    and curr_path_in_junction_id == in_junction_id
                ):
                    count_in_junction += 1
            if count_in_junction == 0:
                condition_risk_tag.cond_entry_and_exit_b_not_pass_junction = True
            else:
                condition_risk_tag.num_pass_junction_w_cond_entry_and_exit = (
                    count_in_junction
                )

    return condition_risk_tag.as_dict()
