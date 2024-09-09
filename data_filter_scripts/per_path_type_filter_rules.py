import os
from typing import Dict, List

import numpy as np


def valid_cruise(tag, labeled_lane_seq: List, labeled_exit_lane_seq: List):

    for cruise_path_tag in tag["cruise_path_tag"]:

        real_pose_l = cruise_path_tag["real_pose_l"]
        percep_pose_l = cruise_path_tag["percep_pose_l"]

        if np.abs(real_pose_l - percep_pose_l) > 0.3:
            continue

        if np.abs(real_pose_l) < 0.3:
            labeled_lane_seq.append(cruise_path_tag["labeled_lane_seq"])
            labeled_exit_lane_seq.append([])

    if len(labeled_lane_seq) == 0:
        return False

    return True


def valid_lc(tag, labeled_lane_seq: List, labeled_exit_lane_seq: List):
    for lc_path_tag in tag["lc_path_tag"]:
        arrive_final_pose_l = lc_path_tag["arrive_final_pose_l"]
        arrive_percep_pose_l = lc_path_tag["arrive_percep_pose_l"]

        if np.abs(arrive_final_pose_l - arrive_percep_pose_l) > 0.3:
            continue

        if np.abs(arrive_final_pose_l) > 0.4:
            continue

        arrive_length = lc_path_tag["arrive_length"]

        if arrive_length < 57:
            continue

        labeled_lane_seq.append(lc_path_tag["labeled_lane_seq"])
        labeled_exit_lane_seq.append([])

    if len(labeled_lane_seq) == 0:
        return False

    return True


def valid_cross_junction_lc(tag, labeled_lane_seq: List, labeled_exit_lane_seq: List):
    junction_path_tag = tag["junction_path_tag"]

    if (
        junction_path_tag["first_arrive_junction_id"]
        != junction_path_tag["label_junction_id"]
    ):
        return False

    if not junction_path_tag["has_entry_lane"]:
        return False
    if not junction_path_tag["has_real_arrive_entry_lane"]:
        return False

    if (
        np.abs(
            junction_path_tag["real_lane_entry_pose_l"]
            - junction_path_tag["percep_lane_entry_pose_l"]
        )
        > 0.3
    ):
        return False
    if np.abs(junction_path_tag["percep_lane_entry_pose_l"]) > 0.4:
        return False

    if junction_path_tag["has_real_arrive_exit_lane"]:
        if (
            np.abs(junction_path_tag["percep_lane_exit_pose_l"]) > 0.4
            and junction_path_tag["mean_pose_l_2_exit_lane"] > 0.4
        ):
            return False

        if (
            np.abs(
                junction_path_tag["percep_lane_exit_pose_l"]
                - junction_path_tag["real_lane_exit_pose_l"]
            )
            > 0.4
        ):
            return False

        if junction_path_tag["max_length_not_in_exit_lane"] > 3.0:
            return False

        if junction_path_tag["mean_pose_l_2_exit_lane"] > 0.5:
            return False

    labeled_lane_seq.append([junction_path_tag["label_entry_lane_id"]])
    labeled_exit_lane_seq.append([junction_path_tag["label_exit_lane_id"]])

    return True


def valid_cross_junction_unknown(
    tag, labeled_lane_seq: List, labeled_exit_lane_seq: List
):
    junction_path_tag = tag["junction_path_tag"]

    if not junction_path_tag["has_junction_label"]:
        return False

    if (
        junction_path_tag["first_arrive_junction_id"]
        != junction_path_tag["label_junction_id"]
    ):
        return False

    if (
        np.abs(junction_path_tag["percep_lane_exit_pose_l"]) > 0.4
        and junction_path_tag["mean_pose_l_2_exit_lane"] > 0.4
    ):
        return False

    if junction_path_tag["has_real_arrive_exit_lane"]:
        if (
            np.abs(
                junction_path_tag["real_lane_exit_pose_l"]
                - junction_path_tag["percep_lane_exit_pose_l"]
            )
            > 0.4
        ):
            return False

        if junction_path_tag["max_length_not_in_exit_lane"] > 3.0:
            return False

        if junction_path_tag["mean_pose_l_2_exit_lane"] > 0.5:
            return False

    labeled_lane_seq.append([])
    labeled_exit_lane_seq.append([junction_path_tag["label_exit_lane_id"]])

    return True
