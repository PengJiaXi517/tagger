import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

# import ujson
from tqdm import tqdm

from per_path_type_filter_rules import (
    valid_cross_junction_lc,
    valid_cross_junction_unknown,
    valid_cruise,
    valid_lc,
)


def filter_risk(value) -> bool:
    # Filter length < 99
    if value["path_risk"]["tag_ego_path_valid_length_risk"]:
        return True

    # Filter curb cross：
    if value["map_risk"]["tag_ego_path_curb_cross_risk"]:
        return True

    # Filter not in obstacle:
    if value["condition_risk"]["ego_car_not_in_obstacles"]:
        return True

    # # Filter no condition Lane:
    if value["condition_risk"]["no_condition_lane"]:
        return True

    # Filter cond no entry b not in junction:
    # if value["condition_risk"]["cond_no_entry_b_not_in_junction"]:
    #     return True

    if value["condition_risk"]["cond_entry_and_exit_b_not_pass_junction"]:
        return True

    if value["is_backing_up"]:
        return True

    return False


class DataNumRes:
    def __init__(self) -> None:
        self.total_num = 0

        self.num_unknown = 0
        self.num_cruise = 0
        self.num_lane_change = 0
        self.num_cross_junction_lc = 0
        self.num_cross_junction_cruise = 0
        self.num_cross_junction_unknown = 0

        self.num_valid_unknown = 0
        self.num_valid_cruise = 0
        self.num_valid_cruise_turn = 0
        self.num_valid_lane_change = 0
        self.num_valid_cross_junction_lc = 0
        self.num_valid_cross_junction_cruise = 0
        self.num_valid_cross_junction_unknown = 0

        self.valid_turn_unknown = 0
        self.valid_turn_forward = 0
        self.valid_turn_left = 0
        self.valid_turn_right = 0
        self.valid_turn_uturn = 0

        self.invalid_turn_unknown = 0
        self.invalid_turn_forward = 0
        self.invalid_turn_left = 0
        self.invalid_turn_right = 0
        self.invalid_turn_uturn = 0

    def update_path_type_num(self, type_name):
        if type_name == "UNKNOWN":
            self.num_unknown += 1
        elif type_name == "CRUISE":
            self.num_cruise += 1
        elif type_name == "LANE_CHANGE":
            self.num_lane_change += 1
        elif type_name == "CROSS_JUNCTION_LC":
            self.num_cross_junction_lc += 1
        elif type_name == "CROSS_JUNCTION_CRUISE":
            self.num_cross_junction_cruise += 1
        elif type_name == "CROSS_JUNCTION_UNKNWON":
            self.num_cross_junction_unknown += 1

    def update_valid_path_type_num(self, type_name):
        if type_name == "UNKNOWN":
            self.num_valid_unknown += 1
        elif type_name == "CRUISE":
            self.num_valid_cruise += 1
        elif type_name == "CRUISE_TURN":
            self.num_valid_cruise_turn += 1
        elif type_name == "LANE_CHANGE":
            self.num_valid_lane_change += 1
        elif type_name == "CROSS_JUNCTION_LC":
            self.num_valid_cross_junction_lc += 1
        elif type_name == "CROSS_JUNCTION_CRUISE":
            self.num_valid_cross_junction_cruise += 1
        elif type_name == "CROSS_JUNCTION_UNKNWON":
            self.num_valid_cross_junction_unknown += 1

    def add_valid_turn_num(self, tag):
        if tag["junction_path_tag"]["turn_type"] == "UNKNWON":
            self.valid_turn_unknown += 1
        elif tag["junction_path_tag"]["turn_type"] == "FORWARD":
            self.valid_turn_forward += 1
        elif tag["junction_path_tag"]["turn_type"] == "LEFT":
            self.valid_turn_left += 1
        elif tag["junction_path_tag"]["turn_type"] == "RIGHT":
            self.valid_turn_right += 1
        elif tag["junction_path_tag"]["turn_type"] == "UTURN":
            self.valid_turn_uturn += 1

    def add_invalid_turn_num(self, tag):
        if tag["junction_path_tag"]["turn_type"] == "UNKNWON":
            self.invalid_turn_unknown += 1
        elif tag["junction_path_tag"]["turn_type"] == "FORWARD":
            self.invalid_turn_forward += 1
        elif tag["junction_path_tag"]["turn_type"] == "LEFT":
            self.invalid_turn_left += 1
        elif tag["junction_path_tag"]["turn_type"] == "RIGHT":
            self.invalid_turn_right += 1
        elif tag["junction_path_tag"]["turn_type"] == "UTURN":
            self.invalid_turn_uturn += 1

    def as_dict(self):
        return {
            "total_num": self.total_num,
            "num_unknown": self.num_unknown,
            "num_cruise": self.num_cruise,
            "num_lane_change": self.num_lane_change,
            "num_cross_junction_lc": self.num_cross_junction_lc,
            "num_cross_junction_cruise": self.num_cross_junction_cruise,
            "num_cross_junction_unknown": self.num_cross_junction_unknown,
            "num_valid_unknown": self.num_valid_unknown,
            "num_valid_cruise": self.num_valid_cruise,
            "num_valid_cruise_turn": self.num_valid_cruise_turn,
            "num_valid_lane_change": self.num_valid_lane_change,
            "num_valid_cross_junction_lc": self.num_valid_cross_junction_lc,
            "num_valid_cross_junction_cruise": self.num_valid_cross_junction_cruise,
            "num_valid_cross_junction_unknown": self.num_valid_cross_junction_unknown,
            "valid_turn_unknown": self.valid_turn_unknown,
            "valid_turn_forward": self.valid_turn_forward,
            "valid_turn_left": self.valid_turn_left,
            "valid_turn_right": self.valid_turn_right,
            "valid_turn_uturn": self.valid_turn_uturn,
            "invalid_turn_unknown": self.invalid_turn_unknown,
            "invalid_turn_forward": self.invalid_turn_forward,
            "invalid_turn_left": self.invalid_turn_left,
            "invalid_turn_right": self.invalid_turn_right,
            "invalid_turn_uturn": self.invalid_turn_uturn,
        }

    def __repr__(self) -> str:
        return f"""\
TOTAL NUM: {self.total_num}\n\
UNKNOWN: {self.num_unknown}\n\
CRUISE: {self.num_cruise}\n\
LANE_CHANGE: {self.num_lane_change}\n\
CROSS_JUNCTION_LC: {self.num_cross_junction_lc}\n\
CROSS_JUNCTION_CRUISE: {self.num_cross_junction_cruise}\n\
CROSS_JUNCTION_UNKNWON: {self.num_cross_junction_unknown}\n\
\n\
UNKNOWN: {self.num_valid_unknown}\n\
CRUISE: {self.num_valid_cruise}\n\
LANE_CHANGE: {self.num_valid_lane_change}\n\
CROSS_JUNCTION_LC: {self.num_valid_cross_junction_lc}\n\
CROSS_JUNCTION_CRUISE: {self.num_valid_cross_junction_cruise}\n\
CROSS_JUNCTION_UNKNWON: {self.num_valid_cross_junction_unknown}\n\
\n\
VLIAD TURN:  {self.valid_turn_unknown}, {self.valid_turn_forward}, {self.valid_turn_left}, {self.valid_turn_right}, {self.valid_turn_uturn}\n\
INVLIAD TURN:  {self.invalid_turn_unknown}, {self.invalid_turn_forward}, {self.invalid_turn_left}, {self.invalid_turn_right}, {self.invalid_turn_uturn}\
        """


def filter_res():
    return {
        "cruise_straight_res": [],
        "cruise_turn_res": [],
        "lc_res": [],
        "UNKNWON_IN": [],
        "FORWARD_IN": [],
        "LEFT_IN": [],
        "RIGHT_IN": [],
        "UTURN_IN": [],
        "UNKNWON_OUT": [],
        "FORWARD_OUT": [],
        "LEFT_OUT": [],
        "RIGHT_OUT": [],
        "UTURN_OUT": [],
    }


def process_task(tasks, root_dir, save_path, rank=0):

    data_num_res = DataNumRes()

    res = defaultdict(filter_res)
    calculate_res = defaultdict(filter_res)

    for task in tqdm(
        tasks,
        nrows=5,
        position=rank,
        leave=False,
    ):
        if os.path.exists(os.path.join(root_dir, task, "tag.json")):
            with open(os.path.join(root_dir, task, "tag.json"), "r") as f:
                tag = json.load(f)

            for _, value in tag.items():

                if "path_risk" not in value:
                    continue

                data_num_res.total_num += 1

                if filter_risk(value):
                    continue

                path_type = value["path_type"]

                data_num_res.update_path_type_num(path_type)

                labeled_lane_seq = []
                labeled_exit_lane_seq = []

                vel = np.linalg.norm(
                    [
                        value["ego_state_speed"]["ego_speed_odom_x"],
                        value["ego_state_speed"]["ego_speed_odom_y"],
                    ]
                )

                if path_type == "CRUISE" and valid_cruise(
                    value, labeled_lane_seq, labeled_exit_lane_seq
                ):
                    if np.abs(
                        value["basic_path_tag"]["sum_path_curvature"]
                    ) > 0.038 and (
                        np.abs(
                            value["basic_path_tag"]["sum_path_curvature"]
                            - value["basic_path_tag"]["pos_sum_path_curvature"]
                        )
                        < 0.005
                        or np.abs(
                            value["basic_path_tag"]["sum_path_curvature"]
                            - value["basic_path_tag"]["neg_sum_path_curvature"]
                        )
                        < 0.005
                    ):
                        if any(
                            [
                                cruise_path_tag["mean_pose_l"] > 0.4
                                for cruise_path_tag in value["cruise_path_tag"]
                            ]
                        ):
                            continue

                        res[value["data_root"]]["cruise_turn_res"].append(
                            value["file_path"]
                        )
                        calculate_res[value["data_root"]]["cruise_turn_res"].append(vel)
                        data_num_res.update_valid_path_type_num("CRUISE_TURN")
                    else:
                        res[value["data_root"]]["cruise_straight_res"].append(
                            value["file_path"]
                        )
                        calculate_res[value["data_root"]]["cruise_straight_res"].append(
                            vel
                        )
                        data_num_res.update_valid_path_type_num(value["path_type"])
                    continue
                elif path_type == "LANE_CHANGE" and valid_lc(
                    value, labeled_lane_seq, labeled_exit_lane_seq
                ):
                    res[value["data_root"]]["lc_res"].append(value["file_path"])
                    calculate_res[value["data_root"]]["lc_res"].append(vel)
                    data_num_res.update_valid_path_type_num(value["path_type"])
                    continue
                elif path_type in [
                    "CROSS_JUNCTION_LC",
                    "CROSS_JUNCTION_CRUISE",
                ]:
                    if valid_cross_junction_lc(
                        value, labeled_lane_seq, labeled_exit_lane_seq
                    ):
                        data_num_res.add_valid_turn_num(value)
                        res[value["data_root"]][
                            f'{value["junction_path_tag"]["turn_type"]}_OUT'
                        ].append(value["file_path"])
                        calculate_res[value["data_root"]][
                            f'{value["junction_path_tag"]["turn_type"]}_OUT'
                        ].append(vel)
                        data_num_res.update_valid_path_type_num(value["path_type"])
                    else:
                        data_num_res.add_invalid_turn_num(value)
                elif path_type in [
                    "CROSS_JUNCTION_UNKNWON",
                ]:
                    if valid_cross_junction_unknown(
                        value, labeled_lane_seq, labeled_exit_lane_seq
                    ):
                        data_num_res.add_valid_turn_num(value)
                        res[value["data_root"]][
                            f'{value["junction_path_tag"]["turn_type"]}_IN'
                        ].append(value["file_path"])
                        calculate_res[value["data_root"]][
                            f'{value["junction_path_tag"]["turn_type"]}_IN'
                        ].append(vel)
                        data_num_res.update_valid_path_type_num(value["path_type"])
                    else:
                        data_num_res.add_invalid_turn_num(value)

    with open(
        os.path.join(
            save_path,
            f"filter_res_{rank}.json",
        ),
        "w",
    ) as f:
        json.dump(res, f)

    with open(
        os.path.join(
            save_path,
            f"calculate_res_{rank}.json",
        ),
        "w",
    ) as f:
        json.dump(calculate_res, f)

    with open(
        os.path.join(
            save_path,
            f"log_{rank}.json",
        ),
        "w",
    ) as f:
        json.dump(data_num_res.as_dict(), f)


if __name__ == "__main__":
    root_dir = "/mnt/train2/eric.wang/path_tag_res/0801/"

    with open(
        "/mnt/openpai-team/ness.hu/LocalMapLabeling/path-nn-tagger/valid_sets/valid_train_sets.txt",
        "r",
    ) as f:
        valid_tasks = [t.strip("\n") for t in f.readlines()]

    tasks = [
        t for t in os.listdir(root_dir) if not t.startswith("log") if t in valid_tasks
    ]

    num_process = 50
    with Pool(num_process) as pool:
        res = []
        for i in range(num_process):
            res.append(
                pool.apply_async(
                    process_task,
                    args=(
                        tasks[i::num_process],
                        root_dir,
                        "./filter_res_new_2",
                        i,
                    ),
                )
            )

        for r in res:
            print(r.get())

        pool.close()
        pool.join()
