import os
import pickle
from typing import Dict, List, Set, Tuple, Union

import numpy as np


class ConditionRes:
    def __init__(self, file_path: os.PathLike) -> None:
        with open(file_path, "rb") as f:
            condition_res = pickle.load(f)

        self.file_path = condition_res["file_path"]
        self.seq_lane_ids: List[Set[int]] = condition_res["seq_lane_ids"]
        self.seq_lane_ids_raw: List[List[int]] = condition_res["seq_lane_ids_raw"]

        start_lane_seqs_ind = condition_res["start_lane_seqs_ind"]
        end_lane_seqs_ind = condition_res["end_lane_seqs_ind"]
        ego_per_lane_seq_mask = condition_res["ego_per_lane_seq_path_mask"]

        self.lane_seq_pair: List[Tuple[int, int, np.array]] = []
        for idx, (start_idx, end_idx) in enumerate(
            zip(start_lane_seqs_ind, end_lane_seqs_ind)
        ):
            if start_idx == -1 and end_idx == -1:
                continue
            self.lane_seq_pair.append((start_idx, end_idx, ego_per_lane_seq_mask[idx]))

    def __repr__(self) -> str:
        return (
            f"{self.file_path}, {len(self.seq_lane_ids_raw)}, {len(self.lane_seq_pair)}"
        )


class EgoPathInfo:
    def __init__(self, ego_path_dict: Dict) -> None:
        self.future_path: List[Tuple[float, float]] = ego_path_dict["future_path"]
        self.in_junction_id: List[Union[None, str, int]] = ego_path_dict[
            "in_junction_id"
        ]
        self.corr_frame_idx: List[Union[None, int]] = ego_path_dict["corr_frame_idx"]
        self.corr_lane_id: List[List[Union[None, Tuple[Union[str, int], float]]]] = (
            ego_path_dict["corr_lane_id"]
        )
        self.vt_profile: List[float] = ego_path_dict["vt_profile"]
        self.lane_seq_valid_len: List[Tuple[List[str], int, int]] = ego_path_dict[
            "lane_seq_valid_len"
        ]
        self.collision_to_curb: bool = ego_path_dict["collision_to_curb"]


class PercepMap:
    def __init__(self, percep_map_dict: Dict) -> None:
        self.lanes: List[Dict] = []
        self.lane_map: Dict[int, Dict] = {}
        self.junctions: List[Dict] = []
        self.junction_map: Dict[int, Dict] = {}
        self.curbs: List[np.array] = []

        for lane in percep_map_dict["lanes"]:
            self.lanes.append(lane)
            self.lane_map[lane["id"]] = lane

        for junction in percep_map_dict["junctions"]:
            self.junctions.append(junction)
            self.junction_map[junction["id"]] = junction

        for curb in percep_map_dict["curbs"]:
            self.curbs.append(np.array(curb))


class LabelScene:
    def __init__(self, label_path: os.PathLike) -> None:
        with open(label_path, "rb") as f:
            label = pickle.load(f)

        self.ego_path_info: EgoPathInfo = EgoPathInfo(label["ego_path_info"])
        self.percepmap: Dict = PercepMap(label["percepmap"])
        self.obstacles: Dict = label["obstacles"]
