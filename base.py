import os
import pickle
from typing import List, Set, Tuple

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
