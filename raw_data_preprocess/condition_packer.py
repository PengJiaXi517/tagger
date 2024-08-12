import time
from bisect import bisect_right
from random import shuffle
from typing import Any, Dict, List, Set, Tuple, Union

import networkx as nx
import numpy as np
from shapely import LineString, Point
from shapely.ops import unary_union

from .utils import get_dict_data, write_dict_data


class EgoPathPacker2LaneSeq:
    def __init__(
        self,
        src_obs_key: str = "obstacles",
        src_map_key: str = "percepmap",
        src_obs_id_map_key: Union[Tuple, str] = ("obstacle_state", "id_map"),
        src_ego_curr_state: Union[Tuple, str] = ("obstacle_state", "ego_curr_state"),
        src_map_id_map_key: Union[Tuple, str] = ("map_state", "id_map"),
        src_map_seqs_lane_id_key: Union[Tuple, str] = ("map_state", "seq_lane_ids"),
        src_rot_aug_state_key: Union[Tuple, str] = ("obstacle_state", "aug_dict"),
        src_lane_seqs_prob_key: Union[Tuple, str] = (
            "obstacle_state",
            "lane_segs_prob",
        ),
        src_ego_path_key: Union[Tuple, str] = ("ego_path_info", "future_path"),
        src_vt_parofile_key: Union[Tuple, str] = ("ego_path_info", "vt_profile"),
        src_lane_seq_valid_key: Union[Tuple, str] = (
            "ego_path_info",
            "lane_seq_valid_len",
        ),
        src_in_junction_id_key: Union[Tuple, str] = ("ego_path_info", "in_junction_id"),
        src_corr_lane_id_key: Union[Tuple, str] = ("ego_path_info", "corr_lane_id"),
        tgt_ego_path_key: Union[Tuple, str] = (
            "obstacle_state",
            "ego_path",
        ),
        tgt_ego_path_weights_key: Union[Tuple, str] = (
            "obstacle_state",
            "ego_path_weights",
        ),
        tgt_ego_path_mask_key: Union[Tuple, str] = (
            "obstacle_state",
            "ego_path_mask",
        ),
        tgt_ego_future_vt_profile_key: Union[Tuple, str] = (
            "obstacle_state",
            "ego_vt_profile",
        ),
        tgt_ego_future_vt_profile_mask_key: Union[Tuple, str] = (
            "obstacle_state",
            "ego_vt_profile_mask",
        ),
        tgt_ego_start_lane_seqs_ind_key: Union[Tuple, str] = (
            "obstacle_state",
            "start_lane_seqs_ind",
        ),
        tgt_ego_end_lane_seqs_ind_key: Union[Tuple, str] = (
            "obstacle_state",
            "end_lane_seqs_ind",
        ),
        tgt_ego_per_lane_seq_path_mask_key: Union[Tuple, str] = (
            "obstacle_state",
            "ego_per_lane_seq_path_mask",
        ),
        tgt_ego_per_lane_seq_vt_profile_mask_key: Union[Tuple, str] = (
            "obstacle_state",
            "ego_per_lane_seq_vt_profile_mask",
        ),
        lane_seq_num: int = 48,
        sample_distance: float = 3.0,
        max_path_sample_points: int = 34,
        max_vt_profile_sample_point: int = 50,
        max_start_lane_seq_num: int = 4,
        max_end_lane_seq_num: int = 4,
        use_distance_coe: bool = True,
        use_curvature_coe: bool = False,
        use_new_curvature_coe: bool = True,
        use_min_entry_and_exit: bool = True,
        use_longest_condition_lane_only: bool = True,
        use_lane_seq_clean: bool = True,
        ignore_label: int = 255,
    ) -> None:
        self.src_obs_key = src_obs_key
        self.src_map_key = src_map_key
        self.src_obs_id_map_key = src_obs_id_map_key
        self.src_ego_curr_state = src_ego_curr_state
        self.src_map_id_map_key = src_map_id_map_key
        self.src_map_seqs_lane_id_key = src_map_seqs_lane_id_key
        self.src_rot_aug_state_key = src_rot_aug_state_key
        self.src_lane_seqs_prob_key = src_lane_seqs_prob_key

        self.src_ego_path_key = src_ego_path_key
        self.src_vt_parofile_key = src_vt_parofile_key
        self.src_lane_seq_valid_key = src_lane_seq_valid_key
        self.src_in_junction_id_key = src_in_junction_id_key
        self.src_corr_lane_id_key = src_corr_lane_id_key

        self.tgt_ego_future_vt_profile_key = tgt_ego_future_vt_profile_key
        self.tgt_ego_future_vt_profile_mask_key = tgt_ego_future_vt_profile_mask_key

        self.tgt_ego_path_key = tgt_ego_path_key
        self.tgt_ego_path_weights_key = tgt_ego_path_weights_key
        self.tgt_ego_path_mask_key = tgt_ego_path_mask_key
        self.tgt_ego_start_lane_seqs_ind_key = tgt_ego_start_lane_seqs_ind_key
        self.tgt_ego_end_lane_seqs_ind_key = tgt_ego_end_lane_seqs_ind_key

        self.tgt_ego_per_lane_seq_path_mask_key = tgt_ego_per_lane_seq_path_mask_key
        self.tgt_ego_per_lane_seq_vt_profile_mask_key = (
            tgt_ego_per_lane_seq_vt_profile_mask_key
        )

        self.lane_seq_num = lane_seq_num

        self.sample_distance = sample_distance
        self.max_path_sample_points = max_path_sample_points
        self.max_vt_profile_sample_point = max_vt_profile_sample_point

        self.max_start_lane_seq_num = max_start_lane_seq_num
        self.max_end_lane_seq_num = max_end_lane_seq_num

        self.use_distance_coe = use_distance_coe
        self.use_curvature_coe = use_curvature_coe
        self.use_new_curvature_coe = use_new_curvature_coe
        self.use_min_entry_and_exit = use_min_entry_and_exit
        self.use_longest_condition_lane_only = use_longest_condition_lane_only
        self.use_lane_seq_clean = use_lane_seq_clean

        self.ignore_label = ignore_label

    def transform_to_curr(self, ego_curr_state, ego_path, rot_aug_state):
        rot_theta = rot_aug_state["rot_theta"]
        trans_vec = rot_aug_state["trans_vec"]

        rot_mat = np.array(
            [
                [
                    np.cos(rot_theta),
                    -np.sin(rot_theta),
                ],
                [
                    np.sin(rot_theta),
                    np.cos(rot_theta),
                ],
            ]
        ).T
        t_vec = -np.array([ego_curr_state["x"], ego_curr_state["y"]]) @ rot_mat.T
        ego_path[..., :2] = ego_path[..., :2] @ rot_mat.T + t_vec + trans_vec
        # ego_path[..., 2:4] = ego_path[..., 2:4] @ rot_mat.T

        return ego_path

    def match_lane_seqs(self, lane_seg_ids: Set, all_lane_seqs: List[Set]):
        max_match = 0
        curr_match_ids = []
        for idx, lane_seq in enumerate(all_lane_seqs):
            match_nums = len(lane_seg_ids & lane_seq)
            if match_nums > max_match:
                max_match = match_nums
                curr_match_ids = [idx]
            elif match_nums == max_match and max_match > 0:
                curr_match_ids.append(idx)
        return curr_match_ids

    def pack_ego_path(self, obstacles, ego_curr_state, aug_dict, ego_path_raw):
        ego_path = np.zeros((self.max_path_sample_points, 2), dtype=np.float32)
        ego_path_mask = np.zeros((self.max_path_sample_points, 1), dtype=np.float32)

        vt_profile = np.zeros((self.max_vt_profile_sample_point, 1), dtype=np.float32)
        vt_profile_mask = np.zeros(
            (self.max_vt_profile_sample_point, 1), dtype=np.float32
        )

        valid_len = len(ego_path_raw)

        future_states = obstacles[-9]["future_trajectory"]["future_states"]

        for i in range(min(valid_len, self.max_path_sample_points)):
            # TODO(Ness): resample path
            ego_path[i, 0] = ego_path_raw[i][0]
            ego_path[i, 1] = ego_path_raw[i][1]
            ego_path_mask[i] = 1

        ego_path = self.transform_to_curr(ego_curr_state, ego_path, aug_dict)
        ego_path[ego_path_mask[:, 0] < 0.5] = 0.0

        for i in range(min(len(future_states), self.max_vt_profile_sample_point)):
            vt_profile[i] = np.linalg.norm(
                [future_states[i]["vx"], future_states[i]["vy"]]
            )
            vt_profile_mask[i] = 1

        return ego_path, ego_path_mask, vt_profile, vt_profile_mask

    def pack_ego_path_condition_corr_junction(
        self, junction_info, in_junction_info, seq_lane_ids
    ):

        seq_lane_ids_sets = [set(lane_ids) for lane_ids in seq_lane_ids]

        start_ind = (
            np.ones(
                (self.max_start_lane_seq_num * self.max_end_lane_seq_num),
                dtype=np.int64,
            )
            * -1
        )
        end_ind = (
            np.ones(
                (self.max_start_lane_seq_num * self.max_end_lane_seq_num),
                dtype=np.int64,
            )
            * -1
        )

        valid_mask = np.zeros(
            (self.max_start_lane_seq_num * self.max_end_lane_seq_num), dtype=np.float32
        )

        corr_path_mask = np.zeros(
            (
                self.max_start_lane_seq_num * self.max_end_lane_seq_num,
                self.max_path_sample_points,
            ),
            dtype=np.float32,
        )
        corr_vt_mask = np.zeros(
            (
                self.max_start_lane_seq_num * self.max_end_lane_seq_num,
                self.max_vt_profile_sample_point,
            ),
            dtype=np.float32,
        )

        start_lane_ids = [e["lane_id"] for e in junction_info["entry_lanes"]]

        end_lane_ids = [e["lane_id"] for e in junction_info["exit_lanes"]]

        if self.use_min_entry_and_exit:

            start_lane_pose_l = [s["pose_l"] for s in junction_info["entry_lanes"]]
            if len(start_lane_pose_l) > 0:
                min_start_select = np.argmin([np.abs(l) for l in start_lane_pose_l])
                start_lane_ids = [start_lane_ids[min_start_select]]

            end_lane_pose_l = [s["pose_l"] for s in junction_info["exit_lanes"]]
            if len(end_lane_pose_l) > 0:
                min_end_select = np.argmin([np.abs(l) for l in end_lane_pose_l])
                end_lane_ids = [end_lane_ids[min_end_select]]

        start_match_idx = []
        if not junction_info["in_junction"]:
            for start_lane_id in start_lane_ids:
                start_match_idx.extend(
                    self.match_lane_seqs(set((start_lane_id,)), seq_lane_ids_sets)
                )
            start_match_idx = list(set(start_match_idx))
            start_match_idx = start_match_idx[: self.max_start_lane_seq_num]

        end_match_idx = []
        for end_lane_id in end_lane_ids:
            end_match_idx.extend(
                self.match_lane_seqs(set((end_lane_id,)), seq_lane_ids_sets)
            )
        end_match_idx = list(set(end_match_idx))
        end_match_idx = end_match_idx[: self.max_end_lane_seq_num]

        if len(start_match_idx) == 0 and len(end_match_idx) == 0:
            return start_ind, end_ind, valid_mask, corr_path_mask, corr_vt_mask

        if len(start_match_idx) > 0 and len(end_match_idx) == 0:
            for i, lane_seq_idx in enumerate(start_match_idx):
                start_ind[i] = lane_seq_idx
                valid_mask[i] = 1.0

                for j, in_junction_id in enumerate(in_junction_info):
                    if in_junction_id is not None:
                        break
                    if j >= self.max_path_sample_points:
                        break
                    corr_path_mask[i, j] = 1.0
                corr_vt_mask[i, :] = 1.0

            return start_ind, end_ind, valid_mask, corr_path_mask, corr_vt_mask

        if len(start_match_idx) == 0:
            start_match_idx = [
                -1,
            ]

        target_junction_id = junction_info["junction_id"]

        for i, start_idx in enumerate(start_match_idx):
            for j, end_idx in enumerate(end_match_idx):
                row = i * self.max_end_lane_seq_num + j
                start_ind[row] = start_idx
                end_ind[row] = end_idx

                for k, in_junction_id in enumerate(in_junction_info):
                    if (
                        in_junction_id is not None
                        and in_junction_id != target_junction_id
                    ):
                        break
                    if k >= self.max_path_sample_points:
                        break
                    corr_path_mask[row, k] = 1.0
                corr_vt_mask[row, :] = 1.0

        return start_ind, end_ind, valid_mask, corr_path_mask, corr_vt_mask

    def crop_lane_seq_by_category(self, lane_seq, lane_dict):
        if len(lane_seq) == 0:
            return False, lane_seq

        if lane_seq[0] not in lane_dict:
            return False, lane_seq

        target_lane_seq = lane_seq

        if lane_dict[lane_seq[0]]["lane_category"] in [
            "INTERSECTION_VIRTUAL",
            "M2N_VIRTUAL",
        ]:
            if len(lane_seq) == 1:
                return False, lane_seq
            else:
                target_lane_seq = lane_seq[1:]

        lane_seq_crop = []

        for lane_id in target_lane_seq:
            if lane_dict[lane_id]["lane_category"] in [
                "INTERSECTION_VIRTUAL",
                "M2N_VIRTUAL",
            ]:
                break
            lane_seq_crop.append(lane_id)

        if len(lane_seq_crop) == 0:
            return False, lane_seq_crop

        return True, lane_seq_crop

    def pack_ego_path_condition_corr_normal_in_junction(
        self, junction_info, in_junction_info, lane_seq_valid_info, seq_lane_ids
    ):
        seq_lane_ids_sets = [set(lane_ids) for lane_ids in seq_lane_ids]

        start_ind = (
            np.ones(
                (self.max_start_lane_seq_num * self.max_end_lane_seq_num),
                dtype=np.int64,
            )
            * -1
        )
        end_ind = (
            np.ones(
                (self.max_start_lane_seq_num * self.max_end_lane_seq_num),
                dtype=np.int64,
            )
            * -1
        )

        valid_mask = np.zeros(
            (self.max_start_lane_seq_num * self.max_end_lane_seq_num), dtype=np.float32
        )

        corr_path_mask = np.zeros(
            (
                self.max_start_lane_seq_num * self.max_end_lane_seq_num,
                self.max_path_sample_points,
            ),
            dtype=np.float32,
        )
        corr_vt_mask = np.zeros(
            (
                self.max_start_lane_seq_num * self.max_end_lane_seq_num,
                self.max_vt_profile_sample_point,
            ),
            dtype=np.float32,
        )

        lane_seq_valid_info.sort(key=lambda x: -x[1])

        collected_match_idx = []
        match_idxs = []
        for lane_seq, path_valid_num, vt_profile_valid_num in lane_seq_valid_info:
            lane_seq_set = set(lane_seq)
            match_idx = self.match_lane_seqs(lane_seq_set, seq_lane_ids_sets)

            for idx in match_idx:
                if idx not in match_idxs:
                    match_idxs.append(idx)
                    collected_match_idx.append(
                        (idx, path_valid_num, vt_profile_valid_num)
                    )

        for row_idx, (idx, path_valid_num, vt_profile_valid_num) in enumerate(
            collected_match_idx[
                : self.max_start_lane_seq_num * self.max_end_lane_seq_num
            ]
        ):
            end_ind[row_idx] = idx
            for i in range(min(path_valid_num, self.max_path_sample_points)):
                if (
                    in_junction_info[i] is not None
                    and in_junction_info[i] != junction_info["junction_id"]
                ):
                    break
                corr_path_mask[row_idx, i] = 1.0
            corr_vt_mask[:, :vt_profile_valid_num] = 1.0

        return start_ind, end_ind, valid_mask, corr_path_mask, corr_vt_mask

    def pack_ego_path_condition_corr_normal(
        self, in_junction_info, lane_seq_valid_info, seq_lane_ids, lane_dict
    ):

        seq_lane_ids_sets = [set(lane_ids) for lane_ids in seq_lane_ids]

        start_ind = (
            np.ones(
                (self.max_start_lane_seq_num * self.max_end_lane_seq_num),
                dtype=np.int64,
            )
            * -1
        )
        end_ind = (
            np.ones(
                (self.max_start_lane_seq_num * self.max_end_lane_seq_num),
                dtype=np.int64,
            )
            * -1
        )

        valid_mask = np.zeros(
            (self.max_start_lane_seq_num * self.max_end_lane_seq_num), dtype=np.float32
        )

        corr_path_mask = np.zeros(
            (
                self.max_start_lane_seq_num * self.max_end_lane_seq_num,
                self.max_path_sample_points,
            ),
            dtype=np.float32,
        )
        corr_vt_mask = np.zeros(
            (
                self.max_start_lane_seq_num * self.max_end_lane_seq_num,
                self.max_vt_profile_sample_point,
            ),
            dtype=np.float32,
        )

        lane_seq_valid_info.sort(key=lambda x: -x[1])
        if self.use_longest_condition_lane_only and len(lane_seq_valid_info) > 0:
            max_num = lane_seq_valid_info[0][1]
            max_ind = 0
            for i in range(len(lane_seq_valid_info)):
                if lane_seq_valid_info[i][1] == max_num:
                    max_ind = i
                else:
                    break
            lane_seq_valid_info = lane_seq_valid_info[0 : (max_ind + 1)]

        collected_match_idx = []
        match_idxs = []
        for lane_seq, path_valid_num, vt_profile_valid_num in lane_seq_valid_info:

            is_valid, lane_seq_crop = self.crop_lane_seq_by_category(
                lane_seq, lane_dict
            )

            if is_valid:
                lane_seq_set = set(lane_seq_crop)
                match_idx = self.match_lane_seqs(lane_seq_set, seq_lane_ids_sets)

                for idx in match_idx:
                    if idx not in match_idxs:
                        match_idxs.append(idx)
                        collected_match_idx.append(
                            (idx, path_valid_num, vt_profile_valid_num)
                        )

        for row_idx, (idx, path_valid_num, vt_profile_valid_num) in enumerate(
            collected_match_idx[
                : self.max_start_lane_seq_num * self.max_end_lane_seq_num
            ]
        ):
            start_ind[row_idx] = idx
            for i in range(min(path_valid_num, self.max_path_sample_points)):
                if in_junction_info[i] is not None:
                    break
                corr_path_mask[row_idx, i] = 1.0
            corr_vt_mask[:, :vt_profile_valid_num] = 1.0

        return start_ind, end_ind, valid_mask, corr_path_mask, corr_vt_mask

    def cal_ego_path_weights(self, ego_path_raw):

        weights = np.ones((self.max_path_sample_points, 1), dtype=np.float32)

        if not (self.use_curvature_coe or self.use_distance_coe):
            return weights

        valid_len = len(ego_path_raw)

        for i in range(min(valid_len, self.max_path_sample_points)):
            # TODO cal curvature here
            # TODO cal time efficient here

            if i <= 0 or i >= (min(valid_len, self.max_path_sample_points) - 2):
                weights[i] = 3.0
                continue

            prev_point = ego_path_raw[i - 1]
            now_point = ego_path_raw[i]
            next_point = ego_path_raw[i + 1]

            dir_0 = np.array(
                [now_point[0] - prev_point[0], now_point[1] - prev_point[1]]
            )
            dir_1 = np.array(
                [next_point[0] - now_point[0], next_point[1] - now_point[1]]
            )

            theta_diff = np.arctan2(dir_1[1], dir_1[0]) - np.arctan2(dir_0[1], dir_0[0])
            while theta_diff >= np.pi:
                theta_diff -= 2 * np.pi
            while theta_diff <= -np.pi:
                theta_diff += 2 * np.pi

            if self.use_distance_coe:
                time_coef = 2 * np.exp(-(i - 1) / 30)
            else:
                time_coef = 1.0
            if self.use_curvature_coe:
                curv_coef = min(np.exp(np.abs(theta_diff)), 4)
            elif self.use_new_curvature_coe:
                curv_coef = min(np.exp(10.0 * np.abs(theta_diff)), 4)
            else:
                curv_coef = 1.0

            weights[i] = time_coef * curv_coef

        return weights

    def clean_lane_seq_valid_info(
        self, percep_map, lane_seq_valid_info, corr_lane_id_info, corr_frame_idx
    ):
        lane_seq_valid_info_clean = []

        lane_dict = {}
        for lane in percep_map["lanes"]:
            lane_dict[lane["id"]] = lane

        for lane_seq, _, _ in lane_seq_valid_info:
            lane_ids = []
            for lane_id in lane_seq:
                if lane_id not in lane_dict or lane_dict[lane_id]["lane_category"] in [
                    "UNKNOWN_LANECATEGORY",
                    "INTERSECTION_VIRTUAL",
                    "M2N_VIRTUAL",
                ]:
                    break
                lane_ids.append(lane_id)
            if len(lane_ids) == 0:
                continue

            idx = 0
            for i, corr_lane_info in enumerate(corr_lane_id_info):
                if corr_lane_info is None:
                    break
                curr_lane_ids = [t[0] for t in corr_lane_info]
                if any([l_d in lane_ids for l_d in curr_lane_ids]):
                    idx = i

            if idx == 0:
                continue

            valid_len = idx + 1
            valid_vt_len = corr_frame_idx[idx] - corr_frame_idx[0]

            lane_seq_valid_info_clean.append((lane_ids, valid_len, valid_vt_len))

        return lane_seq_valid_info_clean

    def __call__(self, data):
        obstacles = get_dict_data(data, self.src_obs_key)
        percep_map = get_dict_data(data, self.src_map_key)
        obs_id_map = get_dict_data(data, self.src_obs_id_map_key)
        ego_curr_state = get_dict_data(data, self.src_ego_curr_state)
        seq_lane_ids = get_dict_data(data, self.src_map_seqs_lane_id_key)
        # lane_segs_prob = get_dict_data(data, self.src_lane_seqs_prob_key)
        aug_dict = get_dict_data(data, self.src_rot_aug_state_key)

        ego_path_raw = get_dict_data(data, self.src_ego_path_key)
        vt_profile = get_dict_data(data, self.src_vt_parofile_key)
        lane_seq_valid_info = get_dict_data(data, self.src_lane_seq_valid_key)
        in_junction_info = get_dict_data(data, self.src_in_junction_id_key)
        corr_lane_id_info = get_dict_data(data, self.src_corr_lane_id_key)

        if self.use_lane_seq_clean:
            corr_frame_idx = get_dict_data(data, ("ego_path_info", "corr_frame_idx"))
            lane_seq_valid_info = self.clean_lane_seq_valid_info(
                percep_map, lane_seq_valid_info, corr_lane_id_info, corr_frame_idx
            )

        lane_dict = {}
        for lane in percep_map["lanes"]:
            lane_dict[lane["id"]] = lane

        assert -9 in obstacles and obs_id_map[-9] == 0, "Need ego obs feature"

        ego_path, ego_path_mask, vt_profile, vt_profile_mask = self.pack_ego_path(
            obstacles, ego_curr_state, aug_dict, ego_path_raw
        )

        ego_path_weights = self.cal_ego_path_weights(ego_path_raw)

        junction_info = obstacles[-9].get(
            "junction_info", dict(in_junction=False, exit_lanes=[])
        )

        if (
            len(junction_info["exit_lanes"]) > 0
            and (
                any(
                    [
                        (
                            in_junction_id is not None
                            and in_junction_id == junction_info["junction_id"]
                        )
                        for in_junction_id in in_junction_info
                    ]
                )
            )
            and junction_info["junction_id"]
            not in [
                735,
            ]
        ):
            start_ind, end_ind, _, corr_path_mask, corr_vt_mask = (
                self.pack_ego_path_condition_corr_junction(
                    junction_info, in_junction_info, seq_lane_ids
                )
            )
        else:
            # if junction_info["in_junction"]:
            #     start_ind, end_ind, _, corr_path_mask, corr_vt_mask = self.pack_ego_path_condition_corr_normal_in_junction(junction_info, in_junction_info, lane_seq_valid_info, seq_lane_ids)
            # else:
            start_ind, end_ind, _, corr_path_mask, corr_vt_mask = (
                self.pack_ego_path_condition_corr_normal(
                    in_junction_info, lane_seq_valid_info, seq_lane_ids, lane_dict
                )
            )

        write_dict_data(data, self.tgt_ego_path_key, ego_path)
        write_dict_data(data, self.tgt_ego_path_weights_key, ego_path_weights)
        write_dict_data(data, self.tgt_ego_path_mask_key, ego_path_mask)
        write_dict_data(data, self.tgt_ego_future_vt_profile_key, vt_profile)
        write_dict_data(data, self.tgt_ego_future_vt_profile_mask_key, vt_profile_mask)
        write_dict_data(data, self.tgt_ego_start_lane_seqs_ind_key, start_ind)
        write_dict_data(data, self.tgt_ego_end_lane_seqs_ind_key, end_ind)
        write_dict_data(data, self.tgt_ego_per_lane_seq_path_mask_key, corr_path_mask)
        write_dict_data(
            data, self.tgt_ego_per_lane_seq_vt_profile_mask_key, corr_vt_mask
        )
        return data
