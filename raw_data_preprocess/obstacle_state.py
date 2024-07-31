import traceback
from typing import Dict, Tuple, Union

import numpy as np

from .utils import get_dict_data, write_dict_data


class StateFeaturePacker:
    def __init__(
        self,
        src_key: str = "obstacles",
        tgt_state_key: Union[Tuple, str] = ("obstacle_state", "state"),
        tgt_mask_key: Union[Tuple, str] = ("obstacle_state", "state_mask"),
        tgt_id_map_key: Union[Tuple, str] = ("obstacle_state", "id_map"),
        tgt_ego_curr_state_key: Union[Tuple, str] = (
            "obstacle_state",
            "ego_curr_state",
        ),
        sort_by_distance: bool = False,
        all_in_range: bool = False,
        predict_in_range: bool = False,
        perception_range: Dict[str, Tuple] = dict(
            x_range=(-30, 60),
            y_range=(-60, 60),
        ),
        filter_min_history: int = 0,
        filter_interpolate: bool = False,
        filter_low_speed: float = -1,
        filter_low_quality: bool = False,
        low_quality_threshold: float = 3.0,
        pad_timestamp: bool = False,
        history_states: int = 11,
        future_states: int = 30,
        max_obstacles: int = 128,
        types_mapping: Dict[str, int] = {
            "VEHICLE": 0,
            "BICYCLE": 1,
            "PEDESTRIAN": 2,
            "PEDESTRAIN": 2,
        },
        ego_id: str = "-9",
        ignore_predict_ego: bool = False,
        predict_pedestrian: bool = True,
        predict_low_speed_quality_ped: bool = False,
        predict_static: bool = False,
        filter_interpolate_ego: bool = False,
    ) -> None:
        self.src_key = src_key
        self.history_states = history_states
        self.future_states = future_states
        self.total_states = history_states + future_states
        self.types_mapping = types_mapping
        self.max_obstacles = max_obstacles
        self.filter_min_history = filter_min_history
        self.pad_timestamp = pad_timestamp
        self.ego_id = ego_id
        self.tgt_state_key = tgt_state_key
        self.tgt_mask_key = tgt_mask_key
        self.tgt_id_map_key = tgt_id_map_key
        self.tgt_ego_curr_state_key = tgt_ego_curr_state_key
        self.sort_by_distance = sort_by_distance
        self.filter_interpolate = filter_interpolate
        self.ignore_predict_ego = ignore_predict_ego
        self.predict_pedestrian = predict_pedestrian
        self.all_in_range = all_in_range
        self.predict_in_range = predict_in_range
        self.range = perception_range
        self.filter_low_speed = filter_low_speed
        self.filter_low_quality = filter_low_quality
        self.low_quality_threshold = low_quality_threshold
        self.predict_low_speed_quality_ped = predict_low_speed_quality_ped
        self.predict_static = predict_static
        self.filter_interpolate_ego = filter_interpolate_ego

    def valid_for_predict_future(
        self, feature, state_mask, id_in_range=None, trajectory_quality=None
    ):
        if state_mask.sum() < self.filter_min_history:
            return False
        if self.ignore_predict_ego and feature["id"] == int(self.ego_id):
            return False
        if not self.predict_pedestrian and feature["type"] == "PEDESTRIAN":
            return False
        if self.predict_in_range:
            assert id_in_range is not None
            if not id_in_range[feature["id"]]:
                return False
        if self.filter_low_speed > 0:
            curr_state = feature["history_states"][-1]
            curr_speed = np.linalg.norm(np.array([curr_state["vx"], curr_state["vy"]]))
            if curr_speed < self.filter_low_speed:
                if self.predict_static:
                    return True
                return False
        if self.filter_low_quality:
            if trajectory_quality["abnormal_score"] > self.low_quality_threshold:
                if not (
                    self.predict_low_speed_quality_ped
                    and feature["type"] == "PEDESTRIAN"
                ):
                    return False
        return True

    def sort_obs_by_distance(self, obstacles: Dict[str, Dict]):
        ego_curr_state = obstacles[self.ego_id]["features"]["history_states"][-1]
        ego_pos = np.array([ego_curr_state["x"], ego_curr_state["y"]])
        ids = []
        poses = []
        for obs_id in obstacles:
            curr_state = obstacles[obs_id]["features"]["history_states"][-1]
            curr_pos = np.array([curr_state["x"], curr_state["y"]])
            ids.append(obs_id)
            poses.append(curr_pos)
        poses = np.stack(poses)
        ids = np.stack(ids)

        distances = np.linalg.norm(poses - ego_pos, axis=-1)
        sorted_idxs = np.argsort(distances)

        rot_mat = np.array(
            [
                [
                    np.cos(ego_curr_state["theta"]),
                    -np.sin(ego_curr_state["theta"]),
                ],
                [
                    np.sin(ego_curr_state["theta"]),
                    np.cos(ego_curr_state["theta"]),
                ],
            ]
        ).T
        t_vec = -np.array([ego_curr_state["x"], ego_curr_state["y"]]) @ rot_mat.T
        poses_in_ego = poses[sorted_idxs] @ rot_mat.T + t_vec
        in_range_mask = (
            (poses_in_ego[:, 0] > self.range["x_range"][0])
            & (poses_in_ego[:, 0] < self.range["x_range"][1])
            & (poses_in_ego[:, 1] > self.range["y_range"][0])
            & (poses_in_ego[:, 1] < self.range["y_range"][1])
        )

        id_in_range = {}
        for i, idx in enumerate(sorted_idxs):
            if in_range_mask[i]:
                id_in_range[int(ids[idx])] = True
            else:
                id_in_range[int(ids[idx])] = False

        if self.all_in_range:
            sorted_idxs = sorted_idxs[in_range_mask]
        elif self.predict_in_range:
            sorted_idxs = np.concatenate(
                [sorted_idxs[in_range_mask], sorted_idxs[~in_range_mask]]
            )

        ids = ids[sorted_idxs].tolist()
        return ids, id_in_range

    def get_valid_ids(self, obstacles: Dict[str, Dict]):
        if self.sort_by_distance:
            obs_ids, id_in_range = self.sort_obs_by_distance(obstacles)
        else:
            id_in_range = None
            obs_ids = [self.ego_id]
            obs_ids.extend(
                [obs_id for obs_id in list(obstacles.keys()) if obs_id != self.ego_id]
            )
        return obs_ids, id_in_range

    def pack_states(self, obstacles: Dict[str, Dict]) -> Tuple:
        id_map = {}
        all_obj_state = []
        all_obj_state_mask = []
        collected_num = 0
        ego_curr_state = None
        obs_ids, id_in_range = self.get_valid_ids(obstacles)
        for obs_id in obs_ids:
            if collected_num >= self.max_obstacles:
                break
            if obs_id not in obstacles:
                continue
            obs = obstacles[obs_id]
            if "features" not in obs or "future_trajectory" not in obs:
                continue
            feature = obs["features"]
            if feature["history_states"][-1]["is_interpolate"]:
                continue
            id_map[obs_id] = collected_num
            future_trajectory = obs["future_trajectory"]
            # [x, y, theta, vx, vy, length, width, type, ts]
            obj_state = np.zeros((self.total_states, 9))
            if self.pad_timestamp:
                obj_state[: self.history_states - 1, -1] = np.arange(
                    -(self.history_states - 1) * 0.2, 0.0, 0.2
                )
                obj_state[self.history_states - 1 :, -1] = np.arange(
                    0.0, self.future_states * 0.1 + 0.001, 0.1
                )
            state_mask = np.zeros((self.total_states,))
            length = feature["length"]
            width = feature["width"]
            obs_type = self.types_mapping[feature["type"]]
            curr_timestamp = feature["history_states"][-1]["timestamp"]
            for idx, hist_state in enumerate(reversed(feature["history_states"])):
                if self.filter_interpolate and hist_state["is_interpolate"]:
                    obj_state[self.history_states - 1 - idx][8] = (
                        hist_state["timestamp"] - curr_timestamp
                    ) * 1e-6
                    state_mask[self.history_states - 1 - idx] = 0
                    continue
                obj_state[self.history_states - 1 - idx][0] = hist_state["x"]
                obj_state[self.history_states - 1 - idx][1] = hist_state["y"]
                obj_state[self.history_states - 1 - idx][2] = hist_state["theta"]
                obj_state[self.history_states - 1 - idx][3] = hist_state["vx"]
                obj_state[self.history_states - 1 - idx][4] = hist_state["vy"]
                obj_state[self.history_states - 1 - idx][5] = length
                obj_state[self.history_states - 1 - idx][6] = width
                obj_state[self.history_states - 1 - idx][7] = obs_type
                real_rela_ts = (hist_state["timestamp"] - curr_timestamp) * 1e-6
                if (
                    abs(real_rela_ts - obj_state[self.history_states - 1 - idx][8])
                    > 0.02
                ):
                    raise RuntimeError(f"Invalid timestamps: {obs_id}")
                obj_state[self.history_states - 1 - idx][8] = real_rela_ts
                state_mask[self.history_states - 1 - idx] = 1
            if self.valid_for_predict_future(
                feature, state_mask, id_in_range, obs.get("trajectory_quality", None)
            ):
                curr_state = feature["history_states"][-1]
                curr_speed = np.linalg.norm(
                    np.array([curr_state["vx"], curr_state["vy"]])
                )
                for idx, future_state in enumerate(future_trajectory["future_states"]):
                    if idx >= self.future_states:
                        break
                    if self.filter_interpolate and future_state["is_interpolate"]:
                        obj_state[self.history_states + idx][8] = (
                            future_state["timestamp"] - curr_timestamp
                        ) * 1e-6
                        state_mask[self.history_states + idx] = 0
                        continue
                    if self.predict_static and curr_speed <= self.filter_low_speed:
                        obj_state[self.history_states + idx][0] = curr_state["x"]
                        obj_state[self.history_states + idx][1] = curr_state["y"]
                        obj_state[self.history_states + idx][2] = curr_state["theta"]
                        obj_state[self.history_states + idx][3] = curr_state["vx"]
                        obj_state[self.history_states + idx][4] = curr_state["vy"]
                        obj_state[self.history_states + idx][5] = length
                        obj_state[self.history_states + idx][6] = width
                        obj_state[self.history_states + idx][7] = obs_type
                        real_rela_ts = (
                            future_state["timestamp"] - curr_timestamp
                        ) * 1e-6
                    else:
                        obj_state[self.history_states + idx][0] = future_state["x"]
                        obj_state[self.history_states + idx][1] = future_state["y"]
                        obj_state[self.history_states + idx][2] = future_state["theta"]
                        obj_state[self.history_states + idx][3] = future_state["vx"]
                        obj_state[self.history_states + idx][4] = future_state["vy"]
                        obj_state[self.history_states + idx][5] = length
                        obj_state[self.history_states + idx][6] = width
                        obj_state[self.history_states + idx][7] = obs_type
                        real_rela_ts = (
                            future_state["timestamp"] - curr_timestamp
                        ) * 1e-6
                    if (
                        abs(real_rela_ts - obj_state[self.history_states + idx][8])
                        > 0.02
                    ):
                        raise RuntimeError(f"Invalid timestamps: {obs_id}")
                    obj_state[self.history_states + idx][8] = real_rela_ts
                    state_mask[self.history_states + idx] = 1
            if obs_id == self.ego_id:
                ego_curr_state = {
                    "x": feature["history_states"][-1]["x"],
                    "y": feature["history_states"][-1]["y"],
                    "theta": feature["history_states"][-1]["theta"],
                }
            all_obj_state.append(obj_state)
            all_obj_state_mask.append(state_mask)
            collected_num += 1
        if collected_num > 0:
            all_obj_state = np.stack(all_obj_state, axis=0).astype("float64")
            all_obj_state_mask = np.stack(all_obj_state_mask, axis=0).astype("bool")
            if self.max_obstacles - collected_num > 0:
                padding_obj_state = np.zeros(
                    (self.max_obstacles - collected_num, self.total_states, 9),
                    dtype="float64",
                )
                padding_obj_state_mask = np.zeros(
                    (
                        self.max_obstacles - collected_num,
                        self.total_states,
                    ),
                    dtype="bool",
                )
                if self.pad_timestamp:
                    padding_obj_state[:, : self.history_states - 1, -1] = np.arange(
                        -(self.history_states - 1) * 0.2, 0.0, 0.2
                    )
                    padding_obj_state[:, self.history_states - 1 :, -1] = np.arange(
                        0.0, self.future_states * 0.1 + 0.001, 0.1
                    )
                all_obj_state = np.concatenate(
                    [all_obj_state, padding_obj_state], axis=0
                )
                all_obj_state_mask = np.concatenate(
                    [all_obj_state_mask, padding_obj_state_mask], axis=0
                )
        else:
            all_obj_state = np.zeros(
                (self.max_obstacles, self.total_states, 9), dtype="float64"
            )
            all_obj_state_mask = np.zeros(
                (
                    self.max_obstacles,
                    self.total_states,
                ),
                dtype="bool",
            )
            if self.pad_timestamp:
                all_obj_state[:, : self.history_states - 1, -1] = np.arange(
                    -(self.history_states - 1) * 0.2, 0.0, 0.2
                )
                all_obj_state[:, self.history_states - 1 :, -1] = np.arange(
                    0.0, self.future_states * 0.1 + 0.001, 0.1
                )

        assert ego_curr_state is not None
        if (
            self.filter_interpolate_ego
            and not feature["history_states"][-1]["is_interpolate"]
        ):
            ego_valid_mask = np.zeros((self.total_states,))
            ego_obs = obstacles[self.ego_id]
            feature = ego_obs["features"]
            future_trajectory = ego_obs["future_trajectory"]
            for idx, hist_state in enumerate(reversed(feature["history_states"])):
                if not hist_state["is_interpolate"]:
                    ego_valid_mask[self.history_states - 1 - idx] = 1
            for idx, future_state in enumerate(future_trajectory["future_states"]):
                if not future_state["is_interpolate"]:
                    ego_valid_mask[self.history_states + idx] = 1
            all_obj_state_mask &= ego_valid_mask[None, ...]
            all_obj_state[~ego_valid_mask[None, ...], :-1] = 0
        return all_obj_state, all_obj_state_mask, id_map, ego_curr_state

    def __call__(self, data):
        obstacles = get_dict_data(data, self.src_key)
        try:
            states, states_mask, id_map, ego_curr_state = self.pack_states(obstacles)
        except Exception as e:
            print(data["file_path"], f"error: {e.args} \n", f"{traceback.format_exc()}")
            raise e
        write_dict_data(data, self.tgt_state_key, states)
        write_dict_data(data, self.tgt_mask_key, states_mask)
        write_dict_data(data, self.tgt_id_map_key, id_map)
        write_dict_data(data, self.tgt_ego_curr_state_key, ego_curr_state)
        return data


class StateFeatureSplitTypePacker(StateFeaturePacker):
    def __init__(
        self,
        num_vehicle=32,
        num_bicycle=16,
        num_pedestrian=16,
        sort_by_cost=False,
        still_vel_thr=0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_vehicle = num_vehicle
        self.num_bicycle = num_bicycle
        self.num_pedestrian = num_pedestrian
        self.sort_by_cost = sort_by_cost
        self.still_vel_thr = still_vel_thr
        assert self.sort_by_cost ^ self.sort_by_distance

    def get_valid_ids(self, obstacles: Dict[str, Dict]):
        type_list = {"vehicle": [], "bicycle": [], "pedestrian": []}
        if self.sort_by_cost:
            sorted_ids, id_in_range = self.sort_obs_by_costs(obstacles)
        elif self.sort_by_distance:
            sorted_ids, id_in_range = self.sort_obs_by_distance(obstacles)

        for obs_id in sorted_ids:
            obs = obstacles[obs_id]
            feature = obs["features"]
            if feature["type"] == "VEHICLE":
                type_list["vehicle"].append(obs_id)
            elif feature["type"] == "BICYCLE":
                type_list["bicycle"].append(obs_id)
            elif feature["type"] == "PEDESTRIAN":
                type_list["pedestrian"].append(obs_id)

        all_ids = np.ones(self.max_obstacles, dtype=int) * -1
        all_ids[: min(self.num_vehicle, len(type_list["vehicle"]))] = type_list[
            "vehicle"
        ][: self.num_vehicle]
        all_ids[
            self.num_vehicle : self.num_vehicle
            + min(self.num_bicycle, len(type_list["bicycle"]))
        ] = type_list["bicycle"][: self.num_bicycle]
        all_ids[
            self.num_vehicle
            + self.num_bicycle : self.num_vehicle
            + self.num_bicycle
            + min(self.num_pedestrian, len(type_list["pedestrian"]))
        ] = type_list["pedestrian"][: self.num_pedestrian]
        return all_ids, id_in_range

    def pack_states(self, obstacles: Dict[str, Dict]) -> Tuple:
        id_map = {}
        collected_num = 0
        ego_curr_state = None
        obs_ids, id_in_range = self.get_valid_ids(obstacles)

        all_obj_state = np.zeros(
            (self.max_obstacles, self.total_states, 9), dtype="float64"
        )
        all_obj_state_mask = np.zeros(
            (
                self.max_obstacles,
                self.total_states,
            ),
            dtype="bool",
        )
        if self.pad_timestamp:
            all_obj_state[:, : self.history_states - 1, -1] = np.arange(
                -(self.history_states - 1) * 0.2, 0.0, 0.2
            )
            all_obj_state[:, self.history_states - 1 :, -1] = np.arange(
                0.0, self.future_states * 0.1 + 0.001, 0.1
            )

        for i, obs_id in enumerate(obs_ids):
            if collected_num >= self.max_obstacles:
                break
            if obs_id not in obstacles:
                continue
            obs = obstacles[obs_id]
            if "features" not in obs or "future_trajectory" not in obs:
                continue
            feature = obs["features"]
            if feature["history_states"][-1]["is_interpolate"]:
                continue
            id_map[obs_id] = i
            future_trajectory = obs["future_trajectory"]
            # [x, y, theta, vx, vy, length, width, type, ts]
            obj_state = np.zeros((self.total_states, 9))
            if self.pad_timestamp:
                obj_state[: self.history_states - 1, -1] = np.arange(
                    -(self.history_states - 1) * 0.2, 0.0, 0.2
                )
                obj_state[self.history_states - 1 :, -1] = np.arange(
                    0.0, self.future_states * 0.1 + 0.001, 0.1
                )
            state_mask = np.zeros((self.total_states,))
            length = feature["length"]
            width = feature["width"]
            obs_type = self.types_mapping[feature["type"]]
            curr_timestamp = feature["history_states"][-1]["timestamp"]
            for idx, hist_state in enumerate(reversed(feature["history_states"])):
                if self.filter_interpolate and hist_state["is_interpolate"]:
                    obj_state[self.history_states - 1 - idx][8] = (
                        hist_state["timestamp"] - curr_timestamp
                    ) * 1e-6
                    state_mask[self.history_states - 1 - idx] = 0
                    continue
                obj_state[self.history_states - 1 - idx][0] = hist_state["x"]
                obj_state[self.history_states - 1 - idx][1] = hist_state["y"]
                obj_state[self.history_states - 1 - idx][2] = hist_state["theta"]
                obj_state[self.history_states - 1 - idx][3] = hist_state["vx"]
                obj_state[self.history_states - 1 - idx][4] = hist_state["vy"]
                obj_state[self.history_states - 1 - idx][5] = length
                obj_state[self.history_states - 1 - idx][6] = width
                obj_state[self.history_states - 1 - idx][7] = obs_type
                real_rela_ts = (hist_state["timestamp"] - curr_timestamp) * 1e-6
                if (
                    abs(real_rela_ts - obj_state[self.history_states - 1 - idx][8])
                    > 0.02
                ):
                    raise RuntimeError(f"Invalid timestamps: {obs_id}")
                obj_state[self.history_states - 1 - idx][8] = real_rela_ts
                state_mask[self.history_states - 1 - idx] = 1
            if self.valid_for_predict_future(
                feature, state_mask, id_in_range, obs.get("trajectory_quality", None)
            ):
                for idx, future_state in enumerate(future_trajectory["future_states"]):
                    if idx >= self.future_states:
                        break
                    if self.filter_interpolate and future_state["is_interpolate"]:
                        obj_state[self.history_states + idx][8] = (
                            future_state["timestamp"] - curr_timestamp
                        ) * 1e-6
                        state_mask[self.history_states + idx] = 0
                        continue
                    obj_state[self.history_states + idx][0] = future_state["x"]
                    obj_state[self.history_states + idx][1] = future_state["y"]
                    obj_state[self.history_states + idx][2] = future_state["theta"]
                    obj_state[self.history_states + idx][3] = future_state["vx"]
                    obj_state[self.history_states + idx][4] = future_state["vy"]
                    obj_state[self.history_states + idx][5] = length
                    obj_state[self.history_states + idx][6] = width
                    obj_state[self.history_states + idx][7] = obs_type
                    real_rela_ts = (future_state["timestamp"] - curr_timestamp) * 1e-6
                    if (
                        abs(real_rela_ts - obj_state[self.history_states + idx][8])
                        > 0.02
                    ):
                        raise RuntimeError(f"Invalid timestamps: {obs_id}")
                    obj_state[self.history_states + idx][8] = real_rela_ts
                    state_mask[self.history_states + idx] = 1
            if (
                obs_id == self.ego_id
                and not feature["history_states"][-1]["is_interpolate"]
            ):
                ego_curr_state = {
                    "x": feature["history_states"][-1]["x"],
                    "y": feature["history_states"][-1]["y"],
                    "theta": feature["history_states"][-1]["theta"],
                }
            all_obj_state[i] = obj_state
            all_obj_state_mask[i] = state_mask
            collected_num += 1
        assert ego_curr_state is not None
        if self.filter_interpolate_ego:
            ego_valid_mask = np.zeros((self.total_states,), dtype=bool)
            ego_obs = obstacles[self.ego_id]
            feature = ego_obs["features"]
            future_trajectory = ego_obs["future_trajectory"]
            for idx, hist_state in enumerate(reversed(feature["history_states"])):
                if not hist_state["is_interpolate"]:
                    ego_valid_mask[self.history_states - 1 - idx] = True
            for idx, future_state in enumerate(future_trajectory["future_states"]):
                if idx >= self.future_states:
                    break
                if not future_state["is_interpolate"]:
                    ego_valid_mask[self.history_states + idx] = True
            all_obj_state_mask &= ego_valid_mask[None, ...]
            all_obj_state[~all_obj_state_mask, :-1] = 0
        return all_obj_state, all_obj_state_mask, id_map, ego_curr_state

    def ref_pose_cost(self, pos_sl, ego_vel_sl):
        ref_pose_l_sigma = 15.0 * 15.0
        ref_pose_l_sigma_scale = 5.0 * 5.0
        ref_pose_l_bias_scale = 1.5 * 1.5

        ref_pose_s_forward_sigma = 100.0 * 100.0
        ref_pose_s_forward_sigma_scale = 40.0 * 40.0
        ref_pose_s_backward_sigma = 40.0 * 40.0
        ref_pose_s_backward_sigma_scale = 20.0 * 20.0

        n_obs = pos_sl.shape[0]
        cost_s = np.zeros([n_obs])
        pos_s_mask = pos_sl[:, 0] > 0
        cost_s[pos_s_mask] = np.exp(
            -pos_sl[pos_s_mask, 0]
            * pos_sl[pos_s_mask, 0]
            / (
                ref_pose_s_forward_sigma
                + ref_pose_s_forward_sigma_scale * ego_vel_sl[0]
            )
        )
        cost_s[~pos_s_mask] = np.exp(
            -pos_sl[~pos_s_mask, 0] ** 2
            / (
                ref_pose_s_backward_sigma
                + ref_pose_s_backward_sigma_scale * ego_vel_sl[0]
            )
        )
        cost_l = np.exp(
            -((pos_sl[:, 1] - ref_pose_l_bias_scale * ego_vel_sl[1]) ** 2)
            / (ref_pose_l_sigma + ref_pose_l_sigma_scale * np.abs(ego_vel_sl[1]))
        )
        return cost_s * cost_l

    def thw_cost(self, thw: np.array):
        thw_time_sigma = 6.0
        n_obs = thw.shape[0]
        thw_cost = np.zeros(n_obs)
        thw_mask = thw >= 0
        thw_cost[thw_mask] = np.exp(-thw[thw_mask] / thw_time_sigma)
        thw_cost[~thw_mask] = -0.3 * np.exp(thw[~thw_mask] / thw_time_sigma)
        return thw_cost

    def ttc_cost(self, ttc_sl: np.array):
        ttc_s_sigma = 100.0 * 100.0
        ttc_l_sigma = 40.0 * 40.0

        n_obs = ttc_sl.shape[0]
        ttc_s_cost = np.zeros(n_obs)
        ttc_s = ttc_sl[:, 0]
        ttc_s_mask = ttc_s >= 0
        ttc_s_cost = np.exp(-(ttc_s**2) / ttc_s_sigma)
        ttc_s_cost[~ttc_s_mask] *= -0.3

        ttc_l_cost = np.zeros(n_obs)
        ttc_l = ttc_sl[:, 1]
        ttc_l_mask = ttc_l >= 0
        ttc_l_cost = np.exp(-(ttc_l**2) / ttc_l_sigma)
        ttc_l_cost[~ttc_l_mask] *= -0.3

        return (ttc_s_cost + ttc_l_cost) * np.exp(-np.abs(ttc_l - ttc_s) / 5.0)

    def intrusion_cost(
        self,
        pos_s: np.array,
        vel_s: np.array,
        ttc_l: np.array,
        ego_vel_s: np.array,
    ):
        ego_vel_scale = 1.0
        position_s_scale = 0.3
        velocity_s_scale = 0.2

        ego_vel_fac = np.exp(-np.abs(ego_vel_scale * ego_vel_s))
        calc_pos_s = np.copy(pos_s)
        calc_pos_s[pos_s <= 0] = 1e-3
        sqrt_sigma = np.sqrt(position_s_scale * calc_pos_s)

        calc_ttc_l = np.copy(ttc_l)
        calc_ttc_l[ttc_l < 0] = 0
        cost = (
            ego_vel_fac
            * (1.0 / sqrt_sigma)
            * np.exp(-calc_ttc_l / sqrt_sigma)
            * np.exp(-np.abs(velocity_s_scale * vel_s))
        )
        cost[pos_s < 0] = 0
        cost[ttc_l < 0] = 0
        cost[sqrt_sigma < 1e-3] = 0
        return cost

    def calc_ttc(
        self,
        pos_sl: np.array,
        theta_in_ego: np.array,
        ref_vel_sl: np.array,
        length: np.array,
        width: np.array,
        ego_length: np.array,
        ego_width: np.array,
    ):
        n_obs = pos_sl.shape[0]
        cos_theta = np.cos(theta_in_ego)
        sin_theta = np.sin(theta_in_ego)
        corner_forward_x = length / 2
        corner_left_y = width / 2
        max_s = np.max(
            np.stack(
                [
                    np.abs(cos_theta * corner_forward_x + sin_theta * corner_left_y),
                    np.abs(cos_theta * corner_forward_x - sin_theta * corner_left_y),
                ]
            ),
            axis=0,
        )
        max_l = np.max(
            np.stack(
                [
                    np.abs(cos_theta * corner_left_y + sin_theta * corner_forward_x),
                    np.abs(cos_theta * corner_left_y - sin_theta * corner_forward_x),
                ]
            ),
            axis=0,
        )

        ttc_sl = np.zeros([n_obs, 2])
        pos_s_mask = pos_sl[:, 0] > 0
        ref_vel_sl[ref_vel_sl == 0] += 1e-9
        ttc_sl[pos_s_mask, 0] = (
            -(pos_sl[pos_s_mask, 0] - ego_length + max_s[pos_s_mask])
            / ref_vel_sl[pos_s_mask, 0]
        )
        ttc_sl[~pos_s_mask, 0] = (
            -(pos_sl[~pos_s_mask, 0] + ego_length + max_s[~pos_s_mask])
            / ref_vel_sl[~pos_s_mask, 0]
        )
        ttc_sl[np.abs(pos_sl[:, 0]) < ego_length + max_s, 0] = 0

        pos_l_mask = pos_sl[:, 1] > 0
        ttc_sl[pos_l_mask, 1] = (
            -(pos_sl[pos_l_mask, 1] - ego_width + max_l[pos_l_mask])
            / ref_vel_sl[pos_l_mask, 1]
        )
        ttc_sl[~pos_l_mask, 1] = (
            -(pos_sl[~pos_l_mask, 1] + ego_width + max_l[~pos_l_mask])
            / ref_vel_sl[~pos_l_mask, 1]
        )
        ttc_sl[np.abs(pos_sl[:, 1]) < ego_width + max_l, 1] = 0
        ttc_sl[ttc_sl > 1e5] = 1e5
        ttc_sl[ttc_sl < -1e5] = -1e5
        return ttc_sl

    def calc_thw(self, pos_sl: np.array, ref_vel_sl: np.array):
        n_obs = pos_sl.shape[0]
        dis_to_ego = np.linalg.norm(pos_sl, axis=1)
        ref_angle_cos_sin = np.zeros([n_obs, 2])
        ref_angle_cos_sin[:, 0] = 1
        ref_angle_cos_sin[dis_to_ego > 1e-10] = (
            pos_sl[dis_to_ego > 1e-10] / dis_to_ego[dis_to_ego > 1e-10, None]
        )
        approach_vel = (
            ref_vel_sl[:, 0] * ref_angle_cos_sin[:, 0]
            + ref_vel_sl[:, 1] * ref_angle_cos_sin[:, 1]
        )
        approach_vel[approach_vel == 0] += 1e-9
        thw = dis_to_ego / approach_vel
        thw[np.isnan(thw)] = 0
        thw[thw > 1e5] = 1e5
        thw[thw < -1e5] = -1e5
        return thw

    def sort_obs_by_costs(self, obstacles: Dict[str, Dict]):
        # collect ego state
        ego_curr_state = obstacles[self.ego_id]["features"]["history_states"][-1]
        ego_pos = np.array([ego_curr_state["x"], ego_curr_state["y"]])
        ego_theta = ego_curr_state["theta"]
        ego_vel = np.array([ego_curr_state["vx"], ego_curr_state["vy"]])
        ego_length = obstacles[self.ego_id]["features"]["length"]
        ego_width = obstacles[self.ego_id]["features"]["width"]

        # collect other state
        ids = []
        pos = []
        vel = []
        theta = []
        length = []
        width = []
        for obs_id in obstacles:
            feature = obstacles[obs_id]["features"]
            curr_state = feature["history_states"][-1]
            curr_pos = np.array([curr_state["x"], curr_state["y"]])
            curr_vel = np.array([curr_state["vx"], curr_state["vy"]])
            ids.append(obs_id)
            pos.append(curr_pos)
            vel.append(curr_vel)
            length.append(feature["length"])
            width.append(feature["width"])
            theta.append(curr_state["theta"])
        ids = np.stack(ids)
        pos = np.stack(pos)
        vel = np.stack(vel)
        theta = np.stack(theta)
        length = np.stack(length)
        width = np.stack(width)

        rot_mat = np.array(
            [
                [
                    np.cos(ego_theta),
                    -np.sin(ego_theta),
                ],
                [
                    np.sin(ego_theta),
                    np.cos(ego_theta),
                ],
            ]
        ).T
        t_vec = -np.array([ego_pos[0], ego_pos[1]]) @ rot_mat.T

        # [N, 2]
        pos_sl = pos @ rot_mat.T + t_vec
        vel_sl = vel @ rot_mat.T
        theta_in_ego = theta - ego_theta
        ego_vel_sl = ego_vel @ rot_mat.T
        ref_vel_sl = vel_sl - ego_vel_sl

        # calc ttc
        ttc_sl = self.calc_ttc(
            pos_sl,
            theta_in_ego,
            ref_vel_sl,
            length,
            width,
            ego_length,
            ego_width,
        )

        # calc thw
        thw = self.calc_thw(pos_sl, ref_vel_sl)

        ref_pose_cost = self.ref_pose_cost(pos_sl, ego_vel_sl)
        thw_cost = self.thw_cost(thw)
        ttc_cost = self.ttc_cost(ttc_sl)
        intrusion_cost = self.intrusion_cost(
            pos_sl[:, 0], vel_sl[:, 0], ttc_sl[:, 1], ego_vel_sl[0]
        )

        is_still = np.linalg.norm(vel, axis=-1) < self.still_vel_thr

        total_cost = (
            1.0 * ref_pose_cost + 0.8 * thw_cost + 0.5 * ttc_cost + 1.0 * intrusion_cost
        )
        total_cost[is_still] += -0.5

        in_range_mask = (
            (pos_sl[:, 0] > self.range["x_range"][0])
            & (pos_sl[:, 0] < self.range["x_range"][1])
            & (pos_sl[:, 1] > self.range["y_range"][0])
            & (pos_sl[:, 1] < self.range["y_range"][1])
        )

        sorted_idxs = np.flip(
            np.argsort(
                total_cost,
            )
        )

        id_in_range = {}
        for idx in sorted_idxs:
            if in_range_mask[idx]:
                id_in_range[int(ids[idx])] = True
            else:
                id_in_range[int(ids[idx])] = False

        ids = ids[sorted_idxs].tolist()
        ids = [self.ego_id] + [i for i in ids if i != self.ego_id]
        return ids, id_in_range
