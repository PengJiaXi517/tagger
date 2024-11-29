import io
import os
import pickle
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from shapely.geometry import LineString, Point


class ConditionRes:
    def __init__(self, file_path_or_file: Union[os.PathLike, Dict]) -> None:
        if isinstance(file_path_or_file, dict):
            label = file_path_or_file
            condition_res = {
                "file_path": label["file_path"],
                "seq_lane_ids": label["map_state"]["seq_lane_ids"],
                "seq_lane_ids_raw": label["map_state"]["seq_lane_ids_raw"],
                "start_lane_seqs_ind": label["obstacle_state"][
                    "start_lane_seqs_ind"
                ],
                "end_lane_seqs_ind": label["obstacle_state"][
                    "end_lane_seqs_ind"
                ],
                "ego_per_lane_seq_path_mask": label["obstacle_state"][
                    "ego_per_lane_seq_path_mask"
                ],
            }
        else:
            with open(file_path_or_file, "rb") as f:
                condition_res = pickle.load(f)

        self.file_path = condition_res["file_path"]
        self.seq_lane_ids: List[Set[int]] = condition_res["seq_lane_ids"]
        self.seq_lane_ids_raw: List[List[int]] = condition_res[
            "seq_lane_ids_raw"
        ]

        start_lane_seqs_ind = condition_res["start_lane_seqs_ind"]
        end_lane_seqs_ind = condition_res["end_lane_seqs_ind"]
        ego_per_lane_seq_mask = condition_res["ego_per_lane_seq_path_mask"]

        self.lane_seq_pair: List[Tuple[int, int, np.ndarray]] = []
        for idx, (start_idx, end_idx) in enumerate(
            zip(start_lane_seqs_ind, end_lane_seqs_ind)
        ):
            if start_idx == -1 and end_idx == -1:
                continue
            self.lane_seq_pair.append(
                (start_idx, end_idx, ego_per_lane_seq_mask[idx])
            )

    def __repr__(self) -> str:
        return f"{self.file_path}, {len(self.seq_lane_ids_raw)}, {len(self.lane_seq_pair)}"


class EgoPathInfo:
    def __init__(self, ego_path_dict: Dict, consider_length=34) -> None:
        self.future_path: List[Tuple[float, float]] = ego_path_dict[
            "future_path"
        ][:consider_length]
        if len(self.future_path) <= 1:
            self.future_path.append(self.future_path[-1])
        self.future_path_linestring: LineString = LineString(self.future_path)
        self.in_junction_id: List[Union[None, str, int]] = ego_path_dict[
            "in_junction_id"
        ][:consider_length]
        self.corr_frame_idx: List[Union[None, int]] = ego_path_dict[
            "corr_frame_idx"
        ][:consider_length]
        self.corr_lane_id: List[
            List[Union[None, Tuple[Union[str, int], float]]]
        ] = (ego_path_dict["corr_lane_id"])[:consider_length]
        self.vt_profile: List[float] = ego_path_dict["vt_profile"][
            :consider_length
        ]
        self.lane_seq_valid_len: List[
            Tuple[List[str], int, int]
        ] = ego_path_dict["lane_seq_valid_len"]
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


class EgoObstacleLaneSeqInfo:
    def __init__(self, info, percepmap: PercepMap) -> None:
        self.current_lanes: List[int] = info["lane_graph"]["current_lanes"]
        self.current_lane_seqs: List[List[int]] = []
        self.nearby_lane_seqs: List[List[int]] = []
        self.keep: bool = info["lane_graph"]["keep"]

        tmp_current_lane_seqs = []
        tmp_nearby_lane_seqs = []
        for lane_seq in info["lane_graph"]["lane_sequence"]:
            lane_ids = []
            for lane_seg in lane_seq["lane_segment"]:
                lane_info = percepmap.lane_map[lane_seg["lane_id"]]
                if lane_info["lane_category"] != "REALITY":
                    break
                lane_ids.append(lane_seg["lane_id"])
            if len(lane_ids) != 0:
                if lane_ids[0] in self.current_lanes:
                    tmp_current_lane_seqs.append(lane_ids)
                else:
                    tmp_nearby_lane_seqs.append(lane_ids)

        for current_lane_seq in tmp_current_lane_seqs:
            if not any(
                [
                    set(current_lane_seq) == set(l)
                    for l in self.current_lane_seqs
                ]
            ):
                self.current_lane_seqs.append(current_lane_seq)

        for nearby_lane_seq in tmp_nearby_lane_seqs:
            if not any(
                [set(nearby_lane_seq) == set(l) for l in self.nearby_lane_seqs]
            ):
                self.nearby_lane_seqs.append(nearby_lane_seq)


class JunctionLabelInfo:
    def __init__(
        self, junction_info, percep_map: PercepMap, ego_path_info: EgoPathInfo
    ) -> None:
        self.junction_goal = junction_info["junction_goal"]
        self.junction_id = junction_info["junction_id"]
        self.entry_lanes = junction_info["entry_lanes"]
        self.exit_lanes = junction_info["exit_lanes"]
        self.waiting_area_lane_info = {}

        for entry_lane in self.entry_lanes:
            entry_lane_id = entry_lane["lane_id"]
            cur_entry_lane = percep_map.lane_map[entry_lane_id]
            if cur_entry_lane["lane_category"] == "REALITY" and (
                cur_entry_lane["type"]
                in ["WAIT_LEFT", "WAIT_RIGHT", "WAIT_FORWARD"]
            ):
                self.add_waiting_lane_info(
                    ego_path_info,
                    cur_entry_lane,
                    entry_lane_id,
                    entry_lane_id,
                    entry_lane["pose_s"],
                    entry_lane["pose_l"],
                )

            if len(cur_entry_lane["successor_id"]) == 0:
                continue

            for succ_id in cur_entry_lane["successor_id"]:
                succ_lane = percep_map.lane_map[succ_id]
                if succ_lane["lane_category"] == "REALITY" and (
                    succ_lane["type"]
                    in ["WAIT_LEFT", "WAIT_RIGHT", "WAIT_FORWARD"]
                ):
                    self.add_waiting_lane_info(
                        ego_path_info, succ_lane, succ_id, entry_lane_id
                    )

    def add_waiting_lane_info(
        self,
        ego_path_info: EgoPathInfo,
        waiting_lane: Dict,
        waiting_lane_id: int,
        cur_entry_lane_id: int,
        pose_s=None,
        pose_l=None,
    ) -> None:
        if len(waiting_lane["polyline"]) <= 2:
            return

        waiting_lane_linestring = LineString(waiting_lane["polyline"])

        if pose_s is None or pose_l is None:
            corr_lane_id = ego_path_info.corr_lane_id
            waiting_lane_corr_final_future_point_idx = -1

            for idx, corr_lane_info in enumerate(corr_lane_id):
                if corr_lane_info is None or len(corr_lane_info) == 0:
                    break
                if corr_lane_info[0][0] == waiting_lane_id:
                    waiting_lane_corr_final_future_point_idx = idx
                    pose_l = corr_lane_info[0][1]

            if waiting_lane_corr_final_future_point_idx != -1:
                pose_s = waiting_lane_linestring.project(
                    Point(
                        ego_path_info.future_path[
                            waiting_lane_corr_final_future_point_idx
                        ]
                    )
                )

        waiting_lane_info = (
            waiting_lane_id,
            pose_s,
            pose_l.item() if pose_l is not None else None,
            waiting_lane_linestring.length,
        )

        if cur_entry_lane_id not in self.waiting_area_lane_info:
            self.waiting_area_lane_info[cur_entry_lane_id] = []
        self.waiting_area_lane_info[cur_entry_lane_id].append(waiting_lane_info)


class LabelScene:
    def __init__(
        self, label_path: os.PathLike, s3_client, max_valid_point_num: int = 34
    ) -> None:
        if "s3://" not in label_path:
            with open(label_path, "rb") as f:
                label = pickle.load(f)
        else:
            file_obj = io.BytesIO()
            s3_client.download_fileobj(
                "pnd", label_path.split("s3://pnd/")[1], file_obj
            )
            file_obj.seek(0)
            label = pickle.load(file_obj)
            file_obj.close()

        self.ego_path_info: EgoPathInfo = EgoPathInfo(
            label["ego_path_info"], max_valid_point_num
        )
        self.percepmap: Dict = PercepMap(label["percepmap"])
        self.obstacles: Dict = label["obstacles"]
        self.ego_obs_lane_seq_info = EgoObstacleLaneSeqInfo(
            label["obstacles"][-9], self.percepmap
        )
        self.junction_label_info = JunctionLabelInfo(
            label["obstacles"][-9]["junction_info"],
            self.percepmap,
            self.ego_path_info,
        )

        from raw_data_preprocess.compose_pipelines import compose_pipelines

        compose_pipelines[-1].max_path_sample_points = max_valid_point_num

        for pipe in compose_pipelines:
            label["file_path"] = label_path
            pipe_res = pipe(label)

        self.label_res = pipe_res


class TagData:
    def __init__(
        self,
        label_path: os.PathLike,
        condition_path: os.PathLike,
        s3_client,
        max_valid_point_num: int = 34,
    ) -> None:
        self.label_scene: LabelScene = LabelScene(
            label_path, s3_client, max_valid_point_num
        )
        self.condition_res: ConditionRes = ConditionRes(
            self.label_scene.label_res
        )
