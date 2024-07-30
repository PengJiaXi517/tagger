from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from base import PercepMap, TagData
from registry import TAG_FUNCTIONS


@dataclass(repr=False)
class CruisePATHTag:
    real_pose_l: float = 0.0
    percep_pose_l: float = 0.0
    max_pose_l: float = 0.0
    mean_pose_l: float = 0.0
    max_continuous_length_on_lane: float = 0.0
    labeled_lane_seq: List[int] = field(default_factory=list)

    def as_dict(self):
        return {
            "real_pose_l": self.real_pose_l,
            "percep_pose_l": self.percep_pose_l,
            "max_pose_l": self.max_pose_l,
            "mean_pose_l": self.mean_pose_l,
            "max_continuous_length_on_lane": self.max_continuous_length_on_lane,
            "labeled_lane_seq": self.labeled_lane_seq,
        }


@dataclass(repr=False)
class LcPATHTag:
    start_pose_l: float = 0.0
    arrive_final_pose_l: float = 0.0
    arrive_nearest_pose_l: float = 0.0
    arrive_percep_pose_l: float = 0.0
    arrive_length: float = 0.0
    labeled_lane_seq: List[int] = field(default_factory=list)

    def as_dict(self):
        return {
            "start_pose_l": self.start_pose_l,
            "arrive_final_pose_l": self.arrive_final_pose_l,
            "arrive_nearest_pose_l": self.arrive_nearest_pose_l,
            "arrive_percep_pose_l": self.arrive_percep_pose_l,
            "arrive_length": self.arrive_length,
            "labeled_lane_seq": self.labeled_lane_seq,
        }


class JunctionTurnType(Enum):
    UNKNWON = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    UTURN = 4


@dataclass
class JunctionPATHTag:
    turn_type: JunctionTurnType = JunctionTurnType.UNKNWON

    has_entry_lane: bool = False
    label_entry_lane_id: int = 0
    percep_lane_entry_pose_l: float = 0.0
    has_real_arrive_entry_lane: bool = False
    real_lane_entry_pose_l: float = 0.0

    has_exit_lane: bool = False
    label_exit_lane_id: int = 0
    percep_lane_exit_pose_l: float = 0.0
    has_real_arrive_exit_lane: bool = False
    real_lane_exit_pose_l: float = 0.0

    max_length_in_exit_lane: float = 0.0
    max_length_not_in_exit_lane: float = 0.0

    hit_point_num: int = 0
    min_pose_l_2_exit_lane: float = 0.0
    max_pose_l_2_exit_lane: float = 0.0
    mean_pose_l_2_exit_lane: float = 0.0

    has_junction_label: bool = False
    first_arrive_junction_id: int = 0
    label_junction_id: int = 0

    def as_dict(self):
        return {
            "turn_type": self.turn_type.name,
            #
            "has_entry_lane": self.has_entry_lane,
            "label_entry_lane_id": self.label_entry_lane_id,
            "percep_lane_entry_pose_l": self.percep_lane_entry_pose_l,
            "has_real_arrive_entry_lane": self.has_real_arrive_entry_lane,
            "real_lane_entry_pose_l": self.real_lane_entry_pose_l,
            #
            "has_exit_lane": self.has_exit_lane,
            "label_exit_lane_id": self.label_exit_lane_id,
            "percep_lane_exit_pose_l": self.percep_lane_exit_pose_l,
            "has_real_arrive_exit_lane": self.has_real_arrive_exit_lane,
            "real_lane_exit_pose_l": self.real_lane_exit_pose_l,
            #
            "max_length_in_exit_lane": self.max_length_in_exit_lane,
            "max_length_not_in_exit_lane": self.max_length_not_in_exit_lane,
            #
            "hit_point_num": self.hit_point_num,
            "min_pose_l_2_exit_lane": self.min_pose_l_2_exit_lane,
            "max_pose_l_2_exit_lane": self.max_pose_l_2_exit_lane,
            "mean_pose_l_2_exit_lane": self.mean_pose_l_2_exit_lane,
            #
            "has_junction_label": self.has_junction_label,
            "first_arrive_junction_id": self.first_arrive_junction_id,
            "label_junction_id": self.label_junction_id,
        }


@dataclass
class ConditionResTag:
    start_lane_seq_ids: List[List[int]] = field(default_factory=list)
    end_lane_seq_ids: List[List[int]] = field(default_factory=list)

    def as_dict(self):
        return {
            "start_lane_seq_ids": self.start_lane_seq_ids,
            "end_lane_seq_ids": self.end_lane_seq_ids,
        }


def label_condition_res_tag(data: TagData, params: Dict) -> ConditionResTag:
    condition_pair = data.condition_res.lane_seq_pair
    lane_seq_ids = data.condition_res.seq_lane_ids_raw

    condition_res_tag = ConditionResTag()
    for idx_start, idx_end, _ in condition_pair:
        if idx_start != -1:
            condition_res_tag.start_lane_seq_ids.append(lane_seq_ids[idx_start])
        else:
            condition_res_tag.start_lane_seq_ids.append([])
        if idx_end != -1:
            condition_res_tag.end_lane_seq_ids.append(lane_seq_ids[idx_end])
        else:
            condition_res_tag.end_lane_seq_ids.append([])

    return condition_res_tag


class FuturePATHType(Enum):
    UNKNOWN = 0
    CRUISE = 1
    LANE_CHANGE = 2
    CROSS_JUNCTION_LC = 3
    CROSS_JUNCTION_CRUISE = 4
    CROSS_JUNCTION_UNKNWON = 5


@dataclass(repr=False)
class BasicPathTag:
    valid_path_len: float = 0.0
    sum_path_curvature: float = 0.0
    abs_sum_path_curvature: float = 0.0
    pos_sum_path_curvature: float = 0.0
    neg_sum_path_curvature: float = 0.0

    def as_dict(self):
        return {
            "valid_path_len": self.valid_path_len,
            "sum_path_curvature": self.sum_path_curvature,
            "abs_sum_path_curvature": self.abs_sum_path_curvature,
            "pos_sum_path_curvature": self.pos_sum_path_curvature,
            "neg_sum_path_curvature": self.neg_sum_path_curvature,
        }


@dataclass(repr=False)
class FuturePathTag:
    path_type: FuturePATHType = FuturePATHType.UNKNOWN
    basic_path_tag: BasicPathTag = None
    cruise_path_tag: List[CruisePATHTag] = None
    lc_path_tag: List[LcPATHTag] = None
    junction_path_tag: JunctionPATHTag = None
    condition_res_tag: ConditionResTag = None
    is_backing_up: bool = False

    def as_dict(self):
        return {
            "path_type": self.path_type.name,
            "basic_path_tag": (
                self.basic_path_tag.as_dict()
                if self.basic_path_tag is not None
                else None
            ),
            "cruise_path_tag": (
                [tag.as_dict() for tag in self.cruise_path_tag]
                if self.cruise_path_tag is not None
                else None
            ),
            "lc_path_tag": (
                [tag.as_dict() for tag in self.lc_path_tag]
                if self.lc_path_tag is not None
                else None
            ),
            "junction_path_tag": (
                self.junction_path_tag.as_dict()
                if self.junction_path_tag is not None
                else None
            ),
            "condition_res_tag": (
                self.condition_res_tag.as_dict()
                if self.condition_res_tag is not None
                else None
            ),
            "is_backing_up": self.is_backing_up,
        }


def label_cruise_tag(
    data: TagData,
    params: Dict,
    always_on_current_lane_seq,
    percep_map: PercepMap,
) -> List[CruisePATHTag]:
    ego_path_info = data.label_scene.ego_path_info
    num_points_on_lane = 0
    final_corr_lane_ids = None
    for corr_lane_ids in ego_path_info.corr_lane_id:
        if corr_lane_ids is None:
            break
        final_corr_lane_ids = corr_lane_ids
        num_points_on_lane += 1

    if final_corr_lane_ids is None:
        return []

    path_line_string = ego_path_info.future_path_linestring

    cruise_tags = []

    for lane_seq in always_on_current_lane_seq:
        cruise_tag = CruisePATHTag()

        cruise_tag.labeled_lane_seq = lane_seq
        cruise_tag.max_continuous_length_on_lane = max(
            (num_points_on_lane - 1) * 3.0, 0.0
        )

        polylines = [
            LineString(percep_map.lane_map[lane_id]["polyline"]) for lane_id in lane_seq
        ]

        pose_ls = []
        for i in range(num_points_on_lane):
            point = Point(path_line_string.coords[i])

            curr_pose_l = np.min([pl.distance(point) for pl in polylines])

            pose_ls.append(curr_pose_l)
        if len(pose_ls) > 0:
            cruise_tag.percep_pose_l = pose_ls[-1]
            cruise_tag.max_pose_l = np.max(pose_ls)
            cruise_tag.mean_pose_l = np.mean(pose_ls)
        for lane_id, pose_l in final_corr_lane_ids:
            if lane_id in lane_seq:
                cruise_tag.real_pose_l = float(np.abs(pose_l))
        cruise_tags.append(cruise_tag)

    return cruise_tags


def label_lc_tag(
    data: TagData,
    params: Dict,
    arrive_on_nearby_lane_seq,
    percep_map: PercepMap,
) -> LcPATHTag:
    ego_path_info = data.label_scene.ego_path_info
    num_points_on_lane = 0
    final_corr_lane_ids = None
    for corr_lane_ids in ego_path_info.corr_lane_id:
        if corr_lane_ids is None:
            break
        final_corr_lane_ids = corr_lane_ids
        num_points_on_lane += 1

    if final_corr_lane_ids is None:
        return []

    path_line_string = ego_path_info.future_path_linestring

    lc_path_tags = []

    for lane_seq in arrive_on_nearby_lane_seq:
        polylines = [
            LineString(percep_map.lane_map[lane_id]["polyline"]) for lane_id in lane_seq
        ]

        first_arrive_lane_seq_idx = num_points_on_lane - 1

        for i, corr_lane_ids in enumerate(ego_path_info.corr_lane_id):
            if corr_lane_ids is not None:
                for lane_id, _ in corr_lane_ids:
                    if lane_id in lane_seq:
                        first_arrive_lane_seq_idx = i
                        break
            if first_arrive_lane_seq_idx != num_points_on_lane - 1:
                break

        lc_path_tag = LcPATHTag()
        lc_path_tag.labeled_lane_seq = lane_seq
        lc_path_tag.arrive_length = (
            num_points_on_lane - first_arrive_lane_seq_idx
        ) * 3.0

        start_point = Point(path_line_string.coords[0])

        lc_path_tag.start_pose_l = np.min(
            [pl.distance(start_point) for pl in polylines]
        )

        pose_ls = []
        for i in range(first_arrive_lane_seq_idx, num_points_on_lane):
            point = Point(path_line_string.coords[i])
            curr_pose_l = np.min([pl.distance(point) for pl in polylines])
            pose_ls.append(curr_pose_l)

        lc_path_tag.arrive_nearest_pose_l = np.min(pose_ls)
        lc_path_tag.arrive_percep_pose_l = pose_ls[-1]
        for lane_id, pose_l in final_corr_lane_ids:
            if lane_id in lane_seq:
                lc_path_tag.arrive_final_pose_l = float(np.abs(pose_l))

        lc_path_tags.append(lc_path_tag)

    return lc_path_tags


JUNCTION_GOAL_2_TURN_TYPE = {
    "UNKNOWN": JunctionTurnType.UNKNWON,
    "TYPE_FORWARD": JunctionTurnType.FORWARD,
    "TYPE_TURN_LEFT": JunctionTurnType.LEFT,
    "TYPE_TURN_RIGHT": JunctionTurnType.RIGHT,
    "TYPE_U_TURN": JunctionTurnType.UTURN,
}


def label_junction_tag(
    data: TagData, params: Dict, percep_map: PercepMap
) -> JunctionPATHTag:
    junction_path_tag = JunctionPATHTag()

    junction_label_info = data.label_scene.junction_label_info
    ego_path_info = data.label_scene.ego_path_info
    ego_path_linestring = data.label_scene.ego_path_info.future_path_linestring

    junction_path_tag.turn_type = JUNCTION_GOAL_2_TURN_TYPE[
        junction_label_info.junction_goal
    ]

    if junction_label_info.junction_id == "":
        junction_path_tag.has_junction_label = False
        return junction_path_tag

    junction_path_tag.has_junction_label = True
    for in_junction_id in ego_path_info.in_junction_id:
        if in_junction_id is not None:
            junction_path_tag.first_arrive_junction_id = in_junction_id
    junction_path_tag.label_junction_id = junction_label_info.junction_id

    if len(junction_label_info.entry_lanes) > 0:
        junction_path_tag.has_entry_lane = True
        nearest_lane_info = min(
            junction_label_info.entry_lanes, key=lambda x: np.abs(x["pose_l"])
        )
        nearest_lane_id = nearest_lane_info["lane_id"]
        nearest_pose_l = float(np.abs(nearest_lane_info["pose_l"]))

        junction_path_tag.label_entry_lane_id = nearest_lane_id
        junction_path_tag.percep_lane_entry_pose_l = nearest_pose_l

        final_entry_corr_lane_ids = None

        for corr_lane_ids in ego_path_info.corr_lane_id:
            if corr_lane_ids is None:
                break
            final_entry_corr_lane_ids = corr_lane_ids

        if final_entry_corr_lane_ids is not None:
            for lane_id, pose_l in final_entry_corr_lane_ids:
                if lane_id == nearest_lane_id:
                    junction_path_tag.has_real_arrive_entry_lane = True
                    junction_path_tag.real_lane_entry_pose_l = float(np.abs(pose_l))
                    break

    if len(junction_label_info.exit_lanes) > 0:
        junction_path_tag.has_exit_lane = True

        nearest_lane_info = min(
            junction_label_info.exit_lanes, key=lambda x: np.abs(x["pose_l"])
        )
        nearest_lane_id = nearest_lane_info["lane_id"]
        nearest_pose_l = float(np.abs(nearest_lane_info["pose_l"]))

        junction_path_tag.label_exit_lane_id = nearest_lane_id
        junction_path_tag.percep_lane_exit_pose_l = nearest_pose_l

        is_arrive_junction = False

        corr_lane_ids_after_junction = []
        after_in_junction_idx = []

        for i, (corr_lane_ids, in_junction_id) in enumerate(
            zip(ego_path_info.corr_lane_id, ego_path_info.in_junction_id)
        ):
            if (
                in_junction_id is not None
                and in_junction_id == junction_label_info.junction_id
            ):
                is_arrive_junction = True

            if is_arrive_junction:
                if in_junction_id is not None:
                    if in_junction_id == junction_label_info.junction_id:
                        corr_lane_ids_after_junction = []
                        after_in_junction_idx = []
                    else:
                        break
                else:
                    corr_lane_ids_after_junction.append(corr_lane_ids)
                    after_in_junction_idx.append(i)

        if len(corr_lane_ids_after_junction) > 0:
            real_lane_exit_pose_l = None

            max_length_in_exit_lane = 0.0
            max_length_not_in_exit_lane = 0.0
            for corr_lane_ids in corr_lane_ids_after_junction:
                corr_real_lane = False
                for lane_id, pose_l in corr_lane_ids:
                    if lane_id == nearest_lane_id:
                        if real_lane_exit_pose_l is None:
                            real_lane_exit_pose_l = float(np.abs(pose_l))
                            junction_path_tag.real_lane_exit_pose_l = (
                                real_lane_exit_pose_l
                            )
                            junction_path_tag.has_real_arrive_exit_lane = True
                        corr_real_lane = True
                        break
                if corr_real_lane:
                    max_length_in_exit_lane += 3.0
                else:
                    max_length_not_in_exit_lane += 3.0

            junction_path_tag.max_length_in_exit_lane = max_length_in_exit_lane
            junction_path_tag.max_length_not_in_exit_lane = max_length_not_in_exit_lane

        if len(after_in_junction_idx) > 0:
            pose_ls = []
            exit_lane_polyline = LineString(
                percep_map.lane_map[nearest_lane_id]["polyline"]
            )
            for idx in after_in_junction_idx:
                point = Point(ego_path_linestring.coords[idx])
                proj_s = exit_lane_polyline.project(point)
                if proj_s <= 0 or proj_s >= exit_lane_polyline.length:
                    continue
                pose_ls.append(exit_lane_polyline.distance(point))

            junction_path_tag.hit_point_num = len(pose_ls)
            if len(pose_ls) > 0:
                junction_path_tag.min_pose_l_2_exit_lane = np.min(pose_ls)
                junction_path_tag.max_pose_l_2_exit_lane = np.max(pose_ls)
                junction_path_tag.mean_pose_l_2_exit_lane = np.mean(pose_ls)

    return junction_path_tag


def label_backing_up_tag(data: TagData, params: Dict) -> bool:

    return not data.label_scene.ego_path_info.future_path_linestring.is_simple


def judge_path_type(data: TagData, params: Dict) -> FuturePATHType:

    lane_seq_info = data.label_scene.ego_obs_lane_seq_info
    ego_path_info = data.label_scene.ego_path_info

    always_on_current_lane_seq = []
    for lane_seq in lane_seq_info.current_lane_seqs:
        on_curr_lane_seq = False
        for corr_lane_ids, in_junction_id in zip(
            ego_path_info.corr_lane_id, ego_path_info.in_junction_id
        ):
            if corr_lane_ids is not None:
                if any([lane_id in lane_seq for lane_id, pose_l in corr_lane_ids]):
                    on_curr_lane_seq = True
                else:
                    on_curr_lane_seq = False
            else:
                break
        if on_curr_lane_seq:
            always_on_current_lane_seq.append(lane_seq)

    arrive_on_nearby_lane_seq = []
    for lane_seq in lane_seq_info.nearby_lane_seqs:
        on_curr_lane_seq = False
        for corr_lane_ids, in_junction_id in zip(
            ego_path_info.corr_lane_id, ego_path_info.in_junction_id
        ):
            if corr_lane_ids is not None:
                if any([lane_id in lane_seq for lane_id, pose_l in corr_lane_ids]):
                    on_curr_lane_seq = True
                else:
                    on_curr_lane_seq = False
            else:
                break
        if on_curr_lane_seq:
            arrive_on_nearby_lane_seq.append(lane_seq)

    if any(
        [
            in_junction_id is not None
            for in_junction_id in data.label_scene.ego_path_info.in_junction_id
        ]
    ):
        if len(always_on_current_lane_seq) > 0:
            return (
                FuturePATHType.CROSS_JUNCTION_CRUISE,
                always_on_current_lane_seq,
                arrive_on_nearby_lane_seq,
            )
        elif len(arrive_on_nearby_lane_seq) > 0:
            return (
                FuturePATHType.CROSS_JUNCTION_LC,
                always_on_current_lane_seq,
                arrive_on_nearby_lane_seq,
            )
        else:
            return (
                FuturePATHType.CROSS_JUNCTION_UNKNWON,
                always_on_current_lane_seq,
                arrive_on_nearby_lane_seq,
            )

    if len(always_on_current_lane_seq) > 0:
        return (
            FuturePATHType.CRUISE,
            always_on_current_lane_seq,
            arrive_on_nearby_lane_seq,
        )
    elif len(arrive_on_nearby_lane_seq) > 0:
        return (
            FuturePATHType.LANE_CHANGE,
            always_on_current_lane_seq,
            arrive_on_nearby_lane_seq,
        )
    else:
        return (
            FuturePATHType.UNKNOWN,
            always_on_current_lane_seq,
            arrive_on_nearby_lane_seq,
        )


def normal_angle(theta):
    while theta >= np.pi:
        theta -= 2 * np.pi
    while theta <= -np.pi:
        theta += 2 * np.pi
    return theta


applyall = np.vectorize(normal_angle)


def label_basic_tag(data: TagData, params: Dict) -> BasicPathTag:
    def cal_curvature(future_path: List[Tuple[float, float]]) -> Tuple[float, ...]:
        if len(future_path) <= 4:
            return 0.0, 0.0, 0.0, 0.0
        path_points = np.array(future_path)

        diff_points = path_points[1:] - path_points[:-1]

        theta = np.arctan2(diff_points[:, 1], diff_points[:, 0])

        theta_diff = theta[1:] - theta[:-1]

        length = np.linalg.norm(diff_points[:-1], axis=-1)

        theta_diff = applyall(theta_diff)

        curvature = theta_diff / length

        return (
            np.abs(curvature).sum(),
            curvature[curvature > 0.0].sum(),
            curvature[curvature < 0.0].sum(),
            curvature.sum(),
        )

    abs_curvature, pos_curvature, neg_curvature, sum_curvature = cal_curvature(
        data.label_scene.ego_path_info.future_path
    )

    basic_path_tag = BasicPathTag()
    basic_path_tag.valid_path_len = len(data.label_scene.ego_path_info.future_path)
    basic_path_tag.sum_path_curvature = sum_curvature
    basic_path_tag.abs_sum_path_curvature = abs_curvature
    basic_path_tag.pos_sum_path_curvature = pos_curvature
    basic_path_tag.neg_sum_path_curvature = neg_curvature

    return basic_path_tag


@TAG_FUNCTIONS.register()
def future_path_tag(data: TagData, params: Dict) -> Dict:
    future_path_tag = FuturePathTag()
    percep_map = data.label_scene.percepmap

    # Judge Type
    (
        future_path_tag.path_type,
        always_on_current_lane_seq,
        arrive_on_nearby_lane_seq,
    ) = judge_path_type(data, params)

    future_path_tag.basic_path_tag = label_basic_tag(data, params)

    if future_path_tag.path_type in [
        FuturePATHType.CRUISE,
        FuturePATHType.CROSS_JUNCTION_CRUISE,
    ]:
        future_path_tag.cruise_path_tag = label_cruise_tag(
            data,
            params,
            always_on_current_lane_seq,
            percep_map,
        )

    elif future_path_tag.path_type in [
        FuturePATHType.LANE_CHANGE,
        FuturePATHType.CROSS_JUNCTION_LC,
    ]:
        future_path_tag.lc_path_tag = label_lc_tag(
            data, params, arrive_on_nearby_lane_seq, percep_map
        )

    if future_path_tag.path_type in [
        FuturePATHType.CROSS_JUNCTION_CRUISE,
        FuturePATHType.CROSS_JUNCTION_LC,
        FuturePATHType.CROSS_JUNCTION_UNKNWON,
    ]:
        future_path_tag.junction_path_tag = label_junction_tag(data, params, percep_map)

    future_path_tag.condition_res_tag = label_condition_res_tag(data, params)

    future_path_tag.is_backing_up = label_backing_up_tag(data, params)

    return future_path_tag.as_dict()
