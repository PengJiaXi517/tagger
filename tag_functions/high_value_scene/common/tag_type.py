from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


@dataclass(repr=False)
class BypassJunctionCurbTag:
    def __init__(self) -> None:
        self.is_bypass_junction_curb: bool = False

    def as_dict(self):
        return {
            "is_bypass_junction_curb": self.is_bypass_junction_curb,
        }


@dataclass(repr=False)
class LaneCentralAdsorbTag:
    def __init__(self) -> None:
        self.adsorbed_path: List[Tuple[float, float]] = []

    def as_dict(self):
        return {
            "adsorbed_path": self.adsorbed_path,
        }


@dataclass(repr=False)
class QuickLaneChangeTag:
    def __init__(self) -> None:
        self.is_lane_change: bool = False
        self.is_quick_lane_change: bool = False
        self.lane_change_begin_index: int = -1

    def as_dict(self):
        return {
            "is_lane_change": self.is_lane_change,
            "is_quick_lane_change": self.is_quick_lane_change,
            "lane_change_begin_index": self.lane_change_begin_index,
        }


@dataclass(repr=False)
class InteractWithMovingObsTag:
    def __init__(self) -> None:
        self.is_interact_with_moving_obs: bool = False

    def as_dict(self):
        return {
            "is_interact_with_moving_obs": self.is_interact_with_moving_obs,
        }


@dataclass(repr=False)
class NarrowRoadTag:
    def __init__(self) -> None:
        self.is_narrow_road: bool = False

    def as_dict(self):
        return {
            "is_narrow_road": self.is_narrow_road,
        }


@dataclass(repr=False)
class RampTag:
    def __init__(self) -> None:
        self.is_enter_ramp: bool = False
        self.is_exit_ramp: bool = False

    def as_dict(self):
        return {
            "is_enter_ramp": self.is_enter_ramp,
            "is_exit_ramp": self.is_exit_ramp,
        }


@dataclass(repr=False)
class YieldVRUTag:
    def __init__(self) -> None:
        self.is_yield_vru: bool = False

    def as_dict(self):
        return {
            "is_yield_vru": self.is_yield_vru,
        }


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


class LaneChangeDirection(Enum):
    UNKNOWN = -1
    LANE_CHANGE_TO_LEFT = 0
    LANE_CHANGE_TO_RIGHT = 1


class JunctionTurnType(Enum):
    UNKNWON = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    UTURN = 4


JUNCTION_GOAL_2_TURN_TYPE = {
    "UNKNOWN": JunctionTurnType.UNKNWON,
    "TYPE_FORWARD": JunctionTurnType.FORWARD,
    "TYPE_TURN_LEFT": JunctionTurnType.LEFT,
    "TYPE_TURN_RIGHT": JunctionTurnType.RIGHT,
    "TYPE_U_TURN": JunctionTurnType.UTURN,
}


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


@dataclass
class RightTurnOnlyTag:
    is_right_turn_only: bool = False
    right_turn_only_valid_path_len: int = 0

    def as_dict(self):
        return {
            "is_right_turn_only": self.is_right_turn_only,
            "right_turn_only_valid_path_len": self.right_turn_only_valid_path_len,
        }


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


class FuturePATHType(Enum):
    UNKNOWN = 0
    CRUISE = 1
    LANE_CHANGE = 2
    CROSS_JUNCTION_LC = 3
    CROSS_JUNCTION_CRUISE = 4
    CROSS_JUNCTION_UNKNWON = 5


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


@dataclass(repr=False)
class HighValueTag:
    narrow_road_tag: NarrowRoadTag = None
    bypass_junction_curb_tag: BypassJunctionCurbTag = None
    yield_vru_tag: YieldVRUTag = None
    interact_with_moving_obs_tag: InteractWithMovingObsTag = None
    ramp_tag: RampTag = None
    future_path_tag: FuturePathTag = field(default_factory=FuturePathTag)
    lane_central_adsorb_tag: LaneCentralAdsorbTag = None
    quick_lane_change_tag: QuickLaneChangeTag = None
    right_turn_only_tag: RightTurnOnlyTag = None
    max_abs_path_curvature: float = 0.0

    def as_dict(self):
        ret_dict = {
            "narrow_road_tag": self.narrow_road_tag.as_dict()
            if self.narrow_road_tag is not None
            else None,
            "bypass_junction_curb_tag": self.bypass_junction_curb_tag.as_dict()
            if self.bypass_junction_curb_tag is not None
            else None,
            "yield_vru_tag": self.yield_vru_tag.as_dict()
            if self.yield_vru_tag is not None
            else None,
            "interact_with_moving_obs_tag": self.interact_with_moving_obs_tag.as_dict()
            if self.interact_with_moving_obs_tag is not None
            else None,
            "ramp_tag": self.ramp_tag.as_dict()
            if self.ramp_tag is not None
            else None,
            "lane_central_adsorb_tag": self.lane_central_adsorb_tag.as_dict()
            if self.lane_central_adsorb_tag is not None
            else None,
            "quick_lane_change_tag": self.quick_lane_change_tag.as_dict()
            if self.quick_lane_change_tag is not None
            else None,
            "right_turn_only_tag": self.right_turn_only_tag.as_dict()
            if self.right_turn_only_tag is not None
            else None,
            "max_abs_path_curvature": self.max_abs_path_curvature,
        }

        ret_dict.update(self.future_path_tag.as_dict())

        return ret_dict
