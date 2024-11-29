from enum import Enum
from shapely.geometry import LineString, Polygon, Point
from typing import Dict, List, Tuple
import numpy as np
from tag_functions.high_value_scene.common.tag_type import LaneChangeDirection


class ConditionLineCorrType(Enum):
    NONE = 0
    START = 1
    END = 2


class BasicInfo:
    def __init__(self) -> None:
        self.future_narrow_road_states: List[List[bool]] = None
        self.future_narrow_road_states_loose_threshold: List[List[bool]] = None
        self.future_path_nearby_curb_indexes: List[List[int]] = None
        self.future_bypass_junction_curb: List[bool] = None
        self.future_interaction_with_moving_obs: List[List[bool]] = None

        self.static_obstacles_map: Dict[int, Dict] = None
        self.static_obstacles_polygons_map: Dict[int, Polygon] = None

        self.moving_obstacles_map: Dict[int, Dict] = None
        self.moving_obstacles_future_state_polygons_map: Dict[int, Dict] = None

        self.curbs_linestring_map: Dict[int, LineString] = None

        self.future_path_curvature: np.ndarray = None
        self.future_path_turn_type: np.ndarray = None
        self.max_abs_path_curvature: float = 0.0

        self.is_cross_junction: bool = False
        self.is_ego_vehicle_always_moving: bool = False

        self.nearest_condition_linestring: List[LineString] = None
        self.nearest_condition_linestring_corr_type: List[ConditionLineCorrType] = None
        self.future_path_points_sl_coordinate_projected_to_condition: List[
            Tuple[float, float, Point]
        ] = []
        self.future_path_points_sl_coordinate_projected_to_condition_corr_type: List[ConditionLineCorrType] = []

        self.lane_change_direction: LaneChangeDirection = (
            LaneChangeDirection.UNKNOWN
        )

        self.lane_id_to_future_path_waypoint_count: Dict[int, int] = {}
