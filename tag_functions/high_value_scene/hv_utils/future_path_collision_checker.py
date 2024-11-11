from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import Polygon
from base import EgoPathInfo
from tag_functions.high_value_scene.hv_utils.collision_detector import (
    CollisionDetector,
    CollisionResult,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    build_ego_vehicle_polygon,
    build_obstacle_bbox,
)


class FuturePathCollisionChecker:
    def __init__(self):
        pass

    def update_future_narrow_road_states(
        self,
        future_narrow_road_states: List[List[bool]],
        future_narrow_road_states_loose_threshold: List[List[bool]],
        future_narrow_road_curb_index: List[List[int]],
        collision_res: CollisionResult,
        idx: int,
    ) -> None:
        if len(future_narrow_road_states) <= idx:
            future_narrow_road_states.append([False, False])

        if len(future_narrow_road_states_loose_threshold) <= idx:
            future_narrow_road_states_loose_threshold.append([False, False])

        if len(future_narrow_road_curb_index) <= idx:
            future_narrow_road_curb_index.append(
                [collision_res.left_curb_index, collision_res.right_curb_index]
            )

        future_narrow_road_states[idx][0] |= collision_res.has_obs_left_strict
        future_narrow_road_states[idx][1] |= collision_res.has_obs_right_strict
        future_narrow_road_states_loose_threshold[idx][
            0
        ] |= collision_res.has_obs_left_loose
        future_narrow_road_states_loose_threshold[idx][
            1
        ] |= collision_res.has_obs_right_loose

    def check_future_path_distance_to_curb_and_static_obs(
        self,
        params: Dict,
        ego_path_info: EgoPathInfo,
        ego_obstacle: Dict,
        static_obstacles_map: Dict,
        static_obstacles_polygons_map: Dict,
        curbs_linestring_map: Dict,
        curbs_interactive_lat_type: Dict,
    ) -> Tuple[List[List[bool]], List[List[bool]]]:
        future_narrow_road_states = []
        future_narrow_road_states_loose_threshold = []
        future_narrow_road_curb_index = []

        collision_detector = CollisionDetector(
            params["big_car_area"],
            params["near_static_obs_dist_strict"],
            params["near_static_obs_dist_loose"],
            params["near_moving_obs_dist"],
            params["near_caution_obs_dist"],
        )

        for idx, (x, y) in enumerate(ego_path_info.future_path):
            veh_polygon = build_ego_vehicle_polygon(
                ego_path_info.future_path, idx, ego_obstacle
            )
            collision_res = collision_detector.check_distance_to_curb(
                veh_polygon, curbs_linestring_map, curbs_interactive_lat_type
            )
            self.update_future_narrow_road_states(
                future_narrow_road_states,
                future_narrow_road_states_loose_threshold,
                future_narrow_road_curb_index,
                collision_res,
                idx,
            )

            if (
                collision_res.has_obs_left_strict
                and collision_res.has_obs_right_strict
            ):
                continue

            collision_res = collision_detector.check_distance_to_static_obs(
                veh_polygon, static_obstacles_map, static_obstacles_polygons_map
            )
            self.update_future_narrow_road_states(
                future_narrow_road_states,
                future_narrow_road_states_loose_threshold,
                future_narrow_road_curb_index,
                collision_res,
                idx,
            )

        return (
            future_narrow_road_states,
            future_narrow_road_states_loose_threshold,
            future_narrow_road_curb_index
        )

    def check_distance_to_moving_obs_for_future_states(
        self,
        params: Dict,
        ego_obstacle: Dict,
        moving_obstacles_map: Dict,
        moving_obstacles_future_state_polygons_map: Dict,
    ) -> List[List[bool]]:
        collision_detector = CollisionDetector(
            params["big_car_area"],
            params["near_static_obs_dist_strict"],
            params["near_static_obs_dist_loose"],
            params["near_moving_obs_dist"],
            params["near_caution_obs_dist"],
        )

        ego_future_states = ego_obstacle["future_trajectory"]["future_states"]
        ego_length = ego_obstacle["features"]["length"]
        ego_width = ego_obstacle["features"]["width"]

        future_interaction_with_moving_obs = []

        for idx, ego_state in enumerate(ego_future_states):
            ts_us = ego_state["timestamp"]
            ego_bbox = build_obstacle_bbox(
                ego_state["x"],
                ego_state["y"],
                ego_state["theta"],
                ego_length,
                ego_width,
            )
            ego_polygon = Polygon(
                [
                    ego_bbox[0],
                    ego_bbox[1],
                    ego_bbox[2],
                    ego_bbox[3],
                    ego_bbox[0],
                ]
            )
            collision_res = collision_detector.check_distance_to_moving_obs(
                ego_polygon,
                moving_obstacles_map,
                moving_obstacles_future_state_polygons_map[ts_us],
            )
            future_interaction_with_moving_obs.append(
                [
                    collision_res.has_obs_left_strict,
                    collision_res.has_obs_right_strict,
                ]
            )

        return future_interaction_with_moving_obs

    def check_future_path_bypass_static_object_in_junction(
        self,
        large_curvature_threshold: float,
        ego_path_info: EgoPathInfo,
        future_path_curvature: np.ndarray,
        future_path_turn_type: np.ndarray,
        future_narrow_road_states_loose_threshold,
    ) -> List[bool]:
        future_bypass_junction_curb = []
        in_junction_id = ego_path_info.in_junction_id

        for idx, (x, y) in enumerate(ego_path_info.future_path):
            future_bypass_junction_curb.append(False)

            if idx < len(in_junction_id) and in_junction_id[idx] is None:
                continue

            if abs(future_path_curvature[idx]) < large_curvature_threshold:
                continue

            if (
                future_path_turn_type[idx] > 0
                and future_narrow_road_states_loose_threshold[idx][0]
            ) or (
                future_path_turn_type[idx] < 0
                and future_narrow_road_states_loose_threshold[idx][1]
            ):
                future_bypass_junction_curb[idx] = True

        return future_bypass_junction_curb
