from typing import Dict, List
from shapely.geometry import Polygon
import numpy as np


class CollisionResult:
    def __init__(self) -> None:
        self.has_obs_left_strict: bool = False
        self.has_obs_right_strict: bool = False
        self.has_obs_left_loose: bool = False
        self.has_obs_right_loose: bool = False
        self.left_curb_index: int = None
        self.right_curb_index: int = None
        self.left_curb_dist: float = np.inf
        self.right_curb_dist: float = np.inf
        self.left_static_obs_dist: float = np.inf
        self.right_static_obs_dist: float = np.inf
        self.left_moving_obs_dist: float = np.inf
        self.right_moving_obs_dist: float = np.inf


class CollisionDetector:
    def __init__(
        self,
        big_car_area: float,
        near_static_obs_dist_strict_th: float,
        near_static_obs_dist_loose_th: float,
        near_moving_obs_dist_th: float,
        near_caution_obs_dist_th: float,
        near_ramp_curb_dist_th: float,
    ) -> None:
        self.big_car_area: float = big_car_area
        self.near_static_obs_dist_strict_th: float = (
            near_static_obs_dist_strict_th
        )
        self.near_static_obs_dist_loose_th: float = (
            near_static_obs_dist_loose_th
        )
        self.near_moving_obs_dist_th: float = near_moving_obs_dist_th
        self.near_caution_obs_dist_th: float = near_caution_obs_dist_th
        self.near_ramp_curb_dist_th: float = near_ramp_curb_dist_th

    def check_distance_to_curb(
        self,
        veh_polygon: Polygon,
        curbs_linestring_map: Dict,
        curbs_interactive_lat_type: Dict,
    ) -> Dict:
        collision_res = CollisionResult()

        for idx, curb_string in curbs_linestring_map.items():
            lat_decision = curbs_interactive_lat_type[idx]
            if lat_decision == 0:
                continue

            dist = curb_string.distance(veh_polygon)

            if lat_decision == 1 and dist < collision_res.right_curb_dist:
                collision_res.right_curb_dist = dist
            elif lat_decision == 2 and dist < collision_res.left_curb_dist:
                collision_res.left_curb_dist = dist

            self.fill_static_collision_res(
                dist, lat_decision, collision_res, idx
            )
        return collision_res

    def check_distance_to_static_obs(
        self,
        veh_polygon: Polygon,
        static_obstacles_map: Dict,
        static_obstacles_polygons_map: Dict,
    ) -> Dict:
        collision_res = CollisionResult()

        for k, v in static_obstacles_map.items():
            if k == -9:
                continue

            lat_decision = v["decision"]["interactive_lat_type"]
            if lat_decision == 0:
                continue

            if k not in static_obstacles_polygons_map:
                continue

            obs_polygon = static_obstacles_polygons_map[k]
            dist = veh_polygon.distance(obs_polygon)

            if lat_decision == 1 and dist < collision_res.right_static_obs_dist:
                collision_res.right_static_obs_dist = dist
            elif lat_decision == 2 and dist < collision_res.left_static_obs_dist:
                collision_res.left_static_obs_dist = dist

            self.fill_static_collision_res(
                dist, lat_decision, collision_res
            )

        return collision_res

    def check_distance_to_moving_obs(
        self,
        veh_polygon: Polygon,
        moving_obstacles_map: Dict,
        moving_obstacles_polygons_map: Dict,
    ) -> Dict:
        collision_res = CollisionResult()

        for id, obs in moving_obstacles_map.items():
            if id == -9:
                continue

            lat_decision = obs["decision"]["interactive_lat_type"]
            if lat_decision == 0:
                continue

            if id not in moving_obstacles_polygons_map:
                continue

            obs_polygon = moving_obstacles_polygons_map[id]

            dist_th = self.near_moving_obs_dist_th
            if (
                obs["features"]["type"] == "VEHICLE"
                and obs["features"]["length"] * obs["features"]["width"]
                > self.big_car_area
            ):
                dist_th = self.near_caution_obs_dist_th
            elif (
                obs["features"]["type"] == "PEDESTRIAN"
                or obs["features"]["type"] == "BICYCLE"
            ):
                dist_th = self.near_caution_obs_dist_th

            dist = veh_polygon.distance(obs_polygon)
            
            if lat_decision == 1 and dist < collision_res.right_moving_obs_dist:
                collision_res.right_moving_obs_dist = dist
            elif lat_decision == 2 and dist < collision_res.left_moving_obs_dist:
                collision_res.left_moving_obs_dist = dist

            if dist < dist_th:
                if lat_decision == 1:
                    collision_res.has_obs_right_strict = True
                elif lat_decision == 2:
                    collision_res.has_obs_left_strict = True

        return collision_res

    def fill_static_collision_res(
        self,
        dist: float,
        lat_decision: int,
        collision_res: CollisionResult,
        curb_idx=None,
    ) -> bool:
        if dist < self.near_static_obs_dist_strict_th:
            if lat_decision == 1:
                collision_res.has_obs_right_strict = True
            elif lat_decision == 2:
                collision_res.has_obs_left_strict = True

        if dist < self.near_static_obs_dist_loose_th:
            if lat_decision == 1:
                collision_res.has_obs_right_loose = True
                collision_res.right_curb_index = curb_idx
            elif lat_decision == 2:
                collision_res.has_obs_left_loose = True
                collision_res.left_curb_index = curb_idx

        if curb_idx is not None and dist < self.near_ramp_curb_dist_th:
            if lat_decision == 1:
                collision_res.right_curb_index = curb_idx
            elif lat_decision == 2:
                collision_res.left_curb_index = curb_idx

        if (
            collision_res.has_obs_left_strict
            and collision_res.has_obs_right_strict
        ):
            return True

        return False
