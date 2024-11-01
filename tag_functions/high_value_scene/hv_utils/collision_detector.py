from typing import Dict, List
from shapely.geometry import Polygon


class CollisionResult:
    def __init__(self) -> None:
        self.has_obs_left_strict: bool = False
        self.has_obs_left_loose: bool = False
        self.has_obs_right_strict: bool = False
        self.has_obs_right_loose: bool = False


class CollisionDetector:
    def __init__(
        self,
        big_car_area: float,
        near_static_obs_dist_strict: float,
        near_static_obs_dist_loose: float,
        near_moving_obs_dist: float,
        near_caution_obs_dist: float,
    ) -> None:
        self.big_car_area: float = big_car_area
        self.near_static_obs_dist_strict: float = near_static_obs_dist_strict
        self.near_static_obs_dist_loose: float = near_static_obs_dist_loose
        self.near_moving_obs_dist: float = near_moving_obs_dist
        self.near_caution_obs_dist: float = near_caution_obs_dist

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
            if self.fill_static_collision_res(
                dist, lat_decision, collision_res
            ):
                break

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

            if self.fill_static_collision_res(
                dist, lat_decision, collision_res
            ):
                break

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

            dist_th = self.near_moving_obs_dist
            if (
                obs["features"]["type"] == "VEHICLE"
                and obs["features"]["length"] * obs["features"]["width"]
                > self.big_car_area
            ):
                dist_th = self.near_caution_obs_dist
            elif (
                obs["features"]["type"] == "PEDESTRIAN"
                or obs["features"]["type"] == "BICYCLE"
            ):
                dist_th = self.near_caution_obs_dist

            if veh_polygon.distance(obs_polygon) < dist_th:
                if lat_decision == 1:
                    collision_res.has_obs_right_strict = True
                elif lat_decision == 2:
                    collision_res.has_obs_left_strict = True

            if (
                collision_res.has_obs_left_strict
                and collision_res.has_obs_right_strict
            ):
                break

        return collision_res

    def fill_static_collision_res(
        self, dist: float, lat_decision: int, collision_res: CollisionResult
    ) -> bool:
        if dist < self.near_static_obs_dist_strict:
            if lat_decision == 1:
                collision_res.has_obs_right_strict = True
            elif lat_decision == 2:
                collision_res.has_obs_left_strict = True

        if dist < self.near_static_obs_dist_loose:
            if lat_decision == 1:
                collision_res.has_obs_right_loose = True
            elif lat_decision == 2:
                collision_res.has_obs_left_loose = True

        if (
            collision_res.has_obs_left_strict
            and collision_res.has_obs_right_strict
        ):
            return True

        return False
