from typing import Dict, List
from shapely.geometry import LineString, Point, Polygon


class CollisionDetector:
    def __init__(self, params: Dict) -> None:
        self.params: Dict = params
        self.big_car_area: int = 18

    def check_distance_to_curb(
        self,
        veh_polygon: Polygon,
        curbs_linestring: Dict,
        curb_lat_decision: Dict,
    ) -> Dict:
        collision_info = {
            "has_static_obs_left_strict": False,
            "has_static_obs_right_strict": False,
            "has_static_obs_left_loose": False,
            "has_static_obs_right_loose": False,
        }

        for idx, curb_string in curbs_linestring.items():
            lat_decision = curb_lat_decision[idx]
            if lat_decision == 0:
                continue

            dist = curb_string.distance(veh_polygon)
            if self.fill_static_collision_info(
                dist, lat_decision, collision_info
            ):
                break

        return collision_info

    def check_distance_to_static_obs(
        self, veh_polygon: Polygon, obstacles: Dict, id_polygon: Dict
    ) -> Dict:
        collision_info = {
            "has_static_obs_left_strict": False,
            "has_static_obs_right_strict": False,
            "has_static_obs_left_loose": False,
            "has_static_obs_right_loose": False,
        }

        for k, v in obstacles.items():
            if k == -9:
                continue
            lat_decision = v["decision"]["interactive_lat_type"]
            if lat_decision == 0:
                continue
            if k not in id_polygon:
                continue

            obs_polygon = id_polygon[k]
            dist = veh_polygon.distance(obs_polygon)
            if self.fill_static_collision_info(
                dist, lat_decision, collision_info
            ):
                break

        return collision_info

    def check_distance_to_moving_obs(
        self, veh_polygon: Polygon, obstacles: Dict, id_polygon: Dict
    ) -> Dict:
        collision_info = {
            "has_moving_obs_left": False,
            "has_moving_obs_right": False,
        }

        for id, obs in obstacles.items():
            if id == -9:
                continue
            lat_decision = obs["decision"]["interactive_lat_type"]
            if lat_decision == 0:
                continue
            if id not in id_polygon:
                continue
            obs_polygon = id_polygon[id]

            dist_th = self.params.near_moving_obs_dist
            if (
                obs["features"]["type"] == "VEHICLE"
                and obs["features"]["length"] * obs["features"]["width"]
                > self.big_car_area
            ):
                dist_th = self.params.near_caution_obs_dist
            elif (
                obs["features"]["type"] == "PEDESTRIAN"
                or obs["features"]["type"] == "BICYCLE"
            ):
                dist_th = self.params.near_caution_obs_dist

            if veh_polygon.distance(obs_polygon) < dist_th:
                if lat_decision == 1:
                    collision_info["has_moving_obs_right"] = True
                elif lat_decision == 2:
                    collision_info["has_moving_obs_left"] = True

            if (
                collision_info["has_moving_obs_left"]
                and collision_info["has_moving_obs_right"]
            ):
                break

        return collision_info

    def fill_static_collision_info(
        self, dist: float, lat_decision: int, collision_info: Dict
    ) -> bool:
        if dist < self.params.near_static_obs_dist_strict:
            if lat_decision == 1:
                collision_info["has_static_obs_right_strict"] = True
            elif lat_decision == 2:
                collision_info["has_static_obs_left_strict"] = True
        if dist < self.params.near_static_obs_dist_loose:
            if lat_decision == 1:
                collision_info["has_static_obs_right_loose"] = True
            elif lat_decision == 2:
                collision_info["has_static_obs_left_loose"] = True
        if (
            collision_info["has_static_obs_left_strict"]
            and collision_info["has_static_obs_right_strict"]
        ):
            return True

        return False
