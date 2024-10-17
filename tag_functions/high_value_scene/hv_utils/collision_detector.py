from typing import Dict, List
from shapely.geometry import LineString, Point, Polygon


class CollisionDetector:
    def __init__(self, params):
        self.params = params

    def check_collision_curb(
        self, veh_polygon, curbs_linestring, curb_lat_decision
    ):
        collision_info = {
            "has_static_obs_left_strict": False,
            "has_static_obs_right_strict": False,
            "has_static_obs_left_relax": False,
            "has_static_obs_right_relax": False,
        }
        for idx, curb_string in curbs_linestring.items():
            lat_decision = curb_lat_decision[idx]
            if lat_decision == 0:
                continue
            dist = curb_string.distance(veh_polygon)
            if dist < self.params.near_static_obs_dist_strict:
                if lat_decision == 1:
                    collision_info["has_static_obs_right_strict"] = True
                elif lat_decision == 2:
                    collision_info["has_static_obs_left_strict"] = True
            if dist < self.params.near_static_obs_dist_relax:
                if lat_decision == 1:
                    collision_info["has_static_obs_right_relax"] = True
                elif lat_decision == 2:
                    collision_info["has_static_obs_left_relax"] = True
            if (
                collision_info["has_static_obs_left_strict"]
                and collision_info["has_static_obs_right_strict"]
            ):
                break
        return collision_info

    def check_collision_static_obs(self, veh_polygon, obstacles, id_polygon):
        collision_info = {
            "has_static_obs_left_strict": False,
            "has_static_obs_right_strict": False,
            "has_static_obs_left_relax": False,
            "has_static_obs_right_relax": False,
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
            if dist < self.params.near_static_obs_dist_strict:
                if lat_decision == 1:
                    collision_info["has_static_obs_right_strict"] = True
                elif lat_decision == 2:
                    collision_info["has_static_obs_left_strict"] = True
            if dist < self.params.near_static_obs_dist_relax:
                if lat_decision == 1:
                    collision_info["has_static_obs_right_relax"] = True
                elif lat_decision == 2:
                    collision_info["has_static_obs_left_relax"] = True
            if (
                collision_info["has_static_obs_left_strict"]
                and collision_info["has_static_obs_right_strict"]
            ):
                break

        return collision_info

    def check_collision_moving_obs(self, veh_polygon, obstacles, id_polygon):
        collision_info = {
            "has_moving_obs_left": False,
            "has_moving_obs_right": False,
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

            if (
                veh_polygon.distance(obs_polygon)
                < self.params.near_moving_obs_dist
            ):
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
