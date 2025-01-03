from typing import Dict, List
from shapely.geometry import LineString, Point, Polygon
import numpy as np
from tag_functions.high_value_scene.hv_utils.basic_func import (
    is_obstacle_always_static,
    build_obstacle_bbox,
)


class ObstacleFilter:
    def __init__(
        self,
        filter_obs_max_l: float,
        front_vehicle_rel_x: float,
        front_vehicle_rel_y: float,
    ) -> None:
        self.filter_obs_max_l: float = filter_obs_max_l
        self.front_vehicle_rel_x: float = front_vehicle_rel_x
        self.front_vehicle_rel_y: float = front_vehicle_rel_y

    def find_moving_obstacles(self, obstacles: Dict) -> Dict:
        moving_obstacles_map = {}
        for id, obs in obstacles.items():
            if id == -9:
                continue

            if abs(obs["decision"]["obs_l"]) > self.filter_obs_max_l:
                continue

            if (
                obs["decision"]["obs_s"] - obs["decision"]["ego_s"] <= -2.5
                or obs["decision"]["obs_s"] - obs["decision"]["ego_s"] > 100
            ):
                continue

            if not is_obstacle_always_static(obs):
                moving_obstacles_map[id] = obs

        return moving_obstacles_map

    def build_static_obstacle_polygons(
        self, obstacles: Dict, future_path_linestring: LineString
    ) -> Dict:
        static_obstacles_polygons_map = {}
        static_obstacles_map = {}
        for id, obs in obstacles.items():
            if id == -9:
                continue

            if abs(obs["decision"]["obs_l"]) > self.filter_obs_max_l:
                continue

            if (
                obs["decision"]["obs_s"] - obs["decision"]["ego_s"] <= -2.5
                or obs["decision"]["obs_s"] - obs["decision"]["ego_s"] > 100
            ):
                continue

            if is_obstacle_always_static(obs):
                obs_x = obs["features"]["history_states"][-1]["x"]
                obs_y = obs["features"]["history_states"][-1]["y"]
                obs_theta = obs["features"]["history_states"][-1]["theta"]
                length = obs["features"]["length"]
                width = obs["features"]["width"]
                obs_bbox = build_obstacle_bbox(
                    obs_x, obs_y, obs_theta, length, width
                )
                obs_polygon = Polygon(
                    [
                        obs_bbox[0],
                        obs_bbox[1],
                        obs_bbox[2],
                        obs_bbox[3],
                        obs_bbox[0],
                    ]
                )
                if future_path_linestring.intersects(obs_polygon):
                    continue

                static_obstacles_polygons_map[id] = obs_polygon
                static_obstacles_map[id] = obs

        return static_obstacles_map, static_obstacles_polygons_map

    def build_curbs_linestring(self, curb_decision: Dict) -> Dict:
        curbs_linestring = {}
        if curb_decision is None:
            return curbs_linestring
        curb_vec = curb_decision["vec"]
        curb_src = curb_decision["src_point"]
        curb_end = [
            (curb_src[i][0] + curb_vec[i][0], curb_src[i][1] + curb_vec[i][1])
            for i in range(len(curb_src))
        ]
        curbs = [(curb_src[i], curb_end[i]) for i in range(len(curb_src))]
        curbs_l = curb_decision["obs_l"]

        for idx, curb in enumerate(curbs):
            if abs(curbs_l[idx]) > self.filter_obs_max_l:
                continue
            curbs_linestring[idx] = LineString(curb)

        return curbs_linestring

    def has_vehicle_in_front(
        self,
        obstacles: Dict,
        ego_vehicle_stop_point: Point,
        ego_vehicle_stop_idx: int,
    ) -> bool:
        ego_vehicle_stop_point_xy = np.array(
            [ego_vehicle_stop_point.x, ego_vehicle_stop_point.y]
        )
        ego_vehicle_stop_point_theta = obstacles[-9]["future_trajectory"][
            "future_states"
        ][ego_vehicle_stop_idx]["theta"]
        unit_direct = np.array(
            [
                np.cos(ego_vehicle_stop_point_theta),
                np.sin(ego_vehicle_stop_point_theta),
            ]
        )

        for idx, obs in obstacles.items():
            if idx == -9 or obs["features"]["type"] != "VEHICLE":
                continue

            if abs(obs["decision"]["obs_l"]) > self.filter_obs_max_l:
                continue

            obs_future_states = obs["future_trajectory"]["future_states"]
            if len(obs_future_states) <= ego_vehicle_stop_idx:
                continue

            obs_state = obs_future_states[ego_vehicle_stop_idx]
            obs_xy = np.array([obs_state["x"], obs_state["y"]])
            relative_direct = obs_xy - ego_vehicle_stop_point_xy
            relative_x = np.sum(relative_direct * unit_direct)
            relative_y = np.cross(unit_direct, relative_direct)
            if (
                0 < relative_x < self.front_vehicle_rel_x
                and np.abs(relative_y) < self.front_vehicle_rel_y
            ):
                return True

        return False

    def get_ego_point(self, ego_obstacle: Dict) -> Point:
        ego_x = ego_obstacle["features"]["history_states"][-1]["x"]
        ego_y = ego_obstacle["features"]["history_states"][-1]["y"]
        return Point([ego_x, ego_y])
