from typing import Dict, List
from shapely.geometry import LineString, Point, Polygon
import numpy as np
from tag_functions.high_value_scene.hv_utils.basic_func import (
    is_static,
    get_bbox,
)


class ObstacleFilter:
    def __init__(self):
        self.filter_obs_max_l = 5.0
        self.front_vehicle_rel_x = 10.0
        self.front_vehicle_rel_y = 0.5

    def get_moving_obs(self, obstacles):
        moving_obs = {}
        for id, obs in obstacles.items():
            if id == -9:
                continue
            if abs(obs["decision"]["obs_l"]) > self.filter_obs_max_l:
                continue
            if not is_static(obs):
                moving_obs[id] = obs

        return moving_obs

    def get_static_obs_polygon(self, obstacles):
        id_polygon = {}
        static_obs = {}
        for id, obs in obstacles.items():
            if id == -9:
                continue
            if abs(obs["decision"]["obs_l"]) > self.filter_obs_max_l:
                continue
            if is_static(obs):
                obs_x = obs["features"]["history_states"][-1]["x"]
                obs_y = obs["features"]["history_states"][-1]["y"]
                obs_theta = obs["features"]["history_states"][-1]["theta"]
                length = obs["features"]["length"]
                width = obs["features"]["width"]
                obs_bbox = get_bbox(obs_x, obs_y, obs_theta, length, width)
                obs_polygon = Polygon(
                    [
                        obs_bbox[0],
                        obs_bbox[1],
                        obs_bbox[2],
                        obs_bbox[3],
                        obs_bbox[0],
                    ]
                )
                id_polygon[id] = obs_polygon
                static_obs[id] = obs

        return static_obs, id_polygon

    def get_curbs_linestring(self, curb_decision):
        curb_vec = curb_decision["vec"]
        curb_src = curb_decision["src_point"]
        curb_end = [
            (curb_src[i][0] + curb_vec[i][0], curb_src[i][1] + curb_vec[i][1])
            for i in range(len(curb_src))
        ]
        curbs = [(curb_src[i], curb_end[i]) for i in range(len(curb_src))]
        curbs_l = curb_decision["obs_l"]

        curbs_linestring = {}
        for idx, curb in enumerate(curbs):
            if abs(curbs_l[idx]) > self.filter_obs_max_l:
                continue
            curbs_linestring[idx] = LineString(curb)
        return curbs_linestring

    def vehicle_in_front(self, obstacles, stop_point, stop_idx):
        ego_future_xy = np.array([stop_point.x, stop_point.y])
        ego_future_theta = obstacles[-9]["future_trajectory"]["future_states"][
            stop_idx
        ]["theta"]
        unit_direct = np.array(
            [np.cos(ego_future_theta), np.sin(ego_future_theta)]
        )
        for idx, obs in obstacles.items():
            if idx == -9 or obs["features"]["type"] != "VEHICLE":
                continue
            if abs(obs["decision"]["obs_l"]) > self.filter_obs_max_l:
                continue
            obs_future_states = obs["future_trajectory"]["future_states"]
            if len(obs_future_states) <= stop_idx:
                continue
            obs_future_state = obs_future_states[stop_idx]
            obs_future_xy = np.array(
                [obs_future_state["x"], obs_future_state["y"]]
            )
            relative_direct = obs_future_xy - ego_future_xy
            relative_x = np.sum(relative_direct * unit_direct)
            relative_y = np.cross(unit_direct, relative_direct)
            if (
                0 < relative_x < self.front_vehicle_rel_x
                and np.abs(relative_y) < self.front_vehicle_rel_y
            ):
                return True
        return False

    def get_ego_point(self, obstacles):
        ego_x = obstacles[-9]["features"]["history_states"][-1]["x"]
        ego_y = obstacles[-9]["features"]["history_states"][-1]["y"]
        return Point([ego_x, ego_y])
