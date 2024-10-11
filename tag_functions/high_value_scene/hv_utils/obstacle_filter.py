from typing import Dict, List
from shapely.geometry import LineString, Point, Polygon
from tag_functions.high_value_scene.hv_utils.basic_func import (
    is_static,
    get_bbox,
)


class ObstacleFilter:
    def __init__(self, params):
        self.params = params
        self.filter_obs_max_l = 5.0

    def get_moving_obs(self, obstacles):
        moving_obs = {}
        for id, obs in obstacles.items():
            if id == -9:
                continue
            if abs(obs['decision']['obs_l']) > self.filter_obs_max_l:
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
            if abs(obs['decision']['obs_l']) > self.filter_obs_max_l:
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

    def get_curbs_linestring(self, curbs, curbs_l):
        curbs_linestring = {}
        for idx, curb in enumerate(curbs):
            if abs(curbs_l[idx]) > self.filter_obs_max_l:
                continue
            curbs_linestring[idx] = LineString(curb)
        return curbs_linestring
