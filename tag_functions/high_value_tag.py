from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon
from base import PercepMap, TagData
import matplotlib.pyplot as plt
from registry import TAG_FUNCTIONS

@dataclass(repr=False)
class NarrowRoadTag:
    is_narrow_road: bool = False
    def as_dict(self):
        return {
            "is_narrow_road": self.is_narrow_road,
        }


@dataclass(repr=False)
class HighValueTag:
    narrow_road_tag: NarrowRoadTag = None
    def as_dict(self):
        return {
            "narrow_road_tag": self.narrow_road_tag.as_dict(),
        }

def get_bbox(center_x: float, center_y: float, heading: float, length: float, width: float):
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    dx1 = cos_heading * length / 2
    dy1 = sin_heading * length / 2
    dx2 = sin_heading * width / 2
    dy2 = -cos_heading * width / 2
    corner1 = (center_x + dx1 + dx2, center_y + dy1 + dy2)
    corner2 = (center_x + dx1 - dx2, center_y + dy1 - dy2)
    corner3 = (center_x - dx1 - dx2, center_y - dy1 - dy2)
    corner4 = (center_x - dx1 + dx2, center_y - dy1 + dy2)
    bbox = [corner1, corner2, corner3, corner4]
    return bbox

def check_collision(ego_polygon, obstacles, curbs):
    for curb in curbs:
        curb_string = LineString(curb)
        if curb_string.intersects(ego_polygon) or curb_string.crosses(ego_polygon):
            return True 
    for k, v in obstacles.items():
        if k == -9:
            continue
        if not v["features"]["is_still"]:
            continue 
        obs_x = v["features"]["history_states"][-1]["x"]
        obs_y = v["features"]["history_states"][-1]["y"]
        obs_theta = v["features"]["history_states"][-1]["theta"]
        length = v["features"]["length"]
        width = v["features"]["width"]
        obs_bbox = get_bbox(obs_x, obs_y, obs_theta, length, width)
        obs_polygon = Polygon([obs_bbox[0], obs_bbox[1], obs_bbox[2], obs_bbox[3], obs_bbox[0]])
        if ego_polygon.intersects(obs_polygon):
            return True

def label_narrow_road_tag(
    data: TagData, params: Dict) -> NarrowRoadTag:
    narrow_road_tag = NarrowRoadTag()
    ego_path_info = data.label_scene.ego_path_info
    percep_map = data.label_scene.percepmap
    obstacles = data.label_scene.obstacles
    curbs = percep_map.curbs
    for idx, (x, y) in enumerate(ego_path_info.future_path):
        if idx % 3 != 0 or idx == 0:
            continue
        last_x, last_y = ego_path_info.future_path[idx - 1]
        heading = math.atan2(y - last_y, x - last_x)
        shift_dir = (-math.sin(heading), math.cos(heading))
        left_bbox = get_bbox(x + shift_dir[0], y + shift_dir[1], heading, params.veh_length, params.veh_width)
        right_bbox = get_bbox(x - shift_dir[0], y - shift_dir[1], heading, params.veh_length, params.veh_width)
        left_polygon = Polygon([left_bbox[0], left_bbox[1], left_bbox[2], left_bbox[3], left_bbox[0]])
        right_polygon = Polygon([right_bbox[0], right_bbox[1], right_bbox[2], right_bbox[3], right_bbox[0]])
        if check_collision(left_polygon, obstacles, curbs) and (
            check_collision(right_polygon, obstacles, curbs)
        ):
            narrow_road_tag.is_narrow_road = True
            break
    return narrow_road_tag

@TAG_FUNCTIONS.register()
def high_value_tag(data: TagData, params: Dict) -> Dict:
    high_value_tag = HighValueTag()
    
    high_value_tag.narrow_road_tag = label_narrow_road_tag(data, params)

    return high_value_tag.as_dict()
