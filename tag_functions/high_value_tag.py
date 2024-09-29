from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon
from base import PercepMap, TagData
import matplotlib.pyplot as plt
from collections import defaultdict
from registry import TAG_FUNCTIONS

@dataclass(repr=False)
class NarrowRoadTag:
    def __init__(self) -> None:
        self.is_narrow_road: bool = False
        self.future_narrow_road = [[False, False] for i in range(100)]

    def as_dict(self):
        return {
            "is_narrow_road": self.is_narrow_road,
            "future_narrow_road": self.future_narrow_road,
        }

@dataclass(repr=False)
class JunctionBypassTag:
    def __init__(self) -> None:
        self.is_junction_bypass: bool = False
        self.future_junction_bypass = [False for i in range(100)]
    
    def as_dict(self):
        return {
            "is_junction_bypass": self.is_junction_bypass,
            "future_junction_bypass": self.future_junction_bypass,
        }

@dataclass(repr=False)
class YieldVRUTag:
    def __init__(self) -> None:
        self.is_yield_vru: bool = False
    
    def as_dict(self):
        return {
            "is_yield_vru": self.is_yield_vru,
        }

@dataclass(repr=False)
class MixedTrafficTag:
    def __init__(self) -> None:
        self.is_mixed_traffic: bool = False
        self.future_mixed_traffic = [[False, False] for i in range(80)]
    
    def as_dict(self):
        return {
            "is_mixed_traffic": self.is_mixed_traffic,
            "future_mixed_traffic:": self.future_mixed_traffic
        }

@dataclass(repr=False)
class HighValueTag:
    narrow_road_tag: NarrowRoadTag = None
    junction_bypass_tag: JunctionBypassTag = None
    yield_vru_tag: YieldVRUTag = None
    mixed_traffic_tag: MixedTrafficTag = None
    def as_dict(self):
        return {
            "narrow_road_tag": self.narrow_road_tag.as_dict(),
            "junction_bypass_tag": self.junction_bypass_tag.as_dict(),
            "yield_vru_tag": self.yield_vru_tag.as_dict(),
            "mixed_traffic_tag": self.mixed_traffic_tag.as_dict(),
        }

def valid_check(data: TagData)-> bool:
    if len(data.label_scene.ego_path_info.future_path) < 10:
        return False
    return True
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

def get_ego_polygon(future_path, idx, obstacle):
    x, y = future_path[idx]
    if idx == 0:
        next_x, next_y = future_path[1]
        heading = math.atan2(next_y - y, next_x - x)
    else:
        last_x, last_y = future_path[idx - 1]
        heading = math.atan2(y - last_y, x - last_x)
    veh_length = obstacle["features"]["length"]
    veh_width = obstacle["features"]["width"]
    veh_bbox = get_bbox(x, y, heading, veh_length, veh_width)
    veh_polygon = Polygon([veh_bbox[0], veh_bbox[1], veh_bbox[2], veh_bbox[3], veh_bbox[0]])
    return veh_polygon

def is_static(obstacle):
    if not obstacle["features"]["is_still"]:
        return False
    future_states = obstacle['future_trajectory']['future_states']
    if len(future_states) < 1:
        return True
    cur_state = future_states[0]
    for state in future_states:
        if (state['timestamp']) > cur_state['timestamp'] + 5e6:
            break
        if math.fabs(state['vx']) > 0.1 or math.fabs(state['vy']) > 0.1:
            return False
    return True

def check_collision_curb(veh_polygon, curbs, curb_lat_decision):
    collision_left, collision_right = False, False
    for idx, curb in enumerate(curbs):
        lat_decision = curb_lat_decision[idx]
        if lat_decision == 0:
            continue 
        curb_string = LineString(curb)
        if curb_string.distance(veh_polygon) < 1.5:
            if lat_decision == 1:
                collision_right = True
            elif lat_decision == 2:
                collision_left = True
        if collision_left and collision_right:
            break
    return collision_left, collision_right    

def check_collision_obs(veh_polygon, obstacles, id_polygon):
    collision_left, collision_right = False, False
    for k, v in obstacles.items():
        if k == -9:
            continue
        lat_decision = v["decision"]["interactive_lat_type"]
        if lat_decision == 0:
            continue
        if k not in id_polygon:
            continue
        obs_polygon = id_polygon[k]
        if veh_polygon.distance(obs_polygon) < 1.5:
            if lat_decision == 1:
                collision_right = True
            elif lat_decision == 2:
                collision_left = True
        if collision_left and collision_right:
            break       
    return collision_left, collision_right

def get_static_obs_polygon(obstacles):
    moving_obs_id = []
    id_polygon = {}
    for id, obs in obstacles.items():
        if id == -9:
            continue
        if is_static(obs):
            obs_x = obs["features"]["history_states"][-1]["x"]
            obs_y = obs["features"]["history_states"][-1]["y"]
            obs_theta = obs["features"]["history_states"][-1]["theta"]
            length = obs["features"]["length"]
            width = obs["features"]["width"]
            obs_bbox = get_bbox(obs_x, obs_y, obs_theta, length, width)
            obs_polygon = Polygon([obs_bbox[0], obs_bbox[1], obs_bbox[2], obs_bbox[3], obs_bbox[0]])
            id_polygon[id] = obs_polygon
            continue
        moving_obs_id.append(id)
    for i in range(len(moving_obs_id)):
        del obstacles[moving_obs_id[i]]
    return obstacles, id_polygon

def get_moving_obs(obstacles):
    static_obs_id = []
    for id, obs in obstacles.items():
        if id == -9:
            continue
        if is_static(obs):
            static_obs_id.append(id)
    for i in range(len(static_obs_id)):
        del obstacles[static_obs_id[i]]
    return obstacles

def get_curvature(ego_path_info):
    def normal_angle(theta):
        while theta >= np.pi:
            theta -= 2 * np.pi
        while theta <= -np.pi:
            theta += 2 * np.pi
        return theta
    applyall = np.vectorize(normal_angle)
    
    path_points = np.array(ego_path_info.future_path)
    diff_points = path_points[1:] - path_points[:-1]
    theta = np.arctan2(diff_points[:, 1], diff_points[:, 0])
    theta_diff = theta[1:] - theta[:-1]
    length = np.linalg.norm(diff_points[:-1], axis=-1)
    theta_diff = applyall(theta_diff)
    curvature = theta_diff / length
    turn_type = np.sign(theta_diff)

    curvature = np.insert(curvature, 0, [curvature[0], curvature[0]], axis=0)
    turn_type = np.insert(turn_type, 0, [turn_type[0], turn_type[0]], axis=0)
    return curvature, turn_type

def detect_braking_and_stop(ego_future_states):
    speeds = []
    is_braking = False
    stop_point = None
    for idx, state in enumerate(ego_future_states):
        speed = np.linalg.norm([state['vx'], state['vy']])
        speeds.append(speed)
        if speed < 0.2:
            if len(speeds) < 3 or math.fabs(speeds[-1] - speeds[0]) < 0.3:
                return False, None
            speed_diff = [speeds[i] - speeds[i + 1] for i in range(len(speeds) - 1)]
            is_braking = (sum(1 for v in speed_diff if v >= 0) / len(speed_diff)) > 0.7
            stop_point = Point([state['x'], state['y']])
            break
    return is_braking, stop_point

def get_obs_future_polygon(obstacles):
    obs_future_polygon = defaultdict(dict)
    for obs_id, obs in obstacles.items():
        if obs_id == -9:
            continue
        obs_future_states = obs['future_trajectory']['future_states']
        length = obs['features']['length']
        width = obs['features']['width']
        for obs_state in obs_future_states:
            obs_bbox = get_bbox(obs_state['x'], obs_state['y'], obs_state['theta'], length, width)
            obs_polygon = Polygon([obs_bbox[0], obs_bbox[1], obs_bbox[2], obs_bbox[3], obs_bbox[0]])
            obs_future_polygon[obs_state['timestamp']][obs_id] = obs_polygon
    return obs_future_polygon

def label_narrow_road_tag(
    data: TagData) -> NarrowRoadTag:
    narrow_road_tag = NarrowRoadTag()
    if not valid_check(data):
        return narrow_road_tag
    ego_path_info = data.label_scene.ego_path_info
    obstacles = data.label_scene.obstacles
    curb_vec = data.label_scene.label_res['curb_label']['decision']['vec']
    curb_src = data.label_scene.label_res['curb_label']['decision']['src_point']
    curb_end = curb_src + curb_vec
    curbs = [(curb_src[i], curb_end[i]) for i in range(len(curb_src))]
    curbs_lat_decision = data.label_scene.label_res['curb_label']['decision']['interactive_lat_type']
    # 判断自车5s内是否静止
    if is_static(obstacles[-9]):
        return narrow_road_tag

    # 删除非静止的障碍物
    obstacles, id_polygon = get_static_obs_polygon(obstacles)
    
    for idx, (x, y) in enumerate(ego_path_info.future_path):
        veh_polygon = get_ego_polygon(ego_path_info.future_path, idx, obstacles[-9])

        collision_left, collision_right = check_collision_curb(veh_polygon, curbs, curbs_lat_decision)
        narrow_road_tag.future_narrow_road[idx][0] |= collision_left
        narrow_road_tag.future_narrow_road[idx][1] |= collision_right
        
        if collision_left and collision_right:
            continue
        collision_left, collision_right = check_collision_obs(veh_polygon, obstacles, id_polygon)
        narrow_road_tag.future_narrow_road[idx][0] |= collision_left
        narrow_road_tag.future_narrow_road[idx][1] |= collision_right
    
    tmp = [all(obj) for obj in narrow_road_tag.future_narrow_road]
    narrow_road_tag.is_narrow_road = any(tmp)
    return narrow_road_tag

def label_junction_bypass_tag(data: TagData,
    narrow_road_tag: NarrowRoadTag)-> JunctionBypassTag:
    junction_bypass_tag = JunctionBypassTag()
    if not valid_check(data):
        return junction_bypass_tag
    ego_path_info = data.label_scene.ego_path_info
    in_junction_id = ego_path_info.in_junction_id
   
    junction_scene = any(obj is not None for obj in in_junction_id)
    if not junction_scene:
        return junction_bypass_tag

    curvature, turn_type = get_curvature(ego_path_info)

    for idx, (x, y) in enumerate(ego_path_info.future_path):
        if curvature[idx] < 0.03:
            continue
        if turn_type[idx] > 0 and narrow_road_tag.future_narrow_road[idx][0]:
            junction_bypass_tag.future_junction_bypass[idx] = True
        if turn_type[idx] < 0 and narrow_road_tag.future_narrow_road[idx][1]:
            junction_bypass_tag.future_junction_bypass[idx] = True

    junction_bypass_tag.is_junction_bypass = any(junction_bypass_tag.future_junction_bypass)
    return junction_bypass_tag

def label_yield_vru_tag(data: TagData) -> YieldVRUTag:
    yield_vru_tag = YieldVRUTag()
    if not valid_check(data):
        return yield_vru_tag
    obstacles = data.label_scene.obstacles
    future_path = data.label_scene.ego_path_info.future_path
    ego_future_states = obstacles[-9]['future_trajectory']['future_states']
    
    # 判断是否有减速行为
    is_braking, stop_point = detect_braking_and_stop(ego_future_states)
    if not is_braking or stop_point is None:
        return yield_vru_tag

    # 判断vru future_states与ego future_path是否相交，计算交点与刹停点的距离
    future_path_polyline = LineString(future_path)
    for idx, obs in obstacles.items():
        if idx == -9 or (obs['features']['type'] != "PEDESTRIAN" 
                        and obs['features']['type'] != "BICYCLE"):
            continue
        obs_future_traj = [(state['x'], state['y']) for state in obs['future_trajectory']['future_states']]
        if len(obs_future_traj) < 2:
            continue
        obs_polyline = LineString(obs_future_traj)
        intersection_pt = obs_polyline.intersection(future_path_polyline)
        if intersection_pt.is_empty:
            continue
        if stop_point.distance(intersection_pt) < 5:
            yield_vru_tag.is_yield_vru = True
            break
    
    return yield_vru_tag

def label_mixed_traffic_tag(data: TagData)-> MixedTrafficTag:
    mixed_traffic_tag = MixedTrafficTag()
    # 判断自车5s内是否静止
    if is_static(data.label_scene.obstacles[-9]):
        return mixed_traffic_tag
    # 删除静止障碍物
    obstacles = get_moving_obs(data.label_scene.obstacles)
    future_obs_polygon = get_obs_future_polygon(obstacles)

    ego_obs = obstacles[-9]
    ego_future_states = ego_obs['future_trajectory']['future_states']
    ego_length = ego_obs['features']['length']
    ego_width = ego_obs['features']['width']
    for idx, ego_state in enumerate(ego_future_states):
        ts_us = ego_state['timestamp']
        ego_bbox = get_bbox(ego_state['x'], ego_state['y'], ego_state['theta'], ego_length, ego_width)
        ego_polygon = Polygon([ego_bbox[0], ego_bbox[1], ego_bbox[2], ego_bbox[3], ego_bbox[0]])
        collision_left, collision_right = check_collision_obs(ego_polygon, obstacles, future_obs_polygon[ts_us])
        mixed_traffic_tag.future_mixed_traffic[idx][0] |= collision_left
        mixed_traffic_tag.future_mixed_traffic[idx][1] |= collision_right
    
    tmp = [any(obj) for obj in mixed_traffic_tag.future_mixed_traffic]
    mixed_traffic_tag.is_mixed_traffic = any(tmp)
    return mixed_traffic_tag

@TAG_FUNCTIONS.register()
def high_value_tag(data: TagData, params: Dict) -> Dict:
    high_value_tag = HighValueTag()
    # 判断窄路通行
    high_value_tag.narrow_road_tag = label_narrow_road_tag(data)
    # 判断路口内绕障
    high_value_tag.junction_bypass_tag = label_junction_bypass_tag(data, high_value_tag.narrow_road_tag)
    # 判断是否在礼让vru而减速
    high_value_tag.yield_vru_tag = label_yield_vru_tag(data)
    # 判断8s内是否与动目标交互
    high_value_tag.mixed_traffic_tag = label_mixed_traffic_tag(data)

    return high_value_tag.as_dict()