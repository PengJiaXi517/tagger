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
        self.future_narrow_road_relax = [[False, False] for i in range(100)]

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
class RampTag:
    def __init__(self) -> None:
        self.is_enter_ramp: bool = False
        self.is_exit_ramp: bool = False

    def as_dict(self):
        return {
            "is_enter_ramp": self.is_enter_ramp,
            "is_exit_ramp": self.is_exit_ramp,
        }

@dataclass(repr=False)
class HighValueTag:
    narrow_road_tag: NarrowRoadTag = None
    junction_bypass_tag: JunctionBypassTag = None
    yield_vru_tag: YieldVRUTag = None
    mixed_traffic_tag: MixedTrafficTag = None
    ramp_tag: RampTag = None
    def as_dict(self):
        return {
            "narrow_road_tag": self.narrow_road_tag.as_dict(),
            "junction_bypass_tag": self.junction_bypass_tag.as_dict(),
            "yield_vru_tag": self.yield_vru_tag.as_dict(),
            "mixed_traffic_tag": self.mixed_traffic_tag.as_dict(),
            "ramp_tag": self.ramp_tag.as_dict(),
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
    future_states = obstacle['future_trajectory']['future_states']
    if len(future_states) < 2:
        return obstacle["features"]["is_still"]
    pt_start = Point([future_states[0]['x'], future_states[0]['y']])
    pt_end = Point([future_states[-1]['x'], future_states[-1]['y']])
    if pt_start.distance(pt_end) < 0.2:
        return True
    return False

def check_collision_curb(veh_polygon, curbs, curb_lat_decision, params):
    collision_info = {'left_strict': False, 'right_strict': False,
                      'left_relax': False, 'right_relax': False}
    for idx, curb in enumerate(curbs):
        lat_decision = curb_lat_decision[idx]
        if lat_decision == 0:
            continue 
        curb_string = LineString(curb)
        dist = curb_string.distance(veh_polygon)
        if dist < params.near_static_obs_dist_strict:
            if lat_decision == 1:
                collision_info['right_strict'] = True
            elif lat_decision == 2:
                collision_info['left_strict'] = True
        if dist < params.near_static_obs_dist_relax:
            if lat_decision == 1:
                collision_info['right_relax'] = True
            elif lat_decision == 2:
                collision_info['left_relax'] = True
        if collision_info['left_strict'] and collision_info['right_strict']:
            break
    return collision_info  

def check_collision_obs(veh_polygon, obstacles, id_polygon, params):
    collision_info = {'left_strict': False, 'right_strict': False,
                      'left_relax': False, 'right_relax': False}

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
        if dist < params.near_static_obs_dist_strict:
            if lat_decision == 1:
                collision_info['right_strict'] = True
            elif lat_decision == 2:
                collision_info['left_strict'] = True    
                # plt.figure(figsize=(14, 12))
                # plt.fill(*veh_polygon.exterior.xy, color='blue', alpha=0.5)  
                # plt.fill(*obs_polygon.exterior.xy, color='green', alpha=0.5)
                # plt.show()
        if dist < params.near_static_obs_dist_relax:
            if lat_decision == 1:
                collision_info['right_relax'] = True
            elif lat_decision == 2:
                collision_info['left_relax'] = True
        if collision_info['right_strict'] and collision_info['left_strict']:
            break       

    return collision_info

def get_static_obs_polygon(obstacles):
    id_polygon = {}
    static_obs = {}
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
            static_obs[id] = obs

    return static_obs, id_polygon

def get_moving_obs(obstacles):
    moving_obs = {}
    for id, obs in obstacles.items():
        if id == -9:
            continue
        if not is_static(obs):
            moving_obs[id] = obs

    return moving_obs

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

def get_nearest_lane_id(lane_map, lane_seqs, target_point, current_lane_seqs):
    min_dist = 100000
    nearest_lane_id = None
    for lane_seq in lane_seqs:
        for lane_id in lane_seq:
            if any(lane_id in obj for obj in current_lane_seqs):
                continue
            nearest_polyline = LineString(lane_map[lane_id]['polyline'])
            dist = target_point.distance(nearest_polyline)
            if dist < min_dist and dist < 5 and dist > 0:
                min_dist = dist
                nearest_lane_id = lane_seq[0]
    return nearest_lane_id

def get_nearest_waypoint_idx(lane_polyline, ego_point):
    lane_array = np.array(lane_polyline)
    distances = np.sqrt(np.sum((lane_array - [ego_point.x, ego_point.y])**2, axis=1))
    nearest_idx = np.argmin(distances)
    if distances[nearest_idx] > 5:
        nearest_idx = -1
    
    return nearest_idx

def judge_enter_ramp(lane_map, lane_ids, ego_point):
    # 滤除对向车道
    v1 = lane_map[lane_ids[0]]['unit_directions'][0]
    v2 = lane_map[lane_ids[1]]['unit_directions'][0]
    if v1[0] * v2[0] + v1[1] * v2[1] < 0:
        return False

    cur_lane = lane_map[lane_ids[0]]['polyline']
    adjacent_lane = lane_map[lane_ids[1]]['polyline']

    # 在lane上找最近点的索引
    cur_idx = get_nearest_waypoint_idx(cur_lane, ego_point)
    adjacent_idx = get_nearest_waypoint_idx(adjacent_lane, ego_point)
    if cur_idx == -1 or adjacent_idx == -1:
        return False
    
    # 截取自车前方consider_len个点
    consider_len = 100
    if cur_idx < len(cur_lane) - consider_len:
        cur_lane = cur_lane[cur_idx: cur_idx + consider_len]
    else:
        cur_lane = cur_lane[cur_idx:]
        if len(lane_map[lane_ids[0]]['successor_id']) > 0:
            cur_succ_id = lane_map[lane_ids[0]]['successor_id'][0]
            if lane_map[cur_succ_id]['lane_category'] == 'REALITY':
                concate_len = consider_len - len(cur_lane)
                cur_lane = cur_lane + lane_map[cur_succ_id]['polyline'][:concate_len]
    
    if adjacent_idx < len(adjacent_lane) - consider_len:
        adjacent_lane = adjacent_lane[adjacent_idx: adjacent_idx + consider_len]
    else:
        adjacent_lane = adjacent_lane[adjacent_idx:]
        if len(lane_map[lane_ids[1]]['successor_id']) > 0:
            adjacent_succ_id = lane_map[lane_ids[1]]['successor_id'][0]
            if lane_map[adjacent_succ_id]['lane_category'] == 'REALITY':
                concate_len = consider_len - len(adjacent_lane)
                adjacent_lane = adjacent_lane + lane_map[adjacent_succ_id]['polyline'][:concate_len]
    
    min_length = min(len(cur_lane), len(adjacent_lane))
    if min_length < 3:
        return False
    cur_lane = cur_lane[:min_length]
    adjacent_lane = adjacent_lane[:min_length]

    large_dist_num = 0
    adjacent_linestring = LineString(adjacent_lane)
    for idx, point in enumerate(cur_lane):
        if idx % 2 or adjacent_linestring.distance(Point(point)) < 10:
            continue
        large_dist_num += 1
        if large_dist_num > 3:
            return True
    return False

def judge_exit_ramp(lane_map, lane_ids, ego_point, cur_lane_id):
    if cur_lane_id not in lane_ids:
        return False
    # 滤除对向车道
    v1 = lane_map[lane_ids[0]]['unit_directions'][0]
    v2 = lane_map[lane_ids[1]]['unit_directions'][0]
    if v1[0] * v2[0] + v1[1] * v2[1] < 0:
        return False
    
    if lane_ids[1] == cur_lane_id:
        lane_ids[0], lane_ids[1] = lane_ids[1], lane_ids[0]

    cur_lane = lane_map[lane_ids[0]]['polyline']
    adjacent_lane = lane_map[lane_ids[1]]['polyline']
    
    if len(cur_lane) < 10 or len(adjacent_lane) < 10:
        return False
    
    # 在lane上找最近点的索引
    cur_idx = get_nearest_waypoint_idx(cur_lane, ego_point)
    consider_len = 100
    if cur_idx == -1 or cur_idx < len(cur_lane) - consider_len:
        return False
    
    # 无分叉情况，需要对齐两条lane的终点
    if Point(cur_lane[-1]).distance(Point(adjacent_lane[-1])) > 1.0:
        adjacent_idx = get_nearest_waypoint_idx(adjacent_lane, Point(cur_lane[-1]))
        if adjacent_idx == -1:
            return False
        adjacent_lane = adjacent_lane[:adjacent_idx + 1]
    
    # 截取汇入点前的consider_len个点
    if len(cur_lane) > consider_len:
        cur_lane = cur_lane[-consider_len:]
    elif len(lane_map[lane_ids[0]]['predecessor_id']) > 0:
        cur_pred_id = lane_map[lane_ids[0]]['predecessor_id'][0]
        if lane_map[cur_pred_id]['lane_category'] == 'REALITY':
            concate_len = consider_len - len(cur_lane)
            cur_lane = lane_map[cur_pred_id]['polyline'][-concate_len:] + cur_lane
    
    if len(adjacent_lane) > consider_len:
        adjacent_lane = adjacent_lane[-consider_len:]
    elif len(lane_map[lane_ids[1]]['predecessor_id']) > 0:
        adjacent_pred_id = lane_map[lane_ids[1]]['predecessor_id'][0]
        if lane_map[adjacent_pred_id]['lane_category'] == 'REALITY':
            concate_len = consider_len - len(adjacent_lane)
            adjacent_lane = lane_map[adjacent_pred_id]['polyline'][-concate_len:] + adjacent_lane
    
    min_length = min(len(cur_lane), len(adjacent_lane))
    if min_length < 3:
        return False
    cur_lane = cur_lane[-min_length:]
    adjacent_lane = adjacent_lane[-min_length:]

    large_dist_num = 0
    adjacent_linestring = LineString(adjacent_lane)
    for idx, point in enumerate(cur_lane):
        if idx % 2 or adjacent_linestring.distance(Point(point)) < 10:
            continue
        large_dist_num += 1
        if large_dist_num > 3:
            return True
    return False

def label_narrow_road_tag(
    data: TagData, params: Dict) -> NarrowRoadTag:
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

    static_obs, id_polygon = get_static_obs_polygon(obstacles)
    
    for idx, (x, y) in enumerate(ego_path_info.future_path):
        veh_polygon = get_ego_polygon(ego_path_info.future_path, idx, obstacles[-9])

        collision_info = check_collision_curb(veh_polygon, curbs, curbs_lat_decision, params)
        narrow_road_tag.future_narrow_road[idx][0] |= collision_info['left_strict']
        narrow_road_tag.future_narrow_road[idx][1] |= collision_info['right_strict']
        narrow_road_tag.future_narrow_road_relax[idx][0] |= collision_info['left_relax']
        narrow_road_tag.future_narrow_road_relax[idx][1] |= collision_info['right_relax']
        
        if collision_info['left_strict'] and collision_info['right_strict']:
            continue

        collision_info = check_collision_obs(veh_polygon, static_obs, id_polygon, params)
        narrow_road_tag.future_narrow_road[idx][0] |= collision_info['left_strict']
        narrow_road_tag.future_narrow_road[idx][1] |= collision_info['right_strict']
        narrow_road_tag.future_narrow_road_relax[idx][0] |= collision_info['left_relax']
        narrow_road_tag.future_narrow_road_relax[idx][1] |= collision_info['right_relax']
    
    tmp = [all(obj) for obj in narrow_road_tag.future_narrow_road]
    narrow_road_tag.is_narrow_road = any(tmp)
    return narrow_road_tag

def label_junction_bypass_tag(data: TagData, params: Dict,
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
        if abs(curvature[idx]) < params.curvature_th:
            continue
        if in_junction_id[idx] is None:
            continue
        if turn_type[idx] > 0 and narrow_road_tag.future_narrow_road_relax[idx][0]:
            junction_bypass_tag.future_junction_bypass[idx] = True
        if turn_type[idx] < 0 and narrow_road_tag.future_narrow_road_relax[idx][1]:
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

def label_mixed_traffic_tag(data: TagData, params: Dict)-> MixedTrafficTag:
    mixed_traffic_tag = MixedTrafficTag()
    obstacles = data.label_scene.obstacles
    # 判断自车8s内是否静止
    if is_static(obstacles[-9]):
        return mixed_traffic_tag

    moving_obs = get_moving_obs(obstacles)
    future_obs_polygon = get_obs_future_polygon(moving_obs)

    ego_obs = obstacles[-9]
    ego_future_states = ego_obs['future_trajectory']['future_states']
    ego_length = ego_obs['features']['length']
    ego_width = ego_obs['features']['width']
    for idx, ego_state in enumerate(ego_future_states):
        ts_us = ego_state['timestamp']
        ego_bbox = get_bbox(ego_state['x'], ego_state['y'], ego_state['theta'], ego_length, ego_width)
        ego_polygon = Polygon([ego_bbox[0], ego_bbox[1], ego_bbox[2], ego_bbox[3], ego_bbox[0]])
        collision_info = check_collision_obs(ego_polygon, moving_obs, future_obs_polygon[ts_us], params)
        mixed_traffic_tag.future_mixed_traffic[idx][0] |= collision_info['left_strict']
        mixed_traffic_tag.future_mixed_traffic[idx][1] |= collision_info['right_strict']
    
    tmp = [any(obj) for obj in mixed_traffic_tag.future_mixed_traffic]
    mixed_traffic_tag.is_mixed_traffic = any(tmp)
    return mixed_traffic_tag

def label_ramp_tag(data):
    ramp_tag = RampTag()
    obstacles = data.label_scene.obstacles
    lane_map = data.label_scene.percepmap.lane_map
    current_lanes = data.label_scene.ego_obs_lane_seq_info.current_lanes
    current_lane_seqs = data.label_scene.ego_obs_lane_seq_info.current_lane_seqs
    nearby_lane_seqs = data.label_scene.ego_obs_lane_seq_info.nearby_lane_seqs
    
    if len(current_lanes) < 1 or len(current_lane_seqs) < 1:
        return ramp_tag
    for lane_seq in current_lane_seqs:
        for lane_id in lane_seq:
            if lane_map[lane_id]['turn'] != 'NOTURN' or (
               lane_map[lane_id]['lane_category'] != 'REALITY'):
                return ramp_tag
    
    ego_x = obstacles[-9]["features"]["history_states"][-1]["x"]
    ego_y = obstacles[-9]["features"]["history_states"][-1]["y"]
    ego_point = Point([ego_x, ego_y])
    
    has_fork = False
    # 判断是否为进匝道: 一分成二的情况
    pred_ids = lane_map[current_lanes[0]]['predecessor_id']
    if len(pred_ids) == 1:
        succ_ids = lane_map[pred_ids[0]]['successor_id']
        if len(succ_ids) == 2:
            has_fork = True
            if judge_enter_ramp(lane_map, succ_ids, ego_point):
                ramp_tag.is_enter_ramp= True
                return ramp_tag
    
    # 判断是否为进匝道: lane无分叉点的情况
    if not has_fork:
        nearest_lane_id = get_nearest_lane_id(lane_map, nearby_lane_seqs, ego_point, current_lane_seqs)
        if nearest_lane_id is not None:
            if judge_enter_ramp(lane_map, [current_lanes[0], nearest_lane_id], ego_point):
                ramp_tag.is_enter_ramp = True
                return ramp_tag
    
    has_merge = False
    #判断是否为出匝道: 二合成一的情况
    succ_ids = lane_map[current_lanes[0]]['successor_id']
    if len(succ_ids) == 1:
        pred_ids = lane_map[succ_ids[0]]['predecessor_id']
        if len(pred_ids) == 2:
            has_merge = True
            if judge_exit_ramp(lane_map, pred_ids, ego_point, current_lanes[0]):
                ramp_tag.is_exit_ramp = True
                return ramp_tag
    
    # 判断是否为出匝道: lane无分叉点的情况
    if not has_merge:
        if len(current_lane_seqs) == 1 and len(current_lane_seqs[0]) == 2:
            # 出口位置的大致坐标
            target_point = Point(lane_map[current_lane_seqs[0][0]]['polyline'][-1])
            # 找到出口处的最近车道
            nearest_lane_id = get_nearest_lane_id(
                lane_map, data.condition_res.seq_lane_ids_raw, target_point, current_lane_seqs)
            if nearest_lane_id is not None:
                if judge_exit_ramp(lane_map, [current_lanes[0], nearest_lane_id], ego_point, current_lanes[0]):
                    ramp_tag.is_exit_ramp = True
                    return ramp_tag
                                              
    return ramp_tag    

@TAG_FUNCTIONS.register()
def high_value_tag(data: TagData, params: Dict) -> Dict:
    high_value_tag = HighValueTag()
    # 判断窄路通行
    high_value_tag.narrow_road_tag = label_narrow_road_tag(data, params)
    # 判断路口内绕障
    high_value_tag.junction_bypass_tag = label_junction_bypass_tag(data, params, high_value_tag.narrow_road_tag)
    # 判断是否在礼让vru而减速
    high_value_tag.yield_vru_tag = label_yield_vru_tag(data)
    # 判断8s内是否与动目标交互
    high_value_tag.mixed_traffic_tag = label_mixed_traffic_tag(data, params)
    # 判断汇入汇出匝道
    high_value_tag.ramp_tag = label_ramp_tag(data)

    return high_value_tag.as_dict()