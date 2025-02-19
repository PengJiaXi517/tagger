from base import TagData, LabelScene
from registry import TAG_FUNCTIONS
import numpy as np
from typing import Tuple
import math

# from utils.viz_utils.draw_scene import draw_scene

@TAG_FUNCTIONS.register()
def ramp_merge(tag_data: TagData, cfg: dict) -> dict:
    output = {
        "ramp_merge": {
            "valid": False,             # 合法标签，表示该帧是否符合基本要求
            "obs_ttc": {},              # 轨迹碰撞时间
        }
    }

    speed_th = cfg.get("speed_th", 3.0)     # 最小速度限制，小于这个速度的样本帧忽略

    scene: LabelScene = tag_data.label_scene
    ego_info = scene.obstacles.get(-9, {})
    all_obstacles = scene.obstacles

    if len(ego_info['future_trajectory']['future_states']) == 0:
        return output
    
    if ego_info.get("junction_info", {}).get("in_junction", False):
        return output
    
    ego_speed = math.hypot(ego_info['future_trajectory']['future_states'][0].get("vx",0), 
                           ego_info['future_trajectory']['future_states'][0].get("vy",0))
    if ego_speed < speed_th:
        return output

    # log_str = "_"
    for obs_id, obs_info in all_obstacles.items():
        if obs_id == -9:
            continue    # 跳过自车

        if obs_info['features']['type'] != 'VEHICLE':
            continue    # 跳过非机动车

        if obs_info['trajectory_quality']['abnormal_score'] > cfg.get("trajectory_abnormal_score_th", 3.0):
            continue    # 检查轨迹质量参数

        if obs_info['features']['is_still']:
            continue    # 排除静止目标

        if obs_info.get("junction_info", {}).get("in_junction", False):
            continue    # 排除junction内目标

        ttc = check_interaction(ego_info, obs_info, cfg)

        if ttc > 0:
            output['ramp_merge']['valid'] = True
            output['ramp_merge']['obs_ttc'][obs_id] = ttc

            # obs_speed = math.hypot(obs_info['future_trajectory']['future_states'][0].get("vx",0), 
            #                        obs_info['future_trajectory']['future_states'][0].get("vy",0))
            # log_str = log_str + str(obs_id) + "-" + str(round(ttc,1)) + "-" + str(round(ego_speed,2)) + "-" + str(round(obs_speed,2)) + "_"

    # save_dir = "/home/sti/backbone-traj-nn-tagger/test/ramp/"
    # save_sub_dir = cfg['clip_name']
    # pickle_name = cfg['pickle_name']

    # if output['ramp_merge']['valid']:
    #     draw_scene(scene, save_dir, save_sub_dir, (pickle_name + log_str)[:-1])
    #     print(f"Valid for {pickle_name}")

    return output


def angle_normalize(angle):
    angle = angle % (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle


def check_interaction(ego_info: dict, obs_info: dict, cfg: dict) -> float:
    """判断未来N秒内两车是否会发生碰撞"""
    collision_time_range = cfg.get("collision_time_range", 2.0)     # 计算碰撞的时间
    time_step = cfg.get("time_step", 0.1)                           # 计算碰撞的步长
    min_angle_diff_th = cfg.get("min_angle_diff_threshold", math.pi / 36.0) # 角度差必须大于阈值，排除追尾碰撞
    max_angle_diff_th = cfg.get("max_angle_diff_threshold", math.pi / 1.8)  # 排除对向碰撞
    speed_th = cfg.get("speed_th", 3.0)

    if len(obs_info['future_trajectory']['future_states']) < (collision_time_range / time_step):
        return -1

    ego_length = ego_info['features']['length']
    ego_width = ego_info['features']['width']
    ego_start = get_state_vector(ego_info['future_trajectory']['future_states'][0])

    obs_length = obs_info['features']['length']
    obs_width = obs_info['features']['width']
    obs_start = get_state_vector(obs_info['future_trajectory']['future_states'][0])

    angle_diff = abs(angle_normalize(ego_start[4] - obs_start[4]))
    if angle_diff < min_angle_diff_th or angle_diff > max_angle_diff_th:
        return -1
    
    obs_speed = math.hypot(obs_start[2], obs_start[3])
    if obs_speed < speed_th:
        return -1    # 跳过慢车

    return check_collision_with_rotation(
        ego_start, (ego_length, ego_width),
        obs_start, (obs_length, obs_width),
        collision_time_range,
        time_step
    )


def get_state_vector(state: dict) -> np.ndarray:
    """提取状态向量 [x, y, vx, vy]"""
    return np.array([
        state['x'], state['y'],
        state['vx'], state['vy'],
        math.atan2(state['vy'], state['vx'])
    ])

    
def check_collision_with_rotation(
    ego_state: np.ndarray,
    ego_size: Tuple[float, float],
    obs_state: np.ndarray,
    obs_size: Tuple[float, float],
    t_max: float,
    t_step: float
) -> float:
    """精确碰撞检测（考虑车辆朝向）"""
    for t in np.arange(0, t_max + t_step, t_step):
        # 计算未来状态
        ego_pos = ego_state[:2] + ego_state[2:4] * t
        ego_theta = ego_state[4]
        obs_pos = obs_state[:2] + obs_state[2:4] * t
        obs_theta = obs_state[4]

        # 生成多边形
        ego_poly = get_contour(ego_pos, *ego_size, ego_theta)
        obs_poly = get_contour(obs_pos, *obs_size, obs_theta)

        # 多边形碰撞检测
        if polygon_collision(ego_poly, obs_poly):
            return t
    return -1


# 生成车辆轮廓多边形
def get_contour(pos, l, w, theta):
    # 生成旋转矩形四个顶点
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    half_l = l/2
    half_w = w/2
    points = np.array([
        [-half_l, -half_w],
        [half_l, -half_w],
        [half_l, half_w],
        [-half_l, half_w]
    ])
    return pos[:2] + np.dot(points, rot_mat.T)


def polygon_collision(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """分离轴定理实现多边形碰撞检测"""
    def project(poly, axis):
        dots = np.dot(poly, axis)
        return min(dots), max(dots)

    for poly in [poly1, poly2]:
        for i in range(len(poly)):
            # 获取边向量
            p1 = poly[i]
            p2 = poly[(i+1)%len(poly)]
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])
            axis /= np.linalg.norm(axis)

            # 投影多边形
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)

            # 检查投影重叠
            if max1 < min2 or max2 < min1:
                return False
    return True
