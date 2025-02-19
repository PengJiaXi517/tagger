# tag_functions/curve_driving.py
from base import TagData, LabelScene
from registry import TAG_FUNCTIONS
import numpy as np
from typing import List
import math

# from utils.viz_utils.draw_scene import draw_scene


@TAG_FUNCTIONS.register()
def curve_driving(tag_data: TagData, cfg: dict) -> dict:
    output = {
        "curve_driving": {
            "valid": False,             # 合法标签，表示该帧是否符合基本要求
            "obs_direction": {},        # 每个obs的转弯方向
            "obs_intensity": {},        # 每个obs转弯强度
            "ego_interact": {},          # 每个obs是否和ego有交互
            "ego_direction": "none",    # ego的转弯方向
            "ego_intensity": 0,    # ego的转弯强度
            "is_intersection": False,   # ego是否在路口内
        }
    }    

    speed_th = cfg.get("speed_th", 1.0) # 自车的最小速度限制，小于这个速度的样本帧忽略

    # try:
    scene: LabelScene = tag_data.label_scene
    ego_info = scene.obstacles.get(-9, {})
    all_obstacles = scene.obstacles

    # 处理自车信息 --------------------------------------------------
    ego_states = get_valid_trajectory(ego_info, "ego")
    if ego_states:
        output["curve_driving"].update(
            calc_turn_feature(ego_states, prefix="ego", cfg=cfg)
        )
        output["curve_driving"]["is_intersection"] = ego_info.get(
            "junction_info", {}).get("in_junction", False)
        
        ego_speed = math.hypot(ego_states[0].get("vx",0), ego_states[0].get("vy",0))
        if ego_speed < speed_th or output["curve_driving"]["ego_direction"] == "none":
            return output

    # 处理他车信息 --------------------------------------------------
    # log_str = "_"
    for obs_id, obs_info in all_obstacles.items():
        if obs_id == -9:
            continue    # 跳过自车

        if obs_info['features']['type'] != 'VEHICLE':
            continue    # 跳过非机动车

        # 检查轨迹质量参数
        if obs_info['trajectory_quality']['abnormal_score'] > cfg.get("trajectory_abnormal_score_th", 3.0):
            continue

        obs_states = get_valid_trajectory(obs_info, str(obs_id))
        if not obs_states:
            continue

        obs_speed = math.hypot(obs_states[0].get("vx",0), obs_states[0].get("vy",0))
        if obs_speed < speed_th:
            continue    # 跳过极慢车

        obs_in_junction = obs_info.get("junction_info", {}).get("in_junction", False)

        if not obs_in_junction and not check_curve_lane(obs_info, scene.percepmap, cfg):
            # 不在路口内也不在弯道上，pass
            continue

        # 计算转向特征
        turn_feature = calc_turn_feature(obs_states, prefix="obs", cfg=cfg)
        output["curve_driving"]["obs_direction"][obs_id] = turn_feature["obs_direction"]
        output["curve_driving"]["obs_intensity"][obs_id] = turn_feature["obs_intensity"]

        # 计算交互特征
        output["curve_driving"]["ego_interact"][obs_id] = check_interaction(
            ego_states, obs_states, 
            ego_in_junction=output["curve_driving"]["is_intersection"],
            obs_in_junction=obs_in_junction,
            cfg=cfg
        )

        # 判断合法性：如果有目标转向方向不为none，且强度大于1，且交互不为none，则合法
        # if (output["curve_driving"]["obs_direction"][obs_id] != 'none' and 
        #     output["curve_driving"]["obs_intensity"][obs_id] > 1 and 
        #     output["curve_driving"]["ego_interact"][obs_id] != 'none'):
        #     output["curve_driving"]['valid'] = True 
        #     log_str = log_str + str(obs_id) + "-" + \
        #               str(output["curve_driving"]["obs_direction"][obs_id]) + "-" + \
        #               str(output["curve_driving"]["obs_intensity"][obs_id]) + "-" + \
        #               str(output["curve_driving"]["ego_interact"][obs_id]) + "_"

    # save_dir = "/home/sti/backbone-traj-nn-tagger/test/curve/"
    # save_sub_dir = cfg['clip_name']
    # pickle_name = cfg['pickle_name']

    # if output["curve_driving"]['valid']:
    #     draw_scene(scene, save_dir, save_sub_dir, (pickle_name + log_str)[:-1])
    #     print(f"Valid for {pickle_name}")
    # else:
    #     print(f"Non Valid for {pickle_name}")

    # except Exception as e:
    #     print(f"[curve_driving] Error: {str(e)}")
    
    return output


def get_valid_trajectory(obs_info: dict, identifier: str) -> List[dict]:
    """验证并获取有效轨迹数据"""
    try:
        states = obs_info.get("future_trajectory", {}).get("future_states", [])
        if len(states) < 2:
            # print(f"{identifier} trajectory length < 2")
            return []
        return states
    except AttributeError:
        print(f"{identifier} missing trajectory data")
        return []
    

def angle_normalize(angle):
    angle = angle % (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle
    

def calc_turn_feature(states: List[dict], prefix: str, cfg: dict) -> dict:
    """计算转向特征"""
    if len(states) < 51:
        return {f"{prefix}_direction": "none", f"{prefix}_intensity": 0}
    
    start_theta = states[0].get("theta", 0.0)
    end_theta = states[-1].get("theta", 0.0)
    middle_theta = states[50].get("theta", 0.0)

    # 计算行进距离差，若0到5秒内的行进距离小于阈值，则视为不合法
    sum_distance = 0.0
    for i in range(0, 41, 10):
        dx = states[i]['x'] - states[i+10]['x']
        dy = states[i]['y'] - states[i+10]['y']
        sum_distance += math.hypot(dx, dy)

    if sum_distance < cfg.get("sum_distance_th", 10.0):
        return {f"{prefix}_direction": "none", f"{prefix}_intensity": 0}
    
    # 计算规范化角度差，取5秒和8秒两种情况的最大值，避免S型弯道末端角度回正的情况
    delta_theta = max(angle_normalize(end_theta - start_theta), angle_normalize(middle_theta - start_theta))
    
    # 计算方向和强度
    direction = "none"
    if abs(delta_theta) > cfg.get("none_direction_angle", math.pi / 36.0):  # 忽略微小角度变化
        direction = "left" if delta_theta > 0 else "right"
    
    if direction == "none":
        intensity_level = 0
    else:
        intensity_level = int(abs(delta_theta) // cfg.get("intensity_level_angle", math.pi / 12.0)) + 1
    
    return {
        f"{prefix}_direction": direction,
        f"{prefix}_intensity": intensity_level
    }


def check_curve_lane(obs_info: dict, percep_map, cfg: dict) -> bool:
    """检查车道是否有一定的曲率"""
    angle_threshold = cfg.get("curve_angle_threshold", math.pi / 12.0)  # 角度阈值
    sample_angle_points_diff = cfg.get("sample_angle_points_diff", 15)  # 每几个点计算一次角度差

    obs_cur_lane_ids = obs_info.get('lane_graph', {}).get('current_lanes', [])
    for cur_lane_id in obs_cur_lane_ids:
        lane_info = percep_map.lane_map.get(cur_lane_id)
        if not lane_info or lane_info["type"] == 'UNKNOWN_LANETYPE':
            continue
        directions = lane_info["unit_directions"]
        if len(directions) <= sample_angle_points_diff:
            continue
        
        for i in range(0, len(directions) - sample_angle_points_diff, sample_angle_points_diff):
            cos_theta = np.dot(directions[i], directions[i + sample_angle_points_diff])
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 避免浮点误差
            if angle > angle_threshold:
                return True
    
    return False


def check_interaction(ego_states: List[dict], 
                      obs_states: List[dict],
                      ego_in_junction: bool,
                      obs_in_junction: bool,
                      cfg: dict) -> str:
    """判断自车与他车的交互类型"""

    interaction_type = "none"
    if not ego_states or not obs_states:
        return "none"
    
    # 场景判断条件 ----------------------------------------------
    params = {
        "abs_dis_th": cfg.get("absolutely_distance", 20.0),
        "normal_parallel_th": cfg.get("normal_parallel_distance", 5.0),
        "closing_parallel_th": cfg.get("closing_parallel_distance", 10.0),  # 接近自车的横向距离适当放宽
        "closing_angle_diff": cfg.get("closing_angle_diff", math.pi / 18.0), # 接近阈值角，10度
        "angle_sim_th": cfg.get("angle_similarity", math.pi / 2.0),         # 两车的速度角差最大值，90度
        "min_parallel_th": cfg.get("min_parallel_distance", 1.5),            # 最小平行距离
        "min_angle_diff": cfg.get("min_angle_diff", math.pi / 30.0)          # 最小平行角度，和距离一起作用
    }
    
    # 获取当前时刻状态
    ego_current = ego_states[0]
    obs_current = obs_states[0]
    
    # 坐标系转换 ------------------------------------------------
    # 将障碍物位置转换到ego坐标系
    dx = obs_current["x"] - ego_current["x"]
    dy = obs_current["y"] - ego_current["y"]
    ego_theta = ego_current["theta"]
    rotated_x = dx * math.cos(ego_theta) + dy * math.sin(ego_theta)
    rotated_y = -dx * math.sin(ego_theta) + dy * math.cos(ego_theta)
    
    distance = math.hypot(dx, dy)
    lateral_dist = abs(rotated_y)

    # 绝对距离阈值检查（必须满足）
    if distance > params["abs_dis_th"]:
        return "none"
    
    # 运动方向分析 --------------------------------------------------
    ego_dir = math.atan2(ego_current["vy"], ego_current["vx"])
    obs_dir = math.atan2(obs_current["vy"], obs_current["vx"])
    dir_diff = angle_normalize(obs_dir - ego_dir)

    # 利用横向距离和角度差判断是否位于自车正前后，若是视为无交互
    # if lateral_dist < params["min_parallel_th"] and abs(dir_diff) < params["min_angle_diff"] and not obs_in_junction:
    #     return "none"
    
    # 相对运动趋势计算
    relative_vx = obs_current["vx"] - ego_current["vx"]
    relative_vy = obs_current["vy"] - ego_current["vy"]
    closing_speed = (rotated_x * relative_vx + rotated_y * relative_vy) / max(distance, 1e-5)

    # 并排转向场景判断（分两种情况）------------------------------------
    is_normal_parallel = (
        lateral_dist < params["normal_parallel_th"] 
        and abs(dir_diff) < params["angle_sim_th"]
    )

    is_closing_parallel = (
        lateral_dist < params["closing_parallel_th"]
        and abs(dir_diff) > params["closing_angle_diff"]
        and abs(dir_diff) < params["angle_sim_th"]
        and closing_speed < 0  # 正在接近
    )

    if is_normal_parallel:
        interaction_type = "normal"
    if is_closing_parallel:
        interaction_type = "closing"
    if interaction_type != "none" and not obs_in_junction:
        interaction_type = "curve"

    return interaction_type