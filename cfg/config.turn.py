import math

tag_pipelines = {
    "curve_driving": {
        "speed_th": 1.0,
        "absolutely_distance": 20.0,
        "normal_parallel_distance": 5.0,
        "closing_parallel_distance": 10.0,          # 接近自车的横向距离适当放宽
        "closing_angle_diff": math.pi / 18.0,       # 接近阈值角，10度
        "angle_similarity": math.pi / 2.0,          # 两车的速度角差最大值
        "none_direction_angle": math.pi / 36.0,     # 忽略微小角度变化
        "intensity_level_angle": math.pi / 12.0,    # 转角每15度为一级
        "sum_distance_th": 10.0,                    # 5秒内移动小于该值的obs忽略
        "trajectory_abnormal_score_th": 3.0,        # 轨迹质量异常阈值
        "curve_angle_threshold": math.pi / 12.0,    # 车道采样角度差阈值
        "sample_angle_points_diff": 15,             # 车道采样点数差
    },
    "ramp_merge": {
        "speed_th": 3.0,
        "trajectory_abnormal_score_th": 3.0,
        "collision_time_range": 3.0,                # 检测碰撞时间
        "time_step": 0.1,                           # 检测碰撞步长
        "min_angle_diff_threshold": math.pi / 30.0,     # 排除追尾情况的角度差
        "max_angle_diff_threshold": math.pi / 1.8,  # 排除对向碰撞的角度差
    },
}

max_valid_point_num = 100
