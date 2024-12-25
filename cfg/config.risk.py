tag_pipelines = {
    "path_risk": {
        "valid_path_num": 100,
    },
    "map_risk": {},
    "condition_risk_check": {},
    "ego_state_location": {},
    "ego_state_speed": {},
    "ego_state_occ_environment": {},
    "ego_state_map_environment": {},
    "ego_state_obs_environment": {},
    "interactive_tag": {},
    "park_check": {},
    "traffic_start_slow_check": {},
    "label_high_value_tag": {
        "big_car_area": 18.0,
        "near_static_obs_dist_strict_th": 0.9,
        "near_static_obs_dist_loose_th": 1.5,
        "near_moving_obs_dist_th": 0.75,
        "near_caution_obs_dist_th": 1.0,
        "near_ramp_curb_dist_th": 3.5,
        "large_curvature_threshold": 0.05,
        "time_window_bef_stop": 3e6,
        "sample_point_length": 1.0,
    },
    "abnormal_yield_tag": {},
}

max_valid_point_num = 100
