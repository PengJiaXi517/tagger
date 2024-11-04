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
    "label_high_value_tag": {
        "big_car_area": 18.0,
        "near_static_obs_dist_strict": 0.9,
        "near_static_obs_dist_loose": 1.5,
        "near_moving_obs_dist": 0.75,
        "near_caution_obs_dist": 1.0,
        "large_curvature_threshold": 0.05,
        "time_window_bef_stop": 2e6,
        "sample_point_length": 1.0,
    },
}

max_valid_point_num = 100
