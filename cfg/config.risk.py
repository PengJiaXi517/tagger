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
    "future_path_tag": {
        "sample_point_length": 1.0,
    },
    "interactive_tag": {},
    "near_static_obj_tag": {
        "near_static_obs_dist_strict": 0.9,
        "near_static_obs_dist_relax": 1.5,
        "curvature_th": 0.05,
    },
    "mixed_traffic_tag": {
        "near_static_obs_dist_strict": 0.75,
        "near_static_obs_dist_relax": 1.5,
    },
    "yield_vru_tag": {},
    "ramp_tag": {},
}

max_valid_point_num = 100
