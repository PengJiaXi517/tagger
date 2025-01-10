import os
import pickle

from raw_data_preprocess.condition_packer import EgoPathPacker2LaneSeq
from raw_data_preprocess.map_state_packer import MapLaneSeqPacker, TransformToCurr
from raw_data_preprocess.obstacle_state import StateFeatureSplitTypePacker

num_lanes = 48
num_lane_nodes = 128
num_obstacles = 64
history_steps = 11
future_steps = 50
ignore_label = 255

feat_dim = 128
dim_feedforward = 512
nhead = 4

num_goal_feature = 10
num_lane_feature = 10

compose_pipelines = [
    StateFeatureSplitTypePacker(
        src_key="obstacles",
        history_states=history_steps,
        future_states=future_steps,
        sort_by_cost=True,
        sort_by_distance=False,
        filter_min_history=1,
        filter_interpolate=True,
        pad_timestamp=True,
        max_obstacles=num_obstacles,
        types_mapping={
            "VEHICLE": 0,
            "BICYCLE": 1,
            "PEDESTRIAN": 2,
            "PEDESTRAIN": 2,
        },
        ego_id=-9,
        predict_pedestrian=True,
        filter_low_speed=0.3,
        predict_in_range=True,
        perception_range=dict(
            x_range=(-30, 60),
            y_range=(-60, 60),
        ),
        filter_low_quality=True,
        low_quality_threshold=3.0,
        predict_low_speed_quality_ped=True,
        predict_static=True,
        filter_interpolate_ego=True,
    ),
    MapLaneSeqPacker(
        src_key="percepmap",
        src_ego_curr_state_key=("obstacle_state", "ego_curr_state"),
        tgt_state_key=("map_state", "state"),
        tgt_mask_key=("map_state", "state_mask"),
        tgt_id_map_key=("map_state", "seq_lane_ids"),
        tgt_id_map_raw_key=("map_state", "seq_lane_ids_raw"),
        sample_interval=3,
        filter_lane_type=[
            "UNKNOWN_LANETYPE",
            "GENERAL",
            "EMERGENCY",
            "TIDAL",
            "BUS",
            "UNCONV",
            "WAIT_LEFT",
            "WAIT_RIGHT",
            "WAIT_FORWARD",
            "ABNORMAL_LANE",
            "RIGHT_TURN_ONLY",
            "VARIABLE_LANE",
            "U_TURN_LANE",
        ],
        turn_type_mapping={
            "UNKNOWN_LANEDIRECTION": 0,
            "NOTURN": 1,
            "LEFTTURN": 2,
            "LEFTSTRAIGHT": 3,
            "RIHGTTURN": 4,
            "RIHGTSTRAIGHT": 5,
            "LEFTRIGHTTURN": 6,
            "LEFTRIGHTSTRAIGHT": 7,
            "UTURN": 8,
            "USTRAIGHT": 9,
            "USTRAIGHT_LEFT": 11,
            "USTRAIGHT_RIGHT": 13,
        },
        lane_type_mapping={
            "UNKNOWN_LANETYPE": 0,
            "GENERAL": 1,
            "EMERGENCY": 2,
            "TIDAL": 3,
            "BUS": 4,
            "UNCONV": 5,
        },
        lane_boundary_type_mapping={
            "UNKNOWN_SPANABILITY": 0,
            "SOLID": 1,
            "DASHED": 2,
            "VIRTUAL": 3,
        },
        save_lane_ids_raw=True,
        filter_virtual=True,
        max_num_nodes=num_lane_nodes,
        max_num_lane_seqs=num_lanes,
        x_range=(-50, 150),
        y_range=(-100, 100),
        filter_roi_range=True,
        roi_x_range=(-50, 150),
        roi_y_range=(-50, 50),
        interp_fast=True,
    ),
    TransformToCurr(
        src_obstacle_state_key=("obstacle_state", "state"),
        src_map_state_key=("map_state", "state"),
        src_ego_curr_state=("obstacle_state", "ego_curr_state"),
        tgt_obstacle_state_key=("obstacle_state", "state"),
        tgt_map_state_key=("map_state", "state"),
    ),
    EgoPathPacker2LaneSeq(
        src_obs_key="obstacles",
        src_obs_id_map_key=("obstacle_state", "id_map"),
        src_map_id_map_key=("map_state", "id_map"),
        src_map_seqs_lane_id_key=("map_state", "seq_lane_ids_raw"),
        src_rot_aug_state_key=("obstacle_state", "aug_dict"),
        src_lane_seqs_prob_key=(
            "obstacle_state",
            "lane_segs_prob",
        ),
        tgt_ego_path_key=(
            "obstacle_state",
            "ego_path",
        ),
        tgt_ego_path_weights_key=(
            "obstacle_state",
            "ego_path_weights",
        ),
        tgt_ego_path_mask_key=(
            "obstacle_state",
            "ego_path_mask",
        ),
        tgt_ego_future_vt_profile_key=(
            "obstacle_state",
            "ego_vt_profile",
        ),
        tgt_ego_future_vt_profile_mask_key=(
            "obstacle_state",
            "ego_vt_profile_mask",
        ),
        sample_distance=3.0,
        max_path_sample_points=34,
        max_vt_profile_sample_point=50,
        use_distance_coe=True,
        use_curvature_coe=False,
        use_new_curvature_coe=True,
        use_min_entry_and_exit=True,
        use_longest_condition_lane_only=True,
        use_lane_seq_clean=True,
        ignore_label=255,
    ),
]


if __name__ == "__main__":

    base_data_root = (
        "/mnt/train2/pnd_data/PnPBaseTrainDataTmp/road_percep_demoroad_overfit"
    )

    json_list = [
        "36628_196_WeiLai-006_dave.du_2024-06-23-14-33-47/36628_196_WeiLai-006_dave.du_2024-06-23-14-33-47_20_29/labels/1719125116391310.pickle",
    ]

    with open(os.path.join(base_data_root, json_list[0]), "rb") as f:
        data = pickle.load(f)
    data["file_path"] = json_list[0]

    for pipe in compose_pipelines:
        data = pipe(data)

    print(data["obstacle_state"]["start_lane_seqs_ind"])
