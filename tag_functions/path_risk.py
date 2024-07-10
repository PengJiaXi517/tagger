import os

from registry import TAG_FUNCTIONS

@TAG_FUNCTIONS.register()
def path_risk(data, params, result):
    # print(1)
    tag_ego_path_valid_length_risk = False
    tag_ego_path_endpoint_condition_lane_risk = False
    tag_ego_path_endpoint_

    condition_res = data['condition_res']
    label_scene = data['label_scene']


    lane_seq_pair = condition_res.lane_seq_pair
    for item in lane_seq_pair:
        if item [-1].sum() < 34:
            tag_valid_length_risk = True

    future_path = label_scene.ego_path_info.future_path
    end_point = future_path[-1]


    pass


