import os

import numpy as np
from shapely import geometry

from registry import TAG_FUNCTIONS

VALID_PATH_NUM = 100

@TAG_FUNCTIONS.register()
def path_risk(data, params):

    output = {'path_risk':{}}
    tag_ego_path_valid_length_risk = False
    tag_ego_path_endpoint_condition_lane_risk = False
    tag_ego_path_endpoint_not_in_map_risk = True

    condition_res = data.condition_res
    label_scene = data.label_scene

    lane_seq_pair = condition_res.lane_seq_pair
    for item in lane_seq_pair:
        if item[-1].sum() < VALID_PATH_NUM:
            tag_ego_path_valid_length_risk = True

    future_path = label_scene.ego_path_info.future_path
    if len(future_path) != 0:
        end_point = geometry.Point(future_path[-1])
        for item in lane_seq_pair:
            start_id, end_id = item[0], item[1]
            if end_id != -1:
                condition_lane_ids = condition_res.seq_lane_ids_raw[end_id]
            else:
                condition_lane_ids = condition_res.seq_lane_ids_raw[start_id]
            min_dis = np.Inf
            for lane_id in condition_lane_ids:
                polyline = label_scene.percepmap.lane_map[int(lane_id)]["polyline"]
                if len(polyline) > 1:
                    condition_line_string = geometry.LineString(polyline)
                    dis = end_point.distance(condition_line_string)
                    if dis < min_dis:
                        min_dis = dis
            if min_dis > 10:
                tag_ego_path_endpoint_condition_lane_risk = True

        for k, v in label_scene.percepmap.junction_map.items():
            if len(v["polygon"]) < 4:
                continue
            juction_polygon = geometry.Polygon(v["polygon"])
            if juction_polygon.contains(end_point) or juction_polygon.intersects(end_point):
                tag_ego_path_endpoint_not_in_map_risk = False

        for k, v in label_scene.percepmap.lane_map.items():
            if len(v['polyline']) < 2:
                continue
            line_string = geometry.LineString(v["polyline"])
            if line_string.distance(end_point) < 5:
                tag_ego_path_endpoint_not_in_map_risk = False

    output['path_risk']["tag_ego_path_valid_length_risk"] = tag_ego_path_valid_length_risk
    output['path_risk'][
        "tag_ego_path_endpoint_condition_lane_risk"
    ] = tag_ego_path_endpoint_condition_lane_risk
    output['path_risk'][
        "tag_ego_path_endpoint_not_in_map_risk"
    ] = tag_ego_path_endpoint_not_in_map_risk

    return output
