import os

import numpy as np
from shapely import geometry

from registry import TAG_FUNCTIONS


@TAG_FUNCTIONS.register()
def ego_state_location(data, params, result):

    output = {}

    in_juction = False
    before_juction = False
    after_juction = False

    in_road = False

    condition_res = data.condition_res
    label_scene = data.label_scene

    if len(label_scene.ego_path_info.future_path) == 0:
        output["ego_in_juction"] = in_juction
        output["ego_before_juction"] = before_juction
        output["ego_after_juction"] = after_juction
        output["ego_in_road"] = in_road
        return output

    if label_scene.ego_path_info.in_junction_id[0] is not None:
        in_juction = True

    if label_scene.ego_path_info.in_junction_id[0] is None:
        for i in range(1, len(label_scene.ego_path_info.in_junction_id)):
            if label_scene.ego_path_info.in_junction_id[i] is not None:
                before_juction = True

    if not in_juction:
        ego_history_x = label_scene.obstacles[-9]["features"]["history_states"][0]["x"]
        ego_history_y = label_scene.obstacles[-9]["features"]["history_states"][0]["y"]
        ego_history_point = geometry.Point([ego_history_x, ego_history_y])
        for k, v in label_scene.percepmap.junction_map.items():
            juction_polygon = geometry.Polygon(v["polygon"])
            if juction_polygon.contains(
                ego_history_point
            ) or juction_polygon.intersects(ego_history_point):
                after_juction = True

    future_path = label_scene.ego_path_info.future_path
    start_point = geometry.Point(future_path[0])
    if not in_juction:
        seq_lane_ids_raw = condition_res.seq_lane_ids_raw
        for seq_lane_ids in seq_lane_ids_raw:
            for seq_lane_id in seq_lane_ids:
                left_points = label_scene.percepmap.lane_map[int(seq_lane_id)][
                    "left_boundary"
                ]["polyline"]
                right_points = label_scene.percepmap.lane_map[int(seq_lane_id)][
                    "right_boundary"
                ]["polyline"]
                lane_polygon_points = []
                lane_polygon_points.extend(left_points)
                lane_polygon_points.extend(right_points[::-1])
                lane_polygon_points.append(left_points[0])
                if np.isnan(np.array(lane_polygon_points).sum()):
                    continue
                lane_polygon = geometry.Polygon(lane_polygon_points)
                if lane_polygon.contains(start_point) or lane_polygon.intersects(
                    start_point
                ):
                    in_road = True

    output["ego_in_juction"] = in_juction
    output["ego_before_juction"] = before_juction
    output["ego_after_juction"] = after_juction
    output["ego_in_road"] = in_road

    return output


@TAG_FUNCTIONS.register()
def ego_state_speed(data, params, result):

    output = {}

    label_scene = data.label_scene

    speed_odom_x = label_scene.obstacles[-9]["features"]["history_states"][-1]["vx"]
    speed_odom_y = label_scene.obstacles[-9]["features"]["history_states"][-1]["vy"]

    output["ego_speed_odom_x"] = speed_odom_x
    output["ego_speed_odom_y"] = speed_odom_y

    return output


@TAG_FUNCTIONS.register()
def ego_state_map_environment(data, params, result):

    output = {}

    ego_range_has_intersection = False
    ego_range_nearest_intersection_area = 0
    ego_range_nearest_intersection_exist_lane_nums = 0
    ego_range_nearest_intersection_entry_lane_nums = 0
    ego_range_nearest_intersection_exist_lane_dir_vector = []
    ego_range_nearest_intersection_entry_lane_dir_vector = []

    ego_range_has_m2n = False

    condition_res = data.condition_res
    label_scene = data.label_scene

    ego_range_num_lanes = len(condition_res.seq_lane_ids_raw)

    if len(label_scene.percepmap.junction_map) > 0:
        ego_range_has_intersection = True

        future_path = label_scene.ego_path_info.future_path
        start_point = geometry.Point(future_path[0])

        nearest_polygon_k = -1
        min_dis = np.Inf
        for k, v in label_scene.percepmap.junction_map.items():
            if k < 0:
                ego_range_has_m2n = True
            juction_polygon = geometry.Polygon(v["polygon"])
            if juction_polygon.contains(start_point) or juction_polygon.intersects(
                start_point
            ):
                dis = 0
                if dis < min_dis:
                    min_dis = dis
                    nearest_polygon_k = k
            else:
                dis = juction_polygon.distance(start_point)
                if dis < min_dis:
                    nearest_polygon_k = k
                    min_dis = dis

        nearest_polygon = label_scene.percepmap.junction_map[nearest_polygon_k]
        ego_range_nearest_intersection_area = geometry.Polygon(
            nearest_polygon["polygon"]
        ).area

        entry_lanes = []
        for item in nearest_polygon["entry_groups"]:
            entry_lanes.extend(item)

        exit_lanes = []
        for item in nearest_polygon["exit_groups"]:
            exit_lanes.extend(item)

        ego_range_nearest_intersection_entry_lane_nums = len(entry_lanes)
        ego_range_nearest_intersection_exist_lane_nums = len(exit_lanes)

        for entry_lane_id in entry_lanes:
            ego_range_nearest_intersection_entry_lane_dir_vector.append(
                label_scene.percepmap.lane_map[entry_lane_id]["unit_directions"][-1]
            )

        for exit_lane_id in exit_lanes:
            ego_range_nearest_intersection_exist_lane_dir_vector.append(
                label_scene.percepmap.lane_map[exit_lane_id]["unit_directions"][-1]
            )

    output["ego_range_has_intersection"] = ego_range_has_intersection
    output["ego_range_nearest_intersection_area"] = ego_range_nearest_intersection_area
    output[
        "ego_range_nearest_intersection_exist_lane_nums"
    ] = ego_range_nearest_intersection_exist_lane_nums
    output[
        "ego_range_nearest_intersection_entry_lane_nums"
    ] = ego_range_nearest_intersection_entry_lane_nums
    output[
        "ego_range_nearest_intersection_exist_lane_dir_vector"
    ] = ego_range_nearest_intersection_exist_lane_dir_vector
    output[
        "ego_range_nearest_intersection_entry_lane_dir_vector"
    ] = ego_range_nearest_intersection_entry_lane_dir_vector
    output["ego_range_has_m2n"] = ego_range_has_m2n
    output["ego_range_num_lanes"] = ego_range_num_lanes

    return output


@TAG_FUNCTIONS.register()
def ego_state_obs_environment(data, params, result):

    output = {}

    ego_range_5m_obs_nums = 0
    ego_range_5m_vehicle_nums = 0

    condition_res = data.condition_res
    label_scene = data.label_scene

    future_path = label_scene.ego_path_info.future_path
    start_point = geometry.Point(future_path[0])

    for k, v in label_scene.obstacles.items():
        if k != -9:
            if v["features"]["type"] == "VEHICLE":
                obs_x = v["features"]["history_states"][-1]["x"]
                obs_y = v["features"]["history_states"][-1]["y"]
                obs_point = geometry.Point([obs_x, obs_y])
                if start_point.distance(obs_point) < 5:
                    ego_range_5m_vehicle_nums += 1
                    ego_range_5m_obs_nums += 1
            else:
                obs_x = v["features"]["history_states"][-1]["x"]
                obs_y = v["features"]["history_states"][-1]["y"]
                obs_point = geometry.Point([obs_x, obs_y])
                if start_point.distance(obs_point) < 5:
                    ego_range_5m_obs_nums += 1

    output["ego_range_5m_obs_nums"] = ego_range_5m_obs_nums
    output["ego_range_5m_vehicle_nums"] = ego_range_5m_vehicle_nums
    return output
