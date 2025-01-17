import os

import numpy as np
from shapely import geometry

from utils.trans import trans_odom2ego
from registry import TAG_FUNCTIONS


@TAG_FUNCTIONS.register()
def ego_state_location(data, params):

    output = {'ego_state_location':{}}

    in_juction = False
    before_juction = False
    after_juction = False

    in_road = False

    condition_res = data.condition_res
    label_scene = data.label_scene

    if len(label_scene.ego_path_info.future_path) == 0:
        output['ego_state_location']["ego_in_juction"] = in_juction
        output['ego_state_location']["ego_before_juction"] = before_juction
        output['ego_state_location']["ego_after_juction"] = after_juction
        output['ego_state_location']["ego_in_road"] = in_road
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

    output['ego_state_location']["ego_in_juction"] = in_juction
    output['ego_state_location']["ego_before_juction"] = before_juction
    output['ego_state_location']["ego_after_juction"] = after_juction
    output['ego_state_location']["ego_in_road"] = in_road

    ego_lla = label_scene.label_res.get('frame_info', {}).get('ego_lla', [])
    output['ego_state_location']["lla"] = ego_lla

    return output


@TAG_FUNCTIONS.register()
def ego_state_speed(data, params):

    output = {'ego_state_speed':{}}

    label_scene = data.label_scene

    speed_odom_x = label_scene.obstacles[-9]["features"]["history_states"][-1]["vx"]
    speed_odom_y = label_scene.obstacles[-9]["features"]["history_states"][-1]["vy"]

    output['ego_state_speed']["ego_speed_odom_x"] = speed_odom_x
    output['ego_state_speed']["ego_speed_odom_y"] = speed_odom_y
    if len(label_scene.obstacles[-9]['future_trajectory']['future_states']) > 51:
        future_states = label_scene.obstacles[-9]['future_trajectory']['future_states']
        start_v = np.sqrt(future_states[0]['vx'] ** 2 + future_states[0]['vy'] ** 2)
        end_v = np.sqrt(future_states[50]['vx'] ** 2 + future_states[50]['vy'] ** 2)
        output['ego_state_speed']["ego_speed_5s_acc"] = (end_v - start_v) / 5.0

    obstacles = label_scene.obstacles
    if len(obstacles[-9]['features']['history_states']) > 1 and len(obstacles[-9]['future_trajectory']['future_states']) > 0:
        cur_state = obstacles[-9]['features']['history_states'][-1]
        pre_state = obstacles[-9]['features']['history_states'][-2]
        next_state = obstacles[-9]['future_trajectory']['future_states'][0]

        cur_x, cur_y, cur_t = cur_state['x'], cur_state['y'], cur_state['timestamp']
        pre_x, pre_y, pre_t = pre_state['x'], pre_state['y'], pre_state['timestamp']
        next_x, next_y, next_t = next_state['x'], next_state['y'], next_state['timestamp']

        cur_x_v = (cur_x - pre_x) / ((cur_t - pre_t) / 10e5)
        cur_y_v = (cur_y - pre_y) / ((cur_t - pre_t) / 10e5)

        next_x_v = (next_x - cur_x) / ((next_t - cur_t) / 10e5)
        next_y_v = (next_y - cur_y) / ((next_t - cur_t) / 10e5)

        cur_x_acc = (next_x_v - cur_x_v) / ((next_t - cur_t) / 10e5)
        cur_y_acc = (next_y_v - cur_y_v) / ((next_t - cur_t) / 10e5)
        output['ego_state_speed']["ego_acc_odom_x"] = cur_x_acc
        output['ego_state_speed']["ego_acc_odom_y"] = cur_y_acc

    return output


@TAG_FUNCTIONS.register()
def ego_state_map_environment(data, params):

    output = {'ego_state_map_environment':{}}

    ego_range_has_intersection = False
    ego_range_nearest_intersection_area = 0
    ego_range_nearest_intersection_exist_lane_nums = 0
    ego_range_nearest_intersection_entry_lane_nums = 0
    ego_range_nearest_intersection_exist_lane_dir_vector = []
    ego_range_nearest_intersection_entry_lane_dir_vector = []
    ego_range_percep_map_loc = {}

    ego_range_has_m2n = False

    condition_res = data.condition_res
    label_scene = data.label_scene

    ego_range_percep_map_loc = label_scene.label_res.get('frame_info', {}).get('percep_map_loc', {})

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

    output['ego_state_map_environment']["ego_range_has_intersection"] = ego_range_has_intersection
    output['ego_state_map_environment']["ego_range_nearest_intersection_area"] = ego_range_nearest_intersection_area
    output['ego_state_map_environment'][
        "ego_range_nearest_intersection_exist_lane_nums"
    ] = ego_range_nearest_intersection_exist_lane_nums
    output['ego_state_map_environment'][
        "ego_range_nearest_intersection_entry_lane_nums"
    ] = ego_range_nearest_intersection_entry_lane_nums
    output['ego_state_map_environment'][
        "ego_range_nearest_intersection_exist_lane_dir_vector"
    ] = ego_range_nearest_intersection_exist_lane_dir_vector
    output['ego_state_map_environment'][
        "ego_range_nearest_intersection_entry_lane_dir_vector"
    ] = ego_range_nearest_intersection_entry_lane_dir_vector
    output['ego_state_map_environment']["ego_range_has_m2n"] = ego_range_has_m2n
    output['ego_state_map_environment']["ego_range_num_lanes"] = ego_range_num_lanes
    output['ego_state_map_environment']["ego_range_percep_map_loc"] = ego_range_percep_map_loc

    return output


@TAG_FUNCTIONS.register()
def ego_state_occ_environment(data, params):

    output = {'ego_state_occ_environment': {}}

    ego_state_intersect_occ = False
    ego_state_nearest_occ_dis = 9999.0

    condition_res = data.condition_res
    label_scene = data.label_scene

    future_path = label_scene.ego_path_info.future_path
    occ_obstacles = label_scene.label_res['occ_obstacles']

    future_path = geometry.LineString(future_path)
    for k, v in occ_obstacles.items():
        polygon_points = v['polygon_points']
        polygon = geometry.Polygon(polygon_points)
        if future_path.intersects(polygon):
            ego_state_intersect_occ = True
            ego_state_nearest_occ_dis = 0
            break
        else:
            polygon_dis = future_path.distance(polygon)
            if polygon_dis < ego_state_nearest_occ_dis:
                ego_state_nearest_occ_dis = polygon_dis

    output['ego_state_occ_environment']['ego_state_intersect_occ'] = ego_state_intersect_occ
    output['ego_state_occ_environment']['ego_state_nearest_occ_dis'] = ego_state_nearest_occ_dis

    return output


@TAG_FUNCTIONS.register()
def ego_state_obs_environment(data, params):

    output = {'ego_state_obs_environment':{}}

    ego_range_5m_obs_nums = 0
    ego_range_5m_vehicle_nums = 0

    condition_res = data.condition_res
    label_scene = data.label_scene

    future_path = label_scene.ego_path_info.future_path
    start_point = geometry.Point(future_path[0])

    ego_state = label_scene.obstacles[-9]['features']['history_states'][-1]
    odom_x, odom_y, odom_theta = ego_state['x'], ego_state['y'], ego_state['theta']

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

    output['ego_state_obs_environment']["ego_range_5m_obs_nums"] = ego_range_5m_obs_nums
    output['ego_state_obs_environment']["ego_range_5m_vehicle_nums"] = ego_range_5m_vehicle_nums

    ego_range_polygon1 = geometry.Polygon([[-5, 10], [30, 10], [30, -10], [-5, -10], [-5, 10]])
    ego_range_polygon2 = geometry.Polygon([[-5, 10], [50, 10], [50, -10], [-5, -10], [-5, 10]])
    ego_range_polygon3 = geometry.Polygon([[-5, 10], [70, 10], [70, -10], [-5, -10], [-5, 10]])
    ego_range_30m_vehicle_nums = 0
    ego_range_50m_vehicle_nums = 0
    ego_range_70m_vehicle_nums = 0
    for k, v in label_scene.obstacles.items():
        if k != -9:
            if v["features"]["type"] == "VEHICLE":
                obs_x = v["features"]["history_states"][-1]["x"]
                obs_y = v["features"]["history_states"][-1]["y"]
                obs_ego_loc = trans_odom2ego(odom_x, odom_y, odom_theta, np.array([obs_x, obs_y]))
                obs_point = geometry.Point(obs_ego_loc)
                if obs_point.intersects(ego_range_polygon1) or ego_range_polygon1.contains(obs_point):
                    ego_range_30m_vehicle_nums += 1
                if obs_point.intersects(ego_range_polygon2) or ego_range_polygon2.contains(obs_point):
                    ego_range_50m_vehicle_nums += 1
                if obs_point.intersects(ego_range_polygon3) or ego_range_polygon3.contains(obs_point):
                    ego_range_70m_vehicle_nums += 1
    output['ego_state_obs_environment']["ego_range_30m_vehicle_nums"] = ego_range_30m_vehicle_nums
    output['ego_state_obs_environment']["ego_range_50m_vehicle_nums"] = ego_range_50m_vehicle_nums
    output['ego_state_obs_environment']["ego_range_70m_vehicle_nums"] = ego_range_70m_vehicle_nums

    ego_current_lanes = label_scene.obstacles[-9]['lane_graph']['current_lanes']
    cipv_dis = 9999
    cipv_vx = None
    cipv_vy = None
    if len(ego_current_lanes) != 0:
        for k, v in label_scene.obstacles.items():
            if k != -9:
                if v["features"]["type"] == "VEHICLE":
                    current_lanes = v['lane_graph']['current_lanes']
                    for current_lane in current_lanes:
                        if current_lane in ego_current_lanes:
                            veh_odom_x, veh_odom_y = v['features']['history_states'][-1]['x'], v['features']['history_states'][-1]['y']
                            dis = np.sqrt((veh_odom_x - odom_x) ** 2 + (veh_odom_y - odom_y) ** 2)
                            if dis < cipv_dis:
                                cipv_dis = dis
                                cipv_vx = v['features']['history_states'][-1]['vx']
                                cipv_vy = v['features']['history_states'][-1]['vy']

    else:
        future_path = geometry.LineString(future_path)
        for k, v in label_scene.obstacles.items():
            if k != -9:
                if v["features"]["type"] == "VEHICLE":
                    veh_odom_x, veh_odom_y = v['features']['history_states'][-1]['x'], \
                                             v['features']['history_states'][-1]['y']

                    veh_point = geometry.Point([veh_odom_x, veh_odom_y])
                    if future_path.distance(veh_point) < 0.5:
                        dis = np.sqrt((veh_odom_x - odom_x) ** 2 + (veh_odom_y - odom_y) ** 2)
                        if dis < cipv_dis:
                            cipv_dis = dis
                            cipv_vx = v['features']['history_states'][-1]['vx']
                            cipv_vy = v['features']['history_states'][-1]['vy']

    output['ego_state_obs_environment']["cipv_dis"] = cipv_dis
    output['ego_state_obs_environment']["cipv_vx"] = cipv_vx
    output['ego_state_obs_environment']["cipv_vy"] = cipv_vy


    return output
