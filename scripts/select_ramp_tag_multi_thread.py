import os
import sys
sys.path.append('../path-nn-tagger')

import json
from tqdm import tqdm
from typing import Dict, List, Tuple
from shapely.geometry import LineString, Point, Polygon
from multiprocessing import Pool
import pickle
import random

# import boto3
import numpy as np
import copy
from base import TagData
from main import TagParse
from tag_functions.high_value_scene.hv_utils.basic_func import (
    build_linestring_from_lane_seq_ids,
    calculate_future_path_curvature_and_turn_type,
)
from tag_functions.high_value_scene.right_turn_only_tag import (
    RightTurnOnlyTagHelper,
)
from tag_functions.high_value_scene.hv_utils.future_path_collision_checker import (
    FuturePathCollisionChecker,
)
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)


def filter_by_curvature(data: Dict) -> bool:
    curvature_threshold = 0.1
    if data["ramp_tag"]["is_enter_ramp"]:
        curvature_threshold = 0.25
    elif data["ramp_tag"]["is_exit_ramp"]:
        curvature_threshold = 0.15

    abs_sum_path_curvature = data["basic_path_tag"]["abs_sum_path_curvature"]
    if abs_sum_path_curvature < curvature_threshold:
        return False

    if data["max_abs_path_curvature"] > 1.5:
        return False

    return True


def filter_by_map_risk(data: Dict) -> bool:
    if data["map_risk"]["tag_ego_path_curb_cross_risk"]:
        return False

    if data["map_risk"]["tag_lane_curb_cross_risk"]:
        return False

    return True


def filter_by_path_risk(data: Dict) -> bool:
    if data["path_risk"]["tag_ego_path_endpoint_not_in_map_risk"]:
        return False

    if data["path_risk"]["tag_ego_path_endpoint_condition_lane_risk"]:
        return False

    if data["path_risk"]["tag_ego_path_valid_length_risk"]:
        return False

    return True


def filter_by_other_tags(data: Dict) -> bool:
    if data["right_turn_only_tag"]["is_right_turn_only"]:
        return False

    return True


def filter_invalid_tag(data: Dict) -> bool:
    if not filter_by_curvature(data):
        return False

    if not filter_by_map_risk(data):
        return False

    if not filter_by_path_risk(data):
        return False

    if not filter_by_other_tags(data):
        return False

    return True


def file_path_validity_check(file_path: str) -> bool:
    print("file_path: ", file_path)
    # return True
    parts = file_path.split("/")
    if not parts[-2].startswith("73204_17909") and not parts[-2].startswith("73449_17909"):
        return True

    return False


def find_nearest_cur_lane_seq(
    data: TagData,
    current_lane_seqs: List[List[int]],
) -> List[int]:
    future_path = data.label_scene.ego_path_info.future_path
    lane_map = data.label_scene.percepmap.lane_map

    min_lateral_dist = 1e6
    nearest_lane_seq_ids = None

    for lane_seq_ids in current_lane_seqs:
        lane_seq_linestring = build_linestring_from_lane_seq_ids(
            lane_map, lane_seq_ids
        )

        sum_proj_l = 0
        for point in future_path:
            path_point = Point(point)
            proj_s = lane_seq_linestring.project(path_point)
            if 0 < proj_s < lane_seq_linestring.length:
                proj_dis = lane_seq_linestring.distance(path_point)
                sum_proj_l += proj_dis

        if 0 < sum_proj_l < min_lateral_dist:
            min_lateral_dist = sum_proj_l
            nearest_lane_seq_ids = lane_seq_ids

    return nearest_lane_seq_ids


def generate_surpervise_data(
    data: Dict, tag_parser: TagParse, base_data_root, condition_data_root
) -> Dict:
    tag_data = copy.deepcopy(data)

    pickle_sub_path = tag_data["file_path"]
    # if "1729588965591212" not in pickle_sub_path:
    #     return tag_data

    tag_parser.read_data(base_data_root, condition_data_root, pickle_sub_path)
    pickle_data = tag_parser.tag_data
    lane_map = pickle_data.label_scene.percepmap.lane_map
    future_path = pickle_data.label_scene.ego_path_info.future_path
    current_lanes = pickle_data.label_scene.ego_obs_lane_seq_info.current_lanes
    tag_data.update(
        {
            "junction_id": pickle_data.label_scene.obstacles[-9].get(
                "junction_info"
            )["junction_id"]
        }
    )

    # 滤除右转专用道
    # if data["right_turn_only_tag"]["is_right_turn_only"]:
    #     return None

    # right_turn_only_tag_helper = RightTurnOnlyTagHelper(
    #     ego_to_junction_dist_threshold=50.0,
    #     future_path_to_junction_dist_threshold=8.0,
    #     sum_path_curvature_threshold=-0.2,
    # )
    # future_path_linestring = (
    #     pickle_data.label_scene.ego_path_info.future_path_linestring
    # )
    # junction_map = pickle_data.label_scene.percepmap.junction_map
    # if right_turn_only_tag_helper.is_ego_vehicle_near_junction(
    #     future_path_linestring, junction_map, current_lanes
    # ):
    #     return None

    # 判断future path是否有绕curb的行为
    # params = {
    #     "big_car_area": 18.0,
    #     "near_static_obs_dist_strict": 0.9,
    #     "near_static_obs_dist_loose": 1.75,
    #     "near_moving_obs_dist": 0.75,
    #     "near_caution_obs_dist": 1.0,
    # }
    # curb_decision = pickle_data.label_scene.label_res["curb_label"].get(
    #     "decision", None
    # )

    # obstacle_filter = ObstacleFilter(
    #     filter_obs_max_l=5.0, front_vehicle_rel_x=10.0, front_vehicle_rel_y=0.5
    # )
    # curbs_linestring_map = obstacle_filter.build_curbs_linestring(curb_decision)

    # curbs_interactive_lat_type = (
    #     curb_decision["interactive_lat_type"]
    #     if curb_decision is not None
    #     else {}
    # )

    # future_path_collision_checker = FuturePathCollisionChecker()
    # (
    #     _,
    #     future_narrow_road_states_loose_threshold,
    #     _
    # ) = future_path_collision_checker.check_future_path_distance_to_curb_and_static_obs(
    #     params,
    #     pickle_data.label_scene.ego_path_info,
    #     pickle_data.label_scene.obstacles[-9],
    #     {},
    #     {},
    #     curbs_linestring_map,
    #     curbs_interactive_lat_type,
    # )

    # (
    #     future_path_curvature,
    #     future_path_turn_type,
    # ) = calculate_future_path_curvature_and_turn_type(future_path)

    # future_bypass_curb = []

    # for idx, (x, y) in enumerate(future_path):
    #     future_bypass_curb.append(False)

    #     if abs(future_path_curvature[idx]) < 0.01:
    #         continue

    #     if (
    #         future_path_turn_type[idx] > 0
    #         and future_narrow_road_states_loose_threshold[idx][0]
    #     ) or (
    #         future_path_turn_type[idx] < 0
    #         and future_narrow_road_states_loose_threshold[idx][1]
    #     ):
    #         future_bypass_curb[idx] = True

    # if not any(future_bypass_curb):
    #     return None

    # fork_lane_ids = []
    # cur_lane_id = current_lanes[0]
    # succ_ids = lane_map[cur_lane_id]["successor_id"]

    # # 判断lane是否为 一分成二
    # if len(succ_ids) == 2:
    #     fork_lane_ids = succ_ids
    # else:
    #     pred_ids = lane_map[cur_lane_id]["predecessor_id"]
    #     if len(pred_ids) == 1:
    #         succ_ids = lane_map[pred_ids[0]]["successor_id"]
    #         if len(succ_ids) == 2 and cur_lane_id in succ_ids:
    #             fork_lane_ids = succ_ids

    # if len(fork_lane_ids) != 2:
    #     return None

    # if (
    #     Point(lane_map[fork_lane_ids[0]]["polyline"][0]).distance(
    #         Point(future_path[0])
    #     )
    #     > 100
    # ):
    #     return None

    current_lane_seqs = (
        pickle_data.label_scene.ego_obs_lane_seq_info.current_lane_seqs
    )
    if len(current_lane_seqs) == 0:
        return None

    nearest_cur_lane_seq = find_nearest_cur_lane_seq(
        pickle_data, current_lane_seqs
    )

    if nearest_cur_lane_seq is None:
        return None

    nearest_cur_lane_seq_linestring = build_linestring_from_lane_seq_ids(
        lane_map, nearest_cur_lane_seq
    )

    if data["path_type"] == "CRUISE":
        fake_path = [[coord for coord in pt] for pt in future_path]
        valid_length = 0
        for idx, point in enumerate(future_path):
            path_point = Point(point)
            proj_s = nearest_cur_lane_seq_linestring.project(path_point)
            if 0 < proj_s < nearest_cur_lane_seq_linestring.length:
                proj_dis = nearest_cur_lane_seq_linestring.distance(path_point)
                if proj_dis <= 0.5:
                    valid_length = idx
        fake_path = fake_path[:valid_length]
        tag_data.update({"fake_path": fake_path})

    # elif data["path_type"] == "LANE_CHANGE":
        # return tag_data

        # 滤除condition在分叉上的情况
        # for start_lane_seq in tag_data["condition_res_tag"][
        #     "start_lane_seq_ids"
        # ]:
        #     for start_lane_id in start_lane_seq:
        #         if start_lane_id in fork_lane_ids:
        #             return None

        # for end_lane_seq in tag_data["condition_res_tag"]["end_lane_seq_ids"]:
        #     for end_lane_id in end_lane_seq:
        #         if end_lane_id in fork_lane_ids:
        #             return None

        # 约一半的数据不用fake
        # if random.random() > 0.5:
        #     return tag_data

        # 开始fake condition和path
        # current_lane_seqs = (
        #     pickle_data.label_scene.ego_obs_lane_seq_info.current_lane_seqs
        # )
        # if len(current_lane_seqs) == 0:
        #     return None

        # nearest_cur_lane_seq = find_nearest_cur_lane_seq(
        #     pickle_data, current_lane_seqs
        # )

        # if nearest_cur_lane_seq is None:
        #     return None

        # nearest_cur_lane_seq_linestring = build_linestring_from_lane_seq_ids(
        #     lane_map, nearest_cur_lane_seq
        # )

        # 将future 向 cur_lane_seq投影，得到伪监督
        # future_path = pickle_data.label_scene.ego_path_info.future_path

        # fake_path = [[coord for coord in pt] for pt in future_path]
        # valid_length = 0
        # for idx, point in enumerate(future_path):
        #     path_point = Point(point)
        #     proj_s = nearest_cur_lane_seq_linestring.project(path_point)
        #     if 0 < proj_s < nearest_cur_lane_seq_linestring.length:
        #         proj_dis = nearest_cur_lane_seq_linestring.distance(path_point)
        #         if proj_dis <= 0.5:
        #             valid_length = idx
        # fake_path = fake_path[:valid_length]

        # # 将cur_lane_seq作为伪condition
        # fake_condition_lane_seq = nearest_cur_lane_seq

        # # 输出
        # tag_data.update({"fake_path": fake_path})
        # tag_data.update({"fake_condition_lane_seq": fake_condition_lane_seq})
        

    return tag_data


def get_ret_info(tag_data, surpervise_data):
    junction_scene = False
    condition_lane_seq_pairs = []
    if "fake_condition_lane_seq" in surpervise_data:
        condition_lane_seq_pairs = [
            (surpervise_data["fake_condition_lane_seq"], None)
        ]
    else:
        for start_ids, end_ids in zip(
            tag_data["condition_res_tag"]["start_lane_seq_ids"],
            tag_data["condition_res_tag"]["end_lane_seq_ids"],
        ):
            if len(end_ids) > 0:
                junction_scene = True
            condition_lane_seq_pairs.append(
                (
                    start_ids if len(start_ids) > 0 else None,
                    end_ids if len(end_ids) > 0 else None,
                )
            )

    path_info = {}
    if "fake_path" in surpervise_data:
        path_info.update({"future_path": surpervise_data["fake_path"]})

    if junction_scene:
        path_info.update({"corr_junction_id": surpervise_data["junction_id"]})

    ret = (tag_data["file_path"], condition_lane_seq_pairs, path_info)
    return ret


def process_file(file_paths, base_data_root, condition_data_root, save_root, task_idx):
    enter_ramp_tags = {}

    enter_ramp_file_list = []

    transformed_pickle_list = []

    tag_parser = TagParse(
        "/mnt/train2/lenny.chen/Code/path-nn-tagger/cfg/config.risk.py"
    )
    for file_path in tqdm(file_paths):
        with open(file_path, "rb") as f:
            tag = json.load(f)
            for ts, data in tag.items():
                # tag_num += 1
                if data["ramp_tag"]["is_enter_ramp"] and filter_invalid_tag(
                    data
                ):
                    surpervise_data = generate_surpervise_data(
                        data,
                        tag_parser,
                        base_data_root,
                        condition_data_root,
                    )
                    if surpervise_data is None:
                        continue
                    # if len(enter_ramp_tags) < sample_num:

                    ret = get_ret_info(data, surpervise_data)

                    # return ({ts: surpervise_data}, data["file_path"], ret)
                    enter_ramp_tags[ts] = surpervise_data
                    enter_ramp_file_list.append(data["file_path"])
                    # ret = get_ret_info(data, surpervise_data)
                    transformed_pickle_list.append(ret)

    if not os.path.exists(save_root):
            os.makedirs(save_root)

    with open(os.path.join(save_root, str(task_idx) + "enter_ramp_tags.json"), "w") as f:
        json.dump(enter_ramp_tags, f)


    with open(os.path.join(save_root, str(task_idx) + "enter_ramp_file_list.json"), "w") as f:
        json.dump(enter_ramp_file_list, f)

    with open(
        os.path.join(save_root, str(task_idx) + "enter_ramp_sample_no_fake.pickle"), "wb"
    ) as f:
        pickle.dump(transformed_pickle_list, f)


    print("enter_ramp_tags len: ", len(enter_ramp_tags))

    print("enter_ramp_file_list len: ", len(enter_ramp_file_list))

    print("transformed_pickle_list len: ", len(transformed_pickle_list))



def select_ramp_tags_multi_thread(tag_data_root, save_root, num_process):
    tag_num = 0
    enter_ramp_num = 0
    exit_ramp_num = 0
    sample_num = 50000
    trip_num = 500

    base_data_root = "s3://pnd/PnPBaseTrainData/base_label_v2.0.9/"
    condition_data_root = ""

    file_paths = []
    for root, dirs, files in os.walk(tag_data_root):
        if trip_num < 0:
            break
        for file in files:
            if file.endswith("tag.json"):
                full_path = os.path.join(root, file)
                if file_path_validity_check(full_path):
                    file_paths.append(full_path)
                    trip_num -= 1
    
    print("file_paths len: ", len(file_paths))

    with Pool(num_process) as pool:
        for i in range(num_process):
            pool.apply_async(
                process_file,
                args=(
                    file_paths[i::num_process],
                    base_data_root,
                    condition_data_root,
                    save_root,
                    i,
                ),
            )

        pool.close()
        pool.join()



if __name__ == "__main__":
    tag_data_root = "/mnt/train2/lenny.chen/result/1101_increase_1108_17.43"
    save_root = "/mnt/train2/lenny.chen/tag_res/select_ramp_tags/test_temp"

    select_ramp_tags_multi_thread(tag_data_root, save_root, num_process=8)
