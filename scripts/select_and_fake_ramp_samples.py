import os
import sys
import copy
import pickle
import rapidjson
import argparse
import random
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, LineString
from multiprocessing import Pool
import multiprocessing
import traceback

sys.path.append("../path-nn-tagger")
from base import TagData
from main import TagParse
from tag_functions.high_value_scene.hv_utils.basic_func import (
    build_linestring_from_lane_seq_ids,
)


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def error(self, msg, *args):
        return multiprocessing.get_logger().error(msg, *args)

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            self.error(traceback.format_exc())
            raise

        return result


class RampTagSelector:
    def __init__(
        self,
        trip_num: int = 100,
        num_process: int = 8,
        is_fake_path_in_lane_change: bool = True,
        fake_path_min_length: int = 20,
        viz_sample_num: int = 30,
        max_pose_l_diff: float = 1.75,
        max_real_pose_l_diff: float = 1.75,
        max_pose_l: float = 0.7,
        max_mean_pose_l: float = 0.35,
    ) -> None:
        self.trip_num = trip_num
        self.num_process = num_process
        self.is_fake_path_in_lane_change = is_fake_path_in_lane_change
        self.fake_path_min_length = fake_path_min_length
        self.viz_sample_num = viz_sample_num
        self.max_pose_l_diff = max_pose_l_diff
        self.max_real_pose_l_diff = max_real_pose_l_diff
        self.max_pose_l = max_pose_l
        self.max_mean_pose_l = max_mean_pose_l

    def filter_invalid_sample_by_curvature(self, data: Dict) -> bool:
        if data["basic_path_tag"]["abs_sum_path_curvature"] < 0.25:
            return False

        if data["max_abs_path_curvature"] > 1.5:
            return False

        return True

    def filter_invalid_sample_by_map_risk(self, data: Dict) -> bool:
        if data["map_risk"]["tag_ego_path_curb_cross_risk"]:
            return False

        if data["map_risk"]["tag_lane_curb_cross_risk"]:
            return False

        return True

    def filter_invalid_sample_by_path_risk(self, data: Dict) -> bool:
        if data["path_risk"]["tag_ego_path_endpoint_not_in_map_risk"]:
            return False

        if data["path_risk"]["tag_ego_path_endpoint_condition_lane_risk"]:
            return False

        if data["path_risk"]["tag_ego_path_valid_length_risk"]:
            return False

        return True

    def filter_invalid_sample_by_right_turn_only_tag(self, data: Dict) -> bool:
        if data["right_turn_only_tag"]["is_right_turn_only"]:
            return False

        return True

    def filter_invalid_sample_by_lc_path_tag(self, data: Dict) -> bool:
        valid_labeled_lane_seq = []

        for lc_path_tag in data["lc_path_tag"]:
            arrive_final_pose_l = lc_path_tag["arrive_final_pose_l"]
            arrive_percep_pose_l = lc_path_tag["arrive_percep_pose_l"]

            if (
                np.abs(arrive_final_pose_l - arrive_percep_pose_l)
                > self.max_pose_l_diff
            ):
                continue

            valid_labeled_lane_seq.append(lc_path_tag["labeled_lane_seq"])

        if len(valid_labeled_lane_seq) == 0:
            return False

        return True

    def filter_invalid_sample_by_cruise_path_tag(self, data: Dict) -> bool:
        valid_labeled_lane_seq = []

        for cruise_path_tag in data["cruise_path_tag"]:

            real_pose_l = cruise_path_tag["real_pose_l"]
            percep_pose_l = cruise_path_tag["percep_pose_l"]

            if np.abs(real_pose_l - percep_pose_l) > self.max_pose_l_diff:
                continue

            if (
                self.max_pose_l is not None
                and cruise_path_tag["max_pose_l"] > self.max_pose_l
            ):
                continue

            if cruise_path_tag["mean_pose_l"] > self.max_mean_pose_l:
                continue

            if np.abs(real_pose_l) < self.max_real_pose_l_diff:
                valid_labeled_lane_seq.append(
                    cruise_path_tag["labeled_lane_seq"]
                )

        if len(valid_labeled_lane_seq) == 0:
            return False

        return True

    def filter_invalid_sample(self, data: Dict) -> bool:
        if not self.filter_invalid_sample_by_curvature(data):
            return False

        if not self.filter_invalid_sample_by_map_risk(data):
            return False

        if not self.filter_invalid_sample_by_path_risk(data):
            return False

        if not self.filter_invalid_sample_by_right_turn_only_tag(data):
            return False

        if data["path_type"] == "LANE_CHANGE":
            if not self.filter_invalid_sample_by_lc_path_tag(data):
                return False
        elif data["path_type"] == "CRUISE":
            if not self.filter_invalid_sample_by_cruise_path_tag(data):
                return False
        else:
            return False

        return True

    def find_nearest_cur_lane_seq(
        self,
        lane_map: Dict,
        future_path: List[Tuple[float, float]],
        current_lane_seqs: List[List[int]],
    ) -> Tuple[List[int], LineString]:
        min_lateral_dist = 1e6
        nearest_lane_seq_ids = None
        nearest_lane_seq_linestring = None

        for lane_seq_ids in current_lane_seqs:
            lane_seq_linestring = build_linestring_from_lane_seq_ids(
                lane_map, lane_seq_ids
            )
            if lane_seq_linestring is None:
                continue

            sum_proj_l = 0
            sampled_future_path = future_path[::3]
            for point in sampled_future_path:
                path_point = Point(point)
                proj_s = lane_seq_linestring.project(path_point)
                if 0 < proj_s < lane_seq_linestring.length:
                    proj_dis = lane_seq_linestring.distance(path_point)
                    sum_proj_l += proj_dis

            if 0 < sum_proj_l < min_lateral_dist:
                min_lateral_dist = sum_proj_l
                nearest_lane_seq_ids = lane_seq_ids
                nearest_lane_seq_linestring = lane_seq_linestring

        return nearest_lane_seq_ids, nearest_lane_seq_linestring

    def generate_fake_path(
        self,
        future_path: List[Tuple[float, float]],
        nearest_cur_lane_seq_linestring: LineString,
    ) -> List[Tuple[float, float]]:
        fake_path = [[coord for coord in pt] for pt in future_path]
        valid_index = -1
        distances = []

        for idx, point in enumerate(future_path):
            path_point = Point(point)
            dist = nearest_cur_lane_seq_linestring.distance(path_point)
            distances.append(dist)
            if dist > 1.0:
                break
            if dist <= 0.4:
                valid_index = idx

        distances = distances[: max(valid_index + 1, 0)]
        if len(distances) > 0:
            mean_dist = sum(distances) / len(distances)
            max_dist = max(distances)
            if mean_dist > self.max_mean_pose_l or max_dist > self.max_pose_l:
                valid_index = -1

        fake_path = fake_path[: max(valid_index + 1, 0)]

        return fake_path

    def generate_supervise_info_and_viz_data(
        self,
        tag_data: Dict,
        tag_parser: TagParse,
        base_data_root: str,
        condition_data_root: str,
    ) -> Tuple[Dict, Tuple[str, List[Tuple[List[int], List[int]]], Dict]]:
        fake_info = {}
        tag_parser.read_data(
            base_data_root, condition_data_root, tag_data["file_path"]
        )
        auto_labeler_tag_data = tag_parser.tag_data
        lane_map = auto_labeler_tag_data.label_scene.percepmap.lane_map
        future_path = (
            auto_labeler_tag_data.label_scene.ego_path_info.future_path
        )
        current_lane_seqs = (
            auto_labeler_tag_data.label_scene.ego_obs_lane_seq_info.current_lane_seqs
        )

        if len(current_lane_seqs) == 0:
            return None, None

        # 在current_lane_seqs中找一条离future path最近的
        (
            nearest_cur_lane_seq,
            nearest_cur_lane_seq_linestring,
        ) = self.find_nearest_cur_lane_seq(
            lane_map, future_path, current_lane_seqs
        )

        if (
            nearest_cur_lane_seq is None
            or nearest_cur_lane_seq_linestring is None
        ):
            return None, None

        if (
            tag_data["path_type"] == "LANE_CHANGE"
            and self.is_fake_path_in_lane_change
        ):
            fake_path = self.generate_fake_path(
                future_path, nearest_cur_lane_seq_linestring
            )
            fake_info.update({"fake_path": fake_path})
            fake_info.update({"fake_condition_lane_seq": nearest_cur_lane_seq})

        if (
            "fake_path" in fake_info
            and len(fake_info["fake_path"]) < self.fake_path_min_length
        ):
            return None, None

        # 生成网络训练所需的数据格式
        supervise_info = self.get_supervise_info(
            tag_data, fake_info, auto_labeler_tag_data
        )

        # 用于可视化的数据
        tag_data_for_viz = copy.deepcopy(tag_data)
        tag_data_for_viz.update(fake_info)

        return tag_data_for_viz, supervise_info

    def get_supervise_info(
        self, tag_data: Dict, fake_info: Dict, auto_labeler_tag_data: TagData
    ) -> Tuple[str, List[Tuple[List[int], List[int]]], Dict]:
        is_junction_scene = False
        condition_lane_seq_pairs = []
        if "fake_condition_lane_seq" in fake_info:
            condition_lane_seq_pairs = [
                (fake_info["fake_condition_lane_seq"], None)
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

        path_info = {"is_enter_ramp": True}
        if "fake_path" in fake_info:
            path_info.update({"future_path": fake_info["fake_path"]})

        if is_junction_scene:
            path_info.update(
                {
                    "corr_junction_id": auto_labeler_tag_data.label_scene.obstacles[
                        -9
                    ].get(
                        "junction_info"
                    )[
                        "junction_id"
                    ]
                }
            )

        ret = (tag_data["file_path"], condition_lane_seq_pairs, path_info)
        return ret

    def save_training_data(
        self,
        save_root: str,
        enter_ramp_tags_for_viz: Dict,
        enter_ramp_file_list_for_viz: List[str],
        surpervise_info_list: List[
            Tuple[str, List[Tuple[List[int], List[int]]], Dict]
        ],
        task_idx: int,
    ) -> None:
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        with open(
            os.path.join(
                save_root, str(task_idx) + "enter_ramp_tags_for_viz.json"
            ),
            "w",
        ) as f:
            f.write(rapidjson.dumps(enter_ramp_tags_for_viz))

        with open(
            os.path.join(
                save_root, str(task_idx) + "enter_ramp_file_list_for_viz.json"
            ),
            "w",
        ) as f:
            f.write(rapidjson.dumps(enter_ramp_file_list_for_viz))

        with open(
            os.path.join(save_root, str(task_idx) + "enter_ramp_sample.pickle"),
            "wb",
        ) as f:
            pickle.dump(surpervise_info_list, f)

        print("enter_ramp_tags_for_viz len: ", len(enter_ramp_tags_for_viz))
        print(
            "enter_ramp_file_list_for_viz len: ",
            len(enter_ramp_file_list_for_viz),
        )
        print("surpervise_info_list len: ", len(surpervise_info_list))

    def collect_all_tag_file_paths(self, tag_data_root: str) -> List[str]:
        trip_num = self.trip_num
        tag_json_file_paths = []

        for root, dirs, files in os.walk(tag_data_root):
            for file in files:
                if file.endswith("tag.json"):
                    full_path = os.path.join(root, file)
                    tag_json_file_paths.append(full_path)

        random.shuffle(tag_json_file_paths)
        tag_json_file_paths = tag_json_file_paths[:trip_num]

        print("tag_json_file_paths len: ", len(tag_json_file_paths))

        return tag_json_file_paths

    def process_subset_files(
        self,
        file_paths: List[str],
        config_file: str,
        base_data_root: str,
        condition_data_root: str,
        save_root: str,
        task_idx=0,
    ) -> None:
        tag_parser = TagParse(config_file)
        enter_ramp_tags_for_viz = {}
        enter_ramp_file_list_for_viz = []
        surpervise_info_list = []

        for file_path in tqdm(file_paths):
            with open(file_path, "rb") as f:
                trip_tag_data = rapidjson.loads(f.read())
                for ts, tag_data in trip_tag_data.items():
                    if tag_data["ramp_tag"][
                        "is_enter_ramp"
                    ] and self.filter_invalid_sample(tag_data):
                        (
                            tag_data_for_viz,
                            supervise_info,
                        ) = self.generate_supervise_info_and_viz_data(
                            tag_data,
                            tag_parser,
                            base_data_root,
                            condition_data_root,
                        )
                        if tag_data_for_viz is None or supervise_info is None:
                            continue

                        if (
                            len(enter_ramp_file_list_for_viz)
                            < self.viz_sample_num
                        ):
                            enter_ramp_tags_for_viz[ts] = tag_data_for_viz
                            enter_ramp_file_list_for_viz.append(
                                tag_data["file_path"]
                            )
                        surpervise_info_list.append(supervise_info)

        self.save_training_data(
            save_root,
            enter_ramp_tags_for_viz,
            enter_ramp_file_list_for_viz,
            surpervise_info_list,
            task_idx,
        )

    def select_and_fake_ramp_samples(
        self, parsed_args: argparse.Namespace
    ) -> None:
        # 获取所有tag文件路径
        tag_json_file_paths = self.collect_all_tag_file_paths(
            parsed_args.tag_data_root
        )

        with Pool(self.num_process) as pool:
            for i in range(self.num_process):
                pool.apply_async(
                    LogExceptions(self.process_subset_files),
                    args=(
                        tag_json_file_paths[i :: self.num_process],
                        parsed_args.config_file,
                        parsed_args.base_data_root,
                        parsed_args.condition_data_root,
                        parsed_args.save_root,
                        i,
                    ),
                )

            pool.close()
            pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tdr",
        "--tag-data-root",
        type=str,
        required=True,
        help="folder path to tags data",
    )
    parser.add_argument(
        "-sr",
        "--save-root",
        type=str,
        required=True,
        help="folder path to save result",
    )
    parser.add_argument(
        "-bdr",
        "--base-data-root",
        type=str,
        default="s3://pnd/PnPBaseTrainData/base_label_v2.0.9/",
        help="folder path to base data",
    )
    parser.add_argument(
        "-cdr",
        "--condition-data-root",
        type=str,
        default="",
        help="folder path to condition data",
    )
    parser.add_argument(
        "-cfg",
        "--config-file",
        type=str,
        default="/mnt/train2/lenny.chen/Code/path-nn-tagger/cfg/config.risk.py",
        help="path to config file",
    )
    parser.add_argument(
        "-np",
        "--num-process",
        type=int,
        default=50,
        help="multi thread processer number",
    )
    parser.add_argument(
        "-tn",
        "--trip-num",
        type=int,
        default=500,
        help="only process trip-num trips data",
    )
    parser.add_argument(
        "-fp",
        "--is-fake-path-in-lane-change",
        type=bool,
        default=True,
        help="whether to generate fake path in lane change",
    )

    parsed_args = parser.parse_args()

    ramp_tag_selector = RampTagSelector(
        trip_num=parsed_args.trip_num,
        num_process=parsed_args.num_process,
        is_fake_path_in_lane_change=parsed_args.is_fake_path_in_lane_change,
    )

    ramp_tag_selector.select_and_fake_ramp_samples(parsed_args)
