import argparse
import io
import json
import os
import pickle
import random
from multiprocessing import Pool
from typing import List

import boto3
import cv2
import numpy as np
from tqdm import tqdm

from utils.opencv_viewer import View
from utils.viz_utils.map import draw_map
from utils.viz_utils.obstacles import draw_ego_path, draw_obstacle
from utils.viz_utils.tag import (
    draw_tag_lines,
    get_basic_path_tag_lines,
    get_cruise_tag_lines,
    get_junction_tag_lines,
    get_lc_tag_lines,
)

s3_client = boto3.client(
    "s3",
    endpoint_url=f"http://10.199.199.{random.randint(81, 96)}:8082/",
    aws_access_key_id="UI2WLBBFHV1CE0HZMZG2",
    aws_secret_access_key="sA2ozSrzC1lehJLbB4AuGCwiuLCHVLOcUbn16xiM",
    verify=False,
    use_ssl=False,
)


def process_file(
    file_list: List[str],
    save_dir="/mnt/train2/pnd_data/PersonalData/Ness.hu/TagVizTest",
    tag_path="/mnt/train2/eric.wang/path_tag_res/0801/",
    root_dir="/mnt/train2/pnd_data/PnPEnd2EndTrainData/pnd_label_result/base_label",
    rank=0,
):
    cache_tag = {}
    for file in tqdm(
        file_list,
        nrows=4,
        position=rank,
        leave=False,
    ):
        pickle_file = file
        json_file = file.replace(".pickle", ".json")
        tag_json = os.path.join(tag_path, json_file.split("/")[0], "tag.json")
        if tag_json not in cache_tag:
            with open(
                os.path.join(tag_path, json_file.split("/")[0], "tag.json"), "r"
            ) as f:
                tags = json.load(f)
            cache_tag[tag_json] = tags
        tags = cache_tag[tag_json]
        cache_tag = {}
        # cache_tag = {}  # Tmp remove
        ts_us = pickle_file.split("/")[-1].split(".pickle")[0]
        if ts_us not in tags:
            continue
        tag = tags[ts_us]
        # with open(os.path.join(tag_path, json_file), "r") as f:
        #     tag = json.load(f)
        label_path = os.path.join(root_dir, pickle_file)
        if "s3://" not in label_path:
            if not os.path.exists(label_path):
                continue
            with open(label_path, "rb") as f:
                label = pickle.load(f)
        else:
            try:
                file_obj = io.BytesIO()
                s3_client.download_fileobj(
                    "pnd", label_path.split("s3://pnd/")[1], file_obj
                )
                file_obj.seek(0)
                label = pickle.load(file_obj)
                file_obj.close()
            except Exception:
                continue
        # with open(os.path.join(root_dir, pickle_file), "rb") as f:
        #     label = pickle.load(f)

        ego_path_info = label["ego_path_info"]

        ego_obstacle = label["obstacles"][-9]

        hist_state = ego_obstacle["features"]["history_states"][-1]
        curr_ego_x = hist_state["x"]
        curr_ego_y = hist_state["y"]
        curr_ego_theta = hist_state["theta"]

        condition_res_tag = tag["condition_res_tag"]
        start_lane_seq_ids = condition_res_tag["start_lane_seq_ids"]
        end_lane_seq_ids = condition_res_tag["end_lane_seq_ids"]

        ego_car_view = View(
            1500,
            3000,
            (100, 200),
            (
                curr_ego_x + 50 * np.cos(curr_ego_theta),
                curr_ego_y + 50 * np.sin(curr_ego_theta),
            ),
            np.pi / 2 - curr_ego_theta,
        )

        frame = ego_car_view.build_frame()

        draw_map(
            label["percepmap"],
            ego_car_view,
            frame,
            start_lane_seq_ids,
            end_lane_seq_ids,
        )

        draw_obstacle(label["obstacles"], ego_car_view, frame)

        draw_ego_path(ego_path_info, ego_car_view, frame, max_length=100)

        bag, clip, _, ts_pickle = pickle_file.split("/")
        lines = [
            bag,
            clip,
            ts_pickle,
            tag["path_type"],
        ]
        colors = [
            (0, 0, 255),
            (0, 0, 255),
            (0, 0, 255),
            (0, 255, 255),
        ]

        for i, (start_lane_seq, end_lane_seq) in enumerate(
            zip(start_lane_seq_ids, end_lane_seq_ids)
        ):
            lines.append(f"{i}: {start_lane_seq}, {end_lane_seq}")
            colors.append((0, 255, 255))

        condition_risk_tag = tag["condition_risk"]

        lines.extend(
            [
                f"Cond no entry b not in junction: {condition_risk_tag['cond_no_entry_b_not_in_junction']}",
                f"Cond Entry a Exit b Not Pass junction: {condition_risk_tag['cond_entry_and_exit_b_not_pass_junction']}",
            ]
        )
        colors.extend([(0, 0, 255), (0, 0, 255)])

        if "basic_path_tag" in tag and tag["basic_path_tag"] is not None:
            basic_path_lines, basic_path_colors = get_basic_path_tag_lines(
                tag["basic_path_tag"]
            )
            lines.extend(basic_path_lines)
            colors.extend(basic_path_colors)

        if tag["lc_path_tag"] is not None:
            lc_lines, lc_colors = get_lc_tag_lines(tag["lc_path_tag"])
            lines.extend(lc_lines)
            colors.extend(lc_colors)

        if tag["junction_path_tag"] is not None:
            junction_lines, junction_colors = get_junction_tag_lines(
                tag["junction_path_tag"]
            )
            lines.extend(junction_lines)
            colors.extend(junction_colors)

        if tag["cruise_path_tag"] is not None:
            cruise_lines, cruise_colors = get_cruise_tag_lines(tag["cruise_path_tag"])
            lines.extend(cruise_lines)
            colors.extend(cruise_colors)

        draw_tag_lines(
            frame,
            lines=lines,
            colors=colors,
        )

        file_name = file.replace("/", "_").strip(".pickle")
        save_sub_dir = file[: -(len(file.split("/")[-1]) + 7)]

        os.makedirs(os.path.join(save_dir, save_sub_dir), exist_ok=True)

        cv2.imwrite(
            os.path.join(save_dir, save_sub_dir, f"{file_name}.jpg"),
            frame,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fl",
        "--frame-list",
        type=str,
        required=True,
        help="frame list json",
    )
    parser.add_argument(
        "-lr",
        "--label-root-path",
        type=str,
        required=True,
        help="label-root-path",
    )
    parser.add_argument(
        "-tr",
        "--tag-root-path",
        type=str,
        required=True,
        help="tag-root-path",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        required=True,
        help="save-path",
    )
    parser.add_argument(
        "-n",
        "--num-process",
        type=int,
        default=6,
        help="num process",
    )
    args = parser.parse_args()

    num_process = 50

    json_file = args.frame_list
    save_path = args.save_path
    tag_root_path = args.tag_root_path
    label_root_path = args.label_root_path

    os.makedirs(save_path, exist_ok=True)

    with open(json_file, "r") as f:
        file_list: List[str] = json.load(f)

    # file_list = file_list[::100]
    # print(len(file_list))
    # process_file(file_list, save_path, tag_root_path, label_root_path, 0)
    # exit(0)

    # process_file(file_list, "/mnt/train2/pnd_data/PersonalData/Ness.hu/WaiQieDataCheck")
    # file_list = file_list[:20000]
    # process_file(file_list, save_path, tag_root_path, label_root_path)

    # file_list = [
    #     "63413_16760_WeiLai-011_hongbinwu_2024-08-23-06-49-21/63413_16760_WeiLai-011_hongbinwu_2024-08-23-06-49-21_150_159/labels/1724371483890853.pickle",
    # ]
    # process_file(file_list, save_path, tag_root_path, label_root_path)
    # exit()

    with Pool(num_process) as pool:
        for i in range(num_process):
            pool.apply_async(
                process_file,
                args=(
                    file_list[i::num_process],
                    save_path,
                    tag_root_path,
                    label_root_path,
                    i,
                ),
            )

        pool.close()
        pool.join()
