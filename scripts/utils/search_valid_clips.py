import json
import os
from typing import List

import numpy as np
from tqdm import tqdm


def search_valid_labeled_clip(gps_clips, label_root_path) -> List[str]:
    clips = []

    for clip in gps_clips:

        clip_name = str(clip["clip_name"])
        clips.append(clip_name)

    task_dir = []

    for clip in tqdm(clips, desc="Search valid labeled clips"):
        clip_num = clip.split("_")[-1]
        task_name = clip[: -(len(clip_num) + 1)]
        clip_num = int(clip_num)

        if os.path.exists(os.path.join(label_root_path, task_name)):
            all_valid_clips = os.listdir(os.path.join(label_root_path, task_name))

            for valid_clip in all_valid_clips:

                start_clip = int(valid_clip.split("_")[-2])
                end_clip = int(valid_clip.split("_")[-1])

                if start_clip <= clip_num <= end_clip:
                    task_dir.append(f"{task_name}/{valid_clip}")

    return task_dir


def check_valid_nearby_frames(
    lon, lat, dis_thr, clip_paths, nn_path_tag_path, label_root_path
) -> List[str]:

    valid_path_test = []

    for clip_path in tqdm(clip_paths):

        if not os.path.exists(
            os.path.join(label_root_path, clip_path, "labels/frame_tags.json")
        ):
            continue

        if not os.path.exists(os.path.join(nn_path_tag_path, clip_path, "labels")):
            continue

        with open(
            os.path.join(label_root_path, clip_path, "labels/frame_tags.json"), "r"
        ) as f:
            frame_tags = json.load(f)

        valid_paths = [
            int(t.strip(".json"))
            for t in os.listdir(os.path.join(nn_path_tag_path, clip_path, "labels"))
        ]

        for key, value in frame_tags.items():
            lla_values = value["frame_info"]["ego_lla"]
            ego_lon, ego_lat = lla_values[0], lla_values[1]

            distance = np.linalg.norm([lon - ego_lon, lat - ego_lat])
            # distance = 0.0
            if distance < dis_thr and int(key) in valid_paths:
                valid_path_test.append(
                    os.path.join(clip_path, "labels", f"{key}.pickle")
                )

    return valid_path_test
