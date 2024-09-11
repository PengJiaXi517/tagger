import argparse
import json
import os

from scripts.utils.clip_db import ClipDB
from scripts.utils.search_valid_clips import (
    check_valid_nearby_frames,
    check_valid_nearby_frames_obj,
    search_valid_labeled_clip,
    search_valid_labeled_clip_obj,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lon",
        "--lon",
        type=float,
        required=True,
        help="lon",
    )
    parser.add_argument(
        "-lat",
        "--lat",
        type=float,
        required=True,
        help="lat",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        required=True,
        help="save-path",
    )
    parser.add_argument(
        "-pt",
        "--nn-path-tag",
        type=str,
        required=True,
        help="nn-path-tag",
    )
    parser.add_argument(
        "-lr",
        "--label-root-path",
        type=str,
        required=True,
        help="label-root-path",
    )
    parser.add_argument(
        "-csd",
        "--clip-search-distance",
        type=float,
        default=0.0001,
        help="clip-search-distance",
    )
    parser.add_argument(
        "-fvd",
        "--frame-valid-distance",
        type=float,
        default=0.0008,
        help="frame-valid-distance",
    )
    args = parser.parse_args()

    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)

    nn_path_tag_path = args.nn_path_tag
    label_root_path = args.label_root_path

    assert os.path.exists(save_path), f"Save Path - {save_path} not exits."

    assert os.path.exists(
        nn_path_tag_path
    ), f"NN Path tag - {nn_path_tag_path} not exits."

    if not label_root_path.startswith("s3://pnd/"):
        assert os.path.exists(
            label_root_path
        ), f"Label Root Path - {label_root_path} not exits."

    lon, lat, clip_dis = args.lon, args.lat, args.clip_search_distance

    frame_valid_distance = args.frame_valid_distance

    db = ClipDB()

    gps_clips = db.get_clips_by_gps(lon=lon, lat=lat, distance=clip_dis)

    with open(
        os.path.join(save_path, f"gps_clips.json"),
        "w",
    ) as f:
        json.dump(gps_clips, f, indent=2)

    db.close_pool()

    if label_root_path.startswith("s3://pnd/"):
        valid_labeled_clips = search_valid_labeled_clip_obj(gps_clips, label_root_path)
    else:
        valid_labeled_clips = search_valid_labeled_clip(gps_clips, label_root_path)

    valid_labeled_clips = list(set(valid_labeled_clips))

    with open(os.path.join(save_path, f"gps_labeled_valid_clips.json"), "w") as f:
        json.dump(valid_labeled_clips, f, indent=2)

    if label_root_path.startswith("s3://pnd/"):
        valid_labeld_frames = check_valid_nearby_frames_obj(
            lon,
            lat,
            frame_valid_distance,
            valid_labeled_clips,
            nn_path_tag_path,
            label_root_path,
        )
    else:
        valid_labeld_frames = check_valid_nearby_frames(
            lon,
            lat,
            frame_valid_distance,
            valid_labeled_clips,
            nn_path_tag_path,
            label_root_path,
        )

    with open(os.path.join(save_path, f"valid_labeld_frames.json"), "w") as f:
        json.dump(valid_labeld_frames, f, indent=2)
