import argparse
import json
import os
import os.path as osp
import pickle
import random

import boto3
import numpy as np
from tqdm import tqdm

from base import ConditionRes, LabelScene, TagData
from registry import TAG_FUNCTIONS
from tag_functions.condition_risk import *
from tag_functions.demo import *
from tag_functions.ego_state import *
from tag_functions.future_path_tag import *
from tag_functions.interactive_tag import *
from tag_functions.map_risk import *
from tag_functions.path_risk import *
from tag_functions.high_value_tag import *
from utils.config import Config


class TagParse:
    def __init__(self, cfg_file):
        self.cfg = Config.fromfile(cfg_file)
        self.max_valid_point_num = self.cfg.get("max_valid_point_num", 34)
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=f"http://10.199.199.{random.randint(81, 88)}:8082/",
            aws_access_key_id="UI2WLBBFHV1CE0HZMZG2",
            aws_secret_access_key="sA2ozSrzC1lehJLbB4AuGCwiuLCHVLOcUbn16xiM",
            verify=False,
            use_ssl=False,
        )

    def read_data(self, base_data_root, condition_data_root, pickle_sub_path):
        base_pickle_path = osp.join(base_data_root, pickle_sub_path)
        condition_pickle_path = osp.join(
            condition_data_root, pickle_sub_path.replace(".pickle", "ego_path.pickle")
        )
        self.tag_data = TagData(
            base_pickle_path,
            condition_pickle_path,
            self.s3_client,
            self.max_valid_point_num,
        )

    def process(
        self, base_data_root, condition_data_root, save_data_root, pickle_sub_path
    ):
        self.read_data(base_data_root, condition_data_root, pickle_sub_path)
        self.tag_result = {
            "data_root": base_data_root,
            "file_path": pickle_sub_path,
        }
        for k, v in self.cfg.get("tag_pipelines", {}).items():
            sub_tag_res = TAG_FUNCTIONS.get(k)(self.tag_data, v)
            assert type(sub_tag_res) is dict
            assert len(sub_tag_res.keys() & self.tag_result.keys()) == 0
            self.tag_result.update(sub_tag_res)
        # self.save_data(save_data_root, pickle_sub_path)
        return self.tag_result

    def save_data(self, save_data_root, pickle_sub_path):
        os.makedirs(
            osp.join(save_data_root, "/".join(pickle_sub_path.split("/")[:-1])),
            exist_ok=True,
        )
        with open(
            os.path.join(save_data_root, pickle_sub_path.replace(".pickle", ".json")),
            "w",
        ) as f:
            json.dump(self.tag_result, f, indent=4)


if __name__ == "__main__":
    # base_data_root = "/mnt/train2/pnd_data/PnPEnd2EndTrainData/pnd_label_result/base_label"
    # condition_data_root = (
    #     "/mnt/train2/RoadPercep/eric.wang/tmp_data/0704_ego_path_parse/"
    # )
    # json_file = "/mnt/train2/pnd_data/PnPEnd2EndTrainData/pnd_label_result/TripPathData/output/0726_increase/bug_fix_output/train/train.json"
    # save_root = "/mnt/train2/RoadPercep/eric.wang/path_tag/test"
    # cfg_file = (
    #     "/mnt/train2/RoadPercep/eric.wang/Code/path-nn-tagger/cfg/config.risk.py"
    # )
    #
    # with open(json_file, "r") as f:
    #     json_list = json.load(f)
    #
    # tag_parse = TagParse(cfg_file)
    # tag_parse.process(base_data_root, condition_data_root, save_root, '52918_16164_Guangqi-003_jinshengye_2024-07-16-20-10-05/52918_16164_Guangqi-003_jinshengye_2024-07-16-20-10-05_10_19/labels/1721132125291139.pickle')
    # for json_line in tqdm(json_list):
    #     tag_parse.process(base_data_root, condition_data_root, save_root, json_line)


    base_data_root = "s3://pnd/PnPBaseTrainData/base_label_v2.0.4"
    condition_data_root = (
        "/mnt/train2/RoadPercep/eric.wang/tmp_data/0704_ego_path_parse/"
    )
    json_file = "/mnt/train2/pnd_data/PnPEnd2EndTrainData/pnd_label_result/TripPathData/output/0726_increase/bug_fix_output/train/train.json"
    save_root = "/mnt/train2/RoadPercep/eric.wang/path_tag/test"
    cfg_file = (
        "/mnt/train2/RoadPercep/eric.wang/Code/path-nn-tagger/cfg/config.risk.py"
    )

    tag_parse = TagParse(cfg_file)
    tag_parse.process(base_data_root, condition_data_root, save_root,
                      '50819_246_WeiLai-007_youzhao_2024-07-10-07-02-50/50819_246_WeiLai-007_youzhao_2024-07-10-07-02-50_60_69/labels/1720568065992249.pickle')