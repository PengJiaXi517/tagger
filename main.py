import argparse
import json
import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm

from utils.config import Config
from tag_functions.demo import *
from tag_functions.path_risk import *
from tag_functions.map_risk import *
from tag_functions.ego_state import *
from base import ConditionRes, LabelScene, TagData
from registry import TAG_FUNCTIONS


class TagParse:
    def __init__(self, cfg_file):
        self.cfg = Config.fromfile(cfg_file)

    def read_data(self, base_data_root, condition_data_root, pickle_sub_path):
        base_pickle_path = osp.join(base_data_root, pickle_sub_path)
        condition_pickle_path = osp.join(
            condition_data_root, pickle_sub_path.replace(".pickle", "ego_path.pickle")
        )
        self.tag_data = TagData(base_pickle_path, condition_pickle_path)

    def process(
        self, base_data_root, condition_data_root, save_data_root, pickle_sub_path
    ):
        self.read_data(base_data_root, condition_data_root, pickle_sub_path)
        self.tag_result = {}
        for k, v in self.cfg.get("tag_pipelines", {}).items():
            sub_tag_res = TAG_FUNCTIONS.get(k)(self.tag_data, v, self.tag_result)
            assert type(sub_tag_res) is dict
            assert len(sub_tag_res.keys() & self.tag_result.keys()) == 0
            self.tag_result.update(sub_tag_res)
        self.save_data(save_data_root, pickle_sub_path)

    def save_data(self, save_data_root, pickle_sub_path):
        os.makedirs(
            osp.join(save_data_root, "/".join(pickle_sub_path.split("/")[:-1])),
            exist_ok=True,
        )
        with open(os.path.join(save_data_root, pickle_sub_path), "wb") as f:
            pickle.dump(self.tag_result, f)


if __name__ == "__main__":
    base_data_root = "/mnt/train2/pnd_data/PnPBaseTrainDataTmp/road_percep_demoroad_overfit"
    condition_data_root = (
        "/mnt/train2/RoadPercep/eric.wang/tmp_data/0704_ego_path_parse/"
    )
    json_file = "/mnt/train2/RoadPercep/eric.wang/path_tag_logs/0711/part_0.json"
    save_root = "/mnt/train2/RoadPercep/eric.wang/path_tag/test"
    cfg_file = (
        "/mnt/train2/RoadPercep/eric.wang/Code/path-nn-tagger/cfg/config.debug.py"
    )

    with open(json_file, "r") as f:
        json_list = json.load(f)

    tag_parse = TagParse(cfg_file)
    for json_line in tqdm(json_list):
        tag_parse.process(base_data_root, condition_data_root, save_root, json_line)
