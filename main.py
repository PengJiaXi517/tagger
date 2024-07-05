import argparse
import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

from base import ConditionRes

ROOT_PATH = "/mnt/train2/PnP/road_percep_0605/"
CONDITION_RES = "/mnt/train2/RoadPercep/eric.wang/tmp_data/0704_ego_path_parse/"

if __name__ == "__main__":

    trips = os.listdir(ROOT_PATH)

    for trip in tqdm(trips, nrows=4, position=0, leave=False):
        clips = os.listdir(osp.join(ROOT_PATH, trip))

        for clip in tqdm(clips, desc=trip, nrows=4, position=1, leave=False):
            if osp.exists(osp.join(ROOT_PATH, trip, clip, "labels")):
                pickles = [
                    p
                    for p in os.listdir(osp.join(ROOT_PATH, trip, clip, "labels"))
                    if p.endswith(".pickle")
                ]

                for pickle_name in tqdm(
                    pickles, desc=clip, nrows=4, position=2, leave=False
                ):

                    label_pickle_path = osp.join(
                        ROOT_PATH, trip, clip, "labels", pickle_name
                    )
                    condition_pickle_path = osp.join(
                        CONDITION_RES,
                        trip,
                        clip,
                        "labels",
                        f"{pickle_name.strip('.pickle')}ego_path.pickle",
                    )

                    if not osp.exists(condition_pickle_path):
                        continue

                    condition_res = ConditionRes(condition_pickle_path)

                    print(condition_res)
                    exit(0)
