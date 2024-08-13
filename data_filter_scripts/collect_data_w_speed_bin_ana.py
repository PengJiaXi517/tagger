import json
import os
from collections import defaultdict

from tqdm import tqdm

root_dir = "./filter_res_new_2"

if __name__ == "__main__":

    root_path = "./filter_res_new_2"

    all_log = defaultdict(lambda: 0)

    for i in range(50):

        with open(os.path.join(root_path, f"log_{i}.json"), "r") as f:
            log = json.load(f)

            for key, value in log.items():
                all_log[key] += value

    title = "Cruise, Lane Change, JUNCTION_UNKNOWN, JUNCTION_FORWARD, JUNCTION_LEFT, JUNCTION_RIGHT, JUNCTION_UTURN\n"
    all_num = f"{all_log['num_cruise']}, {all_log['num_lane_change']}, {all_log['valid_turn_unknown'] + all_log['invalid_turn_unknown']}, {all_log['valid_turn_forward'] + all_log['invalid_turn_forward']}, {all_log['valid_turn_left'] + all_log['invalid_turn_left']}, {all_log['valid_turn_right'] + all_log['invalid_turn_right']}, {all_log['valid_turn_uturn'] + all_log['invalid_turn_uturn']}\n"
    valid_num = f"{all_log['num_valid_cruise']}, {all_log['num_valid_lane_change']}, {all_log['valid_turn_unknown']}, {all_log['valid_turn_forward']}, {all_log['valid_turn_left']}, {all_log['valid_turn_right']}, {all_log['valid_turn_uturn']}\n"

    with open("ana.csv", "w") as f:
        f.write(title)
        f.write(all_num)
        f.write(valid_num)

    res_dict = defaultdict(list)

    up_sample = {"lc_res": 2, "RIGHT": 4, "UTURN": 5}
    down_sample = {
        "cruise_straight_res": 0.2,
        "cruise_turn_res": 0.2,
        "FORWARD": 0.2,
        "LEFT": 0.4,
    }

    speed_bin_num = 20
    bin_size = 1.0

    down_sample_speed_bin = defaultdict(
        lambda: {
            "cruise_straight_res": [[] for _ in range(speed_bin_num)],
            "cruise_turn_res": [[] for _ in range(speed_bin_num)],
            "lc_res": [[] for _ in range(speed_bin_num)],
            "FORWARD_IN": [[] for _ in range(speed_bin_num)],
            "LEFT_IN": [[] for _ in range(speed_bin_num)],
            "RIGHT_IN": [[] for _ in range(speed_bin_num)],
            "UTURN_IN": [[] for _ in range(speed_bin_num)],
            "FORWARD_OUT": [[] for _ in range(speed_bin_num)],
            "LEFT_OUT": [[] for _ in range(speed_bin_num)],
            "RIGHT_OUT": [[] for _ in range(speed_bin_num)],
            "UTURN_OUT": [[] for _ in range(speed_bin_num)],
        }
    )

    for i in tqdm(list(range(50))):
        with open(os.path.join(root_path, f"filter_res_{i}.json"), "r") as f:
            res = json.load(f)
        with open(os.path.join(root_path, f"calculate_res_{i}.json"), "r") as f:
            cal_res = json.load(f)

        for key, value in res.items():
            cal_value = cal_res[key]
            for res_key, file_path in value.items():
                # if res_key in up_sample:
                #     res_dict[key].extend(file_path * up_sample[res_key])
                # elif res_key in down_sample:
                # vel = cal_value[res_key]

                if res_key in down_sample_speed_bin[key]:
                    for f_p, vel in zip(file_path, cal_value[res_key]):
                        for i in range(speed_bin_num):
                            if vel < (i + 1) * bin_size or i == speed_bin_num - 1:
                                down_sample_speed_bin[key][res_key][i].append(f_p)
                                break
                    # import random

                    # random.shuffle(file_path)
                    # res_dict[key].extend(
                    #     file_path[: int(len(file_path) * down_sample[res_key])]
                    # )

    import matplotlib.pyplot as plt
    import numpy as np

    sample_bin_0_ratio = {
        "cruise_straight_res": 1.0,
        "cruise_turn_res": 1.0,
        "lc_res": 1.0,
        "FORWARD_IN": 1.0,
        "LEFT_IN": 1.0,
        "RIGHT_IN": 1.0,
        "UTURN_IN": 1.0,
        "FORWARD_OUT": 0.1,
        "LEFT_OUT": 0.1,
        "RIGHT_OUT": 0.25,
        "UTURN_OUT": 0.1,
    }

    sample_ratio = {
        "cruise_straight_res": 0.2,
        "cruise_turn_res": 3,
        "lc_res": 3,
        "FORWARD_IN": 0.3,
        "LEFT_IN": 2,
        "RIGHT_IN": 5,
        "UTURN_IN": 5,
        "FORWARD_OUT": 0.3,
        "LEFT_OUT": 1,
        "RIGHT_OUT": 5,
        "UTURN_OUT": 5,
    }

    type_count = defaultdict(lambda: 0)

    for dirname, sample_per_type in down_sample_speed_bin.items():
        dir_res = res_dict[dirname]
        for type, samples in sample_per_type.items():

            sample_0 = samples[0]
            import random

            random.shuffle(sample_0)
            sample_0 = sample_0[: int(len(sample_0) * sample_bin_0_ratio[type])]

            samples[0] = sample_0

            s_r = sample_ratio[type]

            if s_r >= 1:
                for s in samples:
                    dir_res.extend(s * int(s_r))
            else:
                tmp_sample = []
                for s in samples:
                    tmp_sample.extend(s)

                random.shuffle(tmp_sample)

                dir_res.extend(tmp_sample[: int(len(tmp_sample) * s_r)])

            bin_sizes = [len(l) for l in samples]  # 每个桶的大小
            bin_edges = list(range(len(bin_sizes) + 1))

            all_sample_num = np.sum(bin_sizes)

            type_count[type] += int(all_sample_num)

            plt.figure(figsize=(8, 4))

            plt.title(f"TotalNum: {all_sample_num}")

            # 计算每个桶的宽度
            bin_widths = [
                bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)
            ]

            # 计算每个桶的中心位置
            bin_centers = [
                bin_edges[i] + bin_widths[i] / 2 for i in range(len(bin_widths))
            ]

            # 绘制 bin 图
            plt.bar(
                bin_centers, bin_sizes, width=bin_widths, edgecolor="black", alpha=0.75
            )

            plt.savefig(
                os.path.join(
                    "./speed_bin_viz", f"{type}_{dirname.replace('/', '_')}.jpg"
                )
            )
            plt.close()

    with open("./type_count.json", "w") as f:
        json.dump(type_count, f)

    for key, value in res_dict.items():
        with open(
            os.path.join(
                "./",
                f"{key.replace('/', '_')}.json",
            ),
            "w",
        ) as f:
            print(len(value))
            json.dump(value, f)
