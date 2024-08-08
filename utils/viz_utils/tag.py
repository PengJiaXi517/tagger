from typing import Dict, List, Tuple

import cv2
import numpy as np


def draw_tag_lines(
    frame: np.ndarray, lines: List[str], colors: List[Tuple[float, float]]
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (255, 255, 255)  # 白色
    thickness = 1
    lineType = cv2.LINE_AA
    lineHeight = 21  # 每行的高度（可根据字体大小调整）

    x, y = (50, 10 + lineHeight)
    for line, color in zip(lines, colors):
        cv2.putText(frame, line, (x, y), font, fontScale, color, thickness, lineType)
        y += lineHeight


def get_basic_path_tag_lines(
    tag: Dict,
) -> Tuple[List[str], List[Tuple[float, float, float]]]:

    lines = [
        f"valid_path_len: {tag['valid_path_len']}",
        f"sum_path_curvature: {tag['sum_path_curvature']}",
        f"abs_sum_path_curvature: {tag['abs_sum_path_curvature']}",
        f"pos_sum_path_curvature: {tag['pos_sum_path_curvature']}",
        f"neg_sum_path_curvature: {tag['neg_sum_path_curvature']}",
    ]
    colors = [(255, 255, 255) for _ in range(5)]

    return lines, colors


def get_cruise_tag_lines(
    tags: List[Dict],
) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    lines = []
    colors = []
    for tag in tags:
        lines_ = [
            f"Real pose_l: {tag['real_pose_l']: .2f}",
            f"Percep pose_l: {tag['percep_pose_l']: .2f}",
            f"Mean pose_l: {tag['mean_pose_l']: .2f}",
            f"Continuous length: {tag['max_continuous_length_on_lane']: .2f}",
            f"Target Lane Sequence: {tag['labeled_lane_seq']}",
        ]
        colors_ = [(255, 255, 255) for _ in range(5)]

        lines.extend(lines_)
        colors.extend(colors_)

    return lines, colors


def get_lc_tag_lines(
    tags: List[Dict],
) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    lines = []
    colors = []
    for tag in tags:
        lines_ = [
            f"Start_pose_l: {tag['start_pose_l']: .2f}",
            f"Arrive final pose_l: {tag['arrive_final_pose_l']: .2f}",
            f"Arrive percep pose_l: {tag['arrive_percep_pose_l']: .2f}",
            f"Arrive length: {tag['arrive_length']: .2f}",
            f"Target Lane Sequence: {tag['labeled_lane_seq']}",
        ]
        colors_ = [(255, 255, 255) for _ in range(5)]

        lines.extend(lines_)
        colors.extend(colors_)

    return lines, colors


def get_junction_tag_lines(
    tag: Dict,
) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    if tag["has_junction_label"]:
        lines = [
            f"Turn type: {tag['turn_type']}",
            f"Has real arrive exit lane: {tag['has_real_arrive_exit_lane']}",
            f"Real lane exit pose_l: {tag['real_lane_exit_pose_l']: .2f}",
            f"Percep lane exit pose_l: {tag['percep_lane_exit_pose_l']: .2f}",
            f"Max length in exit lane: {tag['max_length_in_exit_lane']: .2f}",
            f"Max length not in exit lane: {tag['max_length_not_in_exit_lane']: .2f}",
            f"Max pose l 2 exit lane: {tag['max_pose_l_2_exit_lane']: .2f}",
            f"Mean pose l 2 exit lane: {tag['mean_pose_l_2_exit_lane']: .2f}",
        ]
        colors = [(255, 177, 177) for _ in range(8)]

        return lines, colors

    return [], []
