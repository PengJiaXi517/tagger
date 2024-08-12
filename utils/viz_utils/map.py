from typing import Dict, List

import numpy as np

from utils.opencv_viewer import View


def draw_map(
    percep_map: Dict,
    view: View,
    frame: np.ndarray,
    start_lane_seq_ids: List[List[int]] = [],
    end_lane_seq_ids: List[List[int]] = [],
):

    for junction in percep_map["junctions"]:
        polygon = np.array(junction["polygon"])

        view.draw_polyline(
            frame, polygon[:, 0], polygon[:, 1], color=(0, 255, 255), thickness=3
        )

    for curb in percep_map["curbs"]:
        if len(curb) > 0:
            curb = np.array(curb)
            view.draw_polyline(
                frame, curb[:, 0], curb[:, 1], color=(0, 255, 255), thickness=2
            )

    start_lane_ids_set = []
    for lane_ids in start_lane_seq_ids:
        start_lane_ids_set.extend(lane_ids)
    start_lane_ids_set = set(start_lane_ids_set)

    end_lane_seq_ids_set = []
    for lane_ids in end_lane_seq_ids:
        end_lane_seq_ids_set.extend(lane_ids)
    end_lane_seq_ids_set = set(end_lane_seq_ids_set)

    for lane in percep_map["lanes"]:
        polyline = np.array(lane["polyline"])

        polyline_color = (122, 122, 122)
        polyline_thickness = 1
        if lane["id"] in start_lane_ids_set:
            polyline_color = (0, 0, 255)
            polyline_thickness = 2
        elif lane["id"] in end_lane_seq_ids_set:
            polyline_color = (255, 0, 0)
            polyline_thickness = 2

        view.draw_polyline(
            frame,
            polyline[:, 0],
            polyline[:, 1],
            color=polyline_color,
            thickness=polyline_thickness,
        )

        left_boundary = np.array(lane["left_boundary"]["polyline"])
        if len(left_boundary) > 0 and len(left_boundary.shape) == 2:
            left_boundary = left_boundary[~np.isnan(left_boundary).any(axis=1)]
            if len(left_boundary) > 0:
                view.draw_polyline(
                    frame,
                    left_boundary[:, 0],
                    left_boundary[:, 1],
                    color=(255, 255, 255),
                    thickness=2,
                )

        right_boundary = np.array(lane["right_boundary"]["polyline"])
        if len(right_boundary) > 0 and len(right_boundary.shape) == 2:
            right_boundary = right_boundary[~np.isnan(right_boundary).any(axis=1)]
            if len(right_boundary) > 0:
                view.draw_polyline(
                    frame,
                    right_boundary[:, 0],
                    right_boundary[:, 1],
                    color=(255, 255, 255),
                    thickness=2,
                )
