from typing import Dict, List, Tuple
from shapely.geometry import LineString, Point, Polygon
import numpy as np


class RampJudge:
    def __init__(self):
        self.consider_len = 100

    def judge_enter_ramp(
        self, lane_map, cur_lane_id, adjacent_lane_id, ego_point
    ):
        # 滤除对向车道
        if self.is_opposite_lane(lane_map, cur_lane_id, adjacent_lane_id):
            return False

        cur_lane = lane_map[cur_lane_id]["polyline"]
        adjacent_lane = lane_map[adjacent_lane_id]["polyline"]

        # 在lane上找最近点的索引
        cur_idx = self.get_nearest_waypoint_idx(cur_lane, ego_point)
        adjacent_idx = self.get_nearest_waypoint_idx(adjacent_lane, ego_point)
        if cur_idx == -1 or adjacent_idx == -1:
            return False

        # 截取自车前方consider_len个点
        cur_lane = self.stitch_lane_forward(cur_lane, cur_lane_id, cur_idx, lane_map)
        adjacent_lane = self.stitch_lane_forward(
            adjacent_lane, adjacent_lane_id, adjacent_idx, lane_map
        )

        if self.large_diff_lane(cur_lane, adjacent_lane, False):
            return True

        return False

    def judge_exit_ramp(
        self, lane_map, cur_lane_id, adjacent_lane_id, ego_point
    ):
        # 滤除对向车道
        if self.is_opposite_lane(lane_map, cur_lane_id, adjacent_lane_id):
            return False

        cur_lane = lane_map[cur_lane_id]["polyline"]
        adjacent_lane = lane_map[adjacent_lane_id]["polyline"]

        # 在lane上找最近点的索引
        cur_idx = self.get_nearest_waypoint_idx(cur_lane, ego_point)
        if cur_idx == -1 or cur_idx < len(cur_lane) - self.consider_len:
            return False

        # 无分叉情况，需要对齐两条lane的终点
        if Point(cur_lane[-1]).distance(Point(adjacent_lane[-1])) > 1.0:
            adjacent_idx = self.get_nearest_waypoint_idx(
                adjacent_lane, Point(cur_lane[-1])
            )
            if adjacent_idx == -1:
                return False
            adjacent_lane = adjacent_lane[: adjacent_idx + 1]

        # 截取汇入点前的consider_len个点
        cur_lane = self.stitch_lane_backward(cur_lane, cur_lane_id, lane_map)
        adjacent_lane = self.stitch_lane_backward(adjacent_lane, adjacent_lane_id, lane_map)

        if self.large_diff_lane(cur_lane, adjacent_lane, True):
            return True

        return False

    def pre_check(self, lane_map, current_lanes, current_lane_seqs):
        if len(current_lanes) < 1 or len(current_lane_seqs) < 1:
            return False

        for lane_seq in current_lane_seqs:
            for lane_id in lane_seq:
                if lane_map[lane_id]["turn"] != "NOTURN" or (
                    lane_map[lane_id]["lane_category"] != "REALITY"
                ):
                    return False
        return True

    def is_opposite_lane(self, lane_map, cur_lane_id, adjacent_lane_id):
        v1 = lane_map[cur_lane_id]["unit_directions"][0]
        v2 = lane_map[adjacent_lane_id]["unit_directions"][0]
        if v1[0] * v2[0] + v1[1] * v2[1] < 0:
            return True
        return False

    def stitch_lane_forward(self, lane_polyline, lane_id, idx, lane_map):
        if idx < len(lane_polyline) - self.consider_len:
            lane_polyline = lane_polyline[idx : idx + self.consider_len]
        else:
            lane_polyline = lane_polyline[idx:]
            if len(lane_map[lane_id]["successor_id"]) > 0:
                cur_succ_id = lane_map[lane_id]["successor_id"][0]
                if lane_map[cur_succ_id]["lane_category"] == "REALITY":
                    concate_len = self.consider_len - len(lane_polyline)
                    lane_polyline = (
                        lane_polyline
                        + lane_map[cur_succ_id]["polyline"][:concate_len]
                    )
        return lane_polyline

    def stitch_lane_backward(self, lane_polyline, lane_id, lane_map):
        if len(lane_polyline) > self.consider_len:
            lane_polyline = lane_polyline[-self.consider_len :]
        elif len(lane_map[lane_id]["predecessor_id"]) > 0:
            cur_pred_id = lane_map[lane_id]["predecessor_id"][0]
            if lane_map[cur_pred_id]["lane_category"] == "REALITY":
                concate_len = self.consider_len - len(lane_polyline)
                lane_polyline = (
                    lane_map[cur_pred_id]["polyline"][-concate_len:]
                    + lane_polyline
                )
        return lane_polyline

    def large_diff_lane(self, cur_lane, adjacent_lane, is_reverse_order):
        min_length = min(len(cur_lane), len(adjacent_lane))
        if min_length < 3:
            return False

        if is_reverse_order:
            cur_lane = cur_lane[-min_length:]
            adjacent_lane = adjacent_lane[-min_length:]
        else:
            cur_lane = cur_lane[:min_length]
            adjacent_lane = adjacent_lane[:min_length]

        large_dist_num = 0
        adjacent_linestring = LineString(adjacent_lane)
        for idx, point in enumerate(cur_lane):
            if idx % 2 or adjacent_linestring.distance(Point(point)) < 10:
                continue
            large_dist_num += 1
            if large_dist_num > 3:
                return True

    def get_nearest_lane_id(
        self, lane_map, lane_seqs, target_point, current_lane_seqs
    ):
        min_dist = 100000
        nearest_lane_id = None
        for lane_seq in lane_seqs:
            for lane_id in lane_seq:
                if any(lane_id in obj for obj in current_lane_seqs):
                    continue
                nearest_polyline = LineString(lane_map[lane_id]["polyline"])
                dist = target_point.distance(nearest_polyline)
                if dist < min_dist and dist < 5 and dist > 0:
                    min_dist = dist
                    nearest_lane_id = lane_seq[0]
        return nearest_lane_id

    def get_nearest_waypoint_idx(self, lane_polyline, ego_point):
        lane_array = np.array(lane_polyline)
        distances = np.sqrt(
            np.sum((lane_array - [ego_point.x, ego_point.y]) ** 2, axis=1)
        )
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] > 5:
            nearest_idx = -1

        return nearest_idx
