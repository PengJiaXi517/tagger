from typing import Dict, List, Tuple
from shapely.geometry import LineString, Point, Polygon
import numpy as np


class RampJudge:
    def __init__(self):
        self.enter_fork_consider_len = 20  # 考虑的index数量
        self.exit_fork_consider_len = 50
        self.large_dist_th = 10
        self.large_dist_num_th = 3
        self.curb_roi_s_min = -10
        self.curb_roi_s_max = 100
        self.curb_roi_l_max = 10

    def enter_ramp_cruise(self, lane_map, current_lanes, curb_decision):
        fork_lane_ids = []
        cur_lane_id = current_lanes[0]
        succ_ids = lane_map[cur_lane_id]["successor_id"]
        # 判断lane是否为 一分成二
        if len(succ_ids) == 2:
            fork_lane_ids = succ_ids
        else:
            pred_ids = lane_map[cur_lane_id]["predecessor_id"]
            if len(pred_ids) == 1:
                succ_ids = lane_map[pred_ids[0]]["successor_id"]
                if len(succ_ids) == 2 and cur_lane_id in succ_ids:
                    fork_lane_ids = succ_ids

        if len(fork_lane_ids) == 2:
            if self.judge_enter_fork(
                lane_map, fork_lane_ids[0], fork_lane_ids[1], curb_decision
            ):
                return True

        return False

    def enter_ramp_lane_change(
        self,
        lane_map,
        current_lanes,
        curb_decision,
        current_lane_seqs,
        ego_path_info,
    ):
        corr_lane_id = ego_path_info.corr_lane_id
        future_path = ego_path_info.future_path

        # 滤掉过路口场景
        if any(obj is not None for obj in ego_path_info.in_junction_id):
            return False

        # 判断是否为变道场景
        target_lane_id = -1
        lc_idx = -1
        for idx, lane_info in enumerate(corr_lane_id):
            if lane_info is None or len(lane_info) == 0:
                continue
            lane_id = lane_info[0][0]
            if any(lane_id in seq for seq in current_lane_seqs):
                continue
            target_lane_id = lane_id
            lc_idx = idx - 1
            break

        if target_lane_id != -1 and lc_idx != -1:
            lc_point = Point(future_path[lc_idx])
            if self.judge_enter_fork(
                lane_map,
                current_lanes[0],
                target_lane_id,
                curb_decision,
                lc_point,
            ):
                return True

        return False

    def exit_ramp(self, lane_map, current_lanes, ego_point):
        cur_lane_id = current_lanes[0]
        succ_ids = lane_map[cur_lane_id]["successor_id"]
        # 判断lane是否为 二合成一
        if len(succ_ids) == 1:
            pred_ids = lane_map[succ_ids[0]]["predecessor_id"]
            if len(pred_ids) == 2 and cur_lane_id in pred_ids:
                adjacent_lane_id = (
                    pred_ids[0] if pred_ids[0] != cur_lane_id else pred_ids[1]
                )
                if self.judge_exit_fork(
                    lane_map, cur_lane_id, adjacent_lane_id, ego_point
                ):
                    return True

        return False

    def judge_enter_fork(
        self,
        lane_map,
        cur_lane_id,
        nearby_lane_id,
        curb_decision,
        lc_point=None,
    ):
        # 滤除对向车道
        if self.is_opposite_lane(lane_map, cur_lane_id, nearby_lane_id):
            return False
        # 滤除虚拟车道
        if self.is_virtual_lane(lane_map, cur_lane_id, nearby_lane_id):
            return False

        cur_lane = lane_map[cur_lane_id]["polyline"]
        nearby_lane = lane_map[nearby_lane_id]["polyline"]

        # 截取前方一定距离
        (
            cutoff_cur_lane,
            cutoff_nearby_lane,
        ) = self.cut_off_lane_forward(cur_lane, nearby_lane, lc_point)

        if len(cutoff_cur_lane) < 3 or len(cutoff_nearby_lane) < 3:
            return False

        # 检查两条分叉路之间是否有curb隔离
        return self.is_separate_by_curb(
            cutoff_cur_lane, cutoff_nearby_lane, curb_decision
        )

    def judge_exit_fork(
        self, lane_map, cur_lane_id, adjacent_lane_id, ego_point
    ):
        # 滤除对向车道
        if self.is_opposite_lane(lane_map, cur_lane_id, adjacent_lane_id):
            return False

        cur_lane = lane_map[cur_lane_id]["polyline"]
        adjacent_lane = lane_map[adjacent_lane_id]["polyline"]

        # 在cur lane上找最近点的索引
        cur_idx = self.get_nearest_waypoint_idx(cur_lane, ego_point)
        if (
            cur_idx == -1
            or cur_idx < len(cur_lane) - self.exit_fork_consider_len
        ):
            return False

        # 截取汇入点前的exit_fork_consider_len个点
        stitched_cur_lane = self.stitch_lane_backward(
            cur_lane, cur_lane_id, lane_map
        )
        stitched_adjacent_lane = self.stitch_lane_backward(
            adjacent_lane, adjacent_lane_id, lane_map
        )
        # 判断两条lane是否分叉大于一定距离
        return self.large_diff_lane(stitched_cur_lane, stitched_adjacent_lane)

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

    def is_virtual_lane(self, lane_map, cur_lane_id, nearby_lane_id):
        return (
            lane_map[cur_lane_id]["lane_category"] != "REALITY"
            or lane_map[nearby_lane_id]["lane_category"] != "REALITY"
        )

    def cut_off_lane_forward(self, cur_lane, nearby_lane, ego_point):
        cur_idx = 0
        nearby_idx = 0

        if ego_point is not None:
            cur_idx = self.get_nearest_waypoint_idx(cur_lane, ego_point)
            nearby_idx = self.get_nearest_waypoint_idx(nearby_lane, ego_point)
            if cur_idx == -1 or nearby_idx == -1:
                return [], []

        cutoff_cur_lane = cur_lane[
            cur_idx : cur_idx + self.enter_fork_consider_len
        ]
        cutoff_nearby_polyline = list(
            reversed(
                nearby_lane[
                    nearby_idx : nearby_idx + self.enter_fork_consider_len
                ]
            )
        )

        return cutoff_cur_lane, cutoff_nearby_polyline

    def stitch_lane_backward(self, lane_polyline, lane_id, lane_map):
        stitched_lane_polyline = []
        if len(lane_polyline) > self.exit_fork_consider_len:
            stitched_lane_polyline = lane_polyline[
                -self.exit_fork_consider_len :
            ]
        elif len(lane_map[lane_id]["predecessor_id"]) > 0:
            cur_pred_id = lane_map[lane_id]["predecessor_id"][0]
            if lane_map[cur_pred_id]["lane_category"] == "REALITY":
                concate_len = self.exit_fork_consider_len - len(lane_polyline)
                stitched_lane_polyline = (
                    lane_map[cur_pred_id]["polyline"][-concate_len:]
                    + lane_polyline
                )
        return stitched_lane_polyline

    def large_diff_lane(self, cur_lane, adjacent_lane):
        min_length = min(len(cur_lane), len(adjacent_lane))
        if min_length < 3:
            return False

        cur_lane_seg = cur_lane[-min_length:]
        adjacent_lane_seg = adjacent_lane[-min_length:]

        large_dist_num = 0
        adjacent_linestring = LineString(adjacent_lane_seg)
        for idx, point in enumerate(cur_lane_seg):
            if (
                idx % 2
                or adjacent_linestring.distance(Point(point))
                < self.large_dist_th
            ):
                continue
            large_dist_num += 1
            if large_dist_num > self.large_dist_num_th:
                return True

    def get_nearest_waypoint_idx(self, lane_polyline, ego_point):
        lane_array = np.array(lane_polyline)
        distances = np.sqrt(
            np.sum((lane_array - [ego_point.x, ego_point.y]) ** 2, axis=1)
        )
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] > 5:
            nearest_idx = -1

        return nearest_idx

    def is_separate_by_curb(
        self, cutoff_cur_polyline, cutoff_nearby_polyline, curb_decision
    ):
        curb_src = curb_decision["src_point"]
        ego_s = curb_decision["ego_s"]
        curb_s = curb_decision["obs_s"]
        curb_l = curb_decision["obs_l"]
        fork_curb_s = []

        polygon = Polygon(cutoff_cur_polyline + cutoff_nearby_polyline)
        for idx, s in enumerate(curb_s):
            if (
                s - ego_s[idx] < self.curb_roi_s_min
                or s - ego_s[idx] > self.curb_roi_s_max
                or abs(curb_l[idx]) > self.curb_roi_l_max
            ):
                continue
            if polygon.contains(Point(curb_src[idx])):
                fork_curb_s.append(s)

        if len(fork_curb_s) < 1:
            return False

        return min(fork_curb_s) > ego_s[0]
