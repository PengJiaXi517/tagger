from typing import Dict, List, Tuple
from shapely.geometry import LineString, Point, Polygon
import numpy as np
from base import EgoPathInfo
from tag_functions.high_value_scene.common.basic_info import BasicInfo
from tag_functions.high_value_scene.hv_utils.basic_func import (
    calculate_future_path_curvature_and_turn_type,
)


class RampTagHelper:
    def __init__(
        self,
        enter_fork_consider_len: int,
        exit_fork_consider_len: int,
        large_dist_th: float,
        large_dist_num_th: int,
        curb_roi_s_min: float,
        curb_roi_s_max: float,
        curb_roi_l_max: float,
    ) -> None:
        self.enter_fork_consider_len: int = (
            enter_fork_consider_len  # 考虑的index数量
        )
        self.exit_fork_consider_len: int = exit_fork_consider_len
        self.large_dist_th: int = large_dist_th
        self.large_dist_num_th: int = large_dist_num_th
        self.curb_roi_s_min: int = curb_roi_s_min
        self.curb_roi_s_max: int = curb_roi_s_max
        self.curb_roi_l_max: int = curb_roi_l_max

    def enter_ramp_cruise(
        self,
        lane_map: Dict,
        current_lanes: List[int],
        curb_decision: Dict,
        basic_info: BasicInfo,
    ) -> bool:
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
                basic_info,
                lane_map,
                fork_lane_ids[0],
                fork_lane_ids[1],
                curb_decision,
            ):
                return True

        return False

    def enter_ramp_lane_change(
        self,
        lane_map: Dict,
        current_lanes: List[int],
        curb_decision: Dict,
        current_lane_seqs: List[List[int]],
        ego_path_info: EgoPathInfo,
    ) -> bool:
        corr_lane_id = ego_path_info.corr_lane_id
        future_path = ego_path_info.future_path

        # 滤掉过路口场景
        if any(
            juntion_id is not None
            for juntion_id in ego_path_info.in_junction_id
        ):
            return False

        # 拿到变道后到达的lane id以及对应的点在future path上的index
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

    def exit_ramp(
        self, lane_map: Dict, current_lanes: List[int], ego_point: Point
    ) -> bool:
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
        basic_info: BasicInfo,
        lane_map: Dict,
        cur_lane_id: int,
        nearby_lane_id: int,
        curb_decision: Dict,
        lc_point=None,
    ) -> bool:
        # 检查两条lane id是否在lane_map中
        if not self.is_laneid_in_lanemap(lane_map, cur_lane_id, nearby_lane_id):
            return False

        # 滤除对向车道
        if self.is_opposite_lanes(lane_map, cur_lane_id, nearby_lane_id):
            return False

        # 滤除虚拟车道
        if self.is_virtual_lanes(lane_map, cur_lane_id, nearby_lane_id):
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

        if not self.large_diff_lane(
            cutoff_cur_lane, cutoff_nearby_lane, False, 4.0, 1
        ):
            return False

        if not self.is_future_path_pass_through_lane_with_larger_curvature(
            basic_info,
            cutoff_cur_lane,
            cutoff_nearby_lane,
            cur_lane_id,
            nearby_lane_id,
        ):
            return False

        # 检查两条分叉路之间是否有curb隔离
        return self.is_separate_by_curb(
            cutoff_cur_lane, cutoff_nearby_lane, curb_decision, basic_info
        )

    def judge_exit_fork(
        self,
        lane_map: Dict,
        cur_lane_id: int,
        adjacent_lane_id: int,
        ego_point: Point,
    ) -> bool:
        # 检查两条lane id是否在lane_map中
        if not self.is_laneid_in_lanemap(
            lane_map, cur_lane_id, adjacent_lane_id
        ):
            return False

        # 滤除对向车道
        if self.is_opposite_lanes(lane_map, cur_lane_id, adjacent_lane_id):
            return False

        cur_lane = lane_map[cur_lane_id]["polyline"]
        adjacent_lane = lane_map[adjacent_lane_id]["polyline"]

        # 自车在cur lane上找最近点的索引
        cur_idx = self.find_nearest_waypoint_idx(cur_lane, ego_point)
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
        return self.large_diff_lane(
            stitched_cur_lane, stitched_adjacent_lane, True, 10.0, 4
        )

    def lane_seq_validity_check(
        self,
        lane_map: Dict,
        current_lanes: List[int],
        current_lane_seqs: List[List[int]],
    ) -> bool:
        if len(current_lanes) < 1 or len(current_lane_seqs) < 1:
            return False

        for lane_seq in current_lane_seqs:
            for lane_id in lane_seq:
                if lane_map[lane_id]["turn"] != "NOTURN" or (
                    lane_map[lane_id]["lane_category"] != "REALITY"
                ):
                    # if lane_map[lane_id]["lane_category"] != "REALITY":
                    return False

        return True

    def is_laneid_in_lanemap(
        self, lane_map: Dict, cur_lane_id: int, nearby_lane_id: int
    ) -> bool:
        return cur_lane_id in lane_map and nearby_lane_id in lane_map

    def is_opposite_lanes(
        self, lane_map: Dict, cur_lane_id: int, adjacent_lane_id: int
    ) -> bool:
        cur_lane_dir = lane_map[cur_lane_id]["unit_directions"][0]
        nearby_lane_dir = lane_map[adjacent_lane_id]["unit_directions"][0]

        return (
            cur_lane_dir[0] * nearby_lane_dir[0]
            + cur_lane_dir[1] * nearby_lane_dir[1]
            < 0
        )

    def is_virtual_lanes(
        self, lane_map: Dict, cur_lane_id: int, nearby_lane_id: int
    ) -> bool:
        return (
            lane_map[cur_lane_id]["lane_category"] != "REALITY"
            or lane_map[nearby_lane_id]["lane_category"] != "REALITY"
        )

    def cut_off_lane_forward(
        self,
        cur_lane: List[List[float]],
        nearby_lane: List[List[float]],
        ego_point: Point,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        cur_idx = 0
        nearby_idx = 0

        if ego_point is not None:
            cur_idx = self.find_nearest_waypoint_idx(cur_lane, ego_point)
            nearby_idx = self.find_nearest_waypoint_idx(nearby_lane, ego_point)
            if cur_idx == -1 or nearby_idx == -1:
                return [], []

        min_length = min(
            len(cur_lane) - cur_idx,
            len(nearby_lane) - nearby_idx,
            self.enter_fork_consider_len,
        )

        cutoff_cur_lane = cur_lane[cur_idx : cur_idx + min_length]
        cutoff_nearby_polyline = list(
            reversed(nearby_lane[nearby_idx : nearby_idx + min_length])
        )

        return cutoff_cur_lane, cutoff_nearby_polyline

    def stitch_lane_backward(
        self, lane_polyline: List[List[float]], lane_id: int, lane_map: Dict
    ) -> List[List[float]]:
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

    # def large_diff_lane(
    #     self, cur_lane: List[List[float]], adjacent_lane: List[List[float]]
    # ) -> bool:
    #     min_length = min(len(cur_lane), len(adjacent_lane))
    #     if min_length < 3:
    #         return False

    #     cur_lane_seg = cur_lane[-min_length:]
    #     adjacent_lane_seg = adjacent_lane[-min_length:]

    #     large_dist_num = 0
    #     adjacent_linestring = LineString(adjacent_lane_seg)
    #     for idx, point in enumerate(cur_lane_seg):
    #         if (
    #             idx % 2
    #             or adjacent_linestring.distance(Point(point))
    #             < self.large_dist_th
    #         ):
    #             continue

    #         large_dist_num += 1
    #         if large_dist_num > self.large_dist_num_th:
    #             return True

    #     return False

    def large_diff_lane(
        self,
        cur_lane: List[List[float]],
        adjacent_lane: List[List[float]],
        is_reverse: bool,
        large_dist_th: float,
        large_dist_num_th: int,
    ) -> bool:
        min_length = min(len(cur_lane), len(adjacent_lane))
        if min_length < 3:
            return False

        if is_reverse:
            cur_lane_seg = cur_lane[-min_length:]
            adjacent_lane_seg = adjacent_lane[-min_length:]
        else:
            cur_lane_seg = cur_lane[:min_length]
            adjacent_lane_seg = adjacent_lane[:min_length]

        large_dist_num = 0
        adjacent_linestring = LineString(adjacent_lane_seg)
        for idx, point in enumerate(cur_lane_seg):
            if (
                idx % 2
                or adjacent_linestring.distance(Point(point)) < large_dist_th
            ):
                continue

            large_dist_num += 1
            if large_dist_num >= large_dist_num_th:
                return True

        return False

    def find_nearest_waypoint_idx(
        self, lane_polyline: List[List[float]], point: Point
    ) -> int:
        lane_polyline_array = np.array(lane_polyline)
        point_to_lane_distances = np.sqrt(
            np.sum((lane_polyline_array - [point.x, point.y]) ** 2, axis=1)
        )
        nearest_idx = np.argmin(point_to_lane_distances)

        if point_to_lane_distances[nearest_idx] > 5:
            nearest_idx = -1

        return nearest_idx

    def future_path_bypass_curb(
        self, basic_info: BasicInfo, valid_curb_index: List[int]
    ) -> bool:
        future_path_curvature = basic_info.future_path_curvature
        future_path_turn_type = basic_info.future_path_turn_type
        future_narrow_road_curb_index = basic_info.future_narrow_road_curb_index

        is_bypass_curb = False
        for i in range(len(future_path_curvature)):
            if abs(future_path_curvature[i]) < 0.007:
                continue

            if future_path_turn_type[i] > 0:
                if future_narrow_road_curb_index[i][0] in valid_curb_index:
                    is_bypass_curb = True
                    break

            if future_path_turn_type[i] < 0:
                if future_narrow_road_curb_index[i][1] in valid_curb_index:
                    is_bypass_curb = True
                    break

        return is_bypass_curb

    def is_separate_by_curb(
        self,
        current_polyline: List[List[float]],
        nearby_polyline: List[List[float]],
        curb_decision: Dict,
        basic_info: BasicInfo,
    ) -> bool:
        curb_src = curb_decision["src_point"]
        curb_vec = curb_decision["vec"]
        ego_lon_positions = curb_decision["ego_s"]
        curb_lon_positions = curb_decision["obs_s"]
        curb_lat_positions = curb_decision["obs_l"]
        valid_curb_index = []
        valid_curb_lon_positions = []
        curb_segment_lengths = []

        enclosed_polygon = Polygon(current_polyline + nearby_polyline)

        if False:
            from matplotlib import pyplot as plt

            plt.figure(figsize=(14, 12))
            # lane1 = lane_map[178425]['polyline']
            # lane2 = lane_map[178160]['polyline']
            x1, y1 = zip(*current_polyline)
            x2, y2 = zip(*nearby_polyline)
            plt.plot(x1, y1, color="black")
            plt.plot(x2, y2, color="blue")
            plt.fill(*enclosed_polygon.exterior.xy, color="green", alpha=0.5)
            plt.show()

        for idx, curb_lon_position in enumerate(curb_lon_positions):
            if (
                curb_lon_position - ego_lon_positions[idx] < self.curb_roi_s_min
                or curb_lon_position - ego_lon_positions[idx]
                > self.curb_roi_s_max
                or abs(curb_lat_positions[idx]) > self.curb_roi_l_max
            ):
                continue

            if enclosed_polygon.contains(Point(curb_src[idx])):
                valid_curb_index.append(idx)
                valid_curb_lon_positions.append(curb_lon_position)
                curb_segment_lengths.append(
                    np.linalg.norm(np.array(curb_vec[idx]))
                )

        if len(valid_curb_lon_positions) <= 0:
            return False

        if sum(curb_segment_lengths) < 5.0:
            return False

        if not self.future_path_bypass_curb(basic_info, valid_curb_index):
            return False

        return min(valid_curb_lon_positions) > ego_lon_positions[0]

    def is_future_path_pass_through_lane_with_larger_curvature(
        self,
        basic_info: BasicInfo,
        current_polyline: List[List[float]],
        nearby_polyline: List[List[float]],
        cur_lane_id: int,
        nearby_lane_id: int,
    ) -> bool:
        if False:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(14, 12))
            # lane1 = lane_map[178425]['polyline']
            # lane2 = lane_map[178160]['polyline']
            x1, y1 = zip(*current_polyline)
            x2, y2 = zip(*nearby_polyline)
            plt.plot(x1, y1, color="black")
            plt.plot(x2, y2, color="blue")
            # plt.fill(*enclosed_polygon.exterior.xy, color="green", alpha=0.5)
            plt.show()


        laneid_corr_waypoint_num_map = basic_info.laneid_corr_waypoint_num_map
        cur_lane_id_corr_point_num = laneid_corr_waypoint_num_map.get(
            cur_lane_id, 0
        )
        nearby_lane_id_corr_point_num = laneid_corr_waypoint_num_map.get(
            nearby_lane_id, 0
        )

        if (
            cur_lane_id_corr_point_num == 0
            and nearby_lane_id_corr_point_num == 0
        ):
            return False

        (
            cur_polyline_curvature,
            _,
        ) = calculate_future_path_curvature_and_turn_type(current_polyline)
        (
            nearby_polyline_curvature,
            _,
        ) = calculate_future_path_curvature_and_turn_type(nearby_polyline)

        cur_polyline_sum_curvature = np.sum(np.abs(cur_polyline_curvature))
        nearby_polyline_sum_curvature = np.sum(
            np.abs(nearby_polyline_curvature)
        )

        if (
            cur_lane_id_corr_point_num >= nearby_lane_id_corr_point_num
            and cur_polyline_sum_curvature > nearby_polyline_sum_curvature
        ):
            return True

        if (
            nearby_lane_id_corr_point_num > cur_lane_id_corr_point_num
            and nearby_polyline_sum_curvature > cur_polyline_sum_curvature
        ):
            return True

        return False
