from typing import Dict, List, Tuple, Union
import numpy as np
from shapely.geometry import LineString, Point
from tag_functions.high_value_scene.common.tag_type import (
    BypassJunctionCurbTag,
)
from tag_functions.high_value_scene.common.basic_info import BasicInfo


class BypassJunctionCurbTagHelper:
    def __init__(
        self, bypass_curvature_thr: float = 0.05, arrive_dist_thr: float = 0.35
    ) -> None:
        self.bypass_curvature_thr = bypass_curvature_thr
        self.arrive_dist_thr = arrive_dist_thr

    def sort_exit_group_from_left_to_right(
        self, lane_map: Dict, exit_group: List[int]
    ) -> List[int]:
        anchor_unit_directions = lane_map[exit_group[0]]["unit_directions"]
        anchor_polyline = lane_map[exit_group[0]]["polyline"]
        if len(anchor_unit_directions) < 1 or len(anchor_polyline) < 1:
            return []

        anchor_unit_direct = anchor_unit_directions[0]
        anchor_point = anchor_polyline[0]

        lane_relative_y = []
        for exit_lane_id in exit_group:
            polyline = lane_map[exit_lane_id]["polyline"]
            if len(polyline) < 1:
                continue

            relative_direct = np.array(polyline[0]) - np.array(anchor_point)
            relative_y = np.cross(anchor_unit_direct, relative_direct)
            lane_relative_y.append((exit_lane_id, relative_y))

        lane_relative_y.sort(key=lambda x: x[1], reverse=True)

        return [x[0] for x in lane_relative_y]

    def get_corr_lane_id_after_junction(self, data: Dict) -> Tuple[List, List]:
        corr_lane_id = data.label_scene.ego_path_info.corr_lane_id
        in_junction_id = data.label_scene.ego_path_info.in_junction_id
        labeled_junction_id = data.label_scene.junction_label_info.junction_id

        corr_lane_ids_after_junction = []
        after_in_junction_idx = []
        is_arrive_junction = False

        for i, (corr_lane_ids, in_junction_id) in enumerate(
            zip(corr_lane_id, in_junction_id)
        ):
            if (
                in_junction_id is not None
                and in_junction_id == labeled_junction_id
            ):
                is_arrive_junction = True

            if is_arrive_junction:
                if in_junction_id is not None:
                    if in_junction_id == labeled_junction_id:
                        corr_lane_ids_after_junction = []
                        after_in_junction_idx = []
                    else:
                        break
                else:
                    corr_lane_ids_after_junction.append(corr_lane_ids)
                    after_in_junction_idx.append(i)

        return corr_lane_ids_after_junction, after_in_junction_idx

    def is_bypass_curb_in_junction_exit(
        self,
        basic_info: BasicInfo,
        is_consider_curvature: bool,
        is_right: int,
        curb_range_near: float,
        curb_range_far: float,
        dist_to_curb_thr: float,
    ) -> bool:
        future_path_nearest_curb_dist = basic_info.future_path_nearest_curb_dist
        future_path_curvature = basic_info.future_path_curvature
        future_path_turn_type = basic_info.future_path_turn_type

        for idx, dist in enumerate(future_path_nearest_curb_dist):
            if idx < curb_range_near or idx > curb_range_far:
                continue

            if dist[is_right] > dist_to_curb_thr:
                continue

            if is_consider_curvature:
                if abs(future_path_curvature[idx]) < self.bypass_curvature_thr:
                    continue

                if (is_right and future_path_turn_type[idx] > 0) or (
                    not is_right and future_path_turn_type[idx] < 0
                ):
                    continue

            return True

        return False

    def cal_consider_curb_range(
        self,
        path_distance_to_entry: float,
        path_distance_to_exit_lane: float,
        thr_entry: float,
        thr_exit: float,
    ) -> Tuple[float, float]:
        curb_range_far = path_distance_to_exit_lane - thr_exit
        curb_range_near = min(
            path_distance_to_entry + thr_entry, curb_range_far
        )

        return curb_range_near, curb_range_far

    def is_interact_with_leftmost_or_rightmost_curb(
        self,
        junction_goal: str,
        arrive_exit_lane_id: int,
        sorted_exit_group: List[int],
        path_distance_to_entry,
        path_distance_to_exit_lane,
        basic_info: BasicInfo,
    ) -> bool:
        if junction_goal == "TYPE_TURN_LEFT" or junction_goal == "TYPE_U_TURN":
            if sorted_exit_group.index(arrive_exit_lane_id) == 0:
                curb_range_near, curb_range_far = self.cal_consider_curb_range(
                    path_distance_to_entry,
                    path_distance_to_exit_lane,
                    10,
                    5,
                )
                if self.is_bypass_curb_in_junction_exit(
                    basic_info,
                    is_consider_curvature=True,
                    is_right=0,
                    curb_range_near=curb_range_near,
                    curb_range_far=curb_range_far,
                    dist_to_curb_thr=2.2,
                ):
                    return True
        elif junction_goal == "TYPE_TURN_RIGHT":
            if (
                sorted_exit_group.index(arrive_exit_lane_id)
                == len(sorted_exit_group) - 1
            ):
                curb_range_near, curb_range_far = self.cal_consider_curb_range(
                    path_distance_to_entry,
                    path_distance_to_exit_lane,
                    10,
                    5,
                )
                if self.is_bypass_curb_in_junction_exit(
                    basic_info,
                    is_consider_curvature=True,
                    is_right=1,
                    curb_range_near=curb_range_near,
                    curb_range_far=curb_range_far,
                    dist_to_curb_thr=2.0,
                ):
                    return True
            elif sorted_exit_group.index(arrive_exit_lane_id) == 0:
                curb_range_near, curb_range_far = self.cal_consider_curb_range(
                    path_distance_to_entry,
                    path_distance_to_exit_lane,
                    10,
                    8,
                )
                if self.is_bypass_curb_in_junction_exit(
                    basic_info,
                    is_consider_curvature=False,
                    is_right=0,
                    curb_range_near=curb_range_near,
                    curb_range_far=curb_range_far,
                    dist_to_curb_thr=2.0,
                ):
                    return True

        return False

    def get_arrive_exit_lane_id(
        self,
        corr_lane_ids_after_junction: List[
            List[Union[None, Tuple[Union[str, int], float]]]
        ],
    ) -> Tuple[int, float]:
        arrive_exit_lane_id = -1
        lane_id_corr_arrive_dist = {}

        for idx, corr_lane_info in enumerate(corr_lane_ids_after_junction):
            if corr_lane_info is None or len(corr_lane_info) == 0:
                continue

            if abs(corr_lane_info[0][1]) < self.arrive_dist_thr:
                arrive_exit_lane_id = corr_lane_info[0][0]
                if (
                    lane_id_corr_arrive_dist.get(arrive_exit_lane_id, None)
                    is None
                ):
                    lane_id_corr_arrive_dist[arrive_exit_lane_id] = idx

        return arrive_exit_lane_id, lane_id_corr_arrive_dist.get(
            arrive_exit_lane_id, -1
        )

    def calculate_corrected_junction_path_info(
        self,
        corr_lane_ids_after_junction: List[
            List[Union[None, Tuple[Union[str, int], float]]]
        ],
        after_in_junction_idx: List[int],
        lane_map: Dict,
        arrive_exit_lane_id: int,
        ego_path_linestring: LineString,
    ) -> Dict:
        bypass_junction_curb_tag = BypassJunctionCurbTag()

        if len(corr_lane_ids_after_junction) > 0:
            real_lane_exit_pose_l = None
            max_length_in_exit_lane = 0.0
            max_length_not_in_exit_lane = 0.0

            for corr_lane_ids in corr_lane_ids_after_junction:
                corr_real_lane = False
                for lane_id, pose_l in corr_lane_ids:
                    if lane_id == arrive_exit_lane_id:
                        if real_lane_exit_pose_l is None:
                            real_lane_exit_pose_l = float(np.abs(pose_l))
                            bypass_junction_curb_tag.real_lane_exit_pose_l = (
                                real_lane_exit_pose_l
                            )
                            bypass_junction_curb_tag.has_real_arrive_exit_lane = (
                                True
                            )
                        corr_real_lane = True
                        break
                if corr_real_lane:
                    max_length_in_exit_lane += 1
                else:
                    max_length_not_in_exit_lane += 1

            bypass_junction_curb_tag.max_length_in_exit_lane = (
                max_length_in_exit_lane
            )

            bypass_junction_curb_tag.max_length_not_in_exit_lane = (
                max_length_not_in_exit_lane
            )

        if len(after_in_junction_idx) > 0:
            pose_ls = []
            exit_lane_polyline = LineString(
                lane_map[arrive_exit_lane_id]["polyline"]
            )
            has_calculated_percep_pose_l = False
            for idx in after_in_junction_idx:
                point = Point(ego_path_linestring.coords[idx])
                proj_s = exit_lane_polyline.project(point)
                if proj_s <= 0 or proj_s >= exit_lane_polyline.length:
                    continue

                dist_to_exit_lane_polyline = exit_lane_polyline.distance(point)
                pose_ls.append(dist_to_exit_lane_polyline)

                if not has_calculated_percep_pose_l:
                    has_calculated_percep_pose_l = True
                    bypass_junction_curb_tag.percep_lane_exit_pose_l = (
                        dist_to_exit_lane_polyline
                    )

            bypass_junction_curb_tag.hit_point_num = len(pose_ls)
            if len(pose_ls) > 0:
                bypass_junction_curb_tag.min_pose_l_2_exit_lane = np.min(
                    pose_ls
                )
                bypass_junction_curb_tag.max_pose_l_2_exit_lane = np.max(
                    pose_ls
                )
                bypass_junction_curb_tag.mean_pose_l_2_exit_lane = np.mean(
                    pose_ls
                )

        return bypass_junction_curb_tag

    def cal_path_dist_to_junction(
        self, in_junction_id: List[int], labeled_junction_id: int
    ) -> Tuple[float, float]:
        path_distance_to_exit_lane = -1
        path_distance_to_entry = -1
        is_arrive_junction_entry = False

        for i, junction_id in enumerate(in_junction_id):
            if junction_id is not None and junction_id == labeled_junction_id:
                path_distance_to_exit_lane = i
                if not is_arrive_junction_entry:
                    is_arrive_junction_entry = True
                    path_distance_to_entry = i - 1

        return path_distance_to_entry, path_distance_to_exit_lane


def label_bypass_junction_curb_tag(
    data: Dict,
    basic_info: BasicInfo,
) -> BypassJunctionCurbTag:
    bypass_junction_curb_tag = BypassJunctionCurbTag()
    bypass_junction_curb_tag_helper = BypassJunctionCurbTagHelper()

    if basic_info.is_ego_vehicle_always_moving:
        bypass_junction_curb_tag.is_bypass_junction_curb = any(
            basic_info.future_bypass_junction_curb
        )

    junction_label_info = data.label_scene.junction_label_info
    lane_map = data.label_scene.percepmap.lane_map
    ego_path_linestring = data.label_scene.ego_path_info.future_path_linestring

    if not basic_info.is_cross_junction:
        return bypass_junction_curb_tag

    if junction_label_info.junction_goal not in [
        "TYPE_TURN_LEFT",
        "TYPE_TURN_RIGHT",
        "TYPE_U_TURN",
    ]:
        return bypass_junction_curb_tag

    if (
        junction_label_info.junction_id == ""
        or junction_label_info.junction_id < 0
    ):
        return bypass_junction_curb_tag

    exit_groups = data.label_scene.percepmap.junction_map.get(
        junction_label_info.junction_id, {}
    ).get("exit_groups", [])

    # 出口之后的corr lane ids信息
    (
        corr_lane_ids_after_junction,
        after_in_junction_idx,
    ) = bypass_junction_curb_tag_helper.get_corr_lane_id_after_junction(data)
    if (
        len(corr_lane_ids_after_junction) == 0
        or len(after_in_junction_idx) == 0
    ):
        return bypass_junction_curb_tag

    # 获取future path出路口后最终到达的exit lane id
    (
        arrive_exit_lane_id,
        arrive_dist_from_junction_exit,
    ) = bypass_junction_curb_tag_helper.get_arrive_exit_lane_id(
        corr_lane_ids_after_junction
    )
    if arrive_exit_lane_id == -1 or arrive_dist_from_junction_exit == -1:
        return bypass_junction_curb_tag

    (
        path_distance_to_entry,
        path_distance_to_exit_lane,
    ) = bypass_junction_curb_tag_helper.cal_path_dist_to_junction(
        data.label_scene.ego_path_info.in_junction_id,
        junction_label_info.junction_id,
    )

    for exit_group in exit_groups:
        if arrive_exit_lane_id not in exit_group:
            continue

        # 把exit group的lane id从左到右排序
        sorted_exit_group = (
            bypass_junction_curb_tag_helper.sort_exit_group_from_left_to_right(
                lane_map, exit_group
            )
        )

        # 判断是否走左一或右一，且有绕行curb行为
        if bypass_junction_curb_tag_helper.is_interact_with_leftmost_or_rightmost_curb(
            junction_label_info.junction_goal,
            arrive_exit_lane_id,
            sorted_exit_group,
            path_distance_to_entry,
            path_distance_to_exit_lane,
            basic_info,
        ):
            bypass_junction_curb_tag = bypass_junction_curb_tag_helper.calculate_corrected_junction_path_info(
                corr_lane_ids_after_junction,
                after_in_junction_idx,
                lane_map,
                arrive_exit_lane_id,
                ego_path_linestring,
            )
            # 记录修正后的出口condition以及相关path信息
            bypass_junction_curb_tag.is_interact_with_leftmost_or_rightmost_curb = (
                True
            )
            bypass_junction_curb_tag.corrected_exit_lane_id = (
                arrive_exit_lane_id
            )
            bypass_junction_curb_tag.arrive_dist_from_junction_exit = (
                arrive_dist_from_junction_exit
            )
            break

    return bypass_junction_curb_tag
