from typing import Dict, List, Tuple, Union
import numpy as np
from shapely.geometry import LineString, Point
from tag_functions.high_value_scene.common.tag_type import (
    BypassJunctionCurbTag,
)
from tag_functions.high_value_scene.common.basic_info import BasicInfo


def sort_exit_group_from_left_to_right(
    lane_map: Dict, exit_group: List[int]
) -> List[int]:
    anchor_unit_directions = lane_map[exit_group[0]]["unit_directions"]
    anchor_polyline = lane_map[exit_group[0]]["polyline"]
    if len(anchor_unit_directions) < 1 or len(anchor_polyline) < 1:
        return exit_group

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


def get_corr_lane_id_after_junction(data: Dict) -> Tuple[List, List]:
    corr_lane_id = data.label_scene.ego_path_info.corr_lane_id
    in_junction_id = data.label_scene.ego_path_info.in_junction_id
    labeled_junction_id = data.label_scene.junction_label_info.junction_id

    corr_lane_ids_after_junction = []
    after_in_junction_idx = []
    is_arrive_junction = False

    for i, (corr_lane_ids, in_junction_id) in enumerate(
        zip(corr_lane_id, in_junction_id)
    ):
        if in_junction_id is not None and in_junction_id == labeled_junction_id:
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
    basic_info: BasicInfo,
    after_in_junction_idx: List[int],
    is_consider_curvature: bool,
    is_right: int,
    dist_to_exit_up: float,
    dist_to_exit_low: float,
    dist_thr: float,
) -> bool:
    future_path_nearest_curb_dist = basic_info.future_path_nearest_curb_dist
    future_path_curvature = basic_info.future_path_curvature
    future_path_turn_type = basic_info.future_path_turn_type

    for idx, dist in enumerate(future_path_nearest_curb_dist):
        if (
            idx < after_in_junction_idx[0] - dist_to_exit_up
            or idx > after_in_junction_idx[0] - dist_to_exit_low
        ):
            continue
        if dist[is_right] > dist_thr:
            continue

        if is_consider_curvature:
            if abs(future_path_curvature[idx]) < 0.05:
                continue

            if (is_right and future_path_turn_type[idx] > 0) or (
                not is_right and future_path_turn_type[idx] < 0
            ):
                continue

        return True

    return False


def is_interact_with_leftmost_or_rightmost_curb(
    junction_goal: str,
    arrive_exit_lane_id: int,
    sorted_exit_group: List[int],
    after_in_junction_idx: List[int],
    basic_info: BasicInfo,
) -> bool:
    if junction_goal == "TYPE_TURN_LEFT" or junction_goal == "TYPE_U_TURN":
        if sorted_exit_group.index(arrive_exit_lane_id) == 0:
            if is_bypass_curb_in_junction_exit(
                basic_info,
                after_in_junction_idx,
                is_consider_curvature=True,
                is_right=0,
                dist_to_exit_up=12,
                dist_to_exit_low=5,
                dist_thr=2.2,
            ):
                return True
    elif junction_goal == "TYPE_TURN_RIGHT":
        if (
            sorted_exit_group.index(arrive_exit_lane_id)
            == len(sorted_exit_group) - 1
        ):
            if is_bypass_curb_in_junction_exit(
                basic_info,
                after_in_junction_idx,
                is_consider_curvature=True,
                is_right=1,
                dist_to_exit_up=12,
                dist_to_exit_low=5,
                dist_thr=2.0,
            ):
                return True
        elif sorted_exit_group.index(arrive_exit_lane_id) == 0:
            if is_bypass_curb_in_junction_exit(
                basic_info,
                after_in_junction_idx,
                is_consider_curvature=False,
                is_right=0,
                dist_to_exit_up=12,
                dist_to_exit_low=8,
                dist_thr=2.0,
            ):
                return True

    return False


def get_arrive_exit_lane_id(
    corr_lane_ids_after_junction: List[
        List[Union[None, Tuple[Union[str, int], float]]]
    ]
) -> Tuple[int, float]:
    arrive_exit_lane_id = -1
    lane_id_corr_arrive_dist = {}
    for idx, corr_lane_info in enumerate(corr_lane_ids_after_junction):
        if corr_lane_info is None or len(corr_lane_info) == 0:
            continue

        if abs(corr_lane_info[0][1]) < 0.35:
            arrive_exit_lane_id = corr_lane_info[0][0]
            if lane_id_corr_arrive_dist.get(arrive_exit_lane_id, None) is None:
                lane_id_corr_arrive_dist[arrive_exit_lane_id] = idx

    return arrive_exit_lane_id, lane_id_corr_arrive_dist.get(
        arrive_exit_lane_id, None
    )


def calculate_corrected_junction_path_info(
    corr_lane_ids_after_junction: List[
        List[Union[None, Tuple[Union[str, int], float]]]
    ],
    after_in_junction_idx: List[int],
    lane_map: Dict,
    arrive_exit_lane_id: int,
    ego_path_linestring: LineString,
) -> Dict:
    corrected_junction_path_info = {}

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
                        corrected_junction_path_info[
                            "real_lane_exit_pose_l"
                        ] = real_lane_exit_pose_l

                        corrected_junction_path_info[
                            "has_real_arrive_exit_lane"
                        ] = True
                    corr_real_lane = True
                    break
            if corr_real_lane:
                max_length_in_exit_lane += 1
            else:
                max_length_not_in_exit_lane += 1

        corrected_junction_path_info[
            "max_length_in_exit_lane"
        ] = max_length_in_exit_lane

        corrected_junction_path_info[
            "max_length_not_in_exit_lane"
        ] = max_length_not_in_exit_lane

    if len(after_in_junction_idx) > 0:
        pose_ls = []
        exit_lane_polyline = LineString(
            lane_map[arrive_exit_lane_id]["polyline"]
        )
        for idx in after_in_junction_idx:
            point = Point(ego_path_linestring.coords[idx])
            proj_s = exit_lane_polyline.project(point)
            if proj_s <= 0 or proj_s >= exit_lane_polyline.length:
                continue
            pose_ls.append(exit_lane_polyline.distance(point))

        corrected_junction_path_info["hit_point_num"] = len(pose_ls)
        if len(pose_ls) > 0:
            corrected_junction_path_info["min_pose_l_2_exit_lane"] = np.min(
                pose_ls
            )
            corrected_junction_path_info["max_pose_l_2_exit_lane"] = np.max(
                pose_ls
            )
            corrected_junction_path_info["mean_pose_l_2_exit_lane"] = np.mean(
                pose_ls
            )

    return corrected_junction_path_info


def label_bypass_junction_curb_tag(
    data: Dict,
    basic_info: BasicInfo,
) -> BypassJunctionCurbTag:
    bypass_junction_curb_tag = BypassJunctionCurbTag()

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

    # 获取出口之后的corr lane ids信息
    (
        corr_lane_ids_after_junction,
        after_in_junction_idx,
    ) = get_corr_lane_id_after_junction(data)
    if (
        len(corr_lane_ids_after_junction) == 0
        or len(after_in_junction_idx) == 0
    ):
        return bypass_junction_curb_tag

    # 获取future path出路口后最终到达的exit lane id
    (
        arrive_exit_lane_id,
        arrive_dist_from_junction_exit,
    ) = get_arrive_exit_lane_id(corr_lane_ids_after_junction)
    if arrive_exit_lane_id == -1 or arrive_dist_from_junction_exit is None:
        return bypass_junction_curb_tag

    for exit_group in exit_groups:
        if arrive_exit_lane_id not in exit_group:
            continue

        # 把exit group的lane id从左到右排序
        sorted_exit_group = sort_exit_group_from_left_to_right(
            lane_map, exit_group
        )

        # 判断是否走左一或右一，且有绕行curb行为
        if is_interact_with_leftmost_or_rightmost_curb(
            junction_label_info.junction_goal,
            arrive_exit_lane_id,
            sorted_exit_group,
            after_in_junction_idx,
            basic_info,
        ):
            # 记录修正后的出口condition以及相关path信息
            for exit_lane_info in junction_label_info.exit_lanes:
                if exit_lane_info["lane_id"] == arrive_exit_lane_id:
                    bypass_junction_curb_tag.is_interact_with_leftmost_or_rightmost_curb = (
                        True
                    )
                    bypass_junction_curb_tag.corrected_exit_lane_id = (
                        arrive_exit_lane_id
                    )
                    bypass_junction_curb_tag.arrive_dist_from_junction_exit = (
                        arrive_dist_from_junction_exit
                    )
                    bypass_junction_curb_tag.corrected_exit_percep_pose_l = (
                        float(exit_lane_info["pose_l"])
                    )
                    bypass_junction_curb_tag.corrected_junction_path_info = (
                        calculate_corrected_junction_path_info(
                            corr_lane_ids_after_junction,
                            after_in_junction_idx,
                            lane_map,
                            arrive_exit_lane_id,
                            ego_path_linestring,
                        )
                    )
                    break
            break

    return bypass_junction_curb_tag
