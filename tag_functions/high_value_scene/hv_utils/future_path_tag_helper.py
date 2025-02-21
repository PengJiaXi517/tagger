from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import LineString, Point
from base import PercepMap, TagData
from tag_functions.high_value_scene.common.tag_type import (
    ConditionResTag,
    CruisePATHTag,
    LcPATHTag,
    JunctionPATHTag,
    FuturePATHType,
    BasicPathTag,
    JUNCTION_GOAL_2_TURN_TYPE,
)


class FuturePathTagHelper:
    def __init__(self) -> None:
        pass

    def label_condition_res_tag(
        self, data: TagData, params: Dict
    ) -> ConditionResTag:
        condition_pair = data.condition_res.lane_seq_pair
        lane_seq_ids = data.condition_res.seq_lane_ids_raw

        condition_res_tag = ConditionResTag()
        for idx_start, idx_end, _ in condition_pair:
            if idx_start != -1:
                condition_res_tag.start_lane_seq_ids.append(
                    lane_seq_ids[idx_start]
                )
            else:
                condition_res_tag.start_lane_seq_ids.append([])
            if idx_end != -1:
                condition_res_tag.end_lane_seq_ids.append(lane_seq_ids[idx_end])
            else:
                condition_res_tag.end_lane_seq_ids.append([])

        return condition_res_tag

    def label_cruise_tag(
        self,
        data: TagData,
        params: Dict,
        always_on_current_lane_seq,
        percep_map: PercepMap,
        sample_point_length: float = 3.0,
    ) -> List[CruisePATHTag]:
        ego_path_info = data.label_scene.ego_path_info
        num_points_on_lane = 0
        final_corr_lane_ids = None
        for corr_lane_ids in ego_path_info.corr_lane_id:
            if corr_lane_ids is None:
                break
            final_corr_lane_ids = corr_lane_ids
            num_points_on_lane += 1

        if final_corr_lane_ids is None:
            return []

        path_line_string = ego_path_info.future_path_linestring

        cruise_tags = []

        for lane_seq in always_on_current_lane_seq:
            cruise_tag = CruisePATHTag()

            cruise_tag.labeled_lane_seq = lane_seq
            cruise_tag.max_continuous_length_on_lane = max(
                (num_points_on_lane - 1) * sample_point_length, 0.0
            )

            polylines = [
                LineString(percep_map.lane_map[lane_id]["polyline"])
                for lane_id in lane_seq
            ]

            pose_ls = []
            latest_on_percep_lane_point_idx = -1
            for i in range(num_points_on_lane):
                point = Point(path_line_string.coords[i])

                l_dis = [
                    pl.distance(point)
                    for pl in polylines
                    if 0 < pl.project(point) < pl.length
                ]

                if len(l_dis) > 0:
                    curr_pose_l = np.min(l_dis)

                    latest_on_percep_lane_point_idx = i
                    pose_ls.append(curr_pose_l)
            cruise_tag.latest_on_percep_lane_point_idx = (
                latest_on_percep_lane_point_idx
            )

            if len(pose_ls) > 0:
                cruise_tag.percep_pose_l = pose_ls[-1]
                cruise_tag.max_pose_l = np.max(pose_ls)
                cruise_tag.mean_pose_l = np.mean(pose_ls)
            for lane_id, pose_l in final_corr_lane_ids:
                if lane_id in lane_seq:
                    cruise_tag.real_pose_l = float(np.abs(pose_l))
            cruise_tags.append(cruise_tag)

        return cruise_tags

    def label_lc_tag(
        self,
        data: TagData,
        params: Dict,
        arrive_on_nearby_lane_seq,
        percep_map: PercepMap,
        sample_point_length: float = 3.0,
    ) -> LcPATHTag:
        ego_path_info = data.label_scene.ego_path_info
        num_points_on_lane = 0
        final_corr_lane_ids = None
        for corr_lane_ids in ego_path_info.corr_lane_id:
            if corr_lane_ids is None:
                break
            final_corr_lane_ids = corr_lane_ids
            num_points_on_lane += 1

        if final_corr_lane_ids is None:
            return []

        path_line_string = ego_path_info.future_path_linestring

        lc_path_tags = []

        for lane_seq in arrive_on_nearby_lane_seq:
            polylines = [
                LineString(percep_map.lane_map[lane_id]["polyline"])
                for lane_id in lane_seq
            ]

            first_arrive_lane_seq_idx = num_points_on_lane - 1

            for i, corr_lane_ids in enumerate(ego_path_info.corr_lane_id):
                if corr_lane_ids is not None:
                    for lane_id, _ in corr_lane_ids:
                        if lane_id in lane_seq:
                            first_arrive_lane_seq_idx = i
                            break
                if first_arrive_lane_seq_idx != num_points_on_lane - 1:
                    break

            lc_path_tag = LcPATHTag()
            lc_path_tag.labeled_lane_seq = lane_seq
            lc_path_tag.arrive_length = (
                num_points_on_lane - first_arrive_lane_seq_idx
            ) * sample_point_length

            start_point = Point(path_line_string.coords[0])

            lc_path_tag.start_pose_l = np.min(
                [pl.distance(start_point) for pl in polylines]
            )

            pose_ls = [np.inf]
            for i in range(first_arrive_lane_seq_idx, num_points_on_lane):
                point = Point(path_line_string.coords[i])
                curr_pose_l = np.min([pl.distance(point) for pl in polylines])
                pose_ls.append(curr_pose_l)

            lc_path_tag.arrive_nearest_pose_l = np.min(pose_ls)
            lc_path_tag.arrive_percep_pose_l = pose_ls[-1]
            for lane_id, pose_l in final_corr_lane_ids:
                if lane_id in lane_seq:
                    lc_path_tag.arrive_final_pose_l = float(np.abs(pose_l))

            lc_path_tags.append(lc_path_tag)

        return lc_path_tags

    def label_junction_tag(
        self,
        data: TagData,
        params: Dict,
        percep_map: PercepMap,
        sample_point_length: float = 3.0,
    ) -> JunctionPATHTag:
        junction_path_tag = JunctionPATHTag()

        junction_label_info = data.label_scene.junction_label_info
        ego_path_info = data.label_scene.ego_path_info
        ego_path_linestring = (
            data.label_scene.ego_path_info.future_path_linestring
        )

        junction_path_tag.turn_type = JUNCTION_GOAL_2_TURN_TYPE[
            junction_label_info.junction_goal
        ]

        if junction_label_info.junction_id == "":
            junction_path_tag.has_junction_label = False
            return junction_path_tag

        junction_path_tag.has_junction_label = True
        for in_junction_id in ego_path_info.in_junction_id:
            if in_junction_id is not None:
                junction_path_tag.first_arrive_junction_id = in_junction_id
                break
        junction_path_tag.label_junction_id = junction_label_info.junction_id

        path_distance_to_exit_lane = -1
        is_arrive_junction_entry = False
        for i, in_junction_id in enumerate(ego_path_info.in_junction_id):
            if (
                in_junction_id is not None
                and in_junction_id == junction_label_info.junction_id
            ):
                path_distance_to_exit_lane = i
                if not is_arrive_junction_entry:
                    is_arrive_junction_entry = True
                    junction_path_tag.path_distance_to_entry = i - 1

        junction_path_tag.path_distance_to_exit_lane = (
            path_distance_to_exit_lane
        )

        if len(junction_label_info.entry_lanes) > 0:
            junction_path_tag.has_entry_lane = True
            nearest_lane_info = min(
                junction_label_info.entry_lanes,
                key=lambda x: np.abs(x["pose_l"]),
            )
            nearest_lane_id = nearest_lane_info["lane_id"]
            nearest_pose_l = float(np.abs(nearest_lane_info["pose_l"]))

            junction_path_tag.label_entry_lane_id = nearest_lane_id
            junction_path_tag.percep_lane_entry_pose_l = nearest_pose_l

            final_entry_corr_lane_ids = None

            for corr_lane_ids in ego_path_info.corr_lane_id:
                if corr_lane_ids is None:
                    break

                if len(corr_lane_ids) == 0:
                    continue
                
                final_entry_corr_lane_ids = corr_lane_ids

            if final_entry_corr_lane_ids is not None:
                for lane_id, pose_l in final_entry_corr_lane_ids:
                    if lane_id == nearest_lane_id:
                        junction_path_tag.has_real_arrive_entry_lane = True
                        junction_path_tag.real_lane_entry_pose_l = float(
                            np.abs(pose_l)
                        )
                        break

            junction_path_tag.waiting_area_lane_info = (
                junction_label_info.waiting_area_lane_info.get(
                    nearest_lane_id, []
                )
            )

            if len(junction_path_tag.waiting_area_lane_info) > 0:
                junction_path_tag.has_waiting_area = True

            junction_path_tag.entry_dashline_length = (
                self.get_dash_boundary_length(
                    percep_map.lane_map[nearest_lane_id], True
                )
            )

        if len(junction_label_info.exit_lanes) > 0:
            junction_path_tag.has_exit_lane = True

            nearest_lane_info = min(
                junction_label_info.exit_lanes,
                key=lambda x: np.abs(x["pose_l"]),
            )
            nearest_lane_id = nearest_lane_info["lane_id"]
            nearest_pose_l = float(np.abs(nearest_lane_info["pose_l"]))

            junction_path_tag.label_exit_lane_id = nearest_lane_id
            junction_path_tag.percep_lane_exit_pose_l = nearest_pose_l

            is_arrive_junction = False

            corr_lane_ids_after_junction = []
            after_in_junction_idx = []

            for i, (corr_lane_ids, in_junction_id) in enumerate(
                zip(ego_path_info.corr_lane_id, ego_path_info.in_junction_id)
            ):
                if (
                    in_junction_id is not None
                    and in_junction_id == junction_label_info.junction_id
                ):
                    is_arrive_junction = True

                if is_arrive_junction:
                    if in_junction_id is not None:
                        if in_junction_id == junction_label_info.junction_id:
                            corr_lane_ids_after_junction = []
                            after_in_junction_idx = []
                        else:
                            break
                    else:
                        corr_lane_ids_after_junction.append(corr_lane_ids)
                        after_in_junction_idx.append(i)

            if len(corr_lane_ids_after_junction) > 0:
                real_lane_exit_pose_l = None

                max_length_in_exit_lane = 0.0
                max_length_not_in_exit_lane = 0.0
                for corr_lane_ids in corr_lane_ids_after_junction:
                    corr_real_lane = False
                    for lane_id, pose_l in corr_lane_ids:
                        if lane_id == nearest_lane_id:
                            if real_lane_exit_pose_l is None:
                                real_lane_exit_pose_l = float(np.abs(pose_l))
                                junction_path_tag.real_lane_exit_pose_l = (
                                    real_lane_exit_pose_l
                                )
                                junction_path_tag.has_real_arrive_exit_lane = (
                                    True
                                )
                            corr_real_lane = True
                            break
                    if corr_real_lane:
                        max_length_in_exit_lane += sample_point_length
                    else:
                        max_length_not_in_exit_lane += sample_point_length

                junction_path_tag.max_length_in_exit_lane = (
                    max_length_in_exit_lane
                )
                junction_path_tag.max_length_not_in_exit_lane = (
                    max_length_not_in_exit_lane
                )

            if len(after_in_junction_idx) > 0:
                pose_ls = []
                exit_lane_polyline = LineString(
                    percep_map.lane_map[nearest_lane_id]["polyline"]
                )
                for idx in after_in_junction_idx:
                    point = Point(ego_path_linestring.coords[idx])
                    proj_s = exit_lane_polyline.project(point)
                    if proj_s <= 0 or proj_s >= exit_lane_polyline.length:
                        continue
                    pose_ls.append(exit_lane_polyline.distance(point))

                junction_path_tag.hit_point_num = len(pose_ls)
                if len(pose_ls) > 0:
                    junction_path_tag.min_pose_l_2_exit_lane = np.min(pose_ls)
                    junction_path_tag.max_pose_l_2_exit_lane = np.max(pose_ls)
                    junction_path_tag.mean_pose_l_2_exit_lane = np.mean(pose_ls)

            junction_path_tag.exit_dashline_length = (
                self.get_dash_boundary_length(
                    percep_map.lane_map[nearest_lane_id], False
                )
            )

        return junction_path_tag

    def label_backing_up_tag(self, data: TagData, params: Dict) -> bool:

        return (
            not data.label_scene.ego_path_info.future_path_linestring.is_simple
        )

    def label_basic_tag(self, data: TagData, params: Dict) -> BasicPathTag:
        def normal_angle(theta):
            while theta >= np.pi:
                theta -= 2 * np.pi
            while theta <= -np.pi:
                theta += 2 * np.pi
            return theta

        applyall = np.vectorize(normal_angle)

        def cal_curvature(
            future_path: List[Tuple[float, float]]
        ) -> Tuple[float, ...]:
            if len(future_path) <= 4:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            path_points = np.array(future_path)

            diff_points = path_points[1:] - path_points[:-1]

            theta = np.arctan2(diff_points[:, 1], diff_points[:, 0])

            theta_diff = theta[1:] - theta[:-1]

            length = np.linalg.norm(diff_points[:-1], axis=-1)

            theta_diff = applyall(theta_diff)

            curvature = theta_diff / length

            return (
                np.abs(curvature).sum(),
                curvature[curvature > 0.0].sum(),
                curvature[curvature < 0.0].sum(),
                curvature.sum(),
                np.max(curvature),
                np.min(curvature),
            )

        (
            abs_curvature,
            pos_curvature,
            neg_curvature,
            sum_curvature,
            max_curvature,
            min_curvature,
        ) = cal_curvature(data.label_scene.ego_path_info.future_path)

        basic_path_tag = BasicPathTag()
        basic_path_tag.valid_path_len = len(
            data.label_scene.ego_path_info.future_path
        )
        basic_path_tag.sum_path_curvature = sum_curvature
        basic_path_tag.abs_sum_path_curvature = abs_curvature
        basic_path_tag.pos_sum_path_curvature = pos_curvature
        basic_path_tag.neg_sum_path_curvature = neg_curvature
        basic_path_tag.max_path_curvature = max_curvature
        basic_path_tag.min_path_curvature = min_curvature

        return basic_path_tag

    def judge_path_type(self, data: TagData, params: Dict) -> FuturePATHType:

        lane_seq_info = data.label_scene.ego_obs_lane_seq_info
        ego_path_info = data.label_scene.ego_path_info

        always_on_current_lane_seq = []
        for lane_seq in lane_seq_info.current_lane_seqs:
            on_curr_lane_seq = False
            for corr_lane_ids, in_junction_id in zip(
                ego_path_info.corr_lane_id, ego_path_info.in_junction_id
            ):
                if corr_lane_ids is None:
                    break
                    
                if len(corr_lane_ids) == 0:
                    continue

                if any(
                    [lane_id in lane_seq for lane_id, pose_l in corr_lane_ids]
                ):
                    on_curr_lane_seq = True
                else:
                    on_curr_lane_seq = False

            if on_curr_lane_seq:
                always_on_current_lane_seq.append(lane_seq)

        arrive_on_nearby_lane_seq = []
        for lane_seq in lane_seq_info.nearby_lane_seqs:
            on_curr_lane_seq = False
            for corr_lane_ids, in_junction_id in zip(
                ego_path_info.corr_lane_id, ego_path_info.in_junction_id
            ):
                if corr_lane_ids is None:
                    break

                if len(corr_lane_ids) == 0:
                    continue
                
                if any(
                    [lane_id in lane_seq for lane_id, pose_l in corr_lane_ids]
                ):
                    on_curr_lane_seq = True
                else:
                    on_curr_lane_seq = False

            if on_curr_lane_seq:
                arrive_on_nearby_lane_seq.append(lane_seq)

        if any(
            [
                in_junction_id is not None
                for in_junction_id in data.label_scene.ego_path_info.in_junction_id
            ]
        ):
            if len(always_on_current_lane_seq) > 0:
                return (
                    FuturePATHType.CROSS_JUNCTION_CRUISE,
                    always_on_current_lane_seq,
                    arrive_on_nearby_lane_seq,
                )
            elif len(arrive_on_nearby_lane_seq) > 0:
                return (
                    FuturePATHType.CROSS_JUNCTION_LC,
                    always_on_current_lane_seq,
                    arrive_on_nearby_lane_seq,
                )
            else:
                return (
                    FuturePATHType.CROSS_JUNCTION_UNKNWON,
                    always_on_current_lane_seq,
                    arrive_on_nearby_lane_seq,
                )

        if len(always_on_current_lane_seq) > 0:
            return (
                FuturePATHType.CRUISE,
                always_on_current_lane_seq,
                arrive_on_nearby_lane_seq,
            )
        elif len(arrive_on_nearby_lane_seq) > 0:
            return (
                FuturePATHType.LANE_CHANGE,
                always_on_current_lane_seq,
                arrive_on_nearby_lane_seq,
            )
        else:
            return (
                FuturePATHType.UNKNOWN,
                always_on_current_lane_seq,
                arrive_on_nearby_lane_seq,
            )

    def get_dash_boundary_length(
        self, lane: Dict, is_entry_lane: bool
    ) -> List[float]:
        left_boundary_type = (
            lane["left_boundary"]["boundary_type"][::-1]
            if is_entry_lane
            else lane["left_boundary"]["boundary_type"]
        )
        right_boundary_type = (
            lane["right_boundary"]["boundary_type"][::-1]
            if is_entry_lane
            else lane["right_boundary"]["boundary_type"]
        )

        left_dash_length = 0
        right_dash_length = 0

        for left_type in left_boundary_type:
            if left_type[1][0] == "SOLID":
                break
            left_dash_length = abs(left_type[0] - left_boundary_type[0][0])

        for right_type in right_boundary_type:
            if right_type[1][0] == "SOLID":
                break
            right_dash_length = abs(right_type[0] - right_boundary_type[0][0])

        return [left_dash_length, right_dash_length]
