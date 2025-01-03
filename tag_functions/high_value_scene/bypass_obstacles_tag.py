from typing import Dict, List, Tuple, Set, Union
from shapely.geometry import LineString, Point
from base import TagData
import numpy as np
from tag_functions.high_value_scene.common.tag_type import (
    BypassObstaclesTag,
)
from tag_functions.high_value_scene.common.basic_info import BasicInfo


def is_distance_within_range(
    distances: List[List[float]], dist_thr: float
) -> List[List[bool]]:
    return [[d < dist_thr for d in dist] for dist in distances]


def is_future_path_point_near_obstacle(
    future_interaction_with_moving_obs: List[List[bool]],
    future_narrow_road_states: List[List[bool]],
    corr_frame_idx: List[Union[None, int]],
    future_path_idx: int,
    is_left: int,
) -> bool:
    consider_index_range = 15
    future_time_idx = corr_frame_idx[future_path_idx]

    consider_obs_static = future_narrow_road_states[
        max(0, future_path_idx - consider_index_range) : min(
            len(future_narrow_road_states),
            future_path_idx + consider_index_range,
        )
    ]

    if future_time_idx <= 50:
        consider_obs_moving = future_interaction_with_moving_obs[
            max(0, future_time_idx - consider_index_range) : min(
                len(future_interaction_with_moving_obs),
                future_time_idx + consider_index_range,
            )
        ]
    else:
        consider_obs_moving = []

    if any(obj[is_left] for obj in consider_obs_static):
        return True
    if any(obj[is_left] for obj in consider_obs_moving):
        return True

    return False


def is_bypass_obstacles_in_road(
    future_path: List[Tuple[float, float]],
    corr_frame_idx: List[Union[None, int]],
    basic_info: BasicInfo,
) -> Tuple[bool, int]:
    future_path_points_sl_coordinate_projected_to_condition = (
        basic_info.future_path_points_sl_coordinate_projected_to_condition
    )

    future_interaction_with_moving_obs = is_distance_within_range(
        basic_info.future_path_nearest_moving_obs_dist, 0.6
    )
    future_path_nearest_curb_states = is_distance_within_range(
        basic_info.future_path_nearest_curb_dist, 0.6
    )
    future_path_nearest_static_obs_states = is_distance_within_range(
        basic_info.future_path_nearest_static_obs_dist, 0.6
    )
    future_interaction_with_static_obj = [
        [a or b for a, b in zip(state1, state2)]
        for state1, state2 in zip(
            future_path_nearest_curb_states,
            future_path_nearest_static_obs_states,
        )
    ]
    future_path_curvature = basic_info.future_path_curvature
    future_path_turn_type = basic_info.future_path_turn_type

    is_lat_deviation = False
    first_bypass_obs_in_road_ind = -1

    if len(future_path_points_sl_coordinate_projected_to_condition) == len(
        future_path
    ):
        for idx, _ in enumerate(future_path):
            # path_point = Point(point)
            if (
                idx < len(future_path_curvature)
                and abs(future_path_curvature[idx]) < 0.05
            ):
                continue

            (
                proj_s,
                proj_l,
                corr_point,
            ) = future_path_points_sl_coordinate_projected_to_condition[idx]
            if proj_s is None or proj_l is None:
                continue

            # 判断横向是否有障碍物导致侧偏
            if (
                proj_l > 0.7
                and future_path_turn_type[idx] < 0
                and is_future_path_point_near_obstacle(
                    future_interaction_with_moving_obs,
                    future_interaction_with_static_obj,
                    corr_frame_idx,
                    idx,
                    1,
                )
            ) or (
                proj_l < -0.7
                and future_path_turn_type[idx] > 0
                and is_future_path_point_near_obstacle(
                    future_interaction_with_moving_obs,
                    future_interaction_with_static_obj,
                    corr_frame_idx,
                    idx,
                    0,
                )
            ):
                is_lat_deviation = True
                first_bypass_obs_in_road_ind = idx
                break

    return is_lat_deviation, first_bypass_obs_in_road_ind


def is_bypass_obstacles_in_junction(
    future_path: List[Tuple[float, float]],
    in_junction_id: List[int],
    corr_frame_idx: List[Union[None, int]],
    basic_info: BasicInfo,
) -> Tuple[bool, int]:

    future_path_curvature = basic_info.future_path_curvature
    future_path_turn_type = basic_info.future_path_turn_type
    future_path_nearest_moving_obs_dist = (
        basic_info.future_path_nearest_moving_obs_dist
    )
    future_path_nearest_static_obs_dist = (
        basic_info.future_path_nearest_static_obs_dist
    )
    future_path_nearest_curb_dist = basic_info.future_path_nearest_curb_dist

    if len(future_path) == len(in_junction_id):
        for idx, point in enumerate(future_path):
            if in_junction_id[idx] is None:
                continue

            if abs(future_path_curvature[idx]) < 0.05:
                continue

            if future_path_turn_type[idx] > 0:
                is_right = 0
            else:
                is_right = 1

            if future_path_nearest_static_obs_dist[idx][is_right] < 0.6:
                return True, idx

            if future_path_nearest_curb_dist[idx][is_right] < 0.6:
                return True, idx

            if (
                idx < len(corr_frame_idx)
                and corr_frame_idx[idx] <= 50
                and corr_frame_idx[idx]
                < len(future_path_nearest_moving_obs_dist)
            ):
                if (
                    future_path_nearest_moving_obs_dist[corr_frame_idx[idx]][
                        is_right
                    ]
                    < 0.6
                ):
                    return True, idx

    return False, -1


def label_bypass_obstacles_tag(
    data: TagData, basic_info: BasicInfo
) -> BypassObstaclesTag:
    future_path = data.label_scene.ego_path_info.future_path
    in_junction_id = data.label_scene.ego_path_info.in_junction_id
    corr_frame_idx = data.label_scene.ego_path_info.corr_frame_idx

    bypass_obstacles_tag = BypassObstaclesTag()

    (
        bypass_obstacles_tag.is_bypass_obstacles_in_road,
        bypass_obstacles_tag.first_bypass_obs_in_road_ind,
    ) = is_bypass_obstacles_in_road(future_path, corr_frame_idx, basic_info)

    if basic_info.is_cross_junction:
        (
            bypass_obstacles_tag.is_bypass_obstacles_in_junction,
            bypass_obstacles_tag.first_bypass_obs_in_junction_ind,
        ) = is_bypass_obstacles_in_junction(
            future_path, in_junction_id, corr_frame_idx, basic_info
        )

    # second_order_grad = np.gradient(np.gradient(np.array(future_path)))
    second_order_grad = np.gradient(basic_info.future_path_curvature)
    bypass_obstacles_tag.max_curvature_change_rate = np.max(
        np.abs(second_order_grad)
    )

    # save extra info
    # future_path_nearest_moving_obs_dist = (
    #     basic_info.future_path_nearest_moving_obs_dist
    # )

    # bypass_obstacles_tag.future_path_curvature = (
    #     basic_info.future_path_curvature.tolist()
    # )
    # bypass_obstacles_tag.future_path_nearest_curb_dist = (
    #     basic_info.future_path_nearest_curb_dist
    # )
    # bypass_obstacles_tag.future_path_nearest_static_obs_dist = (
    #     basic_info.future_path_nearest_static_obs_dist
    # )

    # nearest_moving_obs_dist = []

    # for idx, _ in enumerate(future_path):
    #     if idx < len(corr_frame_idx) and corr_frame_idx[idx] < len(
    #         future_path_nearest_moving_obs_dist
    #     ):
    #         nearest_moving_obs_dist.append(
    #             future_path_nearest_moving_obs_dist[corr_frame_idx[idx]]
    #         )
    #     else:
    #         nearest_moving_obs_dist.append([np.inf, np.inf])

    # bypass_obstacles_tag.future_path_nearest_moving_obs_dist = (
    #     nearest_moving_obs_dist
    # )

    # distance_to_condition = []
    # for (
    #     _,
    #     proj_l,
    #     _,
    # ) in basic_info.future_path_points_sl_coordinate_projected_to_condition:
    #     distance_to_condition.append(proj_l)

    # bypass_obstacles_tag.distance_to_condition = distance_to_condition

    return bypass_obstacles_tag
