from typing import Dict, List, Tuple, Set, Union
from shapely.geometry import LineString, Point
from base import TagData
import numpy as np
from tag_functions.high_value_scene.common.tag_type import (
    DeadCarBypassTag,
)
from tag_functions.high_value_scene.common.basic_info import BasicInfo


class DeadCarBypassTagHelper:
    def __init__(
        self,
        bypass_index_range_curb: int = 2,
        bypass_index_range_static_obs: int = 15,
        bypass_index_range_moving_obs: int = 15,
        moving_obs_index_window: int = 50,
        in_road_moving_obs_dist_thr: float = 0.6,
        in_road_static_obs_dist_thr: float = 0.6,
        in_road_curb_dist_thr: float = 0.6,
        in_junction_static_obs_dist_thr: float = 0.6,
        curvature_thr: float = 0.05,
        path_point_min_dist_to_condition: float = 0.7,
        path_point_max_dist_to_condition: float = 6.0,
    ):
        # 判断当前点/当前时刻是否绕行时，考虑的前后索引范围,每个类别阈值可不同
        self.bypass_index_range_curb = bypass_index_range_curb
        self.bypass_index_range_static_obs = bypass_index_range_static_obs
        self.bypass_index_range_moving_obs = bypass_index_range_moving_obs

        # 不考虑超过该时间阈值的交互动目标
        self.moving_obs_index_window = moving_obs_index_window

        # 道路上绕行 与障碍物的距离阈值，每个类别阈值可不同
        self.in_road_moving_obs_dist_thr = in_road_moving_obs_dist_thr
        self.in_road_static_obs_dist_thr = in_road_static_obs_dist_thr
        self.in_road_curb_dist_thr = in_road_curb_dist_thr

        # 路口内绕行 与静态障碍物的距离阈值
        self.in_junction_static_obs_dist_thr = in_junction_static_obs_dist_thr

        # 判断绕行时的曲率阈值
        self.curvature_thr = curvature_thr

        # path点偏离condition的距离阈值
        self.path_point_min_dist_to_condition = path_point_min_dist_to_condition
        self.path_point_max_dist_to_condition = path_point_max_dist_to_condition

    def is_distance_within_range(
        self, distances: List[List[float]], dist_thr: float
    ) -> List[List[bool]]:
        return [[d < dist_thr for d in dist] for dist in distances]

    def is_future_path_point_near_obstacle(
        self,
        future_interaction_with_moving_obs: List[List[bool]],
        future_path_nearest_curb_states: List[List[bool]],
        future_path_nearest_static_obs_states: List[List[bool]],
        corr_frame_idx: List[Union[None, int]],
        future_path_idx: int,
        is_left: int,
    ) -> bool:
        future_time_idx = corr_frame_idx[future_path_idx]

        consider_curb = future_path_nearest_curb_states[
            max(0, future_path_idx - self.bypass_index_range_curb) : min(
                len(future_path_nearest_curb_states),
                future_path_idx + self.bypass_index_range_curb,
            )
        ]

        consider_obs_static = future_path_nearest_static_obs_states[
            max(0, future_path_idx - self.bypass_index_range_static_obs) : min(
                len(future_path_nearest_static_obs_states),
                future_path_idx + self.bypass_index_range_static_obs,
            )
        ]

        if future_time_idx <= self.moving_obs_index_window:
            consider_obs_moving = future_interaction_with_moving_obs[
                max(
                    0, future_time_idx - self.bypass_index_range_moving_obs
                ) : min(
                    len(future_interaction_with_moving_obs),
                    future_time_idx + self.bypass_index_range_moving_obs,
                )
            ]
        else:
            consider_obs_moving = []

        if any(curb[is_left] for curb in consider_curb):
            return True
        if any(obs[is_left] for obs in consider_obs_static):
            return True
        if any(obs[is_left] for obs in consider_obs_moving):
            return True

        return False

    def is_bypass_dead_car_in_road(
        self,
        future_path: List[Tuple[float, float]],
        corr_frame_idx: List[Union[None, int]],
        basic_info: BasicInfo,
    ) -> Tuple[bool, int]:
        future_path_points_sl_coordinate_projected_to_condition = (
            basic_info.future_path_points_sl_coordinate_projected_to_condition
        )
        future_interaction_with_moving_obs = self.is_distance_within_range(
            basic_info.future_path_nearest_moving_obs_dist,
            self.in_road_moving_obs_dist_thr,
        )
        future_path_nearest_curb_states = self.is_distance_within_range(
            basic_info.future_path_nearest_curb_dist, self.in_road_curb_dist_thr
        )
        future_path_nearest_static_obs_states = self.is_distance_within_range(
            basic_info.future_path_nearest_static_obs_dist,
            self.in_road_static_obs_dist_thr,
        )

        is_bypass_dead_car = False
        first_bypass_dead_car_in_road_ind = -1

        if len(future_path_points_sl_coordinate_projected_to_condition) == len(
            future_path
        ):
            for idx, _ in enumerate(future_path):
                if (
                    idx < len(basic_info.future_path_curvature)
                    and abs(basic_info.future_path_curvature[idx])
                    < self.curvature_thr
                ):
                    continue

                (
                    proj_s,
                    proj_l,
                    corr_point,
                ) = future_path_points_sl_coordinate_projected_to_condition[idx]
                if proj_s is None or proj_l is None:
                    continue

                if abs(proj_l) > self.path_point_max_dist_to_condition:
                    continue

                if (
                    proj_l > self.path_point_min_dist_to_condition
                    and basic_info.future_path_turn_type[idx] < 0
                    and self.is_future_path_point_near_obstacle(
                        future_interaction_with_moving_obs,
                        future_path_nearest_curb_states,
                        future_path_nearest_static_obs_states,
                        corr_frame_idx,
                        idx,
                        1,
                    )
                ) or (
                    proj_l < -self.path_point_min_dist_to_condition
                    and basic_info.future_path_turn_type[idx] > 0
                    and self.is_future_path_point_near_obstacle(
                        future_interaction_with_moving_obs,
                        future_path_nearest_curb_states,
                        future_path_nearest_static_obs_states,
                        corr_frame_idx,
                        idx,
                        0,
                    )
                ):
                    is_bypass_dead_car = True
                    first_bypass_dead_car_in_road_ind = idx
                    break

        return is_bypass_dead_car, first_bypass_dead_car_in_road_ind

    def is_bypass_dead_car_in_junction(
        self,
        future_path: List[Tuple[float, float]],
        in_junction_id: List[int],
        basic_info: BasicInfo,
    ) -> Tuple[bool, int]:
        future_path_nearest_static_obs_dist = (
            basic_info.future_path_nearest_static_obs_dist
        )

        if len(future_path) == len(in_junction_id):
            for idx, point in enumerate(future_path):
                if in_junction_id[idx] is None:
                    continue

                if (
                    abs(basic_info.future_path_curvature[idx])
                    < self.curvature_thr
                ):
                    continue

                if basic_info.future_path_turn_type[idx] > 0:
                    is_turn_right = 0
                else:
                    is_turn_right = 1

                if (
                    future_path_nearest_static_obs_dist[idx][is_turn_right]
                    < self.in_junction_static_obs_dist_thr
                ):
                    return True, idx

        return False, -1


def label_dead_car_bypass_tag(
    data: TagData, params: Dict, basic_info: BasicInfo
) -> DeadCarBypassTag:
    future_path = data.label_scene.ego_path_info.future_path
    in_junction_id = data.label_scene.ego_path_info.in_junction_id
    corr_frame_idx = data.label_scene.ego_path_info.corr_frame_idx

    dead_car_bypass_tag = DeadCarBypassTag()
    dead_car_bypass_tag_helper = DeadCarBypassTagHelper(
        bypass_index_range_curb=params["dead_car_bypass_tag"][
            "bypass_index_range_curb"
        ],
        bypass_index_range_static_obs=params["dead_car_bypass_tag"][
            "bypass_index_range_static_obs"
        ],
        bypass_index_range_moving_obs=params["dead_car_bypass_tag"][
            "bypass_index_range_moving_obs"
        ],
        moving_obs_index_window=params["dead_car_bypass_tag"][
            "moving_obs_index_window"
        ],
        in_road_moving_obs_dist_thr=params["dead_car_bypass_tag"][
            "in_road_moving_obs_dist_thr"
        ],
        in_road_static_obs_dist_thr=params["dead_car_bypass_tag"][
            "in_road_static_obs_dist_thr"
        ],
        in_road_curb_dist_thr=params["dead_car_bypass_tag"][
            "in_road_curb_dist_thr"
        ],
        in_junction_static_obs_dist_thr=params["dead_car_bypass_tag"][
            "in_junction_static_obs_dist_thr"
        ],
        curvature_thr=params["dead_car_bypass_tag"]["curvature_thr"],
        path_point_min_dist_to_condition=params["dead_car_bypass_tag"][
            "path_point_min_dist_to_condition"
        ],
        path_point_max_dist_to_condition=params["dead_car_bypass_tag"][
            "path_point_max_dist_to_condition"
        ],
    )

    # 判断是否有非路口绕行
    (
        dead_car_bypass_tag.is_bypass_dead_car_in_road,
        dead_car_bypass_tag.first_bypass_dead_car_in_road_ind,
    ) = dead_car_bypass_tag_helper.is_bypass_dead_car_in_road(
        future_path, corr_frame_idx, basic_info
    )

    # 判断是否有路口内绕行
    if basic_info.is_cross_junction:
        (
            dead_car_bypass_tag.is_bypass_dead_car_in_junction,
            dead_car_bypass_tag.first_bypass_dead_car_in_junction_ind,
        ) = dead_car_bypass_tag_helper.is_bypass_dead_car_in_junction(
            future_path, in_junction_id, basic_info
        )

    dead_car_bypass_tag.max_curvature_gradient = (
        basic_info.max_curvature_gradient
    )

    return dead_car_bypass_tag
