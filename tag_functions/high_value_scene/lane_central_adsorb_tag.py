from typing import Dict, List, Tuple, Set, Union
from shapely.geometry import LineString, Point
from base import TagData
import numpy as np
from tag_functions.high_value_scene.common.tag_type import (
    LaneCentralAdsorbTag,
)
from tag_functions.high_value_scene.common.basic_info import BasicInfo


class LaneCentralAdsorbTagHelper:
    def __init__(self, consider_index_range: int) -> None:
        self.consider_index_range: int = consider_index_range

    def is_future_path_point_near_obstacle(
        self,
        future_interaction_with_moving_obs: List[List[bool]],
        future_narrow_road_states: List[List[bool]],
        corr_frame_idx: List[Union[None, int]],
        future_path_idx: int,
        is_left: int,
    ) -> bool:
        future_time_idx = corr_frame_idx[future_path_idx]

        consider_obs_static = future_narrow_road_states[
            max(0, future_path_idx - self.consider_index_range) : min(
                len(future_narrow_road_states),
                future_path_idx + self.consider_index_range,
            )
        ]
        consider_obs_moving = future_interaction_with_moving_obs[
            max(0, future_time_idx - self.consider_index_range) : min(
                len(future_interaction_with_moving_obs),
                future_time_idx + self.consider_index_range,
            )
        ]

        if any(obj[is_left] for obj in consider_obs_static):
            return True
        if any(obj[is_left] for obj in consider_obs_moving):
            return True

        return False

    def lane_central_adsorb_with_nudge(
        self,
        future_path: List[Tuple[float, float]],
        corr_frame_idx: List[Union[None, int]],
        basic_info: BasicInfo,
    ) -> List[Tuple[float, float]]:
        future_path_points_sl_coordinate_projected_to_condition = (
            basic_info.future_path_points_sl_coordinate_projected_to_condition
        )
        future_interaction_with_moving_obs = (
            basic_info.future_interaction_with_moving_obs
        )
        future_narrow_road_states = basic_info.future_narrow_road_states

        adsorbed_path = [[coord for coord in pt] for pt in future_path]

        if len(future_path_points_sl_coordinate_projected_to_condition) == len(
            future_path
        ):
            for idx, point in enumerate(future_path):
                path_point = Point(point)

                (
                    proj_s,
                    proj_l,
                    corr_point,
                ) = future_path_points_sl_coordinate_projected_to_condition[idx]
                if proj_s is None or abs(proj_l) > 1:
                    continue

                # 判断横向是否有障碍物导致侧偏
                is_lat_deviation = (
                    proj_l > 0
                    and self.is_future_path_point_near_obstacle(
                        future_interaction_with_moving_obs,
                        future_narrow_road_states,
                        corr_frame_idx,
                        idx,
                        1,
                    )
                ) or (
                    proj_l < 0
                    and self.is_future_path_point_near_obstacle(
                        future_interaction_with_moving_obs,
                        future_narrow_road_states,
                        corr_frame_idx,
                        idx,
                        0,
                    )
                )

                # 吸附
                if not is_lat_deviation:
                    vec = np.array(
                        [
                            corr_point.x - path_point.x,
                            corr_point.y - path_point.y,
                        ]
                    )
                    ratio = (1 - (abs(proj_l) - 0.25) / 0.75) ** 2
                    adsorbed_path[idx] = (
                        [corr_point.x, corr_point.y]
                        if abs(proj_l) < 0.25
                        else [
                            point[0] + vec[0] * ratio,
                            point[1] + vec[1] * ratio,
                        ]
                    )
        return adsorbed_path


def label_lane_central_adsorb_tag(
    data: TagData, basic_info: BasicInfo
) -> LaneCentralAdsorbTag:
    future_path = data.label_scene.ego_path_info.future_path
    corr_frame_idx = data.label_scene.ego_path_info.corr_frame_idx
    lane_central_adsorb_tag = LaneCentralAdsorbTag()
    lane_central_adsorb_tag_helper = LaneCentralAdsorbTagHelper(
        consider_index_range=15
    )

    # 向condition_linestring吸附，同时考虑有障碍物时的侧偏
    lane_central_adsorb_tag.adsorbed_path = (
        lane_central_adsorb_tag_helper.lane_central_adsorb_with_nudge(
            future_path, corr_frame_idx, basic_info
        )
    )

    return lane_central_adsorb_tag
