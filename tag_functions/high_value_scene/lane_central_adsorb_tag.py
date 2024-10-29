from typing import Dict, List, Tuple, Set, Union
from shapely.geometry import LineString, Point
from base import TagData
import numpy as np
from tag_functions.high_value_scene.hv_utils.tag_type import (
    NarrowRoadTag,
    InteractWithMovingObsTag,
    LaneCentralAdsorbTag,
    FuturePathTag,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    xy_to_sl,
)


class LaneCentralAdsorbTagHelper:
    def __init__(self) -> None:
        self.consider_index_range: int = 15

    def is_future_path_point_near_obstacle(
        self,
        interact_with_moving_obs_tag: InteractWithMovingObsTag,
        narrow_road_tag: NarrowRoadTag,
        corr_frame_idx: List[Union[None, int]],
        future_path_idx: int,
        is_left: int,
    ) -> bool:
        future_time_idx = corr_frame_idx[future_path_idx]
        future_narrow_road_states = narrow_road_tag.future_narrow_road_states
        future_interaction_with_moving_obs = (
            interact_with_moving_obs_tag.future_interaction_with_moving_obs
        )
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
        nearest_condition_linestring: List[LineString],
        interact_with_moving_obs_tag: InteractWithMovingObsTag,
        narrow_road_tag: NarrowRoadTag,
    ) -> List[Tuple[float, float]]:
        adsorbed_path = [[coord for coord in pt] for pt in future_path]

        for idx, point in enumerate(future_path):
            path_point = Point(point)
            for linestring in nearest_condition_linestring:

                # 计算sl坐标
                proj_s, proj_l = xy_to_sl(linestring, path_point)
                if proj_s is None or abs(proj_l) > 1:
                    continue

                # 判断横向是否有障碍物导致侧偏
                is_lat_deviation = (
                    proj_l > 0
                    and self.is_future_path_point_near_obstacle(
                        interact_with_moving_obs_tag,
                        narrow_road_tag,
                        corr_frame_idx,
                        idx,
                        1,
                    )
                ) or (
                    proj_l < 0
                    and self.is_future_path_point_near_obstacle(
                        interact_with_moving_obs_tag,
                        narrow_road_tag,
                        corr_frame_idx,
                        idx,
                        0,
                    )
                )

                # 吸附
                if not is_lat_deviation:
                    corr_point = linestring.interpolate(proj_s)
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
                break
        return adsorbed_path


def label_lane_central_adsorb_tag(
    data: TagData,
    params: Dict,
    interact_with_moving_obs_tag: InteractWithMovingObsTag,
    narrow_road_tag: NarrowRoadTag,
    future_path_tag: FuturePathTag,
) -> LaneCentralAdsorbTag:
    future_path = data.label_scene.ego_path_info.future_path
    corr_frame_idx = data.label_scene.ego_path_info.corr_frame_idx
    lane_central_adsorb_tag = LaneCentralAdsorbTag(future_path)
    lane_central_adsorb_tag_helper = LaneCentralAdsorbTagHelper()

    # 获取离future path横向距离最近的condtion_lane_seq linestring
    nearest_condition_linestring = (
        future_path_tag.condition_res_tag.nearest_condition_linestring
    )

    if nearest_condition_linestring is None:
        return lane_central_adsorb_tag

    # 向condition_linestring吸附，同时考虑有障碍物时的侧偏
    lane_central_adsorb_tag.adsorbed_path = (
        lane_central_adsorb_tag_helper.lane_central_adsorb_with_nudge(
            future_path,
            corr_frame_idx,
            nearest_condition_linestring,
            interact_with_moving_obs_tag,
            narrow_road_tag,
        )
    )

    return lane_central_adsorb_tag
