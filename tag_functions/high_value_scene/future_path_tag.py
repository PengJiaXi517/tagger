from typing import Dict, List
from base import TagData
from tag_functions.high_value_scene.hv_utils.tag_type import (
    FuturePathTag,
    FuturePATHType,
)
from tag_functions.high_value_scene.hv_utils.future_path_tag_helper import (
    FuturePathTagHelper,
)


def label_future_path_tag(data: TagData, params: Dict) -> FuturePathTag:
    future_path_tag = FuturePathTag()
    future_path_tag_helper = FuturePathTagHelper()
    percep_map = data.label_scene.percepmap

    # Judge Type
    (
        future_path_tag.path_type,
        always_on_current_lane_seq,
        arrive_on_nearby_lane_seq,
    ) = future_path_tag_helper.judge_path_type(data, params)

    future_path_tag.basic_path_tag = future_path_tag_helper.label_basic_tag(
        data, params
    )

    if future_path_tag.path_type in [
        FuturePATHType.CRUISE,
        FuturePATHType.CROSS_JUNCTION_CRUISE,
    ]:
        future_path_tag.cruise_path_tag = (
            future_path_tag_helper.label_cruise_tag(
                data,
                params,
                always_on_current_lane_seq,
                percep_map,
                params["sample_point_length"],
            )
        )

    elif future_path_tag.path_type in [
        FuturePATHType.LANE_CHANGE,
        FuturePATHType.CROSS_JUNCTION_LC,
    ]:
        future_path_tag.lc_path_tag = future_path_tag_helper.label_lc_tag(
            data,
            params,
            arrive_on_nearby_lane_seq,
            percep_map,
            params["sample_point_length"],
        )

    if future_path_tag.path_type in [
        FuturePATHType.CROSS_JUNCTION_CRUISE,
        FuturePATHType.CROSS_JUNCTION_LC,
        FuturePATHType.CROSS_JUNCTION_UNKNWON,
    ]:
        future_path_tag.junction_path_tag = (
            future_path_tag_helper.label_junction_tag(
                data,
                params,
                percep_map,
                params["sample_point_length"],
            )
        )

    future_path_tag.condition_res_tag = (
        future_path_tag_helper.label_condition_res_tag(data, params)
    )

    future_path_tag.is_backing_up = future_path_tag_helper.label_backing_up_tag(
        data, params
    )

    return future_path_tag
