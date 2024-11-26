import numpy as np

from tag_functions.high_value_scene.common.tag_type import (
    NarrowRoadTag,
)
from tag_functions.high_value_scene.common.basic_info import BasicInfo


def label_narrow_road_tag(basic_info: BasicInfo) -> NarrowRoadTag:
    narrow_road_tag = NarrowRoadTag()

    if basic_info.is_ego_vehicle_always_moving:
        narrow_road_tag.narrow_road_count = float(np.sum(
            [all(obj) for obj in basic_info.future_narrow_road_states]
        ))
        narrow_road_tag.is_narrow_road = bool(narrow_road_tag.narrow_road_count > 50)

    return narrow_road_tag
