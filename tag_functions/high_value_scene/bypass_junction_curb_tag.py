from tag_functions.high_value_scene.common.tag_type import (
    BypassJunctionCurbTag,
)
from tag_functions.high_value_scene.common.basic_info import BasicInfo


def label_bypass_junction_curb_tag(
    basic_info: BasicInfo,
) -> BypassJunctionCurbTag:
    bypass_junction_curb_tag = BypassJunctionCurbTag()

    if basic_info.is_ego_vehicle_always_moving:
        bypass_junction_curb_tag.is_bypass_junction_curb = any(
            basic_info.future_bypass_junction_curb
        )

    return bypass_junction_curb_tag
