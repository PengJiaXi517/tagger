from tag_functions.high_value_scene.common.tag_type import (
    InteractWithMovingObsTag,
)
from tag_functions.high_value_scene.common.basic_info import BasicInfo


def label_interact_with_moving_obs_tag(
    basic_info: BasicInfo,
) -> InteractWithMovingObsTag:
    interact_with_moving_obs_tag = InteractWithMovingObsTag()

    if basic_info.is_ego_vehicle_always_moving:
        interact_with_moving_obs_tag.is_interact_with_moving_obs = any(
            [any(obj) for obj in basic_info.future_interaction_with_moving_obs]
        )

    return interact_with_moving_obs_tag
