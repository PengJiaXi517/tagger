from typing import Dict
from base import TagData
from registry import TAG_FUNCTIONS
from tag_functions.high_value_scene.hv_utils.tag_type import HighValueTag
from tag_functions.high_value_scene.narrow_road_tag import label_narrow_road_tag
from tag_functions.high_value_scene.bypass_junction_curb_tag import (
    label_bypass_junction_curb_tag,
)
from tag_functions.high_value_scene.interact_with_moving_obs import (
    label_interact_with_moving_obs_tag,
)
from tag_functions.high_value_scene.yield_vru_tag import label_yield_vru_tag
from tag_functions.high_value_scene.ramp_tag import label_ramp_tag
from tag_functions.high_value_scene.lane_central_adsorb_tag import (
    label_lane_central_adsorb_tag,
)
from tag_functions.high_value_scene.future_path_tag import label_future_path_tag
from tag_functions.high_value_scene.quick_lane_change_tag import (
    label_quick_lane_change_tag,
)
from tag_functions.high_value_scene.right_turn_only_tag import (
    label_right_turn_only_tag,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    future_path_validity_check,
)


@TAG_FUNCTIONS.register()
def label_high_value_tag(data: TagData, params: Dict) -> Dict:
    high_value_tag = HighValueTag()

    if future_path_validity_check(data):
        # 判断窄路通行
        high_value_tag.narrow_road_tag = label_narrow_road_tag(data, params)

        # 判断路口内绕curb或静态障碍物
        high_value_tag.bypass_junction_curb_tag = (
            label_bypass_junction_curb_tag(
                data, params, high_value_tag.narrow_road_tag
            )
        )

        # 判断是否在礼让vru而减速
        high_value_tag.yield_vru_tag = label_yield_vru_tag(data, params)

        # 判断未来8s内是否与动目标距离较近
        high_value_tag.interact_with_moving_obs_tag = (
            label_interact_with_moving_obs_tag(data, params)
        )

        # 判断汇入汇出匝道
        high_value_tag.ramp_tag = label_ramp_tag(data, params)

        # 判断future path的各种属性(变道、曲率等)
        high_value_tag.future_path_tag = label_future_path_tag(data, params)

        # 判断右转专用道; 出右转专用道时，记录变道前的valid length
        high_value_tag.right_turn_only_tag = label_right_turn_only_tag(
            data, params, high_value_tag.future_path_tag
        )

        # 结合动静障碍物距离的打标结果，输出向中心线吸附的future path
        high_value_tag.lane_central_adsorb_tag = label_lane_central_adsorb_tag(
            data,
            params,
            high_value_tag.interact_with_moving_obs_tag,
            high_value_tag.narrow_road_tag,
            high_value_tag.future_path_tag
        )

        # 结合动静障碍物距离的打标结果，判断变道缓急的合理性
        high_value_tag.quick_lane_change_tag = label_quick_lane_change_tag(
            data,
            params,
            high_value_tag.interact_with_moving_obs_tag,
            high_value_tag.narrow_road_tag,
            high_value_tag.future_path_tag,
        )

    return high_value_tag.as_dict()
