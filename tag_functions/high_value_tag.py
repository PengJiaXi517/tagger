from typing import Dict
from base import TagData
from registry import TAG_FUNCTIONS
from tag_functions.high_value_scene.common.tag_type import HighValueTag
from tag_functions.high_value_scene.narrow_road_tag import label_narrow_road_tag
from tag_functions.high_value_scene.bypass_junction_curb_tag import (
    label_bypass_junction_curb_tag,
)
from tag_functions.high_value_scene.interact_with_moving_obs import (
    label_interact_with_moving_obs_tag,
)
from tag_functions.high_value_scene.yield_vru_tag import label_yield_vru_tag
from tag_functions.high_value_scene.ramp_tag import label_ramp_tag
from tag_functions.high_value_scene.dead_car_bypass_tag import (
    label_dead_car_bypass_tag,
)
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
from tag_functions.high_value_scene.common.basic_info_generator import (
    BasicInfoGenerartor,
)


@TAG_FUNCTIONS.register()
def label_high_value_tag(data: TagData, params: Dict) -> Dict:
    high_value_tag = HighValueTag()

    # 提前计算基础信息
    basic_info_generator = BasicInfoGenerartor()
    basic_info = basic_info_generator.process(data, params)

    # 判断future path的各种属性(变道、曲率等)
    high_value_tag.future_path_tag = label_future_path_tag(data, params)

    # 判断窄路通行
    high_value_tag.narrow_road_tag = label_narrow_road_tag(basic_info)

    # 判断路口内绕curb或静态障碍物
    high_value_tag.bypass_junction_curb_tag = label_bypass_junction_curb_tag(
        data, basic_info
    )

    # 判断未来8s内是否与动目标距离较近
    high_value_tag.interact_with_moving_obs_tag = (
        label_interact_with_moving_obs_tag(basic_info)
    )

    # 判断是否绕行死车等障碍物
    high_value_tag.dead_car_bypass_tag = label_dead_car_bypass_tag(
        data, params, basic_info
    )

    # 结合动静障碍物距离，输出向中心线吸附的future path
    high_value_tag.lane_central_adsorb_tag = label_lane_central_adsorb_tag(
        data, basic_info
    )

    # 结合动静障碍物距离，判断变道缓急的合理性
    high_value_tag.quick_lane_change_tag = label_quick_lane_change_tag(
        data, basic_info, high_value_tag.future_path_tag
    )

    # 判断右转专用道; 出右转专用道时，记录变道前的valid length
    high_value_tag.right_turn_only_tag = label_right_turn_only_tag(
        data, basic_info, high_value_tag.future_path_tag
    )

    # 判断是否在礼让vru而减速
    high_value_tag.yield_vru_tag = label_yield_vru_tag(data, params)

    # 判断汇入汇出匝道
    high_value_tag.ramp_tag = label_ramp_tag(data, basic_info)

    # future path曲率绝对值的最大值
    high_value_tag.max_abs_path_curvature = basic_info.max_abs_path_curvature

    return high_value_tag.as_dict()
