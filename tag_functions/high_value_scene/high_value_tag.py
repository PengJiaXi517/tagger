from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from base import PercepMap, TagData
from registry import TAG_FUNCTIONS
from tag_functions.high_value_scene.narrow_road_tag import *
from tag_functions.high_value_scene.junction_bypass_tag import *
from tag_functions.high_value_scene.mixed_traffic_tag import *
from tag_functions.high_value_scene.yield_vru_tag import *
from tag_functions.high_value_scene.ramp_tag import *
from tag_functions.high_value_scene.lateral_deviation_tag import *
from tag_functions.future_path_tag import *
from tag_functions.high_value_scene.lc_valid_tag import *


@dataclass(repr=False)
class HighValueTag:
    narrow_road_tag: NarrowRoadTag = None
    junction_bypass_tag: JunctionBypassTag = None
    yield_vru_tag: YieldVRUTag = None
    mixed_traffic_tag: MixedTrafficTag = None
    ramp_tag: RampTag = None
    future_path_tag: FuturePathTag = None
    lateral_deviation_tag: LateralDeviationTag = None
    lc_valid_tag: LcValidTag = None

    def as_dict(self):
        dict1 = {
            "narrow_road_tag": self.narrow_road_tag.as_dict(),
            "junction_bypass_tag": self.junction_bypass_tag.as_dict(),
            "yield_vru_tag": self.yield_vru_tag.as_dict(),
            "mixed_traffic_tag": self.mixed_traffic_tag.as_dict(),
            "ramp_tag": self.ramp_tag.as_dict(),
            "lateral_deviation_tag": self.lateral_deviation_tag.as_dict(),
            "lc_valid_tag": self.lc_valid_tag.as_dict(),
        }
        dict1.update(self.future_path_tag.as_dict())
        return dict1


@TAG_FUNCTIONS.register()
def high_value_tag(data: TagData, params: Dict) -> Dict:
    high_value_tag = HighValueTag()
    # 判断窄路通行
    high_value_tag.narrow_road_tag = narrow_road_tag(data, params)
    # 判断路口内绕障
    high_value_tag.junction_bypass_tag = junction_bypass_tag(
        data, params, high_value_tag.narrow_road_tag
    )
    # 判断是否在礼让vru而减速
    high_value_tag.yield_vru_tag = yield_vru_tag(data, params)
    # 判断8s内是否与动目标交互
    high_value_tag.mixed_traffic_tag = mixed_traffic_tag(data, params)
    # 判断汇入汇出匝道
    high_value_tag.ramp_tag = ramp_tag(data, params)

    high_value_tag.future_path_tag = future_path_tag(data, params)

    # 输出侧偏修正后的轨迹
    high_value_tag.lateral_deviation_tag = lateral_deviation_tag(
        data,
        high_value_tag.mixed_traffic_tag,
        high_value_tag.narrow_road_tag,
        params,
    )
    # 判断变道缓急合理性
    high_value_tag.lc_valid_tag = lc_valid_tag(
        data,
        high_value_tag.mixed_traffic_tag,
        high_value_tag.narrow_road_tag,
        high_value_tag.future_path_tag,
        params,
    )
    return high_value_tag.as_dict()
