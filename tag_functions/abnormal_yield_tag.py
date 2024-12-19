import numpy as np
from base import TagData
from typing import Dict, Union, List, Tuple
from dataclasses import dataclass, field
from registry import TAG_FUNCTIONS

from shapely.geometry import LineString, Point

from base import LaneLightsInfo, LaneLightInfo, JunctionLabelInfo


@dataclass(repr=True)
class AbnormalYieldTag:
    is_still: bool = False
    ego_acc_1s: float = None
    ego_acc_2s: float = None
    dis_to_stopline: int = 150
    nearest_front_car_distance: float = None
    nearest_front_car_vel: float = None
    nearest_front_car_length: float = None
    cutin_in_5_seconds_distance: float = 0.0
    cutin_in_5_seconds_width: float = 0.0
    has_pass_signal: List[int] = field(default_factory=list)
    has_yield_signal: List[int] = field(default_factory=list)
    pass_type: int = None
    
    def as_dict(self):
        return {
            "abnormal_yield_tag": {
                "is_still": self.is_still,
                "ego_acc_1s": self.ego_acc_1s,
                "ego_acc_2s": self.ego_acc_2s,
                "dis_to_stopline": self.dis_to_stopline,
                "nearest_front_car_distance": self.nearest_front_car_distance,
                "nearest_front_car_vel": self.nearest_front_car_vel,
                "nearest_front_car_length": self.nearest_front_car_length,
                "cutin_in_5_seconds_distance": self.cutin_in_5_seconds_distance,
                "cutin_in_5_seconds_width": self.cutin_in_5_seconds_width,
                "has_pass_signal": self.has_pass_signal,
                "has_yield_signal": self.has_yield_signal,
                "pass_type": self.pass_type,
            }
        }


def get_nearest_front_car(obstacles) -> Union[int, None]:
    """
    :param obstacles: data.label_scene.label_res["obstacles"]
    :return None if no front car else front car OBSID
    """
    # init
    obs_info = np.empty(
        (0, 5),
        dtype=[
            ("s", float),
            ("l", float),
            ("key", int),
            ("v", float),
            ("length", float),
        ],
    )
    ego_s = None

    # forall obs
    for key in obstacles:
        if key == -9:
            continue

        # get obs_v
        obs_v = np.sqrt(
            obstacles[key]["features"]["history_states"][-1]["vx"] ** 2
            + obstacles[key]["features"]["history_states"][-1]["vy"] ** 2
        )

        # append obs_info
        obs_info = np.append(
            obs_info,
            np.array(
                [
                    (obstacles[key]["decision"]["obs_s"],
                    obstacles[key]["decision"]["obs_l"],
                    key,
                    obs_v,
                    obstacles[key]["features"]["length"]),
                ],
                dtype=obs_info.dtype,
            ),
        )

        # get ego_s only once
        if ego_s is None and "ego_s" in obstacles[key]["decision"]:
            ego_s = obstacles[key]["decision"]["ego_s"]

    # relative obs s
    if ego_s is None:
        return None
    obs_info["s"] = obs_info["s"] - ego_s

    # filter obs
    filtered_obs_info = obs_info[(np.abs(obs_info["l"]) < 1.5) & (obs_info["s"] > 0)]

    # find argmin obs_s
    if len(filtered_obs_info) > 0:
        min_obs = filtered_obs_info[np.argmin(filtered_obs_info["s"])]
        return min_obs
    else:
        return None


def get_nearest_cutin_cat(obstacles, ego_path_linestring: LineString):
    ret = {
        "s": None,
        "w": None,
    }

    for obs_id, obs in obstacles.items():
        if obs_id == -9:
            continue

        obs_decision = obs["decision"]
        interact_lon_type = obs_decision["interactive_lon_type"]
        
        if interact_lon_type != 2:
            continue

        obs_ttc = obs_decision["obs_ttc"]

        if 0 < obs_ttc <= 50:
            future_states = obs["future_trajectory"]["future_states"]

            if obs_ttc > len(future_states):
                continue

            future_state = future_states[obs_ttc - 1]

            future_point = Point(future_state["x"], future_state["y"])

            collision_s = ego_path_linestring.project(future_point)

            if collision_s <= 0 or collision_s >= ego_path_linestring.length:
                continue

            if ret["s"] is None or collision_s < ret["s"]:
                ret["s"] = float(collision_s)
                ret["w"] = float(obs["features"]["width"])
    
    return ret


def check_has_pass_yield_signal(junction_label_info: JunctionLabelInfo, lane_lights_info: LaneLightsInfo):
    has_yield_signal = []
    has_pass_signal = []

    goal = junction_label_info.junction_goal
    """
    goal:  str
        UNKNOWN
        TYPE_FORWARD
        TYPE_TURN_LEFT
        TYPE_TURN_RIGHT
        TYPE_UTURN
    color:  int
        enum Color {  
        UNKNOWN = 0; 
        RED = 1;  
        RED_FLASH = 2;  
        YELLOW = 3;  
        YELLOW_FLASH = 4;  
        GREEN = 5;  
        GREEN_FLASH = 6;  
        BLACK = 7;};
    
    navi_type: int
        enum Type {
        FORWARD = 0;
        TURN_LEFT = 1;
        TURN_RIGHT = 2;
        UTURN_LEFT = 3;
        FORWARD_LEFT = 4;
        FORWARD_RIGHT = 5;
        DUMMY_FORWARD = 6};
        灯类别：直行，左转，右转，掉头，直行+左转，直行+右转，冗余灯
    
    """
    
    def is_color_yield(color):
        if color in [1, 2, 3, 4]:
            return True
        return False
    
    def is_color_passed(color):
        if color in [5, 6]:
            return True
        return False
    
    def is_navi_type_corr_goal_pass(goal, navi_type, color):
        if goal == "TYPE_FORWARD":
            if navi_type in [0, 6, 4, 5]:
                if is_color_passed(color):
                    return True
        elif goal == "TYPE_TURN_LEFT":
            if navi_type in [1, 3, 4]:
                if is_color_passed(color):
                    return True
        elif goal == "TYPE_TURN_RIGHT":
            if navi_type in [2, 5]:
                if is_color_passed(color):
                    return True
        elif goal == "TYPE_TURN_UTURN":
            if navi_type in [1, 3]:
                if is_color_passed(color):
                    return True
        
        return False
    
    def is_navi_type_corr_goal_yield(goal, navi_type, color):
        if goal == "TYPE_FORWARD":
            if navi_type in [0, 6, 4, 5]:
                if is_color_yield(color):
                    return True
        elif goal == "TYPE_TURN_LEFT":
            if navi_type in [1, 3, 4]:
                if is_color_yield(color):
                    return True
        elif goal == "TYPE_TURN_RIGHT":
            if navi_type in [2, 5]:
                if is_color_yield(color):
                    return True
        elif goal == "TYPE_TURN_UTURN":
            if navi_type in [1, 3]:
                if is_color_yield(color):
                    return True
        
        return False

    for lane_info in junction_label_info.entry_lanes:
        lane_id = lane_info['lane_id']
        lane_ids = [lane_id]
        
        if lane_id in junction_label_info.waiting_area_lane_info:
            lane_ids.extend([l[0] for l in junction_label_info.waiting_area_lane_info[lane_id]])
        for lane_light_info in lane_lights_info.lane_lights_info:
            if any([l in lane_light_info.associated_lane_id for l in lane_ids]):
                if is_navi_type_corr_goal_pass(goal, lane_light_info.navi_type, lane_light_info.color):
                    has_pass_signal.append(lane_id)
                if is_navi_type_corr_goal_yield(goal, lane_light_info.navi_type, lane_light_info.color):
                    has_yield_signal.append(lane_id)

    return list(set(has_pass_signal)), list(set(has_yield_signal))


@TAG_FUNCTIONS.register()
def abnormal_yield_tag(data: TagData, params: Dict) -> Dict:
    abnormal_yield_tag = AbnormalYieldTag()
    ego_vel = float(
        (
            data.label_scene.obstacles[-9]["features"]["history_states"][-1]["vx"] ** 2
            + data.label_scene.obstacles[-9]["features"]["history_states"][-1]["vy"]
            ** 2
        )
        ** 0.5
    )

    # 1. cal is still
    if ego_vel < 0.5:
        abnormal_yield_tag.is_still = True

    # 2. cal future acc
    future_states = data.label_scene.obstacles[-9]["future_trajectory"]["future_states"]
    if len(future_states) >= 10:
        abnormal_yield_tag.ego_acc_1s = float(
            ((future_states[9]["vx"] ** 2 + future_states[9]["vy"] ** 2) ** 0.5)
            - ego_vel
        )
    if len(future_states) >= 20:
        abnormal_yield_tag.ego_acc_2s = (
            float(
                ((future_states[19]["vx"] ** 2 + future_states[19]["vy"] ** 2) ** 0.5)
                - ego_vel
            )
            / 2.0
        )

    # 3. cal dis to stop line
    junction_label_info = data.label_scene.junction_label_info
    ego_path_info = data.label_scene.ego_path_info

    if junction_label_info.junction_id != "":
        distance_to_stop_line = 0.0
        for i, in_junction_id in enumerate(ego_path_info.in_junction_id):
            if (
                in_junction_id is not None
                and in_junction_id == junction_label_info.junction_id
            ):
                break
            distance_to_stop_line = i + 1

        abnormal_yield_tag.dis_to_stopline = distance_to_stop_line

    # 4. cal nearest_front_car
    nearest_obs_info = get_nearest_front_car(data.label_scene.obstacles)

    if nearest_obs_info is not None:
        abnormal_yield_tag.nearest_front_car_distance = float(nearest_obs_info["s"])
        abnormal_yield_tag.nearest_front_car_vel = float(nearest_obs_info["v"])
        abnormal_yield_tag.nearest_front_car_length = float(nearest_obs_info["length"])

    # 5. cal nearest cutin in 5s
    nearest_cutin_obs_info = get_nearest_cutin_cat(
        data.label_scene.obstacles, ego_path_info.future_path_linestring
    )

    if nearest_cutin_obs_info is not None:
        abnormal_yield_tag.cutin_in_5_seconds_distance = nearest_cutin_obs_info["s"]
        abnormal_yield_tag.cutin_in_5_seconds_width = nearest_cutin_obs_info["w"]

    # 6. has pass signal and yield signal
    if (
        junction_label_info.junction_id != ""
        and abnormal_yield_tag.dis_to_stopline >= 1
        and len(junction_label_info.entry_lanes) > 0
    ):
        abnormal_yield_tag.has_pass_signal, abnormal_yield_tag.has_yield_signal = (
            check_has_pass_yield_signal(junction_label_info, data.label_scene.lane_lights_info)
        )
        abnormal_yield_tag.pass_type = junction_label_info.junction_goal
    
    return abnormal_yield_tag.as_dict()
