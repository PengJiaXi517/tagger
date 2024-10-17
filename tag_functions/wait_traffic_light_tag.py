import numpy as np
from base import TagData
from typing import Dict, Union
from dataclasses import dataclass
from registry import TAG_FUNCTIONS


def get_front_car_info(obstacles) -> Union[Dict, None]:
    """
    :param obstacles: data.label_scene.label_res["obstacles"]
    :return None if no front car else front car OBSID
    """
    # init
    obs_info = np.empty(
        (0, 4), dtype=[("s", float), ("l", float), ("key", int), ("v", float)]
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
                    obstacles[key]["decision"]["obs_s"],
                    obstacles[key]["decision"]["obs_l"],
                    key,
                    obs_v,
                ],
                dtype=obs_info.dtype,
            ),
        )

        # get ego_s only once
        if ego_s is None and "ego_s" in obstacles[key]["decision"]:
            ego_s = obstacles[key]["decision"]["ego_s"]

    # relative obs s
    if ego_s is not None:
        obs_info["s"] -= ego_s

    # filter obs
    filtered_obs_info = obs_info[(np.abs(obs_info["l"]) < 1.5) & (obs_info["s"] > 0)]

    # find argmin obs_s
    if len(filtered_obs_info) > 0:
        min_obs = filtered_obs_info[np.argmin(filtered_obs_info["s"])]
        return min_obs
    else:
        return None


@dataclass(repr=False)
class WaitTrafficLightTag:
    start_slow: bool = None
    ego_velocity: float = None
    dis_to_stopline: float = None
    dis_to_front_car: float = None
    front_car_id: int = None
    front_car_velocity: float = None

    def as_dict(self) -> Dict:
        return {
            "traffic_start_slow_check": {
                "start_slow": self.start_slow,
                "ego_velocity": self.ego_velocity,
                "dis_to_stopline": self.dis_to_stopline,
                "dis_to_front_car": self.dis_to_front_car,
                "front_car_id": self.front_car_id,
                "front_car_velocity": self.front_car_velocity,
            }
        }


@TAG_FUNCTIONS.register()
def traffic_start_slow_check(data: TagData, params: Dict) -> Dict:
    wait_traffic_light_tag = WaitTrafficLightTag()

    # 1. get ego velocity
    ego_vx = data.label_scene.obstacles[-9]["features"]["history_states"][-1]["vx"]
    ego_vy = data.label_scene.obstacles[-9]["features"]["history_states"][-1]["vy"]
    wait_traffic_light_tag.ego_velocity = (ego_vx**2 + ego_vy**2) ** 0.5

    # 2. get dis to stopline
    wait_traffic_light_tag.dis_to_stopline = data.label_scene.label_res["frame_info"][
        "ego_curr_status"
    ]["dis_to_stopline_by_polygon"]

    # 3. get front car info
    front_car_info = get_front_car_info(data.label_scene.obstacles)
    if front_car_info is not None:
        wait_traffic_light_tag.dis_to_front_car = front_car_info["s"]
        wait_traffic_light_tag.front_car_id = front_car_info["key"]
        wait_traffic_light_tag.front_car_velocity = front_car_info["v"]

    # 4. check if start slow
    if (
        wait_traffic_light_tag.dis_to_front_car is None
        or wait_traffic_light_tag.dis_to_stopline is None
        or wait_traffic_light_tag.ego_velocity is None
        or wait_traffic_light_tag.front_car_velocity is None
    ):
        wait_traffic_light_tag.start_slow = False
    elif (
        wait_traffic_light_tag.dis_to_front_car
        < wait_traffic_light_tag.dis_to_stopline
        < 50.0
        and wait_traffic_light_tag.ego_velocity < 3.0
        and wait_traffic_light_tag.dis_to_front_car > 10.0
        and wait_traffic_light_tag.front_car_velocity - 2.5
        > wait_traffic_light_tag.ego_velocity
    ):
        wait_traffic_light_tag.start_slow = True
    else:
        wait_traffic_light_tag.start_slow = False

    return wait_traffic_light_tag.as_dict()
