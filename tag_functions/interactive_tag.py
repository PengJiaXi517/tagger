from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from base import PercepMap, TagData
from registry import TAG_FUNCTIONS


@dataclass
class TypeLonInteractTag:
    has_yield: bool = False
    min_yield_speed: float = -1.0
    min_yield_ttc_frame: int = -1
    min_yield_ttk_frame: int = -1
    yield_ttc_ego_frame: int = -1
    rel_yield_s: float = -1.0
    rel_yield_l: float = -1.0

    has_follow: bool = False
    follow_speed: float = -1.0
    follow_ttc_ego_frame: int = -1
    rel_follow_s: float = -1.0
    rel_follow_l: float = -1.0

    has_overtake: bool = False
    max_overtake_speed: float = -1.0
    min_overtake_obs_ttk_frame: int = -1
    min_overtake_obs_ttc_frame: int = -1
    overtake_ego_ttc_frame: int = -1
    rel_overtake_s: float = 0.0
    rel_overtake_l: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "has_yield": self.has_yield,
            "min_yield_speed": self.min_yield_speed,
            "min_yield_ttc_frame": self.min_yield_ttc_frame,
            "min_yield_ttk_frame": self.min_yield_ttk_frame,
            "yield_ttc_ego_frame": self.yield_ttc_ego_frame,
            "rel_yield_s": self.rel_yield_s,
            "rel_yield_l": self.rel_yield_l,
            # ---
            "has_follow": self.has_follow,
            "follow_speed": self.follow_speed,
            "follow_ttc_ego_frame": self.follow_ttc_ego_frame,
            "rel_follow_s": self.rel_follow_s,
            "rel_follow_l": self.rel_follow_l,
            # ---
            "has_overtake": self.has_overtake,
            "max_overtake_speed": self.max_overtake_speed,
            "min_overtake_obs_ttk_frame": self.min_overtake_obs_ttk_frame,
            "min_overtake_obs_ttc_frame": self.min_overtake_obs_ttc_frame,
            "overtake_ego_ttc_frame": self.overtake_ego_ttc_frame,
            "rel_overtake_s": self.rel_overtake_s,
            "rel_overtake_l": self.rel_overtake_l,
        }


@dataclass
class TypeLatInteractTag:
    has_bypass: bool = False
    nearest_rel_s: float = 0.0
    nearest_rel_l: float = 0.0
    nearest_vel: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "has_bypass": self.has_bypass,
            "nearest_rel_s": self.nearest_rel_s,
            "nearest_rel_l": self.nearest_rel_l,
            "nearest_vel": self.nearest_vel,
        }


@dataclass
class InteractiveTag:
    veh_lon_interact_tag: TypeLonInteractTag = field(default_factory=TypeLonInteractTag)
    ped_lon_interact_tag: TypeLonInteractTag = field(default_factory=TypeLonInteractTag)
    cyc_lon_interact_tag: TypeLonInteractTag = field(default_factory=TypeLonInteractTag)

    ped_lat_interact_tag: TypeLatInteractTag = field(default_factory=TypeLatInteractTag)
    cyc_lat_interact_tag: TypeLatInteractTag = field(default_factory=TypeLatInteractTag)

    def as_dict(self):
        return {
            "veh_lon_interact_tag": self.veh_lon_interact_tag.as_dict(),
            "ped_lon_interact_tag": self.ped_lon_interact_tag.as_dict(),
            "cyc_lon_interact_tag": self.cyc_lon_interact_tag.as_dict(),
            # ---
            "ped_lat_interact_tag": self.ped_lat_interact_tag.as_dict(),
            "cyc_lat_interact_tag": self.cyc_lat_interact_tag.as_dict(),
        }


def fill_sub_lon_tag(
    lon_interact_tag: TypeLonInteractTag,
    obs_state: Dict,
    obs_decision: Dict,
    lon_type: int,
):
    if lon_type == 1:
        if not lon_interact_tag.has_follow or (
            obs_decision["obs_s"] < lon_interact_tag.rel_follow_s
        ):
            curr_state = obs_state["features"]["history_states"][-1]

            lon_interact_tag.has_follow = True
            lon_interact_tag.follow_speed = np.linalg.norm(
                np.array([curr_state["vx"], curr_state["vy"]])
            )
            lon_interact_tag.follow_ttc_ego_frame = obs_decision["ego_ttc"]
            lon_interact_tag.rel_follow_s = obs_decision["obs_s"]
            lon_interact_tag.rel_follow_l = obs_decision["obs_l"]

    elif lon_type == 2:
        if not lon_interact_tag.has_yield or (
            obs_decision["obs_ttk"] != -1
            and obs_decision["obs_ttk"] < lon_interact_tag.min_yield_ttk_frame
        ):
            curr_state = obs_state["features"]["history_states"][-1]

            lon_interact_tag.has_yield = True
            lon_interact_tag.min_yield_speed = np.linalg.norm(
                np.array([curr_state["vx"], curr_state["vy"]])
            )
            lon_interact_tag.min_yield_ttk_frame = obs_decision["obs_ttk"]
            lon_interact_tag.min_yield_ttc_frame = obs_decision["obs_ttc"]
            lon_interact_tag.yield_ttc_ego_frame = obs_decision["ego_ttc"]
            lon_interact_tag.rel_yield_s = obs_decision["obs_s"]
            lon_interact_tag.rel_yield_l = obs_decision["obs_l"]

    elif lon_type == 3:
        if not lon_interact_tag.has_overtake or (
            obs_decision["obs_ttk"] != -1
            and obs_decision["obs_ttk"] < lon_interact_tag.min_overtake_obs_ttk_frame
        ):
            curr_state = obs_state["features"]["history_states"][-1]

            lon_interact_tag.has_overtake = True
            lon_interact_tag.max_overtake_speed = np.linalg.norm(
                np.array([curr_state["vx"], curr_state["vy"]])
            )
            lon_interact_tag.min_overtake_obs_ttk_frame = obs_decision["obs_ttk"]
            lon_interact_tag.min_overtake_obs_ttc_frame = obs_decision["obs_ttc"]
            lon_interact_tag.overtake_ego_ttc_frame = obs_decision["ego_ttc"]
            lon_interact_tag.rel_overtake_s = obs_decision["obs_s"]
            lon_interact_tag.rel_overtake_l = obs_decision["obs_l"]


def fill_lon_interact_tag(
    interact_tag: InteractiveTag, obs_state: Dict, obs_decision: Dict, lon_type: int
):
    obs_type = obs_state["features"]["type"]
    if obs_type == "PEDESTRIAN":
        fill_sub_lon_tag(
            interact_tag.ped_lon_interact_tag, obs_state, obs_decision, lon_type
        )
    elif obs_type == "BICYCLE":
        fill_sub_lon_tag(
            interact_tag.cyc_lon_interact_tag, obs_state, obs_decision, lon_type
        )
    elif obs_type == "VEHICLE":
        fill_sub_lon_tag(
            interact_tag.veh_lon_interact_tag, obs_state, obs_decision, lon_type
        )


def fill_sub_lat_tag(
    lat_interact_tag: TypeLatInteractTag,
    obs_state: Dict,
    obs_decision: Dict,
    lat_type: int,
):
    if lat_type != 0:
        if not lat_interact_tag.has_bypass or (
            np.abs(obs_decision["obs_l"]) < np.abs(lat_interact_tag.nearest_rel_l)
        ):
            curr_state = obs_state["features"]["history_states"][-1]

            lat_interact_tag.has_bypass = True
            lat_interact_tag.nearest_rel_s = obs_decision["obs_s"]
            lat_interact_tag.nearest_rel_l = obs_decision["obs_l"]
            lat_interact_tag.nearest_vel = np.linalg.norm(
                np.array([curr_state["vx"], curr_state["vy"]])
            )


def fill_lat_interact_tag(
    interact_tag: InteractiveTag, obs_state: Dict, obs_decision: Dict, lat_type: int
):
    obs_type = obs_state["features"]["type"]
    if obs_type == "PEDESTRIAN":
        fill_sub_lat_tag(
            interact_tag.ped_lat_interact_tag, obs_state, obs_decision, lat_type
        )
    elif obs_type == "BICYCLE":
        fill_sub_lat_tag(
            interact_tag.cyc_lat_interact_tag, obs_state, obs_decision, lat_type
        )


@TAG_FUNCTIONS.register()
def interactive_tag(data: TagData, params: Dict) -> Dict:

    ret = InteractiveTag()

    for obs_id, obs_state in data.label_scene.obstacles.items():
        if obs_id == -9:
            continue

        obs_decision = obs_state["decision"]

        obs_lat = 0
        obs_lon = 0  # 1 follow 2 yield 3 overtake

        # Get lat tag
        if (
            obs_decision["interactive_lat_type"] != 0
            and -2.5 < obs_decision["obs_s"] - obs_decision["ego_s"] < 100
        ):
            if obs_decision["obs_l"] < 0:
                obs_lat = 1
            else:
                obs_lat = 2

        # Get lon tag
        obs_lon = obs_decision["interactive_lon_type"]
        if obs_lon != 0:
            obs_lat = 0

        if obs_lon != 0:
            fill_lon_interact_tag(ret, obs_state, obs_decision, obs_lon)
        elif obs_lat != 0:
            fill_lat_interact_tag(ret, obs_state, obs_decision, obs_lat)

    return ret.as_dict()
