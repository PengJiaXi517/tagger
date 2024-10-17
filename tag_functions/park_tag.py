import numpy as np
from base import TagData
from typing import Dict, Union
from dataclasses import dataclass, field
from registry import TAG_FUNCTIONS


def transform_point_to_ego_axis(ego_info: np.array, points: np.array) -> np.array:
    """
    :param ego_info: (1, 3) np.array contains (x, y, theta)
    :param points: (n, 2) np.array
    :return transed_points: (n, 2) np.array

    """

    trans_mat = np.array([[1, 0, -ego_info[0]], [0, 1, -ego_info[1]], [0, 0, 1]])

    rot_mat = np.array(
        [
            [np.cos(ego_info[2]), -np.sin(ego_info[2]), 0],
            [np.sin(ego_info[2]), np.cos(ego_info[2]), 0],
            [0, 0, 1],
        ]
    )

    # combine rotation and translation
    comb_mat = np.dot(rot_mat, trans_mat)

    # (x, y) -> (x, y, 1)
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])

    transed_points = np.dot(comb_mat, points_homo.T).T

    return transed_points[:, :2]


def is_seg_intersect_aabox(p1: np.array, p2: np.array, aabox: Dict) -> bool:
    """
    :param p1: src point of seg (n, 2) np.array
    :param p2: tgt point of seg (n, 2) np.array
    :param aabox: 2D-AABB {'min': (x_min, y_min), 'max': (x_max, y_max)}
    :return np.any(): True if intersect
    """

    # aabox 4 line of rectangel
    aabox_rect_line = np.array(
        [
            [aabox["min"][0], aabox["min"][1], aabox["max"][0], aabox["min"][1]],
            [aabox["max"][0], aabox["min"][1], aabox["max"][0], aabox["max"][1]],
            [aabox["max"][0], aabox["max"][1], aabox["min"][0], aabox["max"][1]],
            [aabox["min"][0], aabox["max"][1], aabox["min"][0], aabox["min"][1]],
        ]
    )

    # VECTOR D: aabox vector of 4 edge
    aabox_seg_direct = aabox_rect_line[:, 2:] - aabox_rect_line[:, :2]
    # VECTOR B: curb vector
    seg_direct = p2 - p1

    # cross of B and D
    cross_seg_aabox_seg = np.cross(aabox_seg_direct, seg_direct[:, np.newaxis])
    # comment: cross_seg_aabox_seg = 0 means curb and aabox edge are paralle
    #          is set to np.inf to avoid divide by zero

    # VECTOR C: from curb start point to aabox edge start point
    p1_minus_aabox_pt = p1[:, np.newaxis, :] - aabox_rect_line[:, :2]

    # cross of C and B
    cross_seg_p1pt = np.cross(p1_minus_aabox_pt, seg_direct[:, np.newaxis])

    # get intersection ratio of curb vec
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(
            cross_seg_aabox_seg != 0, cross_seg_p1pt / cross_seg_aabox_seg, np.inf
        )

    # cross of C and D
    cross_aabox_seg_p1pt = np.cross(p1_minus_aabox_pt, aabox_seg_direct)

    # get intersection ratio of aabox vec
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(
            cross_seg_aabox_seg != 0, cross_aabox_seg_p1pt / cross_seg_aabox_seg, np.inf
        )

    # two ratio both in [0, 1] means segments intersect, get rid of duplicates
    res_idx = np.where((t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))[0]
    res_idx = np.unique(res_idx)

    return res_idx.size > 0


def check_segs_intersect_aabox(ego_info: np.array, segs: np.array, aabox: Dict) -> bool:
    """
    :param ego_info: (1, 3) np.array contains (x, y, theta)
    :param segs: (n, 4) -> (x1, y1, x2, y2) src to tgt
    :param aabox: 2D-AABB {'min': (x_min, y_min), 'max': (x_max, y_max)}
    :return np.any(): True if intersect
    """

    # transform curb points to ego axis
    points = segs.reshape(-1, 2)
    transformed_points = transform_point_to_ego_axis(ego_info, points)
    transformed_segs = transformed_points.reshape(-1, 4)

    # get curb start and end point
    p1 = transformed_segs[:, :2]
    p2 = transformed_segs[:, 2:]

    # 1. judging whether the start and end points of the curb are in the aabox
    p1_in_aabox = np.where(
        (p1[0] >= aabox["min"][0])
        & (p1[0] <= aabox["max"][0])
        & (p1[1] >= aabox["min"][1])
        & (p1[1] <= aabox["max"][1])
    )[0]
    if p1_in_aabox.size:
        return True
    p2_in_aabox = np.where(
        (p2[0] >= aabox["min"][0])
        & (p2[0] <= aabox["max"][0])
        & (p2[1] >= aabox["min"][1])
        & (p2[1] <= aabox["max"][1])
    )[0]
    if p2_in_aabox.size:
        return True

    # 2. judging whether the curb intersects with the aabox edge
    return is_seg_intersect_aabox(p1, p2, aabox)


def check_has_close_right_curb(ego_infos, curbs, tag_aabox: Dict) -> bool:
    """
    :param ego_infos: data.label_scene.obstacles[-9]["features"]["history_states"]
    :param curbs: data.label_scene.percepmap.curbs
    """
    # prepare data
    ego_info = np.array(
        [ego_infos[-1]["x"], ego_infos[-1]["y"], ego_infos[-1]["theta"]]
    )
    curb_segs = np.array([])
    for curb in curbs:
        if len(curb) <= 1:
            continue
        curb_array = np.array(curb)
        segs = np.concatenate((curb_array[:-1], curb_array[1:]), axis=1)
        curb_segs = (
            np.concatenate((curb_segs, segs), axis=0) if curb_segs.size else segs
        )
    aabox = tag_aabox
    if check_segs_intersect_aabox(ego_info, curb_segs, aabox):
        return True
    else:
        return False


def check_future_change_path(label_res, ego_obs_seq_info) -> bool:
    """
    :param label_res: data.label_scene.label_res
    :param ego_obs_seq_info: data.label_scene.ego_obs_lane_seq_info
    """
    # get ego lane ids
    corr_lane_id = label_res["ego_path_info"]["corr_lane_id"]
    corr_lane_id = np.array(
        [
            lane_id[0][0]
            for lane_id in corr_lane_id
            if lane_id is not None and len(lane_id) > 0
        ]
    )

    # remove duplicates
    ego_lanes = np.unique(corr_lane_id)

    if len(ego_lanes) <= 1:
        return False

    # get current lane and nearby lanes
    current_lane = (
        ego_obs_seq_info.current_lanes[0]
        if len(ego_obs_seq_info.current_lanes) > 0
        else None
    )
    if ego_obs_seq_info.nearby_lane_seqs:
        nearby_lanes = np.concatenate(ego_obs_seq_info.nearby_lane_seqs)
    else:
        nearby_lanes = np.array([])

    # check if change path in future
    if current_lane is not None:
        check_change_path = np.any(
            np.isin(ego_lanes, nearby_lanes) & (ego_lanes != current_lane)
        )
    else:
        check_change_path = False

    return check_change_path


def check_front_car(obstacles) -> Union[int, None]:
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
        return min_obs["key"]
    else:
        return None


@dataclass(repr=False)
class ParkTag:
    abnormal_park: bool = None
    ego_velocity: float = None
    dis_to_stopline: float = None
    dis_to_front_car: float = None
    front_car_id: int = None
    front_car_velocity: float = None
    future_change_path: bool = None
    is_on_rightmost_lane: bool = None
    aabox: dict = field(default_factory=dict)

    def __post_init__(self):
        self.aabox = {"min": np.array([-3, -3]), "max": np.array([3, 0])}

    def as_dict(self) -> Dict:
        return {
            "park_check": {
                "abnormal_park": self.abnormal_park,
                "ego_velocity": self.ego_velocity,
                "dis_to_stopline": self.dis_to_stopline,
                "front_car_id": self.front_car_id,
                "future_change_path": self.future_change_path,
                "is_on_rightmost_lane": self.is_on_rightmost_lane,
            }
        }


@TAG_FUNCTIONS.register()
def park_check(data: TagData, params: Dict) -> Dict:
    park_tag = ParkTag()

    assert len(data.label_scene.obstacles[-9]["features"]["history_states"]) > 0

    # 1. get ego velocity
    park_tag.ego_velocity = (
        data.label_scene.obstacles[-9]["features"]["history_states"][-1]["vx"] ** 2
        + data.label_scene.obstacles[-9]["features"]["history_states"][-1]["vy"] ** 2
    ) ** 0.5

    # 2. get ego dis to stopline (logic based on autolabeler)
    park_tag.dis_to_stopline = data.label_scene.label_res["frame_info"][
        "ego_curr_status"
    ]["dis_to_stopline_by_polygon"]

    # 3. check if on rightmost lane
    #   a. get rightmost from auto_labeler
    park_tag.is_on_rightmost_lane = data.label_scene.label_res["frame_info"][
        "ego_curr_status"
    ]["is_rightmost"]

    #   b. update rightmost by checking existance of right close curb
    if not park_tag.is_on_rightmost_lane:
        park_tag.is_on_rightmost_lane = check_has_close_right_curb(
            data.label_scene.obstacles[-9]["features"]["history_states"],
            data.label_scene.percepmap.curbs,
            park_tag.aabox,
        )

    # 4. check if change lane in future
    park_tag.future_change_path = check_future_change_path(
        data.label_scene.label_res, data.label_scene.ego_obs_lane_seq_info
    )

    # 5. check if exists front car
    park_tag.front_car_id = check_front_car(data.label_scene.obstacles)

    # 6. check abnormal park
    if park_tag.ego_velocity < 0.01 and park_tag.is_on_rightmost_lane:
        if park_tag.future_change_path:
            park_tag.abnormal_park = True
        else:
            if park_tag.front_car_id is None:
                park_tag.abnormal_park = True
            else:
                if (
                    park_tag.dis_to_stopline is None
                    or park_tag.dis_to_stopline == 180.0
                ):
                    park_tag.abnormal_park = True
                else:
                    park_tag.abnormal_park = False
    else:
        park_tag.abnormal_park = False

    return park_tag.as_dict()
