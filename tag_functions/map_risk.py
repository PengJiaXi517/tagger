import os
from shapely import geometry

from registry import TAG_FUNCTIONS


@TAG_FUNCTIONS.register()
def map_risk(data, params, result):

    output = {}
    tag_ego_path_curb_cross_risk = False
    tag_lane_curb_cross_risk = False

    label_scene = data.label_scene

    if len(label_scene.ego_path_info.future_path) > 1:
        future_path = geometry.LineString(label_scene.ego_path_info.future_path)
        for curb in label_scene.percepmap.curbs:
            curb_string = geometry.LineString(curb)
            if curb_string.crosses(future_path) or curb_string.intersects(future_path):
                tag_ego_path_curb_cross_risk = True

    for k, v in label_scene.percepmap.lane_map.items():
        if len(v['polyline']) > 1:
            line_string = geometry.LineString(v["polyline"])
            for curb in label_scene.percepmap.curbs:
                curb_string = geometry.LineString(curb)
                if curb_string.crosses(line_string) or curb_string.crosses(line_string):
                    tag_lane_curb_cross_risk = True

    output["tag_ego_path_curb_cross_risk"] = tag_ego_path_curb_cross_risk
    output["tag_lane_curb_cross_risk"] = tag_lane_curb_cross_risk

    return output
