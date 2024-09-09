import functools
import time
from bisect import bisect_right
from copy import deepcopy
from random import shuffle
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from shapely import LineString, Point
from shapely.ops import unary_union

from .utils import get_dict_data, write_dict_data


def normalize_angles(angles: np.array):
    while (angles > np.pi).any():
        angles[angles > np.pi] -= 2 * np.pi
    while (angles <= -np.pi).any():
        angles[angles <= -np.pi] += 2 * np.pi
    return angles


class MapLaneSeqPacker:
    def __init__(
        self,
        src_key: str = "percepmap",
        src_ego_curr_state_key=("obstacle_state", "ego_curr_state"),
        tgt_state_key: Union[Tuple, str] = ("map_state", "state"),
        tgt_mask_key: Union[Tuple, str] = ("map_state", "state_mask"),
        tgt_id_map_key: Union[Tuple, str] = ("map_state", "seq_lane_ids"),
        tgt_id_map_raw_key: Union[Tuple, str] = ("map_state", "seq_lane_ids_raw"),
        sample_interval: float = 3,
        # lane
        filter_lane_type=[
            "UNKNOWN_LANETYPE",
            "GENERAL",
            "EMERGENCY",
            "TIDAL",
            "BUS",
            "UNCONV",
            "WAIT_LEFT",
            "WAIT_RIGHT",
            "WAIT_FORWARD",
        ],
        turn_type_mapping={
            "UNKNOWN_LANEDIRECTION": 0,
            "NOTURN": 1,
            "LEFTTURN": 2,
            "LEFTSTRAIGHT": 3,
            "RIHGTTURN": 4,
            "RIHGTSTRAIGHT": 5,
            "LEFTRIGHTTURN": 6,
            "LEFTRIGHTSTRAIGHT": 7,
            "UTURN": 8,
            "USTRAIGHT": 9,
            "USTRAIGHT_LEFT": 11,
            "USTRAIGHT_RIGHT": 13,
        },
        lane_type_mapping={
            "UNKNOWN_LANETYPE": 0,
            "GENERAL": 1,
            "EMERGENCY": 2,
            "TIDAL": 3,
            "BUS": 4,
            "UNCONV": 5,
        },
        lane_boundary_type_mapping={
            "UNKNOWN_SPANABILITY": 0,
            "SOLID": 1,
            "DASHED": 2,
            "VIRTUAL": 3,
        },
        save_lane_ids_raw: bool = True,
        filter_virtual: bool = True,
        filter_pd_lane: bool = True,
        max_num_nodes: int = 128,
        max_num_lane_seqs: int = 64,
        x_range=(-50, 150),
        y_range=(-100, 100),
        filter_roi_range=False,
        roi_x_range=(-50, 150),
        roi_y_range=(-50, 50),
        interp_fast: bool = True,
        shuffle: bool = False,
    ) -> None:
        self.src_key = src_key
        self.src_ego_curr_state_key = src_ego_curr_state_key
        self.tgt_state_key = tgt_state_key
        self.tgt_mask_key = tgt_mask_key
        self.tgt_id_map_key = tgt_id_map_key
        self.tgt_id_map_raw_key = tgt_id_map_raw_key

        self.sample_interval = sample_interval
        self.filter_lane_type = filter_lane_type
        self.filter_pd_lane = filter_pd_lane
        self.turn_type_mapping = turn_type_mapping
        self.lane_type_mapping = lane_type_mapping
        self.max_num_nodes = max_num_nodes
        self.max_num_lane_seqs = max_num_lane_seqs
        self.lane_boundary_type_mapping = lane_boundary_type_mapping
        self.filter_virtual = filter_virtual

        self.x_range = x_range
        self.y_range = y_range
        self.filter_roi_range = filter_roi_range
        self.roi_x_range = roi_x_range
        self.roi_y_range = roi_y_range

        self.interp_fast = interp_fast
        self.shuffle = shuffle

        self.save_lane_ids_raw = save_lane_ids_raw

    def is_virtual_lane(self, lane: Dict[str, Any]):
        res = (
            lane["left_boundary"]["is_virtual"] and lane["right_boundary"]["is_virtual"]
        )
        if self.filter_pd_lane:
            if not lane.get("pd_lane", True):
                return True
        if lane.get("lane_category"):
            return res or lane["lane_category"] in (
                "INTERSECTION_VIRTUAL",
                "M2N_VIRTUAL",
            )
        else:
            return res

    def collect_lanes(self, lanes: List[Dict[str, Any]]):
        lane_id_map = dict()
        for idx, lane in enumerate(lanes):
            if self.filter_virtual and self.is_virtual_lane(lane):
                continue
            if lane["type"] not in self.filter_lane_type:
                continue
            lane_id = lane["id"]
            lane_id_map[lane_id] = idx
        return lane_id_map

    def collect_lanes_online(
        self, lanes: List[Dict[str, Any]], ego_curr_state: Dict, search_radius=142.0
    ):
        search_lanes = []
        search_central_point = np.array(
            [
                ego_curr_state["x"] + 50 * np.cos(ego_curr_state["theta"]),
                ego_curr_state["y"] + 50 * np.sin(ego_curr_state["theta"]),
            ]
        ).reshape(1, 2)
        for idx, lane in enumerate(lanes):
            if self.filter_virtual and self.is_virtual_lane(lane):
                continue
            if lane["type"] not in self.filter_lane_type:
                continue
            central_polyline = np.array(lane["polyline"])
            distance = np.min(
                np.linalg.norm(central_polyline - search_central_point, axis=-1)
            )
            if distance > search_radius:
                continue
            search_lanes.append(lane)

        sorted_search_lanes = self.sort_search_lanes(
            search_lanes, search_central_point.reshape((2,))
        )

        return sorted_search_lanes

    def sort_search_lanes(
        self, search_lanes: List[Dict[str, Any]], search_central_point
    ):
        lanes_and_dist = []
        for idx, lane_info in enumerate(search_lanes):
            if len(lane_info["polyline"]) < 2:
                continue
            start_p = np.array(lane_info["polyline"][0])
            end_p = np.array(lane_info["polyline"][-1])
            center_p = (start_p + end_p) / 2.0
            dis_middle = np.linalg.norm((center_p - search_central_point))
            dis_start = np.linalg.norm(start_p - search_central_point)
            dis_end = np.linalg.norm(end_p - search_central_point)
            lanes_and_dist.append([idx, np.min([dis_middle, dis_end, dis_start])])

        def cmp_lane_and_dist(lane_and_dist_a, lane_and_dist_b):
            if lane_and_dist_a[1] == lane_and_dist_b[1]:
                if (
                    search_lanes[lane_and_dist_a[0]]["id"]
                    < search_lanes[lane_and_dist_b[0]]["id"]
                ):
                    return -1
                else:
                    return 1
            if lane_and_dist_a[1] < lane_and_dist_b[1]:
                return -1
            return 1

        lanes_and_dist = sorted(
            lanes_and_dist, key=functools.cmp_to_key(cmp_lane_and_dist)
        )
        # lanes_and_dist.sort(key=lambda x: x[1])

        sorted_search_lanes = []
        for idx, dis in lanes_and_dist:
            sorted_search_lanes.append(search_lanes[idx])

        return sorted_search_lanes

    def dfs_build_lane_seq(
        self,
        now_idx,
        successor_nodes,
        visit_flags: np.ndarray,
        lane_seq: List[int],
        lane_seqs: List[List[int]],
    ):
        if len(successor_nodes[now_idx]) == 0:
            lane_seq.append(now_idx)
            lane_seqs.append(deepcopy(lane_seq))
            del lane_seq[-1]
            return

        next_count = 0
        lane_seq.append(now_idx)
        visit_flags[now_idx] = 1
        for next_idx in successor_nodes[now_idx]:
            if visit_flags[next_idx] == 0:
                self.dfs_build_lane_seq(
                    next_idx, successor_nodes, visit_flags, lane_seq, lane_seqs
                )
                next_count += 1

        if next_count == 0:
            lane_seqs.append(deepcopy(lane_seq))
        visit_flags[now_idx] = 0
        del lane_seq[-1]
        return

    def dfs_build_lane_graph(self, search_lanes: List[Dict[str, Any]]):
        lane_id_2_idx_map = {}
        for idx, lane in enumerate(search_lanes):
            lane_id_2_idx_map[lane["id"]] = idx
        node_num = len(lane_id_2_idx_map)

        successor_nodes = [[] for _ in range(len(lane_id_2_idx_map))]
        in_degree_nodes = np.zeros((len(lane_id_2_idx_map),), dtype=np.int32)
        out_degree_nodes = np.zeros((len(lane_id_2_idx_map),), dtype=np.int32)
        idx_2_lane_map = [None for _ in range(len(lane_id_2_idx_map))]

        for idx, lane in enumerate(search_lanes):
            idx_2_lane_map[idx] = lane

            for succ_id in lane["successor_id"]:
                if succ_id not in lane_id_2_idx_map:
                    continue
                out_idx = lane_id_2_idx_map[succ_id]
                successor_nodes[idx].append(out_idx)
                in_degree_nodes[out_idx] += 1
                out_degree_nodes[idx] += 1

        # TODO: build lane graph here
        lane_seqs: List[List[int]] = []
        lane_seq: List[int] = []
        visit_flag = np.zeros((node_num,), dtype=np.int32)

        for i in range(node_num):
            if in_degree_nodes[i] == 0:
                self.dfs_build_lane_seq(
                    i, successor_nodes, visit_flag, lane_seq, lane_seqs
                )

        return lane_id_2_idx_map, idx_2_lane_map, lane_seqs

    def get_lane_start_distances(
        self, lanes: List[Dict[str, Any]], ego_curr_state: Dict
    ) -> np.array:
        distances = list()
        start_pts = np.array([lane["polyline"][0] for lane in lanes])

        rot_mat = np.array(
            [
                [
                    np.cos(ego_curr_state["theta"]),
                    -np.sin(ego_curr_state["theta"]),
                ],
                [
                    np.sin(ego_curr_state["theta"]),
                    np.cos(ego_curr_state["theta"]),
                ],
            ]
        ).T
        t_vec = -np.array([ego_curr_state["x"], ego_curr_state["y"]]) @ rot_mat.T
        start_in_ego = start_pts[:, :2] @ rot_mat.T + t_vec
        distances = np.linalg.norm(start_in_ego, axis=-1)
        return distances

    def build_lane_graph(
        self, lane_id_map: Dict[str, int], lanes: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        for lane_id in lane_id_map:
            index = lane_id_map[lane_id]
            lane = lanes[index]
            graph.add_node(index)
            predecessors = lane["predecessor_id"]
            for predecessor in predecessors:
                if predecessor not in lane_id_map:
                    continue
                pre_idx = lane_id_map[predecessor]
                graph.add_edge(pre_idx, index)
            successors = lane["successor_id"]
            for successor in successors:
                if successor not in lane_id_map:
                    continue
                suc_idx = lane_id_map[successor]
                graph.add_edge(index, suc_idx)
        return graph

    def get_connected_subgraphs(self, graph: nx.DiGraph) -> List[nx.DiGraph]:
        comps = nx.weakly_connected_components(graph)
        comps = [comp for comp in comps]
        sub_graphs = [graph.subgraph(comp) for comp in comps]
        return sub_graphs

    def remove_cycle(self, graph: nx.DiGraph, lane_start_distances: np.array):
        try:
            cycle_edges = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            return [graph]
        # logger = MMLogger.get_current_instance()
        # logger.warning(f"{self.file_path} find loops!")

        start_nodes = []
        for node in graph:
            if graph.in_degree(node) == 0:
                start_nodes.append(node)
        if len(start_nodes) > 0:
            return [nx.dfs_tree(graph, start_node) for start_node in start_nodes]
        else:
            cycle_edges = sorted(cycle_edges, key=lambda x: lane_start_distances[x[1]])
            return [nx.dfs_tree(graph, cycle_edges[-1][-1])]

    def get_lane_sequences_from_connected_graph(
        self, graph: nx.DiGraph, lane_start_distances: np.array
    ) -> List[List[str]]:
        dfs_trees = self.remove_cycle(graph, lane_start_distances)
        res = []
        for tree in dfs_trees:
            if len(tree.nodes) == 1:
                res.append([node for node in tree.nodes])
                continue
            start_nodes = []
            end_nodes = []
            for node in tree:
                if tree.in_degree(node) == 0:
                    start_nodes.append(node)
                if tree.out_degree(node) == 0:
                    end_nodes.append(node)

            for start_node in start_nodes:
                for end_node in end_nodes:
                    paths = nx.all_simple_paths(tree, start_node, end_node)
                    res.extend(paths)
        return res

    def cut_lane_seq(
        self,
        lane_states: np.array,
        lane_ids: np.array,
        ego_curr_state: Dict,
    ):
        rot_mat = np.array(
            [
                [
                    np.cos(ego_curr_state["theta"]),
                    -np.sin(ego_curr_state["theta"]),
                ],
                [
                    np.sin(ego_curr_state["theta"]),
                    np.cos(ego_curr_state["theta"]),
                ],
            ]
        ).T
        t_vec = -np.array([ego_curr_state["x"], ego_curr_state["y"]]) @ rot_mat.T

        state_in_ego = lane_states[:, :2] @ rot_mat.T + t_vec

        valid_mask = (
            (state_in_ego[:, 0] > self.x_range[0])
            & (state_in_ego[:, 0] < self.x_range[1])
            & (state_in_ego[:, 1] > self.y_range[0])
            & (state_in_ego[:, 1] < self.y_range[1])
        )

        start = 0
        end = 0
        seg_idxs = list()
        for i in range(len(valid_mask)):
            if valid_mask[i]:
                end += 1
            else:
                if end > start:
                    seg_idxs.append([start, end])
                start = i + 1
                end = start
        if end > start:
            seg_idxs.append([start, end])

        segs = list()
        for seg in seg_idxs:
            if self.filter_roi_range:
                seg_ego_state = state_in_ego[seg[0] : seg[1]]
                valid_mask = (
                    (seg_ego_state[:, 0] > self.roi_x_range[0])
                    & (seg_ego_state[:, 0] < self.roi_x_range[1])
                    & (seg_ego_state[:, 1] > self.roi_y_range[0])
                    & (seg_ego_state[:, 1] < self.roi_y_range[1])
                )
                if valid_mask.sum() == 0:
                    continue
            seg_state = lane_states[seg[0] : seg[1]]
            seg_lane_ids = list(set(lane_ids[seg[0] : seg[1]]))
            segs.append((seg_state, seg_lane_ids))
        return segs

    def pack_lane_seq(
        self,
        lane_seq: List[str],
        lanes: List[Dict[str, Any]],
        state_cache: Dict,
    ):
        start_s = 0
        lane_states = list()
        lane_ids = list()
        for lane_idx in lane_seq:
            if lane_idx not in state_cache:
                state_cache[lane_idx] = dict()
            lane = lanes[lane_idx]
            # use unary_union for duplicated points
            polyline = unary_union(LineString(lane["polyline"]))

            if start_s > polyline.length:
                start_s -= polyline.length
                continue
            curr_start_s = start_s
            if curr_start_s not in state_cache[lane_idx]:
                for i, pt in enumerate(lane["left_boundary"]["polyline"]):
                    if np.isnan(pt).any():
                        lane["left_boundary"]["polyline"][i] = lane["polyline"][i]
                for i, pt in enumerate(lane["right_boundary"]["polyline"]):
                    if np.isnan(pt).any():
                        lane["right_boundary"]["polyline"][i] = lane["polyline"][i]
                lb_line = unary_union(LineString(lane["left_boundary"]["polyline"]))
                rb_line = unary_union(LineString(lane["right_boundary"]["polyline"]))
                lb_type_seg_s = [
                    type_map[0] for type_map in lane["left_boundary"]["boundary_type"]
                ]
                rb_type_seg_s = [
                    type_map[0] for type_map in lane["right_boundary"]["boundary_type"]
                ]
                original_s = [polyline.project(Point(pt)) for pt in lane["polyline"]]
                ss = np.arange(start_s, polyline.length, self.sample_interval)
                if len(ss) == 0:
                    continue
                start_s = self.sample_interval - (polyline.length - ss[-1])

                # x, y, dir_x, dir_y, lb_width, lb_type, rb_width, rb_type
                curr_lane_states = list()
                for s in ss:
                    state = np.zeros((9,))
                    pt = polyline.interpolate(s)
                    orginal_s_idx = bisect_right(original_s, s) - 1
                    # TODO: linear interpolate directions
                    direction = lane["unit_directions"][orginal_s_idx]
                    # TODO: check virtual lane boudary type
                    if self.is_virtual_lane(lane):
                        lb_width = -1
                        lb_type = self.lane_boundary_type_mapping["UNKNOWN"]
                        rb_width = -1
                        rb_type = self.lane_boundary_type_mapping["UNKNOWN"]
                    else:
                        lb_width = pt.distance(lb_line)
                        lb_s = lb_line.project(pt)
                        lb_s_idx = bisect_right(lb_type_seg_s, lb_s) - 1
                        lb_type = self.lane_boundary_type_mapping[
                            lane["left_boundary"]["boundary_type"][lb_s_idx][1][0]
                        ]
                        rb_width = pt.distance(rb_line)
                        rb_s = rb_line.project(pt)
                        rb_s_idx = bisect_right(rb_type_seg_s, rb_s) - 1
                        rb_type = self.lane_boundary_type_mapping[
                            lane["right_boundary"]["boundary_type"][rb_s_idx][1][0]
                        ]
                    state[0] = pt.x
                    state[1] = pt.y
                    state[2] = direction[0]
                    state[3] = direction[1]
                    state[4] = lb_width
                    state[5] = lb_type
                    state[6] = rb_width
                    state[7] = rb_type
                    state[8] = int(self.is_virtual_lane(lane))
                    curr_lane_states.append(state)
                lane_states.extend(curr_lane_states)
                state_cache[lane_idx][curr_start_s] = curr_lane_states
                lane_ids.extend([lane_idx] * len(curr_lane_states))
            else:
                lane_states.extend(state_cache[lane_idx][curr_start_s])
                lane_ids.extend([lane_idx] * len(state_cache[lane_idx][curr_start_s]))
        lane_states = np.stack(lane_states)

        return lane_states, lane_ids

    def pack_lane_seq_fast(
        self,
        lane_seq: List[str],
        lanes: List[Dict[str, Any]],
        state_cache: Dict,
    ):
        start_s = 0
        lane_states = list()
        lane_ids = list()
        for lane_idx in lane_seq:
            if lane_idx not in state_cache:
                state_cache[lane_idx] = dict()
            lane = lanes[lane_idx]
            # [Removed] use unary_union for duplicated points
            lane_polyline = np.array(lane["polyline"])
            lane_left_boundary = np.array(lane["left_boundary"]["polyline"])
            lane_right_boundary = np.array(lane["right_boundary"]["polyline"])
            if len(lane_polyline) < 2:
                continue

            dist_polyline = np.linalg.norm(
                lane_polyline[1:, :] - lane_polyline[:-1, :], axis=-1
            )
            original_s = np.concatenate([np.array([0]), np.cumsum(dist_polyline)])
            ss = np.arange(start_s, original_s[-1], self.sample_interval)
            if len(ss) == 0:
                start_s -= original_s[-1]
                continue
            start_s_key = int(start_s * 1000)
            start_s = self.sample_interval - (original_s[-1] - ss[-1])  # next start

            if start_s_key not in state_cache[lane_idx]:
                mask = np.isnan(lane_left_boundary).sum(axis=1) > 0
                lane_left_boundary[mask] = lane_polyline[mask]
                mask = np.isnan(lane_right_boundary).sum(axis=1) > 0
                lane_right_boundary[mask] = lane_polyline[mask]

                # x, y, dir_x, dir_y, lb_width, lb_type, rb_width, rb_type
                curr_lane_states = np.zeros((len(ss), 9))
                insert_idx = np.searchsorted(original_s, ss, side="right") - 1
                idx_left = np.clip(insert_idx, 0, len(original_s) - 1)
                idx_right = np.clip(insert_idx + 1, 0, len(original_s) - 1)
                seg_ratio = np.zeros(len(ss))
                seg_offset = original_s[idx_right] - original_s[idx_left]
                mask = seg_offset > 0
                seg_ratio[mask] = (ss - original_s[idx_left])[mask] / seg_offset[mask]
                idx_fp = idx_left + seg_ratio

                pt_x = np.interp(
                    idx_fp, np.arange(len(original_s)), lane_polyline[:, 0]
                )
                pt_y = np.interp(
                    idx_fp, np.arange(len(original_s)), lane_polyline[:, 1]
                )
                curr_lane_states[:, 0] = pt_x
                curr_lane_states[:, 1] = pt_y
                dir_0 = [
                    lane["unit_directions"][idx_left_][0] for idx_left_ in idx_left
                ]
                dir_1 = [
                    lane["unit_directions"][idx_left_][1] for idx_left_ in idx_left
                ]
                curr_lane_states[:, 2] = dir_0
                curr_lane_states[:, 3] = dir_1
                # TODO: check virtual lane boudary type
                if self.is_virtual_lane(lane):
                    curr_lane_states[:, 4] = -1
                    curr_lane_states[:, 5] = self.lane_boundary_type_mapping["UNKNOWN"]
                    curr_lane_states[:, 6] = -1
                    curr_lane_states[:, 7] = self.lane_boundary_type_mapping["UNKNOWN"]
                    curr_lane_states[:, 8] = int(self.is_virtual_lane(lane))
                else:
                    lb_widths = np.linalg.norm(
                        lane_polyline - lane_left_boundary, axis=-1
                    )
                    rb_widths = np.linalg.norm(
                        lane_polyline - lane_right_boundary, axis=-1
                    )
                    interp_lb_widths = np.interp(
                        idx_fp, np.arange(len(original_s)), lb_widths
                    )
                    interp_rb_widths = np.interp(
                        idx_fp, np.arange(len(original_s)), rb_widths
                    )
                    curr_lane_states[:, 4] = interp_lb_widths
                    curr_lane_states[:, 6] = interp_rb_widths
                    # curr_lane_states[:, 4] = lb_widths[idx_left]
                    # curr_lane_states[:, 6] = rb_widths[idx_left]
                    curr_lane_states[:, 5] = [
                        self.lane_boundary_type_mapping[
                            lane["left_boundary"]["boundary_type"][idx_left_][1][0]
                        ]
                        for idx_left_ in idx_left
                    ]
                    curr_lane_states[:, 7] = [
                        self.lane_boundary_type_mapping[
                            lane["right_boundary"]["boundary_type"][idx_left_][1][0]
                        ]
                        for idx_left_ in idx_left
                    ]
                    curr_lane_states[:, 8] = int(self.is_virtual_lane(lane))
                lane_states.append(curr_lane_states)
                state_cache[lane_idx][start_s_key] = curr_lane_states
                lane_ids.extend([lane_idx] * len(curr_lane_states))
            else:
                lane_states.append(state_cache[lane_idx][start_s_key])
                lane_ids.extend([lane_idx] * len(state_cache[lane_idx][start_s_key]))
        lane_states = np.concatenate(lane_states)

        return lane_states, lane_ids

    def get_lane_sequences(
        self, sub_graphs: List[nx.DiGraph], lane_start_distances: np.array
    ):
        lane_seqs = []
        for sub_graph in sub_graphs:
            lane_seqs.extend(
                self.get_lane_sequences_from_connected_graph(
                    sub_graph, lane_start_distances
                )
            )
        return lane_seqs

    def found_same_lane_seq(self, curr_lane_seq, all_lane_seqs):
        for lane_idx, lane_seq in enumerate(all_lane_seqs):
            if len(curr_lane_seq - lane_seq) == 0:
                return True, -1
            elif len(lane_seq - curr_lane_seq) == 0:
                return True, lane_idx
        return False, -1

    def remove_duplicated_lane_seqs(self, exists_lane_seqs):
        candidate_lane_seq_idx = [
            (idx, lane_seq) for idx, lane_seq in enumerate(exists_lane_seqs)
        ]

        def cmp_lane_seq(seq_a, seq_b):
            if len(seq_a[1]) == len(seq_b[1]):
                if seq_a[0] < seq_b[0]:
                    return -1
                else:
                    return 1
            if len(seq_a[1]) > len(seq_b[1]):
                return -1
            else:
                return 1

        def is_lane_seq_belong_to_another(target_lane_seq, source_lane_seq):
            if len(target_lane_seq) < len(source_lane_seq):
                return False
            is_belong = False
            for i in range(len(target_lane_seq) - len(source_lane_seq) + 1):
                is_belong = True
                for j in range(len(source_lane_seq)):
                    if target_lane_seq[i + j] != source_lane_seq[j]:
                        is_belong = False
                        break
                if is_belong:
                    return True

            return False

        candidate_lane_seq_idx = sorted(
            candidate_lane_seq_idx, key=functools.cmp_to_key(cmp_lane_seq)
        )

        filter_index = []
        for i, source_lane_seq in candidate_lane_seq_idx:
            is_keep = True
            for j in filter_index:
                target_lane_seq = exists_lane_seqs[j]
                if is_lane_seq_belong_to_another(target_lane_seq, source_lane_seq):
                    is_keep = False
                    break
            if is_keep:
                filter_index.append(i)

        filter_index = sorted(filter_index)

        return filter_index

    def process(self, lanes, ego_curr_state):
        # reimpl
        search_lanes = self.collect_lanes_online(lanes, ego_curr_state)
        lane_id_2_idx_map, idx_2_lane_map, lane_seqs = self.dfs_build_lane_graph(
            search_lanes
        )

        collected_lane_seqs = 0
        all_lane_states = np.zeros((self.max_num_lane_seqs, self.max_num_nodes, 9))
        all_lane_states_mask = np.zeros((self.max_num_lane_seqs, self.max_num_nodes))
        all_lane_ids = list()
        all_lane_ids_raw = list()
        exists_lane_seqs = list()
        state_cache = dict()
        for lane_seq in lane_seqs:
            if collected_lane_seqs >= self.max_num_lane_seqs:
                break
            if self.interp_fast:
                lane_states, lane_ids_raw = self.pack_lane_seq_fast(
                    lane_seq, search_lanes, state_cache
                )
            else:
                lane_states, lane_ids_raw = self.pack_lane_seq(
                    lane_seq, search_lanes, state_cache
                )
            segs = self.cut_lane_seq(lane_states, lane_ids_raw, ego_curr_state)
            for state, lane_ids in segs:
                if collected_lane_seqs >= self.max_num_lane_seqs:
                    break
                curr_lane_ids = set([search_lanes[idx]["id"] for idx in lane_ids])
                curr_lane_ids_raw = []
                for idx in lane_ids_raw:
                    lane_id = search_lanes[idx]["id"]
                    if lane_id in curr_lane_ids_raw:
                        continue
                    if lane_id in curr_lane_ids:
                        curr_lane_ids_raw.append(lane_id)

                exists_lane_seqs.append(curr_lane_ids_raw)
                all_lane_states[
                    collected_lane_seqs, : min(len(state), self.max_num_nodes)
                ] = state[: self.max_num_nodes]
                all_lane_states_mask[
                    collected_lane_seqs, : min(len(state), self.max_num_nodes)
                ] = 1
                all_lane_ids.append(curr_lane_ids)
                all_lane_ids_raw.append(curr_lane_ids_raw)
                collected_lane_seqs += 1

        # remove duplicated lane seq
        filter_index = self.remove_duplicated_lane_seqs(exists_lane_seqs)
        for idx, filter_idx in enumerate(filter_index):
            all_lane_states[idx] = all_lane_states[filter_idx]
            all_lane_states_mask[idx] = all_lane_states_mask[filter_idx]
            all_lane_ids[idx] = all_lane_ids[filter_idx]
            all_lane_ids_raw[idx] = all_lane_ids_raw[filter_idx]
        all_lane_states[len(filter_index) :] = 0.0
        all_lane_states_mask[len(filter_index) :] = 0.0
        all_lane_ids = all_lane_ids[: len(filter_index)]
        all_lane_ids_raw = all_lane_ids_raw[: len(filter_index)]

        if self.shuffle:
            random_idxs = np.arange(len(all_lane_ids))
            shuffle(random_idxs)
            all_lane_states[: len(all_lane_ids)] = all_lane_states[: len(all_lane_ids)][
                random_idxs
            ]
            all_lane_states_mask[: len(all_lane_ids)] = all_lane_states_mask[
                : len(all_lane_ids)
            ][random_idxs]
            all_lane_ids = np.array(all_lane_ids, dtype=object)[random_idxs].tolist()
        return (
            all_lane_states,
            all_lane_states_mask.astype("bool"),
            all_lane_ids,
            all_lane_ids_raw,
        )

    def __call__(self, data):
        self.file_path = data["file_path"]
        localmap = get_dict_data(data, self.src_key)
        ego_state = get_dict_data(data, self.src_ego_curr_state_key)
        all_lane_states, all_lane_states_mask, lane_ids, lane_ids_raw = self.process(
            localmap["lanes"], ego_state
        )
        write_dict_data(data, self.tgt_state_key, all_lane_states)
        write_dict_data(data, self.tgt_mask_key, all_lane_states_mask)
        write_dict_data(data, self.tgt_id_map_key, lane_ids)
        if self.save_lane_ids_raw:
            write_dict_data(data, self.tgt_id_map_raw_key, lane_ids_raw)
        return data


class TransformToCurr:
    def __init__(
        self,
        src_obstacle_state_key=("obstacle_state", "state"),
        src_obstacle_state_mask_key=("obstacle_state", "state_mask"),
        src_map_state_key=("map_state", "state"),
        src_map_state_mask_key=("map_state", "state_mask"),
        src_ego_curr_state=("obstacle_state", "ego_curr_state"),
        tgt_obstacle_state_key=("obstacle_state", "state"),
        tgt_map_state_key=("map_state", "state"),
        tgt_rot_aug_state_key=("obstacle_state", "aug_dict"),
        filter_abnormal_value: bool = True,
        random_rot: bool = False,
        random_trans: bool = False,
        theta_rot_range: List[float] = [-np.pi, np.pi],
        x_trans_range: List[float] = [-10, 80],
        y_trans_range: List[float] = [-40, 40],
    ) -> None:
        self.src_obstacle_state_key = src_obstacle_state_key
        self.src_obstacle_state_mask_key = src_obstacle_state_mask_key
        self.src_map_state_key = src_map_state_key
        self.src_map_state_mask_key = src_map_state_mask_key
        self.src_ego_curr_state = src_ego_curr_state
        self.tgt_obstacle_state_key = tgt_obstacle_state_key
        self.tgt_map_state_key = tgt_map_state_key
        self.tgt_rot_aug_state_key = tgt_rot_aug_state_key
        self.filter_abnormal_value = filter_abnormal_value
        self.random_rot = random_rot
        self.random_trans = random_trans

        self.theta_rot_range = theta_rot_range
        self.x_trans_range = x_trans_range
        self.y_trans_range = y_trans_range

    def transform_to_curr(
        self,
        ego_curr_state,
        obstacle_state,
        map_state,
        obstacle_state_mask,
        map_state_mask,
    ):
        rot_theta = ego_curr_state["theta"]
        trans_vec = np.array([0.0, 0.0])

        if self.random_rot:
            rot_theta += np.random.uniform(
                self.theta_rot_range[0], self.theta_rot_range[1]
            )
        if self.random_trans:
            trans_vec[0] = np.random.uniform(
                self.x_trans_range[0], self.x_trans_range[1]
            )
            trans_vec[1] = np.random.uniform(
                self.y_trans_range[0], self.y_trans_range[1]
            )

        rot_mat = np.array(
            [
                [
                    np.cos(rot_theta),
                    -np.sin(rot_theta),
                ],
                [
                    np.sin(rot_theta),
                    np.cos(rot_theta),
                ],
            ]
        ).T
        t_vec = -np.array([ego_curr_state["x"], ego_curr_state["y"]]) @ rot_mat.T
        # obstacle x, y
        obstacle_state[:, :, :2] = (
            obstacle_state[:, :, :2] @ rot_mat.T + t_vec + trans_vec
        )
        # obstacle theta
        obstacle_state[:, :, 2] = obstacle_state[:, :, 2] - rot_theta
        obstacle_state[:, :, 2] = normalize_angles(obstacle_state[:, :, 2])
        # obstacle vx, vy
        obstacle_state[:, :, 3:5] = obstacle_state[:, :, 3:5] @ rot_mat.T
        obstacle_state[~obstacle_state_mask, :5] = 0
        if self.filter_abnormal_value:
            invalid = (np.abs(obstacle_state[:, :, :2]) > 500).sum(axis=-1) > 0
            obstacle_state_mask[invalid] = False
            obstacle_state[~obstacle_state_mask, :5] = 0

        # map x, y
        map_state[:, :, :2] = map_state[:, :, :2] @ rot_mat.T + t_vec + trans_vec
        # map dir x, y
        map_state[:, :, 2:4] = map_state[:, :, 2:4] @ rot_mat.T
        map_state[~map_state_mask, :4] = 0

        aug_dict = {
            "rot_theta": rot_theta,
            "trans_vec": trans_vec,
        }

        return obstacle_state, map_state, obstacle_state_mask, aug_dict

    def __call__(self, data: Dict) -> Dict:
        obstacle_state = get_dict_data(data, self.src_obstacle_state_key)
        obstacle_state_mask = get_dict_data(data, self.src_obstacle_state_mask_key)
        map_state = get_dict_data(data, self.src_map_state_key)
        map_state_mask = get_dict_data(data, self.src_map_state_mask_key)
        ego_curr_state = get_dict_data(data, self.src_ego_curr_state)
        obstacle_state, map_state, obstacle_state_mask, aug_dict = (
            self.transform_to_curr(
                ego_curr_state,
                obstacle_state,
                map_state,
                obstacle_state_mask,
                map_state_mask,
            )
        )
        write_dict_data(data, self.tgt_obstacle_state_key, obstacle_state)
        write_dict_data(data, self.src_obstacle_state_mask_key, obstacle_state_mask)
        write_dict_data(data, self.tgt_map_state_key, map_state)
        write_dict_data(data, self.tgt_rot_aug_state_key, aug_dict)
        return data
