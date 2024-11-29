from shapely.geometry import LineString, Point
from typing import Dict, List, Tuple
import numpy as np
from base import TagData
from tag_functions.high_value_scene.common.basic_info import BasicInfo, ConditionLineCorrType
from tag_functions.high_value_scene.hv_utils.obstacle_filter import (
    ObstacleFilter,
)
from tag_functions.high_value_scene.hv_utils.future_path_collision_checker import (
    FuturePathCollisionChecker,
)
from tag_functions.high_value_scene.hv_utils.basic_func import (
    is_ego_vehicle_always_moving,
    build_obstacle_future_state_polygons,
    calculate_future_path_curvature_and_turn_type,
    find_nearest_condition_linestring,
    judge_lane_change_direction,
    project_point_to_linestring_in_sl_coordinate,
)


class BasicInfoGenerartor:
    def __init__(self) -> None:
        self.basic_info: BasicInfo = BasicInfo()
        self.obstacle_filter: ObstacleFilter = ObstacleFilter(
            filter_obs_max_l=5.0,
            front_vehicle_rel_x=10.0,
            front_vehicle_rel_y=0.5,
        )
        self.future_path_collision_checker = FuturePathCollisionChecker()

    def process(self, data: TagData, params: Dict) -> BasicInfo:
        self.calculate_ego_vehicle_basic_info(data)

        self.calculate_future_path_basic_info(data)

        self.calculate_obstacles_and_curbs_basic_info(data)

        self.calculate_future_interaction_with_obstacles_and_curbs_info(
            data, params
        )

        self.calculate_condition_lane_seq_basic_info(data)

        self.calculate_lane_change_basic_info()

        return self.basic_info

    def calculate_ego_vehicle_basic_info(self, data: TagData) -> None:
        obstacles = data.label_scene.obstacles

        self.basic_info.is_ego_vehicle_always_moving = (
            is_ego_vehicle_always_moving(obstacles[-9])
        )

    def calculate_future_path_basic_info(self, data: TagData) -> None:
        ego_path_info = data.label_scene.ego_path_info

        # 是否通过路口
        self.basic_info.is_cross_junction = any(
            juntion_id is not None
            for juntion_id in ego_path_info.in_junction_id
        )

        # 计算future path中每一个点的曲率和转向
        (
            self.basic_info.future_path_curvature,
            self.basic_info.future_path_turn_type,
        ) = calculate_future_path_curvature_and_turn_type(
            ego_path_info.future_path
        )

        # 对于future path所关联的lane id，计算每条lane经过了多少个future path点
        for lane_info in ego_path_info.corr_lane_id:
            if lane_info is None or len(lane_info) == 0:
                continue

            lane_id = lane_info[0][0]
            self.basic_info.lane_id_to_future_path_waypoint_count[lane_id] = (
                self.basic_info.lane_id_to_future_path_waypoint_count.get(
                    lane_id, 0
                )
                + 1
            )

        # 计算曲率绝对值的最大值，过滤曲折轨迹(比如倒车时)
        self.basic_info.max_abs_path_curvature = np.abs(
            self.basic_info.future_path_curvature
        ).max()

    def calculate_obstacles_and_curbs_basic_info(self, data: TagData) -> None:
        obstacles = data.label_scene.obstacles
        curb_decision = data.label_scene.label_res["curb_label"].get(
            "decision", None
        )

        # 过滤l绝对值大的curb，并计算curbs的linestring
        self.basic_info.curbs_linestring_map = (
            self.obstacle_filter.build_curbs_linestring(curb_decision)
        )

        # 筛选出静态障碍物，并计算其polygon
        (
            self.basic_info.static_obstacles_map,
            self.basic_info.static_obstacles_polygons_map,
        ) = self.obstacle_filter.build_static_obstacle_polygons(obstacles)

        # 筛选出动态障碍物
        self.basic_info.moving_obstacles_map = (
            self.obstacle_filter.find_moving_obstacles(obstacles)
        )

        # 计算动态障碍物的未来状态的polygon
        self.basic_info.moving_obstacles_future_state_polygons_map = (
            build_obstacle_future_state_polygons(
                self.basic_info.moving_obstacles_map
            )
        )

    def calculate_future_interaction_with_obstacles_and_curbs_info(
        self, data: TagData, params: Dict
    ) -> None:
        ego_path_info = data.label_scene.ego_path_info
        obstacles = data.label_scene.obstacles
        curb_decision = data.label_scene.label_res["curb_label"].get(
            "decision", None
        )
        curbs_interactive_lat_type = (
            curb_decision["interactive_lat_type"]
            if curb_decision is not None
            else {}
        )

        (
            self.basic_info.future_narrow_road_states,
            self.basic_info.future_narrow_road_states_loose_threshold,
            self.basic_info.future_path_nearby_curb_indexes,
        ) = self.future_path_collision_checker.check_future_path_distance_to_curb_and_static_obs(
            params,
            ego_path_info,
            obstacles[-9],
            self.basic_info.static_obstacles_map,
            self.basic_info.static_obstacles_polygons_map,
            self.basic_info.curbs_linestring_map,
            curbs_interactive_lat_type,
        )

        self.basic_info.future_bypass_junction_curb = self.future_path_collision_checker.check_future_path_bypass_static_object_in_junction(
            params["large_curvature_threshold"],
            ego_path_info,
            self.basic_info.future_path_curvature,
            self.basic_info.future_path_turn_type,
            self.basic_info.future_narrow_road_states_loose_threshold,
        )

        self.basic_info.future_interaction_with_moving_obs = self.future_path_collision_checker.check_distance_to_moving_obs_for_future_states(
            params,
            obstacles[-9],
            self.basic_info.moving_obstacles_map,
            self.basic_info.moving_obstacles_future_state_polygons_map,
        )

    def calculate_condition_lane_seq_basic_info(self, data: TagData) -> None:
        # 获取所有的condition lane seq pair
        condition_pair = data.condition_res.lane_seq_pair
        lane_seq_ids = data.condition_res.seq_lane_ids_raw
        start_lane_seq_ids = []
        end_lane_seq_ids = []
        for idx_start, idx_end, _ in condition_pair:
            start_lane_seq_ids.append(
                lane_seq_ids[idx_start] if idx_start != -1 else []
            )
            end_lane_seq_ids.append(
                lane_seq_ids[idx_end] if idx_end != -1 else []
            )

        # 找到离future path横向距离最近的condition lane seq linestring
        (
            self.basic_info.nearest_condition_linestring,
            self.basic_info.nearest_condition_linestring_corr_type
        ) = (
            find_nearest_condition_linestring(
                data,
                start_lane_seq_ids,
                end_lane_seq_ids,
            )
        )

        # 计算future path的每一个点的sl坐标，以及在condition linestring上的投影点
        if self.basic_info.nearest_condition_linestring is not None:
            future_path = data.label_scene.ego_path_info.future_path
            for point in future_path:
                path_point = Point(point)
                project_success = False
                proj_s_list, proj_l_list, linestring_idx_list, corr_line_type = [], [], [], []

                for idx, (linestring, corr_type) in enumerate(
                    self.basic_info.nearest_condition_linestring,
                    self.basic_info.nearest_condition_linestring_corr_type,
                ):
                    (
                        proj_s,
                        proj_l,
                    ) = project_point_to_linestring_in_sl_coordinate(
                        linestring, path_point
                    )
                    if proj_s is not None and proj_l is not None:
                        proj_s_list.append(proj_s)
                        proj_l_list.append(proj_l)
                        linestring_idx_list.append(idx)
                        corr_line_type.append(corr_type)

                if len(proj_l_list) > 0:
                    min_proj_l_index = np.argmin(np.abs(np.array(proj_l_list)))
                    min_proj_s = proj_s_list[min_proj_l_index]
                    min_proj_l = proj_l_list[min_proj_l_index]
                    corr_point = self.basic_info.nearest_condition_linestring[
                        linestring_idx_list[min_proj_l_index]
                    ].interpolate(min_proj_s)
                    self.basic_info.future_path_points_sl_coordinate_projected_to_condition.append(
                        (min_proj_s, min_proj_l, corr_point)
                    )
                    self.basic_info.future_path_points_sl_coordinate_projected_to_condition_corr_type.append(corr_line_type[min_proj_l_index])
                    project_success = True

                if not project_success:
                    self.basic_info.future_path_points_sl_coordinate_projected_to_condition.append(
                        (None, None, None)
                    )
                    self.basic_info.future_path_points_sl_coordinate_projected_to_condition_corr_type.append(ConditionLineCorrType.NONE)

    def calculate_lane_change_basic_info(self) -> None:
        self.basic_info.lane_change_direction = judge_lane_change_direction(
            self.basic_info.future_path_points_sl_coordinate_projected_to_condition
        )
