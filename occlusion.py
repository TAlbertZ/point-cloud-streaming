import numpy as np
import pdb
import time
import pickle as pk
import logging
import os
import math
from numpy import linalg

import params
from hidden_points_removal import HiddenPointsRemoval

np.random.seed(7)

class Occlusion():
    '''
        multiple point/tile based occlusion algorithm
    '''
    def __init__(self, valid_tiles_obj, fov_traces_obj, origin):
        self.valid_tiles_obj = valid_tiles_obj
        self.fov_traces_obj = fov_traces_obj
        self.origin = origin

    def calculate_distance(self, point1, point2):
        distance = np.linalg.norm(point1 - point2)
        # return 0
        return distance

    def segment_occlusion(self, viewpoints, update_start_idx, update_end_idx, emitting_buffer, overlap_ratio_history):
        # HPR
        viewing_probability = []
        true_viewing_probability = []
        num_visible_pts_per_tile = []
        num_pts_per_tile = []
        distances = []
        cur_visibility = np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME))

        for frame_idx in range(update_start_idx, update_end_idx + 1):
            start_time = time.time()
            print(frame_idx)
            viewing_probability.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            true_viewing_probability.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            num_visible_pts_per_tile.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            num_pts_per_tile.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            distances.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            tile_center_points = []
            points_of_interest = []
            viewpoint = {
                "x": 0,
                'y': 0,
                "z": 0,
                "pitch": 0,
                "yaw": 0,
                "roll": 0
            }
            for key in viewpoint.keys():
                viewpoint[key] = viewpoints[key][frame_idx - update_start_idx]

            if frame_idx < params.BUFFER_LENGTH:
                continue
            valid_tiles = self.valid_tiles_obj.valid_tiles[
                (frame_idx - params.BUFFER_LENGTH) % params.
                NUM_FRAMES]  # cubic array denotes whether a tile is empty or not

            ### fixed / constant obj ############
            # valid_tiles = self.valid_tiles_obj.valid_tiles[0]
            #####################################

            ######### segment based occlusion #################
            if (frame_idx - update_start_idx) % params.OCCLUSION_FREQ:
                viewing_probability[frame_idx - update_start_idx] = cur_visibility * valid_tiles

                # calculate distance
                distances[frame_idx - update_start_idx] = distances[int((frame_idx - update_start_idx) // params.OCCLUSION_FREQ * params.OCCLUSION_FREQ)].copy()
                continue
            else:
                if (frame_idx - update_start_idx) % params.SEGMENT_LEN == 0:
                    cur_visibility = np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))
            #####################################################

            # read from ply files with highest  level of detail
            ply_frame_idx = (frame_idx - params.BUFFER_LENGTH
                             ) % params.NUM_FRAMES + params.FRAME_IDX_BIAS

            # my heuristic: viewport->sort by distance->occlusion: O(n^2) worst case
            # store hash tables of vectors and distances between predicted viewpoint and tiles' center
            tiles_of_interest = []

            viewpoint_position = np.array(
                [viewpoint["x"], viewpoint["y"], viewpoint["z"]])
            viewpoint_position = np.expand_dims(viewpoint_position, axis=0)

            true_viewpoint = self.fov_traces_obj.fov_traces[frame_idx]
            true_position = np.array([
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["x"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["y"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["z"]]
            ])
            true_position = np.expand_dims(true_position, axis=0)
            # print(viewpoint_position)
            print("init--- ", time.time() - start_time, " seconds ---")

            # get tilecenter/point coordinate
            tile_xs, tile_ys, tile_zs = valid_tiles.nonzero()
            for point_idx in range(len(tile_xs)):
                ply_x = tile_xs[point_idx]
                ply_y = tile_ys[point_idx]
                ply_z = tile_zs[point_idx]

                ply_tile_file = params.PLY_PATH + '/' + str(
                    ply_frame_idx) + '/' + params.VIDEO_NAME + '_' + str(
                        ply_frame_idx) + '_' + str(ply_x // 10) + str(
                            ply_x % 10) + '_' + str(ply_y // 10) + str(
                                ply_y % 10) + '_' + str(ply_z // 10) + str(
                                    ply_z % 10) + '_3.ply'

                f = open(ply_tile_file, "r")
                for line in f:
                    # print(line.split())
                    # pdb.set_trace()
                    line_list = line.split()
                    if not line_list[0][0].isdigit():
                        continue
                    pt_x = int(float(line_list[0]))
                    pt_y = int(float(line_list[1]))
                    pt_z = int(float(line_list[2]))
                    num_pts_per_tile[
                        frame_idx - update_start_idx][ply_x][ply_y][ply_z] += 1

                    pt_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate_4_ply(
                        pt_x, pt_y, pt_z)
                    points_of_interest.append(pt_coordinate)

                # tile_center_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate(
                #     ply_x, ply_y, ply_z)
                # tile_center_points.append(tile_center_coordinate)
            print("collect--- ", time.time() - start_time, " seconds ---")

            # # modify object coordinate: origin at obj's bottom center
            # tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(
            #     self.origin, tile_center_points)

            # # mirror x and z axis to let obj face the user start view orientation
            # tile_center_points = np.array(tile_center_points)
            # tile_center_points[:, 0] = -tile_center_points[:, 0]
            # tile_center_points[:, 2] = -tile_center_points[:, 2]
            # pdb.set_trace()

            # modify object coordinate: origin at obj's bottom center
            points_of_interest = self.valid_tiles_obj.change_tile_coordinates_origin(
                self.origin, points_of_interest)

            # mirror x and z axis to let obj face the user start view orientation
            points_of_interest = np.array(points_of_interest)
            points_of_interest[:, 0] = -points_of_interest[:, 0]
            points_of_interest[:, 2] = -points_of_interest[:, 2]

            # vecs_from_vp_to_tilectr = tile_center_points - viewpoint_position
            # dist_from_vp_to_tilectr = linalg.norm(vecs_from_vp_to_tilectr,
            #                                       axis=1)
            # print("pre-processing--- ", time.time() - start_time, " seconds ---")

            # # sort wrt distance
            # sort_dist_idx = np.argsort(dist_from_vp_to_tilectr)
            # print("sort--- ", time.time() - start_time, " seconds ---")

            # ################################ occlusion ###############################
            # set_occluded_idx = {}
            # visible_tilectr_idx = []
            # for i in range(len(sort_dist_idx) - 1):
            #     ref_tilectr_idx = sort_dist_idx[i]
            #     if ref_tilectr_idx in set_occluded_idx and set_occluded_idx[
            #             ref_tilectr_idx] >= 3:
            #         continue
            #     visible_tilectr_idx.append(ref_tilectr_idx)
            #     ref_dist = dist_from_vp_to_tilectr[ref_tilectr_idx]
            #     ref_vec = vecs_from_vp_to_tilectr[ref_tilectr_idx]
            #     ang_thresh = params.TILE_SIDE_LEN / ref_dist
            #     dist_diff_thresh = params.TILE_SIDE_LEN
            #     for j in range(i + 1, len(sort_dist_idx)):
            #         cur_tilectr_idx = sort_dist_idx[j]
            #         if cur_tilectr_idx in set_occluded_idx and set_occluded_idx[
            #                 cur_tilectr_idx] >= 3:
            #             continue
            #         # check if current tile is already occluded or not
            #         cur_dist = dist_from_vp_to_tilectr[cur_tilectr_idx]
            #         if cur_dist - ref_dist <= dist_diff_thresh * 1.5:
            #             continue
            #         cur_vec = vecs_from_vp_to_tilectr[cur_tilectr_idx]
            #         # in radian
            #         ang_between_ref_and_cur = np.arccos(
            #             np.dot(ref_vec, cur_vec) /
            #             (linalg.norm(ref_vec) * linalg.norm(cur_vec)))

            #         vec_from_ref_to_cur = cur_vec - ref_vec
            #         ang_between_refvec_and_refcurvec = np.arccos(
            #             np.dot(ref_vec, vec_from_ref_to_cur) /
            #             (linalg.norm(ref_vec) *
            #              linalg.norm(vec_from_ref_to_cur)))

            #         if ang_between_ref_and_cur < ang_thresh * 1 and ang_between_refvec_and_refcurvec < ang_thresh * 3:
            #             # set_occluded_idx.add(cur_tilectr_idx)
            #             if cur_tilectr_idx not in set_occluded_idx:
            #                 set_occluded_idx[cur_tilectr_idx] = 0
            #             set_occluded_idx[cur_tilectr_idx] += 1

            # # modify object coordinate: origin at viewpoint
            # # tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(viewpoint_position, tile_center_points)
            # print("hpr--- ", time.time() - start_time, " seconds ---")

            # ######################### true occlusion ###########################
            # true_vecs_from_vp_to_tilectr = tile_center_points - true_position
            # true_dist_from_vp_to_tilectr = linalg.norm(
            #     true_vecs_from_vp_to_tilectr, axis=1)

            # # sort wrt distance
            # true_sort_dist_idx = np.argsort(true_dist_from_vp_to_tilectr)

            # # occlusion
            # set_occluded_idx = set()
            # true_visible_tilectr_idx = []
            # for i in range(len(true_sort_dist_idx) - 1):
            #     ref_tilectr_idx = true_sort_dist_idx[i]
            #     if ref_tilectr_idx in set_occluded_idx:
            #         continue
            #     true_visible_tilectr_idx.append(ref_tilectr_idx)
            #     ref_dist = true_dist_from_vp_to_tilectr[ref_tilectr_idx]
            #     ref_vec = true_vecs_from_vp_to_tilectr[ref_tilectr_idx]
            #     ang_thresh = params.TILE_SIDE_LEN / ref_dist
            #     for j in range(i + 1, len(true_sort_dist_idx)):
            #         cur_tilectr_idx = true_sort_dist_idx[j]
            #         if cur_tilectr_idx in set_occluded_idx:
            #             continue
            #         # check if current tile is already occluded or not
            #         cur_dist = true_dist_from_vp_to_tilectr[cur_tilectr_idx]
            #         cur_vec = true_vecs_from_vp_to_tilectr[cur_tilectr_idx]
            #         # in radian
            #         ang_between = np.arccos(
            #             np.dot(ref_vec, cur_vec) /
            #             (linalg.norm(ref_vec) * linalg.norm(cur_vec)))
            #         if ang_between <= ang_thresh:
            #             set_occluded_idx.add(cur_tilectr_idx)
            # print("hpr_true--- ", time.time() - start_time, " seconds ---")

            # HPR
            HPR_obj = HiddenPointsRemoval(points_of_interest)
            # # First subplot
            # fig = plt.figure(figsize = plt.figaspect(0.5))
            # plt.title('Test Case With A Sphere (Left) and Visible Sphere Viewed From Well Above (Right)')
            # ax = fig.add_subplot(1,2,1, projection = '3d')
            # ax.scatter(HPR_obj.points[:, 0], HPR_obj.points[:, 1], HPR_obj.points[:, 2], c='r', marker='^') # Plot all points
            # ax.set_xlabel('X Axis')
            # ax.set_ylabel('Y Axis')
            # ax.set_zlabel('Z Axis')
            # plt.show()
            # start_time = time.time()
            print("init--- ", time.time() - start_time, " seconds ---")
            flippedPoints = HPR_obj.sphericalFlip(
                viewpoint_position, math.pi
            )  # Reflect the point cloud about a sphere centered at viewpoint_position
            myHull = HPR_obj.convexHull(
                flippedPoints
            )  # Take the convex hull of the center of the sphere and the deformed point cloud

            print("hpr--- ", time.time() - start_time, " seconds ---")

            true_flippedPoints = HPR_obj.sphericalFlip(
                true_position, math.pi
            )  # Reflect the point cloud about a sphere centered at viewpoint_position
            true_myHull = HPR_obj.convexHull(
                true_flippedPoints
            )  # Take the convex hull of the center of the sphere and the deformed point cloud
            print("hpr_true--- ", time.time() - start_time, " seconds ---")

            # ax = fig.add_subplot(1,2,2, projection = '3d')
            # ax.scatter(flippedPoints[:, 0], flippedPoints[:, 1], flippedPoints[:, 2], c='r', marker='^') # Plot all points
            # ax.set_xlabel('X Axis')
            # ax.set_ylabel('Y Axis')
            # ax.set_zlabel('Z Axis')
            # plt.show()

            # HPR_obj.plot(visible_hull_points=myHull)
            # pdb.set_trace()

            ### TODO by Tongyu: use gradient descent to optimize radius of HPR ####

            ###############################################################

            ############ check which visible points are within fov #############
            predicted_visible_tiles_set = set()
            for vertex in myHull.vertices[:-1]:
                vertex_coordinate = np.array([
                    points_of_interest[vertex, 0],
                    points_of_interest[vertex, 1],
                    points_of_interest[vertex, 2]
                ])
                vector_from_viewpoint_to_tilecenter = vertex_coordinate - viewpoint_position
                pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
                yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
                viewing_ray_unit_vector = np.array([
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)
                ])
                intersection_angle = np.arccos(
                    np.dot(vector_from_viewpoint_to_tilecenter,
                           viewing_ray_unit_vector) /
                    np.linalg.norm(vector_from_viewpoint_to_tilecenter))
                if intersection_angle <= params.FOV_DEGREE_SPAN:
                    # # viewable => viewing probability = 1
                    # viewable_tile_idx = (tile_xs[vertex], tile_ys[vertex],
                    #                      tile_zs[vertex]
                    #                      )  # position among all tiles
                    point_x = (
                        -vertex_coordinate[0] + self.origin[0] -
                        params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    point_y = (
                        vertex_coordinate[1] + self.origin[1] -
                        params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    point_z = (
                        -vertex_coordinate[2] + self.origin[2] -
                        params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)

                    viewable_tile_idx = np.array([
                        int(point_x) // params.NUM_VOXEL_TILESIDE,
                        int(point_y) // params.NUM_VOXEL_TILESIDE,
                        int(point_z) // params.NUM_VOXEL_TILESIDE
                    ])

            # for tilectr_idx in visible_tilectr_idx:
            #     vertex_coordinate = tile_center_points[tilectr_idx]
            #     vector_from_viewpoint_to_tilecenter = vecs_from_vp_to_tilectr[
            #         tilectr_idx]
            #     pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
            #     yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
            #     viewing_ray_unit_vector = np.array([
            #         np.sin(yaw) * np.cos(pitch),
            #         np.sin(pitch),
            #         np.cos(yaw) * np.cos(pitch)
            #     ])
            #     intersection_angle = np.arccos(
            #         np.dot(vector_from_viewpoint_to_tilecenter,
            #                viewing_ray_unit_vector) /
            #         np.linalg.norm(vector_from_viewpoint_to_tilecenter))
            #     if intersection_angle <= params.FOV_DEGREE_SPAN:
            #         viewable_tile_idx = np.array([
            #             tile_xs[tilectr_idx], tile_ys[tilectr_idx],
            #             tile_zs[tilectr_idx]
            #         ])
                    # as long as the tile is visible, the viewing probability is 1 (which means the overlap ratio is 100%)
                    if np.max(viewable_tile_idx
                              ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
                        # pdb.set_trace()
                        viewable_tile_idx[np.where(
                            viewable_tile_idx ==
                            params.NUM_TILES_PER_SIDE_IN_A_FRAME
                        )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1
                    viewing_probability[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = 1
                    visible_tiles_pos = viewing_probability[frame_idx - update_start_idx].nonzero()
                    cur_visibility[visible_tiles_pos] = 1
                    cur_visibility *= valid_tiles
                    viewing_probability[frame_idx - update_start_idx] = cur_visibility.copy()
                    # num_visible_pts_per_tile[frame_idx - update_start_idx][
                    #     viewable_tile_idx[0]][viewable_tile_idx[1]][
                    #         viewable_tile_idx[2]] += 1

                    tile_center_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate(
                        ply_x, ply_y, ply_z)
                    tile_center_points = [tile_center_coordinate]

                    # modify object coordinate: origin at obj's bottom center
                    tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(
                        self.origin, tile_center_points)

                    # mirror x and z axis to let obj face the user start view orientation
                    tile_center_points = np.array(tile_center_points)
                    tile_center_points[:, 0] = -tile_center_points[:, 0]
                    tile_center_points[:, 2] = -tile_center_points[:, 2]

                    tile_center_coordinate = tile_center_points[0]

                    distances[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = self.calculate_distance(
                                tile_center_coordinate, viewpoint_position)

                    predicted_visible_tiles_set.add(tuple(viewable_tile_idx))
            print("check fov--- ", time.time() - start_time, " seconds ---")

            # visible_tiles_idx = num_visible_pts_per_tile[
            #     frame_idx - update_start_idx].nonzero()

            # print(
            #     num_visible_pts_per_tile[frame_idx -
            #                              update_start_idx][visible_tiles_idx])
            # print(
            #     num_visible_pts_per_tile[frame_idx -
            #                              update_start_idx][visible_tiles_idx] /
            #     num_pts_per_tile[frame_idx -
            #                      update_start_idx][visible_tiles_idx])
            # pdb.set_trace()

            true_visible_tiles_set = set()
            for vertex in true_myHull.vertices[:-1]:
                vertex_coordinate = np.array([
                    points_of_interest[vertex, 0],
                    points_of_interest[vertex, 1], points_of_interest[vertex,
                                                                      2]
                ])
                vector_from_viewpoint_to_tilecenter = vertex_coordinate - true_position
                pitch = true_viewpoint[params.MAP_6DOF_TO_HMD_DATA[
                    "pitch"]] / params.RADIAN_TO_DEGREE
                yaw = true_viewpoint[params.MAP_6DOF_TO_HMD_DATA[
                    "yaw"]] / params.RADIAN_TO_DEGREE
                viewing_ray_unit_vector = np.array([
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)
                ])
                intersection_angle = np.arccos(
                    np.dot(vector_from_viewpoint_to_tilecenter,
                           viewing_ray_unit_vector) /
                    np.linalg.norm(vector_from_viewpoint_to_tilecenter))
                if intersection_angle <= params.FOV_DEGREE_SPAN:
                    # # viewable => viewing probability = 1
                    # viewable_tile_idx = (tile_xs[vertex], tile_ys[vertex],
                    #                      tile_zs[vertex]
                    #                      )  # position among all tiles
                    # as long as the tile is visiblle, the viewing probability is 1 (which means the overlap ratio is 100%)
                    point_x = (
                        -vertex_coordinate[0] + self.origin[0] -
                        params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    point_y = (
                        vertex_coordinate[1] + self.origin[1] -
                        params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    point_z = (
                        -vertex_coordinate[2] + self.origin[2] -
                        params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)

                    viewable_tile_idx = np.array([
                        int(point_x) // params.NUM_VOXEL_TILESIDE,
                        int(point_y) // params.NUM_VOXEL_TILESIDE,
                        int(point_z) // params.NUM_VOXEL_TILESIDE
                    ])
                    if np.max(viewable_tile_idx
                              ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
                        # pdb.set_trace()
                        viewable_tile_idx[np.where(
                            viewable_tile_idx ==
                            params.NUM_TILES_PER_SIDE_IN_A_FRAME
                        )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1
                    # true_viewing_probability[frame_idx - update_start_idx][
                    #     viewable_tile_idx[0]][viewable_tile_idx[1]][
                    #         viewable_tile_idx[2]] = 1
                    # distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)

            # for tilectr_idx in true_visible_tilectr_idx:
            #     vertex_coordinate = tile_center_points[tilectr_idx]
            #     vector_from_viewpoint_to_tilecenter = true_vecs_from_vp_to_tilectr[
            #         tilectr_idx]
            #     pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
            #     yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
            #     viewing_ray_unit_vector = np.array([
            #         np.sin(yaw) * np.cos(pitch),
            #         np.sin(pitch),
            #         np.cos(yaw) * np.cos(pitch)
            #     ])
            #     intersection_angle = np.arccos(
            #         np.dot(vector_from_viewpoint_to_tilecenter,
            #                viewing_ray_unit_vector) /
            #         np.linalg.norm(vector_from_viewpoint_to_tilecenter))
            #     if intersection_angle <= params.FOV_DEGREE_SPAN:
            #         viewable_tile_idx = np.array([
            #             tile_xs[tilectr_idx], tile_ys[tilectr_idx],
            #             tile_zs[tilectr_idx]
            #         ])
            #         if np.max(viewable_tile_idx
            #                   ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
            #             # pdb.set_trace()
            #             viewable_tile_idx[np.where(
            #                 viewable_tile_idx ==
            #                 params.NUM_TILES_PER_SIDE_IN_A_FRAME
            #             )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1
                    true_viewing_probability[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = 1
                    # distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)
                    true_visible_tiles_set.add(tuple(viewable_tile_idx))
            print("check true fov--- ", time.time() - start_time, " seconds ---")

            ########################################################################
            if not emitting_buffer:
                # update overlap_ratio history
                overlap_tiles_set = true_visible_tiles_set.intersection(
                    predicted_visible_tiles_set)
                overlap_ratio = len(overlap_tiles_set) / len(
                    true_visible_tiles_set)
                if len(predicted_visible_tiles_set) == 0:
                    overlap_ratio = 0
                    # pdb.set_trace()
                else:
                    overlap_ratio = len(overlap_tiles_set) / len(
                        predicted_visible_tiles_set)
                if frame_idx >= params.BUFFER_LENGTH:
                    overlap_ratio_history[
                        frame_idx - update_start_idx].append(overlap_ratio)
                # self.overlap_ratio_history[frame_idx - update_start_idx].pop(0)
                # if overlap_ratio < 1:
                # 	pdb.set_trace()
                # if self.update_step >= params.BUFFER_LENGTH // params.FPS:
                #     pdb.set_trace()
            # print("overlap--- ", time.time() - start_time, " seconds ---")
            print("frame hpr--- ", time.time() - start_time, " seconds ---")

        return viewing_probability, distances, true_viewing_probability

    def frame_occlusion(self, viewpoints, update_start_idx, update_end_idx, emitting_buffer, overlap_ratio_history):
        # HPR
        viewing_probability = []
        true_viewing_probability = []
        num_visible_pts_per_tile = []
        num_pts_per_tile = []
        distances = []

        for frame_idx in range(update_start_idx, update_end_idx + 1):
            start_time = time.time()
            # print(frame_idx)
            viewing_probability.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            true_viewing_probability.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            num_visible_pts_per_tile.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            num_pts_per_tile.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            distances.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            tile_center_points = []
            points_of_interest = []
            viewpoint = {
                "x": 0,
                'y': 0,
                "z": 0,
                "pitch": 0,
                "yaw": 0,
                "roll": 0
            }
            for key in viewpoint.keys():
                viewpoint[key] = viewpoints[key][frame_idx - update_start_idx]

            if frame_idx < params.BUFFER_LENGTH:
                continue
            valid_tiles = self.valid_tiles_obj.valid_tiles[
                (frame_idx - params.BUFFER_LENGTH) % params.
                NUM_FRAMES]  # cubic array denotes whether a tile is empty or not

            ### fixed / constant obj ############
            # valid_tiles = self.valid_tiles_obj.valid_tiles[0]
            #####################################

            # read from ply files with highest  level of detail
            ply_frame_idx = (frame_idx - params.BUFFER_LENGTH
                             ) % params.NUM_FRAMES + params.FRAME_IDX_BIAS

            # my heuristic: viewport->sort by distance->occlusion: O(n^2) worst case
            # store hash tables of vectors and distances between predicted viewpoint and tiles' center
            tiles_of_interest = []

            viewpoint_position = np.array(
                [viewpoint["x"], viewpoint["y"], viewpoint["z"]])
            viewpoint_position = np.expand_dims(viewpoint_position, axis=0)

            true_viewpoint = self.fov_traces_obj.fov_traces[frame_idx]
            true_position = np.array([
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["x"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["y"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["z"]]
            ])
            true_position = np.expand_dims(true_position, axis=0)
            # print(viewpoint_position)
            # print("init--- ", time.time() - start_time, " seconds ---")

            # get tilecenter/point coordinate
            # TODO load points_of_interest and tile_center_points from pickle.
            tile_xs, tile_ys, tile_zs = valid_tiles.nonzero()
            for point_idx in range(len(tile_xs)):
                ply_x = tile_xs[point_idx]
                ply_y = tile_ys[point_idx]
                ply_z = tile_zs[point_idx]

                # ply_tile_file = params.PLY_PATH + '/' + str(
                #     ply_frame_idx) + '/' + params.VIDEO_NAME + '_' + str(
                #         ply_frame_idx) + '_' + str(ply_x // 10) + str(
                #             ply_x % 10) + '_' + str(ply_y // 10) + str(
                #                 ply_y % 10) + '_' + str(ply_z // 10) + str(
                #                     ply_z % 10) + '_3.ply'

                # f = open(ply_tile_file, "r")
                # for line in f:
                #     # print(line.split())
                #     # pdb.set_trace()
                #     line_list = line.split()
                #     if not line_list[0][0].isdigit():
                #         continue
                #     pt_x = int(float(line_list[0]))
                #     pt_y = int(float(line_list[1]))
                #     pt_z = int(float(line_list[2]))
                #     num_pts_per_tile[
                #         frame_idx - update_start_idx][ply_x][ply_y][ply_z] += 1

                #     pt_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate_4_ply(
                #         pt_x, pt_y, pt_z)
                #     points_of_interest.append(pt_coordinate)

                tile_center_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate(
                    ply_x, ply_y, ply_z)
                tile_center_points.append(tile_center_coordinate)
            # print("collect--- ", time.time() - start_time, " seconds ---")

            # modify object coordinate: origin at obj's bottom center
            tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(
                self.origin, tile_center_points)

            # mirror x and z axis to let obj face the user start view orientation
            tile_center_points = np.array(tile_center_points)
            tile_center_points[:, 0] = -tile_center_points[:, 0]
            tile_center_points[:, 2] = -tile_center_points[:, 2]
            # pdb.set_trace()

            # # modify object coordinate: origin at obj's bottom center
            # points_of_interest = self.valid_tiles_obj.change_tile_coordinates_origin(
            #     self.origin, points_of_interest)

            # # mirror x and z axis to let obj face the user start view orientation
            # points_of_interest = np.array(points_of_interest)
            # points_of_interest[:, 0] = -points_of_interest[:, 0]
            # points_of_interest[:, 2] = -points_of_interest[:, 2]

            # vecs_from_vp_to_tilectr = tile_center_points - viewpoint_position
            # dist_from_vp_to_tilectr = linalg.norm(vecs_from_vp_to_tilectr,
            #                                       axis=1)
            # print("pre-processing--- ", time.time() - start_time, " seconds ---")

            # sort wrt distance
            # sort_dist_idx = np.argsort(dist_from_vp_to_tilectr)
            # print("sort--- ", time.time() - start_time, " seconds ---")

            ################################ occlusion ###############################
            # set_occluded_idx = {}
            # visible_tilectr_idx = []
            # for i in range(len(sort_dist_idx) - 1):
            #     ref_tilectr_idx = sort_dist_idx[i]
            #     if ref_tilectr_idx in set_occluded_idx and set_occluded_idx[
            #             ref_tilectr_idx] >= 3:
            #         continue
            #     visible_tilectr_idx.append(ref_tilectr_idx)
            #     ref_dist = dist_from_vp_to_tilectr[ref_tilectr_idx]
            #     ref_vec = vecs_from_vp_to_tilectr[ref_tilectr_idx]
            #     ang_thresh = params.TILE_SIDE_LEN / ref_dist
            #     dist_diff_thresh = params.TILE_SIDE_LEN
            #     for j in range(i + 1, len(sort_dist_idx)):
            #         cur_tilectr_idx = sort_dist_idx[j]
            #         if cur_tilectr_idx in set_occluded_idx and set_occluded_idx[
            #                 cur_tilectr_idx] >= 3:
            #             continue
            #         # check if current tile is already occluded or not
            #         cur_dist = dist_from_vp_to_tilectr[cur_tilectr_idx]
            #         if cur_dist - ref_dist <= dist_diff_thresh * 1.5:
            #             continue
            #         cur_vec = vecs_from_vp_to_tilectr[cur_tilectr_idx]
            #         # in radian
            #         ang_between_ref_and_cur = np.arccos(
            #             np.dot(ref_vec, cur_vec) /
            #             (linalg.norm(ref_vec) * linalg.norm(cur_vec)))

            #         vec_from_ref_to_cur = cur_vec - ref_vec
            #         ang_between_refvec_and_refcurvec = np.arccos(
            #             np.dot(ref_vec, vec_from_ref_to_cur) /
            #             (linalg.norm(ref_vec) *
            #              linalg.norm(vec_from_ref_to_cur)))

            #         if ang_between_ref_and_cur < ang_thresh * 1 and ang_between_refvec_and_refcurvec < ang_thresh * 3:
            #             # set_occluded_idx.add(cur_tilectr_idx)
            #             if cur_tilectr_idx not in set_occluded_idx:
            #                 set_occluded_idx[cur_tilectr_idx] = 0
            #             set_occluded_idx[cur_tilectr_idx] += 1

            # ######################### true occlusion ###########################
            # true_vecs_from_vp_to_tilectr = tile_center_points - true_position
            # true_dist_from_vp_to_tilectr = linalg.norm(
            #     true_vecs_from_vp_to_tilectr, axis=1)

            # # sort wrt distance
            # true_sort_dist_idx = np.argsort(true_dist_from_vp_to_tilectr)

            # # occlusion
            # set_occluded_idx = set()
            # true_visible_tilectr_idx = []
            # for i in range(len(true_sort_dist_idx) - 1):
            #     ref_tilectr_idx = true_sort_dist_idx[i]
            #     if ref_tilectr_idx in set_occluded_idx:
            #         continue
            #     true_visible_tilectr_idx.append(ref_tilectr_idx)
            #     ref_dist = true_dist_from_vp_to_tilectr[ref_tilectr_idx]
            #     ref_vec = true_vecs_from_vp_to_tilectr[ref_tilectr_idx]
            #     ang_thresh = params.TILE_SIDE_LEN / ref_dist
            #     for j in range(i + 1, len(true_sort_dist_idx)):
            #         cur_tilectr_idx = true_sort_dist_idx[j]
            #         if cur_tilectr_idx in set_occluded_idx:
            #             continue
            #         # check if current tile is already occluded or not
            #         cur_dist = true_dist_from_vp_to_tilectr[cur_tilectr_idx]
            #         cur_vec = true_vecs_from_vp_to_tilectr[cur_tilectr_idx]
            #         # in radian
            #         ang_between = np.arccos(
            #             np.dot(ref_vec, cur_vec) /
            #             (linalg.norm(ref_vec) * linalg.norm(cur_vec)))
            #         if ang_between <= ang_thresh:
            #             set_occluded_idx.add(cur_tilectr_idx)
            # print("hpr_true--- ", time.time() - start_time, " seconds ---")

            # HPR
            HPR_obj = HiddenPointsRemoval(tile_center_points)
            # # First subplot
            # fig = plt.figure(figsize = plt.figaspect(0.5))
            # plt.title('Test Case With A Sphere (Left) and Visible Sphere Viewed From Well Above (Right)')
            # ax = fig.add_subplot(1,2,1, projection = '3d')
            # ax.scatter(HPR_obj.points[:, 0], HPR_obj.points[:, 1], HPR_obj.points[:, 2], c='r', marker='^') # Plot all points
            # ax.set_xlabel('X Axis')
            # ax.set_ylabel('Y Axis')
            # ax.set_zlabel('Z Axis')
            # plt.show()
            # start_time = time.time()
            # print("init--- ", time.time() - start_time, " seconds ---")
            flippedPoints = HPR_obj.sphericalFlip(
                viewpoint_position, math.pi
            )  # Reflect the point cloud about a sphere centered at viewpoint_position
            myHull = HPR_obj.convexHull(
                flippedPoints
            )  # Take the convex hull of the center of the sphere and the deformed point cloud

            # print("hpr--- ", time.time() - start_time, " seconds ---")

            true_flippedPoints = HPR_obj.sphericalFlip(
                true_position, math.pi
            )  # Reflect the point cloud about a sphere centered at viewpoint_position
            true_myHull = HPR_obj.convexHull(
                true_flippedPoints
            )  # Take the convex hull of the center of the sphere and the deformed point cloud
            # print("hpr_true--- ", time.time() - start_time, " seconds ---")

            # # ax = fig.add_subplot(1,2,2, projection = '3d')
            # # ax.scatter(flippedPoints[:, 0], flippedPoints[:, 1], flippedPoints[:, 2], c='r', marker='^') # Plot all points
            # # ax.set_xlabel('X Axis')
            # # ax.set_ylabel('Y Axis')
            # # ax.set_zlabel('Z Axis')
            # # plt.show()

            # # HPR_obj.plot(visible_hull_points=myHull)
            # # pdb.set_trace()

            # ### TODO by Tongyu: use gradient descent to optimize radius of HPR ####

            # ###############################################################

            # ############ check which visible points are within fov #############
            predicted_visible_tiles_set = set()
            for vertex in myHull.vertices[:-1]:
                vertex_coordinate = np.array([
                    tile_center_points[vertex, 0],
                    tile_center_points[vertex, 1],
                    tile_center_points[vertex, 2]
                ])
                vector_from_viewpoint_to_tilecenter = vertex_coordinate - viewpoint_position
                pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
                yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
                viewing_ray_unit_vector = np.array([
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)
                ])
                intersection_angle = np.arccos(
                    np.dot(vector_from_viewpoint_to_tilecenter,
                           viewing_ray_unit_vector) /
                    np.linalg.norm(vector_from_viewpoint_to_tilecenter))
                if intersection_angle <= params.FOV_DEGREE_SPAN:
                    # viewable => viewing probability = 1
                    viewable_tile_idx = (tile_xs[vertex], tile_ys[vertex],
                                         tile_zs[vertex]
                                         )  # position among all tiles
                    # point_x = (
                    #     -vertex_coordinate[0] + self.origin[0] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    # point_y = (
                    #     vertex_coordinate[1] + self.origin[1] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    # point_z = (
                    #     -vertex_coordinate[2] + self.origin[2] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)

                    # viewable_tile_idx = np.array([
                    #     int(point_x) // params.NUM_VOXEL_TILESIDE,
                    #     int(point_y) // params.NUM_VOXEL_TILESIDE,
                    #     int(point_z) // params.NUM_VOXEL_TILESIDE
                    # ])

            # for tilectr_idx in visible_tilectr_idx:
            #     vertex_coordinate = tile_center_points[tilectr_idx]
            #     vector_from_viewpoint_to_tilecenter = vecs_from_vp_to_tilectr[
            #         tilectr_idx]
            #     pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
            #     yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
            #     viewing_ray_unit_vector = np.array([
            #         np.sin(yaw) * np.cos(pitch),
            #         np.sin(pitch),
            #         np.cos(yaw) * np.cos(pitch)
            #     ])
            #     intersection_angle = np.arccos(
            #         np.dot(vector_from_viewpoint_to_tilecenter,
            #                viewing_ray_unit_vector) /
            #         np.linalg.norm(vector_from_viewpoint_to_tilecenter))
            #     if intersection_angle <= params.FOV_DEGREE_SPAN:
            #         viewable_tile_idx = np.array([
            #             tile_xs[tilectr_idx], tile_ys[tilectr_idx],
            #             tile_zs[tilectr_idx]
            #         ])
            #         # as long as the tile is visible, the viewing probability is 1 (which means the overlap ratio is 100%)
            #         if np.max(viewable_tile_idx
            #                   ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
            #             # pdb.set_trace()
            #             viewable_tile_idx[np.where(
            #                 viewable_tile_idx ==
            #                 params.NUM_TILES_PER_SIDE_IN_A_FRAME
            #             )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1

                    viewing_probability[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = 1
                    num_visible_pts_per_tile[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] += 1
                    distances[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = self.calculate_distance(
                                vertex_coordinate, viewpoint_position)

                    predicted_visible_tiles_set.add(tuple(viewable_tile_idx))
            # print("check fov--- ", time.time() - start_time, " seconds ---")

            # visible_tiles_idx = num_visible_pts_per_tile[
            #     frame_idx - update_start_idx].nonzero()

            # print(
            #     num_visible_pts_per_tile[frame_idx -
            #                              update_start_idx][visible_tiles_idx])
            # print(
            #     num_visible_pts_per_tile[frame_idx -
            #                              update_start_idx][visible_tiles_idx] /
            #     num_pts_per_tile[frame_idx -
            #                      update_start_idx][visible_tiles_idx])
            # pdb.set_trace()

            true_visible_tiles_set = set()
            for vertex in true_myHull.vertices[:-1]:
                vertex_coordinate = np.array([
                    tile_center_points[vertex, 0],
                    tile_center_points[vertex, 1], tile_center_points[vertex,
                                                                      2]
                ])
                vector_from_viewpoint_to_tilecenter = vertex_coordinate - true_position
                pitch = true_viewpoint[params.MAP_6DOF_TO_HMD_DATA[
                    "pitch"]] / params.RADIAN_TO_DEGREE
                yaw = true_viewpoint[params.MAP_6DOF_TO_HMD_DATA[
                    "yaw"]] / params.RADIAN_TO_DEGREE
                viewing_ray_unit_vector = np.array([
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)
                ])
                intersection_angle = np.arccos(
                    np.dot(vector_from_viewpoint_to_tilecenter,
                           viewing_ray_unit_vector) /
                    np.linalg.norm(vector_from_viewpoint_to_tilecenter))
                if intersection_angle <= params.FOV_DEGREE_SPAN:
                    # viewable => viewing probability = 1
                    viewable_tile_idx = (tile_xs[vertex], tile_ys[vertex],
                                         tile_zs[vertex]
                                         )  # position among all tiles
                    # as long as the tile is visiblle, the viewing probability is 1 (which means the overlap ratio is 100%)
                    # point_x = (
                    #     -vertex_coordinate[0] + self.origin[0] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    # point_y = (
                    #     vertex_coordinate[1] + self.origin[1] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    # point_z = (
                    #     -vertex_coordinate[2] + self.origin[2] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)

                    # viewable_tile_idx = np.array([
                    #     int(point_x) // params.NUM_VOXEL_TILESIDE,
                    #     int(point_y) // params.NUM_VOXEL_TILESIDE,
                    #     int(point_z) // params.NUM_VOXEL_TILESIDE
                    # ])
                    # if np.max(viewable_tile_idx
                    #           ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
                    #     # pdb.set_trace()
                    #     viewable_tile_idx[np.where(
                    #         viewable_tile_idx ==
                    #         params.NUM_TILES_PER_SIDE_IN_A_FRAME
                    #     )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1
                    # true_viewing_probability[frame_idx - update_start_idx][
                    #     viewable_tile_idx[0]][viewable_tile_idx[1]][
                    #         viewable_tile_idx[2]] = 1
                    # distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)

            # for tilectr_idx in true_visible_tilectr_idx:
            #     vertex_coordinate = tile_center_points[tilectr_idx]
            #     vector_from_viewpoint_to_tilecenter = true_vecs_from_vp_to_tilectr[
            #         tilectr_idx]
            #     pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
            #     yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
            #     viewing_ray_unit_vector = np.array([
            #         np.sin(yaw) * np.cos(pitch),
            #         np.sin(pitch),
            #         np.cos(yaw) * np.cos(pitch)
            #     ])
            #     intersection_angle = np.arccos(
            #         np.dot(vector_from_viewpoint_to_tilecenter,
            #                viewing_ray_unit_vector) /
            #         np.linalg.norm(vector_from_viewpoint_to_tilecenter))
            #     if intersection_angle <= params.FOV_DEGREE_SPAN:
            #         viewable_tile_idx = np.array([
            #             tile_xs[tilectr_idx], tile_ys[tilectr_idx],
            #             tile_zs[tilectr_idx]
            #         ])
            #         if np.max(viewable_tile_idx
            #                   ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
            #             # pdb.set_trace()
            #             viewable_tile_idx[np.where(
            #                 viewable_tile_idx ==
            #                 params.NUM_TILES_PER_SIDE_IN_A_FRAME
            #             )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1

                    true_viewing_probability[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = 1
                    # distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)
                    true_visible_tiles_set.add(tuple(viewable_tile_idx))
            # print("check true fov--- ", time.time() - start_time, " seconds ---")

            ########################################################################
            if not emitting_buffer:
                # update overlap_ratio history
                overlap_tiles_set = true_visible_tiles_set.intersection(
                    predicted_visible_tiles_set)
                # overlap_ratio = len(overlap_tiles_set) / len(
                #     true_visible_tiles_set)
                if len(predicted_visible_tiles_set) == 0:
                    overlap_ratio = 0
                    # pdb.set_trace()
                else:
                    overlap_ratio = len(overlap_tiles_set) / len(
                        predicted_visible_tiles_set)
                if frame_idx >= params.BUFFER_LENGTH:
                    overlap_ratio_history[
                        frame_idx - update_start_idx].append(overlap_ratio)
                # self.overlap_ratio_history[frame_idx - update_start_idx].pop(0)
                # if overlap_ratio < 1:
                # 	pdb.set_trace()
                # if self.update_step >= params.BUFFER_LENGTH // params.FPS:
                #     pdb.set_trace()
            # print("overlap--- ", time.time() - start_time, " seconds ---")
            # print("frame hpr--- ", time.time() - start_time, " seconds ---")

        return viewing_probability, distances, true_viewing_probability


    def KATZ(self, viewpoints, update_start_idx, update_end_idx, emitting_buffer, overlap_ratio_history):
        viewing_probability = []
        true_viewing_probability = []
        num_visible_pts_per_tile = []
        num_pts_per_tile = []
        distances = []

        for frame_idx in range(update_start_idx, update_end_idx + 1):
            start_time = time.time()
            # print(frame_idx)
            viewing_probability.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            true_viewing_probability.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            num_visible_pts_per_tile.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            num_pts_per_tile.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            distances.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            tile_center_points = []
            points_of_interest = []
            viewpoint = {
                "x": 0,
                'y': 0,
                "z": 0,
                "pitch": 0,
                "yaw": 0,
                "roll": 0
            }
            for key in viewpoint.keys():
                viewpoint[key] = viewpoints[key][frame_idx - update_start_idx]

            if frame_idx < params.BUFFER_LENGTH:
                continue
            valid_tiles = self.valid_tiles_obj.valid_tiles[
                (frame_idx - params.BUFFER_LENGTH) % params.
                NUM_FRAMES]  # cubic array denotes whether a tile is empty or not

            ### fixed / constant obj ############
            # valid_tiles = self.valid_tiles_obj.valid_tiles[0]
            #####################################

            # read from ply files with highest  level of detail
            ply_frame_idx = (frame_idx - params.BUFFER_LENGTH
                             ) % params.NUM_FRAMES + params.FRAME_IDX_BIAS

            # my heuristic: viewport->sort by distance->occlusion: O(n^2) worst case
            # store hash tables of vectors and distances between predicted viewpoint and tiles' center
            tiles_of_interest = []

            viewpoint_position = np.array(
                [viewpoint["x"], viewpoint["y"], viewpoint["z"]])
            viewpoint_position = np.expand_dims(viewpoint_position, axis=0)

            true_viewpoint = self.fov_traces_obj.fov_traces[frame_idx]
            true_position = np.array([
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["x"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["y"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["z"]]
            ])
            true_position = np.expand_dims(true_position, axis=0)
            # print(viewpoint_position)
            # print("init--- ", time.time() - start_time, " seconds ---")

            # get tilecenter/point coordinate
            tile_xs, tile_ys, tile_zs = valid_tiles.nonzero()
            for point_idx in range(len(tile_xs)):
                ply_x = tile_xs[point_idx]
                ply_y = tile_ys[point_idx]
                ply_z = tile_zs[point_idx]

                # ply_tile_file = params.PLY_PATH + '/' + str(
                #     ply_frame_idx) + '/' + params.VIDEO_NAME + '_' + str(
                #         ply_frame_idx) + '_' + str(ply_x // 10) + str(
                #             ply_x % 10) + '_' + str(ply_y // 10) + str(
                #                 ply_y % 10) + '_' + str(ply_z // 10) + str(
                #                     ply_z % 10) + '_3.ply'

                # f = open(ply_tile_file, "r")
                # for line in f:
                #     # print(line.split())
                #     # pdb.set_trace()
                #     line_list = line.split()
                #     if not line_list[0][0].isdigit():
                #         continue
                #     pt_x = int(float(line_list[0]))
                #     pt_y = int(float(line_list[1]))
                #     pt_z = int(float(line_list[2]))
                #     num_pts_per_tile[
                #         frame_idx - update_start_idx][ply_x][ply_y][ply_z] += 1

                #     pt_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate_4_ply(
                #         pt_x, pt_y, pt_z)
                #     points_of_interest.append(pt_coordinate)

                tile_center_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate(
                    ply_x, ply_y, ply_z)
                tile_center_points.append(tile_center_coordinate)
            # print("collect--- ", time.time() - start_time, " seconds ---")

            # modify object coordinate: origin at obj's bottom center
            tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(
                self.origin, tile_center_points)

            # mirror x and z axis to let obj face the user start view orientation
            tile_center_points = np.array(tile_center_points)
            tile_center_points[:, 0] = -tile_center_points[:, 0]
            tile_center_points[:, 2] = -tile_center_points[:, 2]
            # pdb.set_trace()

            # # modify object coordinate: origin at obj's bottom center
            # points_of_interest = self.valid_tiles_obj.change_tile_coordinates_origin(
            #     self.origin, points_of_interest)

            # # mirror x and z axis to let obj face the user start view orientation
            # points_of_interest = np.array(points_of_interest)
            # points_of_interest[:, 0] = -points_of_interest[:, 0]
            # points_of_interest[:, 2] = -points_of_interest[:, 2]

            # vecs_from_vp_to_tilectr = tile_center_points - viewpoint_position
            # dist_from_vp_to_tilectr = linalg.norm(vecs_from_vp_to_tilectr,
            #                                       axis=1)
            # print("pre-processing--- ", time.time() - start_time, " seconds ---")

            # # sort wrt distance
            # sort_dist_idx = np.argsort(dist_from_vp_to_tilectr)
            # print("sort--- ", time.time() - start_time, " seconds ---")

            # ################################ occlusion ###############################
            # set_occluded_idx = {}
            # visible_tilectr_idx = []
            # for i in range(len(sort_dist_idx) - 1):
            #     ref_tilectr_idx = sort_dist_idx[i]
            #     if ref_tilectr_idx in set_occluded_idx and set_occluded_idx[
            #             ref_tilectr_idx] >= 3:
            #         continue
            #     visible_tilectr_idx.append(ref_tilectr_idx)
            #     ref_dist = dist_from_vp_to_tilectr[ref_tilectr_idx]
            #     ref_vec = vecs_from_vp_to_tilectr[ref_tilectr_idx]
            #     ang_thresh = params.TILE_SIDE_LEN / ref_dist
            #     dist_diff_thresh = params.TILE_SIDE_LEN
            #     for j in range(i + 1, len(sort_dist_idx)):
            #         cur_tilectr_idx = sort_dist_idx[j]
            #         if cur_tilectr_idx in set_occluded_idx and set_occluded_idx[
            #                 cur_tilectr_idx] >= 3:
            #             continue
            #         # check if current tile is already occluded or not
            #         cur_dist = dist_from_vp_to_tilectr[cur_tilectr_idx]
            #         if cur_dist - ref_dist <= dist_diff_thresh * 1.5:
            #             continue
            #         cur_vec = vecs_from_vp_to_tilectr[cur_tilectr_idx]
            #         # in radian
            #         ang_between_ref_and_cur = np.arccos(
            #             np.dot(ref_vec, cur_vec) /
            #             (linalg.norm(ref_vec) * linalg.norm(cur_vec)))

            #         vec_from_ref_to_cur = cur_vec - ref_vec
            #         ang_between_refvec_and_refcurvec = np.arccos(
            #             np.dot(ref_vec, vec_from_ref_to_cur) /
            #             (linalg.norm(ref_vec) *
            #              linalg.norm(vec_from_ref_to_cur)))

            #         if ang_between_ref_and_cur < ang_thresh * 1 and ang_between_refvec_and_refcurvec < ang_thresh * 3:
            #             # set_occluded_idx.add(cur_tilectr_idx)
            #             if cur_tilectr_idx not in set_occluded_idx:
            #                 set_occluded_idx[cur_tilectr_idx] = 0
            #             set_occluded_idx[cur_tilectr_idx] += 1

            # # modify object coordinate: origin at viewpoint
            # # tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(viewpoint_position, tile_center_points)
            # print("hpr--- ", time.time() - start_time, " seconds ---")

            # ######################### true occlusion ###########################
            # true_vecs_from_vp_to_tilectr = tile_center_points - true_position
            # true_dist_from_vp_to_tilectr = linalg.norm(
            #     true_vecs_from_vp_to_tilectr, axis=1)

            # # sort wrt distance
            # true_sort_dist_idx = np.argsort(true_dist_from_vp_to_tilectr)

            # # occlusion
            # set_occluded_idx = set()
            # true_visible_tilectr_idx = []
            # for i in range(len(true_sort_dist_idx) - 1):
            #     ref_tilectr_idx = true_sort_dist_idx[i]
            #     if ref_tilectr_idx in set_occluded_idx:
            #         continue
            #     true_visible_tilectr_idx.append(ref_tilectr_idx)
            #     ref_dist = true_dist_from_vp_to_tilectr[ref_tilectr_idx]
            #     ref_vec = true_vecs_from_vp_to_tilectr[ref_tilectr_idx]
            #     ang_thresh = params.TILE_SIDE_LEN / ref_dist
            #     for j in range(i + 1, len(true_sort_dist_idx)):
            #         cur_tilectr_idx = true_sort_dist_idx[j]
            #         if cur_tilectr_idx in set_occluded_idx:
            #             continue
            #         # check if current tile is already occluded or not
            #         cur_dist = true_dist_from_vp_to_tilectr[cur_tilectr_idx]
            #         cur_vec = true_vecs_from_vp_to_tilectr[cur_tilectr_idx]
            #         # in radian
            #         ang_between = np.arccos(
            #             np.dot(ref_vec, cur_vec) /
            #             (linalg.norm(ref_vec) * linalg.norm(cur_vec)))
            #         if ang_between <= ang_thresh:
            #             set_occluded_idx.add(cur_tilectr_idx)
            # print("hpr_true--- ", time.time() - start_time, " seconds ---")

            # HPR
            HPR_obj = HiddenPointsRemoval(tile_center_points)
            # # First subplot
            # fig = plt.figure(figsize = plt.figaspect(0.5))
            # plt.title('Test Case With A Sphere (Left) and Visible Sphere Viewed From Well Above (Right)')
            # ax = fig.add_subplot(1,2,1, projection = '3d')
            # ax.scatter(HPR_obj.points[:, 0], HPR_obj.points[:, 1], HPR_obj.points[:, 2], c='r', marker='^') # Plot all points
            # ax.set_xlabel('X Axis')
            # ax.set_ylabel('Y Axis')
            # ax.set_zlabel('Z Axis')
            # plt.show()
            # start_time = time.time()
            # print("init--- ", time.time() - start_time, " seconds ---")
            flippedPoints = HPR_obj.sphericalFlip(
                viewpoint_position, math.pi
            )  # Reflect the point cloud about a sphere centered at viewpoint_position
            myHull = HPR_obj.convexHull(
                flippedPoints
            )  # Take the convex hull of the center of the sphere and the deformed point cloud

            # print("hpr--- ", time.time() - start_time, " seconds ---")

            true_flippedPoints = HPR_obj.sphericalFlip(
                true_position, math.pi
            )  # Reflect the point cloud about a sphere centered at viewpoint_position
            true_myHull = HPR_obj.convexHull(
                true_flippedPoints
            )  # Take the convex hull of the center of the sphere and the deformed point cloud
            # print("hpr_true--- ", time.time() - start_time, " seconds ---")

            # ax = fig.add_subplot(1,2,2, projection = '3d')
            # ax.scatter(flippedPoints[:, 0], flippedPoints[:, 1], flippedPoints[:, 2], c='r', marker='^') # Plot all points
            # ax.set_xlabel('X Axis')
            # ax.set_ylabel('Y Axis')
            # ax.set_zlabel('Z Axis')
            # plt.show()

            # HPR_obj.plot(visible_hull_points=myHull)
            # pdb.set_trace()

            ### TODO by Tongyu: use gradient descent to optimize radius of HPR ####

            ###############################################################

            ############ check which visible points are within fov #############
            predicted_visible_tiles_set = set()
            for vertex in myHull.vertices[:-1]:
                vertex_coordinate = np.array([
                    tile_center_points[vertex, 0],
                    tile_center_points[vertex, 1],
                    tile_center_points[vertex, 2]
                ])
                vector_from_viewpoint_to_tilecenter = vertex_coordinate - viewpoint_position
                pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
                yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
                viewing_ray_unit_vector = np.array([
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)
                ])
                intersection_angle = np.arccos(
                    np.dot(vector_from_viewpoint_to_tilecenter,
                           viewing_ray_unit_vector) /
                    np.linalg.norm(vector_from_viewpoint_to_tilecenter))
                if intersection_angle <= params.FOV_DEGREE_SPAN:
                    # viewable => viewing probability = 1
                    viewable_tile_idx = (tile_xs[vertex], tile_ys[vertex],
                                         tile_zs[vertex]
                                         )  # position among all tiles
                    # point_x = (
                    #     -vertex_coordinate[0] + self.origin[0] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    # point_y = (
                    #     vertex_coordinate[1] + self.origin[1] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    # point_z = (
                    #     -vertex_coordinate[2] + self.origin[2] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)

                    # viewable_tile_idx = np.array([
                    #     int(point_x) // params.NUM_VOXEL_TILESIDE,
                    #     int(point_y) // params.NUM_VOXEL_TILESIDE,
                    #     int(point_z) // params.NUM_VOXEL_TILESIDE
                    # ])

            # for tilectr_idx in visible_tilectr_idx:
            #     vertex_coordinate = tile_center_points[tilectr_idx]
            #     vector_from_viewpoint_to_tilecenter = vecs_from_vp_to_tilectr[
            #         tilectr_idx]
            #     pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
            #     yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
            #     viewing_ray_unit_vector = np.array([
            #         np.sin(yaw) * np.cos(pitch),
            #         np.sin(pitch),
            #         np.cos(yaw) * np.cos(pitch)
            #     ])
            #     intersection_angle = np.arccos(
            #         np.dot(vector_from_viewpoint_to_tilecenter,
            #                viewing_ray_unit_vector) /
            #         np.linalg.norm(vector_from_viewpoint_to_tilecenter))
            #     if intersection_angle <= params.FOV_DEGREE_SPAN:
            #         viewable_tile_idx = np.array([
            #             tile_xs[tilectr_idx], tile_ys[tilectr_idx],
            #             tile_zs[tilectr_idx]
            #         ])
            #         # as long as the tile is visible, the viewing probability is 1 (which means the overlap ratio is 100%)
            #         if np.max(viewable_tile_idx
            #                   ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
            #             # pdb.set_trace()
            #             viewable_tile_idx[np.where(
            #                 viewable_tile_idx ==
            #                 params.NUM_TILES_PER_SIDE_IN_A_FRAME
            #             )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1
                    viewing_probability[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = 1
                    num_visible_pts_per_tile[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] += 1
                    distances[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = self.calculate_distance(
                                vertex_coordinate, viewpoint_position)

                    predicted_visible_tiles_set.add(tuple(viewable_tile_idx))
            # print("check fov--- ", time.time() - start_time, " seconds ---")

            # visible_tiles_idx = num_visible_pts_per_tile[
            #     frame_idx - update_start_idx].nonzero()

            # print(
            #     num_visible_pts_per_tile[frame_idx -
            #                              update_start_idx][visible_tiles_idx])
            # print(
            #     num_visible_pts_per_tile[frame_idx -
            #                              update_start_idx][visible_tiles_idx] /
            #     num_pts_per_tile[frame_idx -
            #                      update_start_idx][visible_tiles_idx])
            # pdb.set_trace()

            true_visible_tiles_set = set()
            for vertex in true_myHull.vertices[:-1]:
                vertex_coordinate = np.array([
                    tile_center_points[vertex, 0],
                    tile_center_points[vertex, 1], tile_center_points[vertex,
                                                                      2]
                ])
                vector_from_viewpoint_to_tilecenter = vertex_coordinate - true_position
                pitch = true_viewpoint[params.MAP_6DOF_TO_HMD_DATA[
                    "pitch"]] / params.RADIAN_TO_DEGREE
                yaw = true_viewpoint[params.MAP_6DOF_TO_HMD_DATA[
                    "yaw"]] / params.RADIAN_TO_DEGREE
                viewing_ray_unit_vector = np.array([
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)
                ])
                intersection_angle = np.arccos(
                    np.dot(vector_from_viewpoint_to_tilecenter,
                           viewing_ray_unit_vector) /
                    np.linalg.norm(vector_from_viewpoint_to_tilecenter))
                if intersection_angle <= params.FOV_DEGREE_SPAN:
                    # viewable => viewing probability = 1
                    viewable_tile_idx = (tile_xs[vertex], tile_ys[vertex],
                                         tile_zs[vertex]
                                         )  # position among all tiles
                    # as long as the tile is visiblle, the viewing probability is 1 (which means the overlap ratio is 100%)
                    # point_x = (
                    #     -vertex_coordinate[0] + self.origin[0] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    # point_y = (
                    #     vertex_coordinate[1] + self.origin[1] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)
                    # point_z = (
                    #     -vertex_coordinate[2] + self.origin[2] -
                    #     params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE / 2
                    # ) / (params.OBJECT_SIDE_LEN / params.NUM_VOXEL_PER_SIDE)

                    # viewable_tile_idx = np.array([
                    #     int(point_x) // params.NUM_VOXEL_TILESIDE,
                    #     int(point_y) // params.NUM_VOXEL_TILESIDE,
                    #     int(point_z) // params.NUM_VOXEL_TILESIDE
                    # ])
                    # if np.max(viewable_tile_idx
                    #           ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
                    #     # pdb.set_trace()
                    #     viewable_tile_idx[np.where(
                    #         viewable_tile_idx ==
                    #         params.NUM_TILES_PER_SIDE_IN_A_FRAME
                    #     )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1
                    # true_viewing_probability[frame_idx - update_start_idx][
                    #     viewable_tile_idx[0]][viewable_tile_idx[1]][
                    #         viewable_tile_idx[2]] = 1
                    # distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)

            # for tilectr_idx in true_visible_tilectr_idx:
            #     vertex_coordinate = tile_center_points[tilectr_idx]
            #     vector_from_viewpoint_to_tilecenter = true_vecs_from_vp_to_tilectr[
            #         tilectr_idx]
            #     pitch = viewpoint["pitch"] / params.RADIAN_TO_DEGREE
            #     yaw = viewpoint["yaw"] / params.RADIAN_TO_DEGREE
            #     viewing_ray_unit_vector = np.array([
            #         np.sin(yaw) * np.cos(pitch),
            #         np.sin(pitch),
            #         np.cos(yaw) * np.cos(pitch)
            #     ])
            #     intersection_angle = np.arccos(
            #         np.dot(vector_from_viewpoint_to_tilecenter,
            #                viewing_ray_unit_vector) /
            #         np.linalg.norm(vector_from_viewpoint_to_tilecenter))
            #     if intersection_angle <= params.FOV_DEGREE_SPAN:
            #         viewable_tile_idx = np.array([
            #             tile_xs[tilectr_idx], tile_ys[tilectr_idx],
            #             tile_zs[tilectr_idx]
            #         ])
            #         if np.max(viewable_tile_idx
            #                   ) == params.NUM_TILES_PER_SIDE_IN_A_FRAME:
            #             # pdb.set_trace()
            #             viewable_tile_idx[np.where(
            #                 viewable_tile_idx ==
            #                 params.NUM_TILES_PER_SIDE_IN_A_FRAME
            #             )] = params.NUM_TILES_PER_SIDE_IN_A_FRAME - 1
                    true_viewing_probability[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = 1
                    # distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)
                    true_visible_tiles_set.add(tuple(viewable_tile_idx))
            # print("check true fov--- ", time.time() - start_time, " seconds ---")

            ########################################################################
            if not emitting_buffer:
                # update overlap_ratio history
                overlap_tiles_set = true_visible_tiles_set.intersection(
                    predicted_visible_tiles_set)
                overlap_ratio = len(overlap_tiles_set) / len(
                    true_visible_tiles_set)
                if len(predicted_visible_tiles_set) == 0:
                    overlap_ratio = 0
                    # pdb.set_trace()
                else:
                    overlap_ratio = len(overlap_tiles_set) / len(
                        predicted_visible_tiles_set)
                if frame_idx >= params.BUFFER_LENGTH:
                    overlap_ratio_history[
                        frame_idx - update_start_idx].append(overlap_ratio)
                # self.overlap_ratio_history[frame_idx - update_start_idx].pop(0)
                # if overlap_ratio < 1:
                # 	pdb.set_trace()
                # if self.update_step >= params.BUFFER_LENGTH // params.FPS:
                #     pdb.set_trace()
            # print("overlap--- ", time.time() - start_time, " seconds ---")
            # print("frame hpr--- ", time.time() - start_time, " seconds ---")

        return viewing_probability, distances, true_viewing_probability



# directory = '../ply_all_tiles_levels/recons'
# frame_idx = 0
# for dir_ in os.listdir(directory):
#     frame_idx += 1
#     print(frame_idx)
#     new_dir = os.path.join(directory, dir_)
#     frame_pts = []
#     num_frame_pts = 0
#     for filename in os.listdir(new_dir):
#         if filename[-5] != '3':
#             continue
#         # line_idx = 0
#         tile_file_name = os.path.join(new_dir, filename)
#         f = open(tile_file_name, 'r')
#         lines = f.readlines()[12:]
#         f.close()
#         frame_pts.extend(lines)
#         # print(len(lines))
#         num_tile_pts = len(lines)
#         num_frame_pts += num_tile_pts
#         # pdb.set_trace()
# 
#     prefix_lines = ['ply\n',
#                     'format ascii 1.0\n',
#                     'element vertex ' + str(num_frame_pts) + '\n',
#                     'property float x\n',
#                     'property float y\n',
#                     'property float z\n',
#                     'property uchar green\n',
#                     'property uchar blue\n',
#                     'property uchar red\n',
#                     'element face 0\n',
#                     'property list uint8 int32 vertex_index\n',
#                     'end_header\n']
# 
#     save_path = os.path.join(new_dir, 'l3_frame.ply')
#     frame_f = open(save_path, 'w+')
#     frame_f.writelines(prefix_lines)
#     frame_f.writelines(frame_pts)
#     frame_f.close()
