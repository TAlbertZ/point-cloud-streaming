import numpy as np
import pdb
import time
import pickle as pk
import logging
import os
import math
import open3d as o3d
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

    def segment_occlusion(self, viewpoints, update_start_idx, update_end_idx,
                          emitting_buffer, overlap_ratio_history):
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
                viewing_probability[
                    frame_idx -
                    update_start_idx] = cur_visibility * valid_tiles

                # calculate distance
                distances[frame_idx - update_start_idx] = distances[int(
                    (frame_idx - update_start_idx) // params.OCCLUSION_FREQ *
                    params.OCCLUSION_FREQ)].copy()
                continue
            else:
                if (frame_idx - update_start_idx) % params.SEGMENT_LEN == 0:
                    cur_visibility = np.zeros(
                        (params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                         params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                         params.NUM_TILES_PER_SIDE_IN_A_FRAME))
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
                    points_of_interest[vertex, 1], points_of_interest[vertex,
                                                                      2]
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
                    visible_tiles_pos = viewing_probability[
                        frame_idx - update_start_idx].nonzero()
                    cur_visibility[visible_tiles_pos] = 1
                    cur_visibility *= valid_tiles
                    viewing_probability[
                        frame_idx - update_start_idx] = cur_visibility.copy()
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
            print("check true fov--- ",
                  time.time() - start_time, " seconds ---")

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
                    overlap_ratio_history[frame_idx - update_start_idx].append(
                        overlap_ratio)
                # self.overlap_ratio_history[frame_idx - update_start_idx].pop(0)
                # if overlap_ratio < 1:
                # 	pdb.set_trace()
                # if self.update_step >= params.BUFFER_LENGTH // params.FPS:
                #     pdb.set_trace()
            # print("overlap--- ", time.time() - start_time, " seconds ---")
            print("frame hpr--- ", time.time() - start_time, " seconds ---")

        return viewing_probability, distances, true_viewing_probability

    def frame_occlusion(self, viewpoints, update_start_idx, update_end_idx,
                        emitting_buffer, overlap_ratio_history):
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
                    tile_center_points[vertex, 1], tile_center_points[vertex,
                                                                      2]
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
                    overlap_ratio_history[frame_idx - update_start_idx].append(
                        overlap_ratio)
                # self.overlap_ratio_history[frame_idx - update_start_idx].pop(0)
                # if overlap_ratio < 1:
                # 	pdb.set_trace()
                # if self.update_step >= params.BUFFER_LENGTH // params.FPS:
                #     pdb.set_trace()
            # print("overlap--- ", time.time() - start_time, " seconds ---")
            # print("frame hpr--- ", time.time() - start_time, " seconds ---")

        return viewing_probability, distances, true_viewing_probability

    def KATZ(self, viewpoints, update_start_idx, update_end_idx,
             emitting_buffer, overlap_ratio_history):
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
                    tile_center_points[vertex, 1], tile_center_points[vertex,
                                                                      2]
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
                    overlap_ratio_history[frame_idx - update_start_idx].append(
                        overlap_ratio)
                # self.overlap_ratio_history[frame_idx - update_start_idx].pop(0)
                # if overlap_ratio < 1:
                # 	pdb.set_trace()
                # if self.update_step >= params.BUFFER_LENGTH // params.FPS:
                #     pdb.set_trace()
            # print("overlap--- ", time.time() - start_time, " seconds ---")
            # print("frame hpr--- ", time.time() - start_time, " seconds ---")

        return viewing_probability, distances, true_viewing_probability

    def get_camera_intrinsic_matrix(self, image_width, image_height):
        fx, fy = 525, 525  # Focal length
        cx, cy = image_width / 2, image_height / 2  # Principal point
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_camera_extrinsic_matrix_from_yaw_pitch_roll(
            self, yaw_degree, pitch_degree, roll_degree, t):
        # from world coordinate to camera coordinate, R is 3*3, t is 3*1,
        t = np.array(t).reshape(3, 1)

        # t is camera position in world coordinate(numpy array)
        # Define camera extrinsic parameters (example values for rotation and translation)
        # define x,y,z rotation matrix
        # here we use left-hand coordinate system
        def rotation_matrix_x(theta):
            return np.array([[1, 0, 0], [0, np.cos(theta),
                                         np.sin(theta)],
                             [0, -np.sin(theta),
                              np.cos(theta)]])

        def rotation_matrix_y(theta):
            return np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0],
                             [np.sin(theta), 0,
                              np.cos(theta)]])

        def rotation_matrix_z(theta):
            return np.array([[np.cos(theta), np.sin(theta), 0],
                             [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        # get the rotation matrix
        # pitch_degree, yaw_degree, roll_degree = 0, 0, 0
        # here we set a 180 degree offset for pitch,
        # because the camera is looking to the negative z axix in the world coordinate
        pitch, yaw, roll = np.radians(-pitch_degree) + np.radians(
            180), np.radians(-yaw_degree), np.radians(-roll_degree)
        R = rotation_matrix_x(pitch) @ rotation_matrix_y(
            yaw) @ rotation_matrix_z(roll)

        # R = np.eye(3) # Identity matrix for rotation
        # t = np.array([[200], [500], [1000]]) # Translation
        # get 4*4 extrinsic matrix from R and t
        extrinsic_matrix = np.hstack((R, -R @ t))
        extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0,
                                                                  1])))
        return extrinsic_matrix

    def get_points_in_FoV(self, pcd, intrinsic_matrix, extrinsic_matrix,
                          image_width, image_height):
        far_near_plane = np.array([10, 10000])
        # Transform point cloud to camera coordinate system
        points_homogeneous = np.hstack(
            (np.asarray(pcd.points), np.ones(
                (len(pcd.points), 1))))  # shape is (n, 4)
        camera_coord_points = extrinsic_matrix @ points_homogeneous.T  # shape is (4,4) * (4, n) = (4, n)
        camera_coord_points = camera_coord_points[:3, :]  # shape is (3, n)
        # Project points onto the image plane
        projected_points = intrinsic_matrix @ camera_coord_points
        # # Normalize by the third (z) component, only on x,y, not on z, so that we can use projected_points[2, :] to get the far/near plane
        projected_points[0:2, :] /= projected_points[
            2, :]  # Normalize by the third (z) component
        # Filter points based on image dimensions (example dimensions)
        in_fov_indices = np.where(
            (projected_points[0, :] >= 0)
            & (projected_points[0, :] < image_width)
            & (projected_points[1, :] >= 0)
            & (projected_points[1, :] < image_height)
            # )
            & (projected_points[2, :] > far_near_plane[0])
            & (projected_points[2, :] < far_near_plane[1]))
        filtered_points = np.array(pcd.points)[in_fov_indices]
        # Create a new point cloud from filtered points
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        if len(pcd.colors) > 0:
            filtered_pcd.colors = o3d.utility.Vector3dVector(
                np.array(pcd.colors)[in_fov_indices])
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=np.array(t))
        # o3d.visualization.draw([filtered_pcd,coordinate_frame],intrinsic_matrix=intrinsic_matrix,extrinsic_matrix=extrinsic_matrix)
        return filtered_pcd

    def hidden_point_removal(self, pcd, para_eye):
        # if pcd has no points, return empty point cloud
        if len(pcd.points) <= 3:
            return pcd
        centroid = [0, 500, 0]
        # get L2 norm of the vector
        radius = np.linalg.norm(np.array(para_eye) - np.array(centroid)) * 1000
        # remove hidden points
        _, pt_map = pcd.hidden_point_removal(para_eye, radius)
        pcd_remove = pcd.select_by_index(pt_map)
        return pcd_remove

    def get_pcd_data_binary(self,
                            point_cloud_name='longdress',
                            ply_frame_idx=0):
        # downsample and remove hidden points
        #
        if point_cloud_name == 'longdress':
            point_cloud_path = f'./binary_tiles_gt/{point_cloud_name}/frame{ply_frame_idx}_binary_downsample.ply'
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            # origin = np.mean(np.array(pcd.points), axis=0)
            # print(origin)
            # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([origin[0],0,origin[2]]))#longdress
        elif point_cloud_name == 'loot':
            point_cloud_path = f'./data/binary_original/{point_cloud_name}/frame{ply_frame_idx}_binary.ply'
            pcd = o3d.io.read_point_cloud(point_cloud_path)
        elif point_cloud_name == 'redandblack':
            point_cloud_path = f'./data/binary_original/{point_cloud_name}/frame{ply_frame_idx}_binary.ply'
            pcd = o3d.io.read_point_cloud(point_cloud_path)
        elif point_cloud_name == 'soldier':
            point_cloud_path = f'./data/binary_original/{point_cloud_name}/frame{ply_frame_idx}_binary.ply'
            pcd = o3d.io.read_point_cloud(point_cloud_path)
        return pcd

    def open3d_hpr(self, viewpoints, update_start_idx, update_end_idx,
                   emitting_buffer, overlap_ratio_history, if_visible_pts,
                   if_save_render):
        viewing_probability = []
        true_viewing_probability = []
        num_visible_pts_per_tile = []
        num_pts_per_tile = []
        distances = []
        uniq_tile_index = None
        image_width = 1920
        image_height = 1080
        intrinsic_matrix = None
        extrinsic_matrix = None

        for frame_idx in range(update_start_idx, update_end_idx + 1):
            start_time = time.time()
            # print(frame_idx)
            # pdb.set_trace()
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

            # read from ply files with highest level of detail
            ply_frame_idx = (frame_idx -
                             params.BUFFER_LENGTH) % params.NUM_FRAMES
            pcd = self.get_pcd_data_binary(point_cloud_name=params.VIDEO_NAME,
                                           ply_frame_idx=ply_frame_idx)

            tiles_of_interest = []

            viewpoint_position = np.array(
                [viewpoint["x"], viewpoint["y"], viewpoint["z"]])

            true_viewpoint = self.fov_traces_obj.fov_traces[frame_idx]
            true_position = np.array([
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["x"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["y"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["z"]]
            ])
            # print(viewpoint_position)
            # print("init--- ", time.time() - start_time, " seconds ---")

            viewpoint_orientation = np.array(
                [viewpoint["pitch"], viewpoint["yaw"], viewpoint["roll"]])

            true_orientation = np.array([
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["pitch"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["yaw"]],
                true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["roll"]]
            ])

            para_eye = [i * 1024 / 1.8 for i in viewpoint_position]
            para_eye[2] = -para_eye[2]
            para_eye = np.array(para_eye).reshape(3, 1)
            pitch_degree, yaw_degree, roll_degree = viewpoint_orientation  # Convert degrees to radians if necessary
            # pitch_degree, yaw_degree, roll_degree = [0,-85,0]
            # image_width, image_height = np.array([1280, 720])
            image_width, image_height = np.array([1920, 1080])
            intrinsic_matrix = self.get_camera_intrinsic_matrix(
                image_width, image_height)
            # Define camera extrinsic parameters (example values for rotation and translation)
            extrinsic_matrix = self.get_camera_extrinsic_matrix_from_yaw_pitch_roll(
                yaw_degree, pitch_degree, roll_degree, para_eye)
            # downsample and remove hidden points

            if if_visible_pts:
                pcd = self.get_points_in_FoV(pcd, intrinsic_matrix,
                                             extrinsic_matrix, image_width,
                                             image_height)
                # pcd = downsampele_hidden_point_removal(pcd,para_eye,voxel_size=8)
                pcd = self.hidden_point_removal(pcd, para_eye)

            ####################### calculate visible tiles index according to pcd.points ###################
            # 1. get a set of arrays that are visible tile 3d-index
            # 2. update viewing_probability
            # 3. convert tile 3d-index to physical world coordinates, then calculate distance form viewpoint, update distances array.
            visible_pts_shift_back = np.array(pcd.points) + params.POINTS_SHIFT
            uniq_tile_index = np.unique(visible_pts_shift_back //
                                        params.NUM_VOXEL_TILESIDE,
                                        axis=0).astype(int)

            viewing_probability[frame_idx -
                                update_start_idx][uniq_tile_index[:, 0],
                                                  uniq_tile_index[:, 1],
                                                  uniq_tile_index[:, 2]] = 1
            convert_uniq_tile_idx_to_dist = uniq_tile_index * params.TILE_SIDE_LEN + params.TILE_SIDE_LEN / 2
            distances[frame_idx - update_start_idx][
                uniq_tile_index[:, 0], uniq_tile_index[:, 1],
                uniq_tile_index[:, 2]] = np.linalg.norm(
                    convert_uniq_tile_idx_to_dist - viewpoint_position, axis=1)

            ##########################################################################################################
            ########################### Setting up the visualizer ########################
            # if frame_idx == 610:
            #     vis = o3d.visualization.Visualizer()
            #     # vis.create_window(width=image_width, height=image_height)
            #     vis.create_window(visible=not if_save_render)
            #     vis.add_geometry(pcd)

            #     # vis.add_geometry(coordinate_frame)
            #     # print("my customize extrincis matrix:")
            #     # print(extrinsic_matrix,selected_orientation,selected_position,intrinsic_matrix)
            #     view_ctl = vis.get_view_control()
            #     # import pdb; pdb.set_trace()
            #     cam_pose_ctl = view_ctl.convert_to_pinhole_camera_parameters()
            #     cam_pose_ctl.intrinsic.height = image_height
            #     cam_pose_ctl.intrinsic.width = image_width
            #     cam_pose_ctl.intrinsic.intrinsic_matrix = intrinsic_matrix
            #     cam_pose_ctl.extrinsic = extrinsic_matrix
            #     view_ctl.convert_from_pinhole_camera_parameters(
            #         cam_pose_ctl, allow_arbitrary=True)
            #     view_ctl.change_field_of_view()
            #     # render
            #     vis.poll_events()
            #     vis.update_renderer()
            #     if not if_save_render:
            #         vis.run()
            # # w
            # if if_save_render and frame_idx == 610:
            #     if emitting_buffer:
            #         # check path exist or not, if not create it
            #         if not os.path.exists('./result_new_hpr/emit/' + params.VIDEO_NAME +
            #                               '/' + params.USER_FOV_TRACE):
            #             os.makedirs('./result_new_hpr/emit/' + params.VIDEO_NAME + '/' +
            #                         params.USER_FOV_TRACE)
            #         vis.capture_screen_image('./result_new_hpr/emit/' +
            #                                  params.VIDEO_NAME + '/' +
            #                                  params.USER_FOV_TRACE + '/' + 'fov_' +
            #                                  str(frame_idx).zfill(3) + '.png',
            #                                  do_render=False)
            #     else:
            #         # check path exist or not, if not create it
            #         if not os.path.exists('./result_new_hpr/pred/' + params.VIDEO_NAME +
            #                               '/' + params.USER_FOV_TRACE):
            #             os.makedirs('./result_new_hpr/pred/' + params.VIDEO_NAME + '/' +
            #                         params.USER_FOV_TRACE)
            #         vis.capture_screen_image('./result_new_hpr/pred/' +
            #                                  params.VIDEO_NAME + '/' +
            #                                  params.USER_FOV_TRACE + '/' + 'fov_' +
            #                                  str(frame_idx).zfill(3) + '.png',
            #                                  do_render=True)
            #     # pdb.set_trace()
            #     # index should have 3 digits
            #     pdb.set_trace()
            #     vis.destroy_window()
            #####################################################################  finish open3d HPR ###########################

            if not emitting_buffer:
                overlap_ratio = 1
                if frame_idx >= params.BUFFER_LENGTH:
                    overlap_ratio_history[frame_idx - update_start_idx].append(
                        overlap_ratio)

        return viewing_probability, distances, true_viewing_probability, uniq_tile_index, image_width, image_height, intrinsic_matrix, extrinsic_matrix


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
