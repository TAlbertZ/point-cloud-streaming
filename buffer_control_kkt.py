import numpy as np
import pickle as pk
import pdb
import math
import time
import logging
import os
import open3d as o3d
import matplotlib.pyplot as plt

from numpy import linalg

import params
from params import Algo
from params import ActionInitType
from occlusion import Occlusion

np.random.seed(7)


class Buffer():
    '''
		buffer controlling
		include buffer initialization and update
	'''

    def __init__(self,
                 fov_traces_obj,
                 bw_traces_obj,
                 valid_tiles_obj,
                 qr_weights_obj,
                 plot_figs_obj,
                 N,
                 max_reg=1e10,
                 hessians=False):

        if params.RENDER_VIEW:
            self.viewer = o3d.visualization.Visualizer()

        if params.SAVE_TILE_LOD:
            self.frame_lod_list = []

        self.logger = self.set_logger(logger_level=logging.DEBUG)
        self.logger_check_cost = self.set_logger(logger_name=__name__ +
                                                 '.check_cost',
                                                 logger_level=logging.INFO)

        self.buffer_length = params.BUFFER_LENGTH  # frame

        self.update_step = 0

        # self.buffer stores 3d numpy arrays,
        # each 3d array represents one frame,
        # which contains byte sizes of all tiles
        self.buffer = []

        self.max_tile_sizes = qr_weights_obj.rate_versions[0]  # 300x16x16x16

        # self.min_tile_sizes = qr_weights_obj.rate_versions[2]  # 300x16x16x16
        self.min_tile_sizes = np.zeros_like(
            qr_weights_obj.rate_versions[params.NUM_RATE_VERSIONS -
                                         1])  # 300x16x16x16
        self.num_pts_versions = qr_weights_obj.num_pts_versions

        self.fov_traces_obj = fov_traces_obj
        self.valid_tiles_obj = valid_tiles_obj
        self.bw_traces_obj = bw_traces_obj
        self.qr_weights_obj = qr_weights_obj
        self.plot_figs_obj = plot_figs_obj

        # a, b and distance_weight are waiting to be fit
        self.tile_a = self.qr_weights_obj.qr_weights[
            "a"]  # for each tile: [300x16x16x16]
        self.tile_b = self.qr_weights_obj.qr_weights[
            "b"]  # for each tile: [300x16x16x16]
        self.min_rates = self.qr_weights_obj.min_rates  # for each tile: [300x16x16x16]

        self.rate_versions = self.qr_weights_obj.rate_versions  # 6x300x16x16x16

        # sigmoid coefficient c: (1 + exp(-c*d))^(-1)
        self.distance_weight = 1

        # linearly increasing from 0 to 1
        self.frame_weights = None

        # initialize according to fov dataset H1, assume the initial viewpoint is always like this:
        # {x, y, z, roll, yaw, pitch} = {0.05, 1.7868, -1.0947, 6.9163, 350.8206, 359.9912}
        # z-x plane is floor
        # self.history_viewpoints = {
        #     "x": [0.05] * params.FOV_PREDICTION_HISTORY_WIN_LENGTH,
        #     "y": [1.7868] * params.FOV_PREDICTION_HISTORY_WIN_LENGTH,
        #     "z": [-1.0947] * params.FOV_PREDICTION_HISTORY_WIN_LENGTH,
        #     "pitch": [6.9163 + 360] *
        #     params.FOV_PREDICTION_HISTORY_WIN_LENGTH,  # rotate around x axis
        #     "yaw": [350.8206] *
        #     params.FOV_PREDICTION_HISTORY_WIN_LENGTH,  # rotate around y axis
        #     "roll": [359.9912] * params.FOV_PREDICTION_HISTORY_WIN_LENGTH
        # }  # rotate around z axis

        self.history_viewpoints = {
            "x": [self.fov_traces_obj.fov_traces[0][0]],
            "y": [self.fov_traces_obj.fov_traces[0][1]],
            "z": [self.fov_traces_obj.fov_traces[0][2]],
            "pitch":
            [self.fov_traces_obj.fov_traces[0][3]],  # rotate around x axis
            "yaw":
            [self.fov_traces_obj.fov_traces[0][4]],  # rotate around y axis
            "roll": [self.fov_traces_obj.fov_traces[0][5]]
        }  # rotate around z axis

        # initialize bandwidth history according to '../bw_traces/100ms_loss1'
        self.history_bandwidths = [self.bw_traces_obj.bw_trace[0]
                                   ] * params.BW_PREDICTION_HISTORY_WIN_LENGTH

        self.current_viewing_frame_idx = -1

        self.current_bandwidth_idx = -1

        self.origin = self.calculate_origin(
            self.valid_tiles_obj.valid_tiles[0]
        )  # calculate origin according to the first frame

        # # r* = Rmax
        # self.typeI_tiles_set = set()
        # # r* = r0
        # self.typeII_tiles_set = set()
        # # r* = z_weight / lambda
        # self.typeIII_tiles_set = set()
        self.frame_quality = []
        self.frame_quality_per_degree_list = []
        self.frame_quality_var = []
        self.ang_resol_list = []
        self.frame_num_points_list = []
        self.plot_bw_trace = []
        self.plot_predicted_bw_trace = []
        self.buffer_size_trace = []
        self.delta_buffer_size_trace = []
        self.true_viewpoints = {
            "x": [],
            'y': [],
            "z": [],
            "pitch": [],
            "yaw": [],
            "roll": []
        }
        self.fov_predict_accuracy_trace = {
            "x": [],
            'y': [],
            "z": [],
            "pitch": [],
            "yaw": [],
            "roll": []
        }
        self.overlap_ratio_history = []

        for key in self.true_viewpoints.keys():
            for frame_idx in range(
                    self.buffer_length + params.FPS * params.UPDATE_FREQ -
                    params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE *
                    params.FPS):
                self.true_viewpoints[key].append([])
                self.fov_predict_accuracy_trace[key].append([])

        overlap_ratio_hist_len = self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS
        for frame_idx in range(overlap_ratio_hist_len):
            self.overlap_ratio_history.append([])
            # for _ in range(params.OVERLAP_RATIO_HISTORY_WIN_LENGTH):
            #     self.overlap_ratio_history[frame_idx].append(1.0)

        self.bw_predict_accuracy_trace = []
        self.success_download_rate_trace = []
        self.frame_size_list = []
        self.num_valid_tiles_per_frame = []
        self.total_span_per_frame = []  # total degree span of each frame

        self.num_max_tiles_per_frame = []
        self.mean_size_over_tiles_per_frame = []

        self.tile_sizes_sol = []

        self.num_intersect_visible_tiles_trace = []

        self.mean_size_over_tiles_per_fov = []
        self.effective_rate = []
        self.wasted_rate = []
        self.visible_rate = []
        self.wasted_ratio = []

        self.fov_front_accuracy_list = []
        self.fov_end_accuracy_list = []
        self.fov_middle_accuracy_list = []
        self.fov_mean_accuracy_list = []

        self.start_time = 0

    def set_logger(self,
                   logger_name=__name__,
                   logger_level=logging.DEBUG,
                   logger_propagate=False):
        logger = logging.getLogger(logger_name)
        logger.propagate = logger_propagate
        logger.setLevel(logger_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s - %(lineno)d - %(module)s\n%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def calculate_origin(self, valid_tiles):
        tile_xs, tile_ys, tile_zs = valid_tiles.nonzero()
        x_list = []
        y_list = []
        z_list = []
        for point_idx in range(len(tile_xs)):
            tile_center_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate(
                tile_xs[point_idx], tile_ys[point_idx], tile_zs[point_idx])
            x_list.append(tile_center_coordinate[0])
            y_list.append(tile_center_coordinate[1])
            z_list.append(tile_center_coordinate[2])

        origin = [np.mean(x_list), 0, np.mean(z_list)]
        # pdb.set_trace()
        return origin

    def initialize_buffer(self):
        '''
			1-second at the front of buffer will be initialized with highest tile size,
			the rest with lowest size (0)
		'''
        for frame_idx in range(self.buffer_length):
            # first 1s tiles are of largest size,
            # the rest are initialized as 0 byte
            self.buffer.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            # if frame_idx < params.FPS:
            # 	valid_tiles_coordinates = self.valid_tiles_obj.valid_tiles[frame_idx].nonzero()
            # 	self.buffer[frame_idx][valid_tiles_coordinates] = params.MAX_TILE_SIZE

    def true_frame_quality_sum_and_var(self, viewing_probability, distances,
                                       frame_idx_within_video, frame_idx):
        if self.current_viewing_frame_idx + 1 < params.TARGET_LATENCY:
            return 0, 0, 0, 0, 0
        tiles_byte_sizes = self.buffer[0]  # front of buffer: cubic array

        self.tile_sizes_sol.append(tiles_byte_sizes)
        self.frame_size_list.append(np.sum(tiles_byte_sizes))
        self.num_max_tiles_per_frame.append(
            len(
                np.where(tiles_byte_sizes ==
                         self.max_tile_sizes[frame_idx_within_video])[0]))
        if len(viewing_probability.nonzero()[0]) == 0:
            self.mean_size_over_tiles_per_frame.append(float("inf"))
            # return 0
        else:
            if len(tiles_byte_sizes.nonzero()[0]) == 0:
                self.mean_size_over_tiles_per_frame.append(0)
            else:
                self.mean_size_over_tiles_per_frame.append(
                    np.sum(tiles_byte_sizes) /
                    len(tiles_byte_sizes.nonzero()[0]))

        frame_quality = 0

        # overlap tiles between viewing_probability and tiles_byte_sizes
        # are those contributing toward quality
        mask_downloaded_tiles = np.zeros_like(tiles_byte_sizes)
        mask_downloaded_tiles[tiles_byte_sizes.nonzero()] = 1
        # mask_visible_tiles = mask_downloaded_tiles * viewing_probability
        mask_visible_tiles = viewing_probability.copy()
        visible_tiles_pos = mask_visible_tiles.nonzero()
        total_visible_size = np.sum(tiles_byte_sizes[visible_tiles_pos])
        distance_of_interest = distances[visible_tiles_pos]

        theta_of_interest = params.TILE_SIDE_LEN / distance_of_interest * params.RADIAN_TO_DEGREE

        a_tiles_of_interest = self.tile_a[frame_idx_within_video][
            visible_tiles_pos]
        b_tiles_of_interest = self.tile_b[frame_idx_within_video][
            visible_tiles_pos]
        tiles_rates_to_be_watched = tiles_byte_sizes[visible_tiles_pos]

        lod = a_tiles_of_interest * np.log(b_tiles_of_interest *
                                           tiles_rates_to_be_watched + 1)

        num_points_per_degree = 2**lod / theta_of_interest
        quality_tiles = theta_of_interest * np.log(
            params.QR_MODEL_LOG_FACTOR *
            num_points_per_degree) + params.RELATIVE_TILE_UTILITY_CONST

        quality_tiles_in_mat = np.zeros_like(tiles_byte_sizes)
        quality_tiles_in_mat[visible_tiles_pos] = quality_tiles

        ang_resol = np.sum(num_points_per_degree * theta_of_interest) / np.sum(
            params.TILE_SIDE_LEN / distances[viewing_probability.nonzero()] *
            params.RADIAN_TO_DEGREE)
        frame_num_points = np.sum(num_points_per_degree * theta_of_interest)

        frame_quality = np.sum(quality_tiles)
        frame_quality_per_degree = frame_quality / np.sum(
            params.TILE_SIDE_LEN / distances[viewing_probability.nonzero()] *
            params.RADIAN_TO_DEGREE)

        if params.SAVE_TILE_LOD:
            tile_info_list = []
            for tile_idx in range(len(visible_tiles_pos[0])):
                x = visible_tiles_pos[0][tile_idx]
                y = visible_tiles_pos[1][tile_idx]
                z = visible_tiles_pos[2][tile_idx]
                lod_tile = lod[tile_idx]
                if lod_tile > 0 and lod_tile < 2:
                    lod_tile = 2
                lod_tile = int(np.round_(lod_tile))
                if params.LOSS_LESS_VERSION and params.FOV_ORACLE_KNOW:
                    lod_tile = params.TILE_DENSITY_LEVELS[-1]

                tile_info = {'x': x, 'y': y, 'z': z, 'lod': lod_tile}
                tile_info_list.append(tile_info)
            self.frame_lod_list.append(tile_info_list)

        ################# frame_quality_var ##########################
        mask_prev_pop_frame = np.zeros_like(self.tilesizes_prev_pop_frame)
        mask_prev_pop_frame[self.tilesizes_prev_pop_frame.nonzero()] = 1
        mask_prev_visible_tiles = mask_prev_pop_frame * self.prev_viewing_probability

        mask_overlap_tiles_in_mat = mask_visible_tiles * mask_prev_visible_tiles
        overlap_pos = mask_overlap_tiles_in_mat.nonzero()
        num_overlap_tiles = len(overlap_pos[0])

        if num_overlap_tiles != 0:
            quality_diff = quality_tiles_in_mat[
                overlap_pos] - self.tilequality_prev_pop_frame[overlap_pos]
            frame_quality_var = np.sum(quality_diff**2) / num_overlap_tiles
        else:
            frame_quality_var = 0
        self.tilequality_prev_pop_frame = quality_tiles_in_mat.copy()

        if len(viewing_probability.nonzero()[0]) == 0:
            self.mean_size_over_tiles_per_fov.append(float("inf"))
        else:
            self.mean_size_over_tiles_per_fov.append(
                total_visible_size / len(viewing_probability.nonzero()[0]))
        # visible size / frame size
        if np.sum(tiles_byte_sizes) == 0:
            self.effective_rate.append(float("inf"))
        else:
            self.effective_rate.append(total_visible_size /
                                       np.sum(tiles_byte_sizes))
        self.wasted_rate.append(np.sum(tiles_byte_sizes) - total_visible_size)
        self.visible_rate.append(total_visible_size)
        if np.sum(tiles_byte_sizes) == 0:
            self.wasted_ratio.append(0)
        else:
            self.wasted_ratio.append(
                (np.sum(tiles_byte_sizes) - total_visible_size) /
                np.sum(tiles_byte_sizes))

        if frame_idx == params.FRAME_IDX_OF_INTEREST:
            self.logger.debug('tiles rates--- %s (Byte)',
                              tiles_rates_to_be_watched)
            self.logger.debug('lod--- %s', lod)
            self.logger.debug('# points--- %s', 2**lod)
            self.logger.debug('wasted rates--- %f', self.wasted_rate[-1])
            self.logger.debug('effective ratio--- %f', self.effective_rate[-1])

        return frame_quality, frame_quality_var, ang_resol, frame_num_points, frame_quality_per_degree

    def emit_buffer(self, if_save_render):
        '''
			emit params.UPDATE_FREQ*params.FPS frames from front of buffer;;
			Based on their true viewpoint, calculate their HPR, distance, and quality;
			update pointers: buffer, current_viewing_frame_idx, history_viewpoints, history_bandwidths, current_bandwidth_idx
		'''

        # previous_visible_tiles_set = set()

        for frame_idx in range(
                self.current_viewing_frame_idx + 1,
                self.current_viewing_frame_idx +
                params.FPS * params.UPDATE_FREQ + 1):

            frame_idx_within_video = (
                frame_idx - params.TARGET_LATENCY) % params.NUM_FRAMES
            # current_visible_tiles_set = set()
            viewpoint = {"x":[self.fov_traces_obj.fov_traces[frame_idx][0]], \
                "y":[self.fov_traces_obj.fov_traces[frame_idx][1]], \
                "z":[self.fov_traces_obj.fov_traces[frame_idx][2]], \
               "pitch":[self.fov_traces_obj.fov_traces[frame_idx][3]], \
                 "yaw":[self.fov_traces_obj.fov_traces[frame_idx][4]], \
                "roll":[self.fov_traces_obj.fov_traces[frame_idx][5]]}

            # constant/fixed viewpoint
            # viewpoint = {"x":[0.05],
            # 			 "y":[1.7868],
            # 			 "z":[-1.0947],
            # 			 "pitch":[6.9163],
            # 			 "yaw":[350.8206],
            # 			 "roll":[359.9912]}

            viewing_probability, distances, true_viewing_probability, visible_tile_3d_idx, image_width, image_height, intrinsic_matrix, extrinsic_matrix = self.calculate_probability_to_be_viewed(
                viewpoint, frame_idx, frame_idx, emitting_buffer=True)

            if params.RENDER_VIEW and frame_idx == 900:
                self.viewer.create_window(visible=not if_save_render)
                # tile_xs, tile_ys, tile_zs = valid_tiles.nonzero()
                ply_frame_idx = (frame_idx - params.BUFFER_LENGTH
                                 ) % params.NUM_FRAMES + params.FRAME_IDX_BIAS
                for _3d_idx in range(len(visible_tile_3d_idx)):
                    ply_x = visible_tile_3d_idx[_3d_idx][0]
                    ply_y = visible_tile_3d_idx[_3d_idx][1]
                    ply_z = visible_tile_3d_idx[_3d_idx][2]

                    ply_tile_file = params.PLY_PATH + '/' + str(
                        ply_frame_idx) + '/' + params.VIDEO_NAME + '_' + str(
                            ply_frame_idx) + '_' + str(ply_x // 10) + str(
                                ply_x % 10) + '_' + str(ply_y // 10) + str(
                                    ply_y % 10) + '_' + str(ply_z // 10) + str(
                                        ply_z % 10) + '_6.ply'

                    pcd = o3d.io.read_point_cloud(ply_tile_file,
                                                  format='ply',
                                                  remove_nan_points=True,
                                                  remove_infinite_points=True,
                                                  print_progress=False)
                    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - params.POINTS_SHIFT) #longdress
                        # print(pcd)
                        # print(np.asarray(pcd.points))

                    self.viewer.add_geometry(pcd)
                    self.viewer.update_geometry(pcd)

                view_ctl = self.viewer.get_view_control()
                cam_pose_ctl = view_ctl.convert_to_pinhole_camera_parameters()
                cam_pose_ctl.intrinsic.height = image_height
                cam_pose_ctl.intrinsic.width = image_width
                cam_pose_ctl.intrinsic.intrinsic_matrix = intrinsic_matrix
                cam_pose_ctl.extrinsic = extrinsic_matrix
                view_ctl.convert_from_pinhole_camera_parameters(
                    cam_pose_ctl, allow_arbitrary=True)
                view_ctl.change_field_of_view()
                # render
                self.viewer.poll_events()
                self.viewer.update_renderer()
                if not if_save_render:
                    self.viewer.run()

                if if_save_render:
                    # check path exist or not, if not create it
                    if not os.path.exists('./result_new_hpr/emit/' +
                                          params.VIDEO_NAME + '/' +
                                          params.USER_FOV_TRACE):
                        os.makedirs('./result_new_hpr/emit/' +
                                    params.VIDEO_NAME + '/' +
                                    params.USER_FOV_TRACE)
                    self.viewer.capture_screen_image(
                        './result_new_hpr/emit/' + params.VIDEO_NAME + '/' +
                        params.USER_FOV_TRACE + '/' + 'fov_' +
                        str(frame_idx).zfill(3) + '.png',
                        do_render=False)
                # pdb.set_trace()
                # index should have 3 digits
                pdb.set_trace()
                self.viewer.destroy_window()

            # for tile_idx in range(len(viewing_probability[0].nonzero()[0])):
            #     x = viewing_probability[0].nonzero()[0][tile_idx]
            #     y = viewing_probability[0].nonzero()[1][tile_idx]
            #     z = viewing_probability[0].nonzero()[2][tile_idx]
            #     current_visible_tiles_set.add((x, y, z))
            # # if frame_idx >= 1:
            # #     intersect = current_visible_tiles_set.intersection(
            # #         previous_visible_tiles_set)
            # #     self.num_intersect_visible_tiles_trace.append(len(intersect))
            # # previous_visible_tiles_set = current_visible_tiles_set.copy()

            # calculate total span of fov (how many degrees)
            if len(viewing_probability[0].nonzero()[0]):
                self.total_span_per_frame.append(
                    np.sum(1 / distances[0][viewing_probability[0].nonzero()])
                    * params.TILE_SIDE_LEN * params.RADIAN_TO_DEGREE)
            else:
                self.total_span_per_frame.append(0)

            self.num_valid_tiles_per_frame.append(
                len(viewing_probability[0].nonzero()[0]))

            true_frame_quality, true_frame_quality_var, ang_resol, frame_num_points, frame_quality_per_degree = self.true_frame_quality_sum_and_var(
                viewing_probability[0], distances[0], frame_idx_within_video,
                frame_idx)
            self.frame_quality.append(true_frame_quality)
            self.frame_quality_var.append(true_frame_quality_var)
            self.ang_resol_list.append(ang_resol)
            self.frame_num_points_list.append(frame_num_points)
            self.frame_quality_per_degree_list.append(frame_quality_per_degree)

            # pop processed frame and store it as previous frame
            # for calculation of quality_var for next frame
            self.tilesizes_prev_pop_frame = self.buffer[0]
            self.tilequality_prev_pop_frame = np.zeros_like(
                self.tilesizes_prev_pop_frame)
            self.prev_viewing_probability = viewing_probability[0].copy()
            self.buffer.pop(0)

            # update history_viewpoints
            for key in self.history_viewpoints.keys():
                if len(self.history_viewpoints[key]
                       ) == params.FOV_PREDICTION_HISTORY_WIN_LENGTH:
                    self.history_viewpoints[key].pop(0)
                viewpoint_dof = viewpoint[key][0]
                if key == 'pitch' or key == 'yaw' or key == 'roll':
                    if viewpoint_dof < 90:  # user always move from 358 to 20
                        viewpoint_dof += 360

                if self.update_step > params.BUFFER_LENGTH // (
                        params.UPDATE_FREQ * params.FPS):
                    self.history_viewpoints[key].append(viewpoint_dof)

            # fov accuracy lists
            # if frame_idx >= params.BUFFER_LENGTH and params.SAVE_FOV_ACCURACY_PER_FRAME:
            #     # pdb.set_trace()
            #     front_idx = frame_idx - self.current_viewing_frame_idx - 1
            #     end_idx = front_idx + params.BUFFER_LENGTH - params.FPS
            #     middle_idx = int(params.BUFFER_LENGTH / params.FPS // 2 *
            #                      params.FPS + front_idx)
            #     self.fov_front_accuracy_list.append(
            #         self.overlap_ratio_history[front_idx][0])
            #     self.fov_end_accuracy_list.append(
            #         self.overlap_ratio_history[end_idx][0])
            #     self.fov_middle_accuracy_list.append(
            #         self.overlap_ratio_history[middle_idx][0])

            #     fov_accuracy_cur_fr = []
            #     for buf_pos in range(front_idx, params.BUFFER_LENGTH,
            #                          params.FPS):
            #         fov_accuracy_cur_fr.append(
            #             self.overlap_ratio_history[buf_pos][0])
            #         self.overlap_ratio_history[buf_pos].pop(0)
            #     self.fov_mean_accuracy_list.append(
            #         np.mean(fov_accuracy_cur_fr))

        # update current_viewing_frame_idx, history_bandwidths, current_bandwidth_idx
        self.current_viewing_frame_idx = frame_idx
        self.current_bandwidth_idx += 1
        self.history_bandwidths.pop(0)
        self.history_bandwidths.append(
            self.bw_traces_obj.bw_trace[self.current_bandwidth_idx])

        logging.info("finish emitting buffer--- %f seconds ---",
                     time.time() - self.start_time)

    def update_tile_size_in_buffer(self):
        self.update_step += 1
        ##################### predict viewpoint #######################
        if params.PROGRESSIVE_DOWNLOADING:
            update_start_idx = self.current_viewing_frame_idx + params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS + 1
        else:
            update_start_idx = self.current_viewing_frame_idx + params.TARGET_LATENCY + 1

        update_end_idx = self.current_viewing_frame_idx + self.buffer_length + params.FPS * params.UPDATE_FREQ
        self.logger.debug('Updating frame %d to %d', update_start_idx,
                          update_end_idx)

        self.start_time = time.time()

        predict_end_idx = update_end_idx

        predicted_viewpoints = self.predict_viewpoint(
            predict_start_idx=update_start_idx,
            predict_end_idx=predict_end_idx)

        # print("predict_viewpoint--- ",
        #       time.time() - self.start_time, " seconds ---")
        self.logger.debug('predict_viewpoint--- %f seconds',
                          time.time() - self.start_time)

        for key in self.true_viewpoints.keys():
            for frame_idx in range(update_end_idx - update_start_idx + 1):
                # if using csv fov file
                # true_viewpoint_at_this_dof = self.fov_traces_obj.fov_traces.at[frame_idx + update_start_idx, params.MAP_6DOF_TO_HMD_DATA[key]]
                # if using txt fov file
                true_viewpoint_at_this_dof = self.fov_traces_obj.fov_traces[
                    frame_idx +
                    update_start_idx][params.MAP_6DOF_TO_HMD_DATA[key]]
                predicted_viewpoint_at_this_dof = predicted_viewpoints[key][
                    frame_idx]
                self.true_viewpoints[key][frame_idx].append(
                    true_viewpoint_at_this_dof)
                if key == 'pitch' or key == 'yaw' or key == 'roll':
                    predicted_viewpoint_at_this_dof = predicted_viewpoint_at_this_dof if 360 - predicted_viewpoint_at_this_dof >= predicted_viewpoint_at_this_dof else predicted_viewpoint_at_this_dof - 360
                    true_viewpoint_at_this_dof = true_viewpoint_at_this_dof if 360 - true_viewpoint_at_this_dof >= true_viewpoint_at_this_dof else true_viewpoint_at_this_dof - 360
                if self.update_step > params.BUFFER_LENGTH // (
                        params.UPDATE_FREQ * params.FPS):
                    self.fov_predict_accuracy_trace[key][frame_idx].append(
                        abs(predicted_viewpoint_at_this_dof -
                            true_viewpoint_at_this_dof))

        logging.info("fov_predict_accuracy_trace--- %f seconds ---",
                     time.time() - self.start_time)
        #################################################################

        viewing_probability, distances, true_viewing_probability, visible_tile_3d_idx, image_width, image_height, intrinsic_matrix, extrinsic_matrix = self.calculate_probability_to_be_viewed(
            predicted_viewpoints, update_start_idx, update_end_idx)

        logging.info("viewing_probability--- %f seconds ---",
                     time.time() - self.start_time)

        # predict bandwidth of future 1s
        predicted_bandwidth_budget = self.predict_bandwidth(
        ) * params.SCALE_BW  # Mbps

        logging.info("predict_bandwidth--- %f seconds ---",
                     time.time() - self.start_time)

        # calculate distance only for viewable valid tiles
        # distances = self.calculate_distance(predicted_viewpoints)
        if params.ALGO in (Algo.KKT, Algo.RUMA_SCALABLE):
            z_weights = self.calculate_z(viewing_probability,
                                         np.array(distances), update_start_idx,
                                         update_end_idx,
                                         predicted_bandwidth_budget)

            logging.info("calculate_z--- %f seconds ---",
                         time.time() - self.start_time)

        if params.ALGO == Algo.MMSYS_HYBRID_TILING:
            tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0, sorted_z_weights = self.hybrid_tiling(
                np.array(z_weights),
                predicted_bandwidth_budget * params.Mbps_TO_Bps,
                update_start_idx, update_end_idx)
        elif params.ALGO == Algo.RUMA_SCALABLE or params.ALGO == params.Algo.RUMA_NONSCALABLE:
            tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0, sorted_z_weights = self.RUMA(
                distances, np.array(z_weights),
                predicted_bandwidth_budget * params.Mbps_TO_Bps,
                update_start_idx, update_end_idx)
        elif params.ALGO == Algo.KKT:
            tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0, sorted_z_weights = self.kkt(
                np.array(z_weights),
                predicted_bandwidth_budget * params.Mbps_TO_Bps,
                update_start_idx, update_end_idx)
        elif params.ALGO == Algo.AVERAGE:
            tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0 = self._average(
                viewing_probability,
                predicted_bandwidth_budget * params.Mbps_TO_Bps,
                update_start_idx, update_end_idx)
        else:
            pass

        if params.ALGO == Algo.KKT:
            logging.info("kkt--- %f seconds ---",
                         time.time() - self.start_time)
        elif params.ALGO == Algo.RUMA_SCALABLE:
            logging.info("ruma--- %f seconds ---",
                         time.time() - self.start_time)
        else:
            pass

        # update buffer following tiles_rate_solution output by algorithm
        for _ in range(params.FPS * params.UPDATE_FREQ):
            self.buffer.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))

        true_bandwidth_budget = self.bw_traces_obj.bw_trace[
            self.current_bandwidth_idx + 1] * params.SCALE_BW  # Mbps
        success_download_rate = 1
        if params.SCALABLE_CODING:
            consumed_bandwidth = sum_solution_rate - sum_r0
        else:
            consumed_bandwidth = sum_solution_rate

        if consumed_bandwidth != 0:
            success_download_rate = min(
                1, true_bandwidth_budget * params.Mbps_TO_Bps /
                consumed_bandwidth)
        else:
            logging.info("!!!!!!!!! nothing download !!!")
        if success_download_rate < 1 - 1e-4:  # 1e-4 is noise error term
            print("success download rate:", success_download_rate)
            tiles_rate_solution = (tiles_rate_solution - buffered_tiles_sizes[
                (params.BUFFER_LENGTH - tiles_rate_solution.shape[0]
                 ):]) * success_download_rate + buffered_tiles_sizes[
                     (params.BUFFER_LENGTH - tiles_rate_solution.shape[0]):]
            sum_solution_rate = np.sum(tiles_rate_solution)

        logging.info("start updating buffer--- %f seconds ---",
                     time.time() - self.start_time)

        for frame_idx in range(update_start_idx, update_end_idx + 1):
            # see how the tiles's rate of one frame evlolves over time
            if frame_idx == params.FRAME_IDX_OF_INTEREST:
                frame_idx_within_video = (
                    frame_idx - params.TARGET_LATENCY) % params.NUM_FRAMES
                tiles_pos = tiles_rate_solution[frame_idx -
                                                update_start_idx].nonzero()
                a_tiles_of_interest = self.tile_a[frame_idx_within_video][
                    tiles_pos]
                b_tiles_of_interest = self.tile_b[frame_idx_within_video][
                    tiles_pos]
                prev_tiles_rates_to_be_watched = self.buffer[
                    frame_idx - self.current_viewing_frame_idx - 1][tiles_pos]
                tiles_rates_to_be_watched = tiles_rate_solution[
                    frame_idx - update_start_idx][tiles_pos]
                prev_lod = a_tiles_of_interest * np.log(
                    b_tiles_of_interest * prev_tiles_rates_to_be_watched + 1)
                lod = a_tiles_of_interest * np.log(
                    b_tiles_of_interest * tiles_rates_to_be_watched + 1)
                prev_num_points = 2**prev_lod
                num_points = 2**lod
                # quality_tiles = theta_of_interest * np.log(
                #     params.QR_MODEL_LOG_FACTOR * num_points_per_degree)

                self.logger.debug('tiles rates--- %s (Byte)',
                                  tiles_rates_to_be_watched)
                self.logger.debug(
                    'delta rates--- %s (Byte)',
                    tiles_rates_to_be_watched - prev_tiles_rates_to_be_watched)
                self.logger.debug('lod--- %s', lod)
                self.logger.debug('delta lod--- %s', lod - prev_lod)
                self.logger.debug('# points--- %s', num_points)
                self.logger.debug('delta points--- %s',
                                  num_points - prev_num_points)
                # pdb.set_trace()

            self.buffer[frame_idx - self.current_viewing_frame_idx -
                        1] = tiles_rate_solution[frame_idx -
                                                 update_start_idx].copy()

        self.plot_bw_trace.append(true_bandwidth_budget)  # Mbps
        self.plot_predicted_bw_trace.append(predicted_bandwidth_budget)  # Mbps
        self.buffer_size_trace.append(sum_solution_rate)  # byte
        self.delta_buffer_size_trace.append(sum_solution_rate -
                                            buffered_tiles_sizes)  # byte
        self.success_download_rate_trace.append(success_download_rate)
        self.bw_predict_accuracy_trace.append(predicted_bandwidth_budget -
                                              true_bandwidth_budget)  # Mbps
        # pdb.set_trace()

        ########### see how much bw is wasted for each frame/segment due to fov error ###########
        viewing_probability = np.array(viewing_probability)
        true_viewing_probability = np.array(true_viewing_probability)
        wasted_tiles = viewing_probability - true_viewing_probability * viewing_probability
        # wasted bw per tile
        # pdb.set_trace()
        wasted_bw_per_tile = (tiles_rate_solution - buffered_tiles_sizes[
            (params.BUFFER_LENGTH - tiles_rate_solution.shape[0]):]
                              ) * wasted_tiles[:params.BUFFER_LENGTH]
        # wasted bw per frame
        wasted_bw_per_fr = wasted_bw_per_tile.sum(
            axis=tuple(range(1, wasted_bw_per_tile.ndim)))
        # wasted bw per segment
        wasted_bw_per_seg = np.add.reduceat(
            wasted_bw_per_fr,
            np.arange(0, len(wasted_bw_per_fr),
                      params.FPS))  # shape: (latency in sec, )
        self.logger.debug('Wasted BW per segment: %s', wasted_bw_per_seg)
        self.logger.debug('ratio to consumed BW: %s',
                          wasted_bw_per_seg / consumed_bandwidth)

        wasted_bw_cur_step = np.sum(wasted_bw_per_seg)
        self.logger.debug('Wasted BW this step: %f, ratio to consumed BW: %f',
                          wasted_bw_cur_step,
                          wasted_bw_cur_step / consumed_bandwidth)
        # pdb.set_trace()

        logging.info("update buffer ends--- %f seconds ---",
                     time.time() - self.start_time)

    def calculate_z(self,
                    viewing_probability,
                    distances,
                    update_start_idx,
                    update_end_idx,
                    bw_budget,
                    evaluation_flag=False):
        '''
            also need self.frame_weights, self.tile_a and self.distance_weight
        '''
        z_weights = []

        a_updated_tiles_core = np.concatenate(
            (self.tile_a[(update_start_idx - params.TARGET_LATENCY) %
                         params.NUM_FRAMES:],
             self.tile_a[:(update_start_idx - params.TARGET_LATENCY) %
                         params.NUM_FRAMES]),
            axis=0)
        a_updated_tiles = a_updated_tiles_core.copy()

        for _ in range((params.TARGET_LATENCY - 1) // params.NUM_FRAMES):
            a_updated_tiles = np.concatenate(
                (a_updated_tiles, a_updated_tiles_core), axis=0)

        if params.HOMO_TILES:
            # for i in range(a_updated_tiles.shape[0]):
            #     valid_idx = a_updated_tiles[i].nonzero()
            #     a_updated_tiles[i][valid_idx] = np.sum(a_updated_tiles[i]) / len(valid_idx[0])
            valid_idx = a_updated_tiles.nonzero()
            a_updated_tiles[valid_idx] = np.sum(a_updated_tiles) / len(
                valid_idx[0])
            a_updated_tiles[valid_idx] = 1

        if params.UNIFORM_DISTANCE:
            # for i in range(len(distances)):
            #     valid_idx = distances[i].nonzero()
            #     distances[i][valid_idx] = np.sum(distances[i]) / len(valid_idx[0])
            valid_idx = distances.nonzero()
            distances[valid_idx] = np.sum(distances) / len(valid_idx[0])
            distances[valid_idx] = 2

        # frame weight is linear wrt. frame_idx: w_j = a * frame_idx + b
        frame_weight_decrease_speed = 0
        if not params.FOV_ORACLE_KNOW:
            frame_weight_decrease_speed = -(
                params.MAX_FRAME_WEIGHT - params.MIN_FRAME_WEIGHT) / (
                    update_end_idx - update_start_idx)
        frame_weight = 1

        for frame_idx in range(update_start_idx, update_end_idx + 1):
            if params.FRAME_WEIGHT_TYPE == params.FrameWeightType.LINEAR_DECREASE:
                # maximal frame_weight = 1, minimal frame_weight is 0.1
                frame_weight = frame_weight_decrease_speed * (
                    frame_idx - update_start_idx) + params.MAX_FRAME_WEIGHT
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.CONST:
                frame_weight = 1
            # based on fov prediction accuracy: overlap ratio
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.FOV_PRED_ACCURACY:
                frame_weight = np.mean(
                    self.overlap_ratio_history[frame_idx - update_start_idx]
                    [-params.OVERLAP_RATIO_HISTORY_WIN_LENGTH:])
                # overlap ratio with fov oracle
                frame_weight = np.exp(
                    self.overlap_ratio_history[frame_idx -
                                               update_start_idx][-1])
                frame_weight = self.overlap_ratio_history[frame_idx -
                                                          update_start_idx][-1]
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.DYNAMIC:
                # frame_weight = np.mean(
                #     self.overlap_ratio_history[frame_idx - update_start_idx])
                # overlap ratio with fov oracle
                frame_weight = self.overlap_ratio_history[frame_idx -
                                                          update_start_idx][-1]
                # frame_weight = np.exp(self.overlap_ratio_history[frame_idx - update_start_idx][-1])
                patch_win_len = bw_budget / params.FRAME_WEIGHT_DYNAMIC_BW_REF * params.FRAME_WEIGHT_DYNAMIC_WINLEN_REF  # type: float
                # patch_win_len = 300
                if self.update_step > params.TARGET_LATENCY // params.FPS:
                    if frame_idx - update_start_idx >= patch_win_len:
                        frame_weight = 0
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.STEP_FUNC:
                if self.update_step <= params.TARGET_LATENCY // params.FPS:
                    frame_weight = 1
                else:
                    frame_weight = 0
                    if frame_idx - update_start_idx < params.FRAME_WEIGHT_STEP_IDX:
                        frame_weight = 1
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.EXP_DECREASE:
                frame_weight = np.power(
                    params.FRAME_WEIGHT_EXP_BOTTOM,
                    params.FRAME_WEIGHT_EXP_FACTOR *
                    (1 - (frame_idx - update_start_idx) /
                     (update_end_idx - update_start_idx)))

                #### add a step window
                # patch_win_len = 150
                # if self.update_step > params.TARGET_LATENCY // params.FPS:
                #     if frame_idx - update_start_idx >= patch_win_len:
                #         frame_weight = 0
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.BELL_SHAPE:
                assert (
                    params.FRAME_WEIGHT_PEAK_IDX < params.BUFFER_LENGTH // 2
                ), "!!!! frame weight peak index >= 0.5 * buffer len !!!!!"
                # if frame_idx - update_start_idx > params.FRAME_WEIGHT_PEAK_IDX:
                frame_weight = np.power(
                    params.FRAME_WEIGHT_EXP_BOTTOM,
                    params.FRAME_WEIGHT_EXP_FACTOR *
                    (1 - np.abs(frame_idx - update_start_idx -
                                params.FRAME_WEIGHT_PEAK_IDX) /
                     (update_end_idx - update_start_idx -
                      params.FRAME_WEIGHT_PEAK_IDX)))
            else:
                frame_weight = 1

            if params.MMSYS_HYBRID_TILING:
                if update_end_idx - frame_idx < params.UPDATE_FREQ * params.FPS:
                    frame_weight = 1
                else:
                    frame_weight = 0

            z_weights.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            valid_locs = viewing_probability[frame_idx -
                                             update_start_idx].nonzero()
            theta_of_interest = params.TILE_SIDE_LEN / distances[
                frame_idx -
                update_start_idx][valid_locs] * params.RADIAN_TO_DEGREE

            if params.ALGO == Algo.RUMA_SCALABLE:
                z_weights[frame_idx - update_start_idx][valid_locs] = 1
                continue
            if params.DISTANCE_BASED_UTILITY:
                z_weights[frame_idx - update_start_idx][valid_locs] = frame_weight * viewing_probability[frame_idx - update_start_idx][valid_locs] \
                              * a_updated_tiles[frame_idx - update_start_idx][valid_locs] * np.log(2) * theta_of_interest
            else:
                z_weights[
                    frame_idx -
                    update_start_idx][valid_locs] = frame_weight  # q=log(r+1)

        return z_weights

    def calculate_probability_to_be_viewed(self,
                                           viewpoints,
                                           update_start_idx,
                                           update_end_idx,
                                           emitting_buffer=False):
        # probability can be represented by overlap ratio

        occlusion_obj = Occlusion(self.valid_tiles_obj, self.fov_traces_obj,
                                  self.origin)
        if emitting_buffer:
            viewing_probability, distances, true_viewing_probability, visible_tile_3d_idx, image_width, image_height, intrinsic_matrix, extrinsic_matrix = occlusion_obj.open3d_hpr(
                viewpoints,
                update_start_idx,
                update_end_idx,
                emitting_buffer,
                self.overlap_ratio_history,
                if_visible_pts=True,
                if_save_render=True)
        else:
            viewing_probability, distances, true_viewing_probability, visible_tile_3d_idx, image_width, image_height, intrinsic_matrix, extrinsic_matrix = occlusion_obj.open3d_hpr(
                viewpoints,
                update_start_idx,
                update_end_idx,
                emitting_buffer,
                self.overlap_ratio_history,
                if_visible_pts=True,
                if_save_render=True)

        return viewing_probability, list(
            np.array(distances) * 1
        ), true_viewing_probability, visible_tile_3d_idx, image_width, image_height, intrinsic_matrix, extrinsic_matrix

    def calculate_distance(self, point1, point2):
        distance = np.linalg.norm(point1 - point2)
        # return 0
        return distance

    def predict_viewpoint(self, predict_start_idx, predict_end_idx):
        predicted_viewpoints = self.fov_traces_obj.predict_6dof(
            self.current_viewing_frame_idx, predict_start_idx, predict_end_idx,
            self.history_viewpoints)
        return predicted_viewpoints

    def predict_bandwidth(self):
        bandwidth = self.bw_traces_obj.predict_bw(self.current_bandwidth_idx,
                                                  self.history_bandwidths)
        return bandwidth

    def calculate_unweighted_tile_quality(self, rate, distance, num_pts):
        if rate == 0:
            return 0
        # tmp = a * rate + b
        # assert (
        #     tmp > 0
        # ), "!!!! RUMA->calculate_unweighted_tile_quality->rate is too small !!!!!!!!!"

        # return np.log(distance / params.TILE_SIDE_LEN * np.power(tmp, 0.5) /
        #               params.RADIAN_TO_DEGREE)
        return num_pts * np.log(rate)

    def RUMA(self, distances, z_weights, bandwidth_budget, update_start_idx,
             update_end_idx):
        num_frames_to_update = update_end_idx - update_start_idx + 1

        buffered_tiles_sizes = self.buffer.copy()
        for i in range(params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE *
                       params.FPS):
            buffered_tiles_sizes.pop(0)
        for i in range(params.UPDATE_FREQ * params.FPS):
            buffered_tiles_sizes.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
        buffered_tiles_sizes = np.array(buffered_tiles_sizes)

        tiles_rate_solution = buffered_tiles_sizes.copy()

        num_pts_updated_tiles_core = np.concatenate((
            self.num_pts_versions[(update_start_idx - params.TARGET_LATENCY) %
                                  params.NUM_FRAMES:],
            self.num_pts_versions[:(update_start_idx - params.TARGET_LATENCY) %
                                  params.NUM_FRAMES]),
                                                    axis=0)
        num_pts_updated_tiles = num_pts_updated_tiles_core.copy()

        for _ in range((params.TARGET_LATENCY - 1) // params.NUM_FRAMES):
            num_pts_updated_tiles = np.concatenate(
                (num_pts_updated_tiles, num_pts_updated_tiles_core), axis=1)

        z_weight_locations = []
        nonzero_zWeight_locs = np.where(
            (z_weights != 0) & (buffered_tiles_sizes != self.max_tile_sizes))
        nonzero_zWeight_frame_idx = nonzero_zWeight_locs[0]
        nonzero_zWeight_x = nonzero_zWeight_locs[1]
        nonzero_zWeight_y = nonzero_zWeight_locs[2]
        nonzero_zWeight_z = nonzero_zWeight_locs[3]

        # r0 = np.zeros(
        #     (num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
        #      params.NUM_TILES_PER_SIDE_IN_A_FRAME,
        #      params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        # r0[nonzero_zWeight_locs] = np.maximum(
        #     buffered_tiles_sizes[nonzero_zWeight_locs],
        #     self.min_rates_frames[nonzero_zWeight_locs])

        # tiles_rate_solution[nonzero_zWeight_locs] = np.maximum(buffered_tiles_sizes[nonzero_zWeight_locs], self.min_rates_frames[nonzero_zWeight_locs])
        tiles_rate_solution[nonzero_zWeight_locs] = buffered_tiles_sizes[
            nonzero_zWeight_locs].copy()

        utility_rate_slopes = np.zeros(
            (num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        next_version_rates = np.zeros(
            (num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))

        for nonzero_zWeight_idx in range(len(nonzero_zWeight_frame_idx)):
            frame_idx = nonzero_zWeight_frame_idx[nonzero_zWeight_idx]
            frame_idx_within_video = (
                update_start_idx + frame_idx -
                params.TARGET_LATENCY) % params.NUM_FRAMES

            x = nonzero_zWeight_x[nonzero_zWeight_idx]
            y = nonzero_zWeight_y[nonzero_zWeight_idx]
            z = nonzero_zWeight_z[nonzero_zWeight_idx]
            z_weight_locations.append({
                "frame_idx": frame_idx,
                "x": x,
                "y": y,
                "z": z
            })
            current_rate = tiles_rate_solution[frame_idx][x][y][z]
            current_version = self.decide_rate_version(
                current_rate, self.rate_versions[:, frame_idx_within_video, x,
                                                 y, z])

            if current_version == params.NUM_RATE_VERSIONS:  # max rate version
                utility_rate_slopes[frame_idx][x][y][z] = 0
                continue
            next_version = current_version + 1
            next_version_rate = self.rate_versions[params.NUM_RATE_VERSIONS - (
                current_version + 1)][frame_idx_within_video][x][y][z]
            next_version_rates[frame_idx][x][y][z] = next_version_rate
            # try:
            assert (next_version_rate - current_rate >
                    0), "!!! same rate between 2 levels (before loop) !!!!!"
            # except:
            #     pdb.set_trace()
            if current_version == 0:
                num_pts_cur_version = 0
            else:
                num_pts_cur_version = num_pts_updated_tiles[
                    current_version - 1][frame_idx][x][y][z]
            num_pts_next_version = num_pts_updated_tiles[next_version -
                                                         1][frame_idx][x][y][z]

            ############ use lod to calculate num_pts ########
            current_lod = params.TILE_DENSITY_LEVELS[current_version - 1]
            if current_version == 0:
                current_lod = 0
            next_lod = params.TILE_DENSITY_LEVELS[next_version - 1]

            num_pts_cur_version = 2**current_lod
            num_pts_next_version = 2**next_lod
            #######################################

            utility_rate_slopes[frame_idx][x][y][z] = z_weights[frame_idx][x][y][z] * \
                        (self.calculate_unweighted_tile_quality(next_version_rate, distances[frame_idx][x][y][z], num_pts_next_version) \
                       - self.calculate_unweighted_tile_quality(current_rate, distances[frame_idx][x][y][z], num_pts_cur_version)) \
                       / (next_version_rate - current_rate)

            # print(self.calculate_unweighted_tile_quality(next_version_rate, distances[frame_idx][x][y][z], self.tile_a[x][y][z], self.tile_b[x][y][z]))
            # print(self.calculate_unweighted_tile_quality(tiles_rate_solution[frame_idx][x][y][z], distances[frame_idx][x][y][z], self.tile_a[x][y][z], self.tile_b[x][y][z]))
            # print(next_version_rate)
            # print(current_rate)
            # pdb.set_trace()

        # sorted_z_weights = sorted(z_weight_locations, \
        #         key=lambda loc: z_weights[loc["frame_idx"]]\
        #                 [loc["x"]]\
        #                 [loc["y"]]\
        #                 [loc["z"]], reverse=True)
        sorted_z_weights = z_weights

        # total_size = 0
        # if params.RUMA_SCALABLE_CODING:
        # 	locs = np.where(tiles_rate_solution > buffered_tiles_sizes)
        # 	total_size = tiles_rate_solution[locs]
        # else:

        # total_size = np.sum(tiles_rate_solution) - np.sum(buffered_tiles_sizes)
        # total_size = np.sum(tiles_rate_solution)

        # final total_size should be equal to total_size_constraint
        total_size_constraint = bandwidth_budget

        consumed_bandwidth = 0
        # current_total_size = np.sum(buffered_tiles_sizes)

        if consumed_bandwidth >= total_size_constraint:  # total budget cannot satisfy all tiles with lowest rate version
            print(
                "!!!! total budget cannot satisfy all tiles with lowest rate version !!!!!!!"
            )
            return tiles_rate_solution, buffered_tiles_sizes, np.sum(
                tiles_rate_solution), np.sum(
                    buffered_tiles_sizes), sorted_z_weights

        while consumed_bandwidth < total_size_constraint:
            max_slope_frame_idx, max_slope_x, max_slope_y, max_slope_z = np.unravel_index(
                np.argmax(utility_rate_slopes, axis=None),
                utility_rate_slopes.shape)
            max_slope_frame_idx_within_video = (
                update_start_idx + max_slope_frame_idx -
                params.TARGET_LATENCY) % params.NUM_FRAMES

            max_slope = utility_rate_slopes[max_slope_frame_idx][max_slope_x][
                max_slope_y][max_slope_z]
            if max_slope == 0:
                break
            current_rate = tiles_rate_solution[max_slope_frame_idx][
                max_slope_x][max_slope_y][max_slope_z]
            next_version_rate = next_version_rates[max_slope_frame_idx][
                max_slope_x][max_slope_y][max_slope_z]
            # new_total_size = 0
            # if params.RUMA_SCALABLE_CODING:
            # 	new_total_size = total_size + next_version_rate - current_rate
            # else:
            # 	new_total_size = total_size + next_version_rate
            # if new_total_size > total_size_constraint:
            # 	break

            # if params.RUMA_SCALABLE_CODING:
            # total_size += (next_version_rate - current_rate)
            # else:
            # 	total_size += next_version_rate

            tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][
                max_slope_z] = next_version_rate

            consumed_bandwidth += (next_version_rate - current_rate)
            # else:
            #     # locs = np.where(tiles_rate_solution > buffered_tiles_sizes)
            #     # consumed_bandwidth = np.sum(tiles_rate_solution[locs])
            #     # consumed_bandwidth += next_version_rate
            #     if current_rate == buffered_tiles_sizes[max_slope_frame_idx][
            #             max_slope_x][max_slope_y][max_slope_z]:
            #         consumed_bandwidth += next_version_rate
            #     else:
            #         consumed_bandwidth += (next_version_rate - current_rate)

            if consumed_bandwidth > total_size_constraint:
                tiles_rate_solution[max_slope_frame_idx][max_slope_x][
                    max_slope_y][max_slope_z] = current_rate
                break

            if next_version_rate == self.rate_versions[0][
                    max_slope_frame_idx_within_video][max_slope_x][
                        max_slope_y][max_slope_z]:
                utility_rate_slopes[max_slope_frame_idx][max_slope_x][
                    max_slope_y][max_slope_z] = 0
            else:
                current_version = self.decide_rate_version(
                    next_version_rate,
                    self.rate_versions[:, max_slope_frame_idx_within_video,
                                       max_slope_x, max_slope_y, max_slope_z])
                next_version = current_version + 1
                next_version_rate = self.rate_versions[
                    params.NUM_RATE_VERSIONS -
                    (current_version + 1)][max_slope_frame_idx_within_video][
                        max_slope_x][max_slope_y][max_slope_z]
                next_version_rates[max_slope_frame_idx][max_slope_x][
                    max_slope_y][max_slope_z] = next_version_rate
                assert (next_version_rate -
                        tiles_rate_solution[max_slope_frame_idx][max_slope_x]
                        [max_slope_y][max_slope_z] >
                        0), "!!! same rate between 2 levels (in loop) !!!!!"

                if current_version == 0:
                    num_pts_cur_version = 0
                else:
                    num_pts_cur_version = num_pts_updated_tiles[
                        current_version - 1][max_slope_frame_idx][max_slope_x][
                            max_slope_y][max_slope_z]
                num_pts_next_version = num_pts_updated_tiles[next_version - 1][
                    max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z]

                ############ use lod to calculate num_pts ########
                current_lod = params.TILE_DENSITY_LEVELS[current_version - 1]
                if current_version == 0:
                    current_lod = 0
                next_lod = params.TILE_DENSITY_LEVELS[next_version - 1]

                num_pts_cur_version = 2**current_lod
                num_pts_next_version = 2**next_lod
                #######################################

                utility_rate_slopes[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] = z_weights[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] * \
                           (self.calculate_unweighted_tile_quality(next_version_rate, distances[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], num_pts_next_version) \
                          - self.calculate_unweighted_tile_quality(tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], distances[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], num_pts_cur_version)) \
                          / (next_version_rate - tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z])

        return tiles_rate_solution, buffered_tiles_sizes, np.sum(
            tiles_rate_solution), np.sum(
                buffered_tiles_sizes), sorted_z_weights

    def _average(self, viewing_probability, bandwidth_budget, update_start_idx,
                 update_end_idx):
        ##################### get v1 and v2 for each tile: ################################
        # v1 = z_weight / r0; v2 = z_weight / params.MAX_TILE_SIZE.
        num_frames_to_update = update_end_idx - update_start_idx + 1
        # pdb.set_trace()

        # TODO by Tongyu: wherever using max_rates_cur_step, assert max_rates_cur_step[.] != 0 to make sure it matches right tile positions.
        # also for a_update_tiles and b_update_tiles.
        max_rates_cur_step_core = np.concatenate(
            (self.max_tile_sizes[(update_start_idx - params.TARGET_LATENCY) %
                                 params.NUM_FRAMES:],
             self.max_tile_sizes[:(update_start_idx - params.TARGET_LATENCY) %
                                 params.NUM_FRAMES]),
            axis=0)
        max_rates_cur_step = max_rates_cur_step_core.copy()

        for _ in range((params.TARGET_LATENCY - 1) // params.NUM_FRAMES):
            max_rates_cur_step = np.concatenate(
                (max_rates_cur_step, max_rates_cur_step_core), axis=0)

        # tiles' byte size that are already in buffer
        buffered_tiles_sizes = self.buffer.copy()
        for i in range(params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE *
                       params.FPS):
            buffered_tiles_sizes.pop(0)
        for i in range(params.UPDATE_FREQ * params.FPS):
            buffered_tiles_sizes.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
        buffered_tiles_sizes = np.array(buffered_tiles_sizes)

        tiles_rate_solution = buffered_tiles_sizes.copy()
        visible_tiles_pos = np.array(viewing_probability).nonzero()
        num_tiles_to_update = len(visible_tiles_pos[0])
        tiles_rate_solution[visible_tiles_pos] = buffered_tiles_sizes[
            visible_tiles_pos] + bandwidth_budget / num_tiles_to_update
        tiles_rate_solution[
            tiles_rate_solution > max_rates_cur_step] = max_rates_cur_step[
                tiles_rate_solution > max_rates_cur_step]
        total_size = np.sum(tiles_rate_solution)

        return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(
            buffered_tiles_sizes)

    def kkt(self, z_weights, bandwidth_budget, update_start_idx,
            update_end_idx):
        ##################### get v1 and v2 for each tile: ################################
        # v1 = z_weight / r0; v2 = z_weight / params.MAX_TILE_SIZE.
        num_frames_to_update = update_end_idx - update_start_idx + 1
        # pdb.set_trace()

        # TODO by Tongyu: write a function for this concatenate process
        b_updated_tiles_core = np.concatenate(
            (self.tile_b[(update_start_idx - params.TARGET_LATENCY) %
                         params.NUM_FRAMES:],
             self.tile_b[:(update_start_idx - params.TARGET_LATENCY) %
                         params.NUM_FRAMES]),
            axis=0)
        b_updated_tiles = b_updated_tiles_core.copy()

        for _ in range((params.TARGET_LATENCY - 1) // params.NUM_FRAMES):
            b_updated_tiles = np.concatenate(
                (b_updated_tiles, b_updated_tiles_core), axis=0)

        if params.HOMO_TILES:
            # for i in range(b_updated_tiles.shape[0]):
            #     valid_idx = b_updated_tiles[i].nonzero()
            #     b_updated_tiles[i][valid_idx] = np.sum(b_updated_tiles[i]) / len(valid_idx[0])
            valid_idx = b_updated_tiles.nonzero()
            b_updated_tiles[valid_idx] = np.sum(b_updated_tiles) / len(
                valid_idx[0])
            b_updated_tiles[valid_idx] = 0.5

        if not params.DISTANCE_BASED_UTILITY:  # q=log(r+1)
            valid_idx = b_updated_tiles.nonzero()
            b_updated_tiles[valid_idx] = 1

        # TODO by Tongyu: wherever using max_rates_cur_step, assert max_rates_cur_step[.] != 0 to make sure it matches right tile positions.
        # also for a_update_tiles and b_update_tiles.
        max_rates_cur_step_core = np.concatenate(
            (self.max_tile_sizes[(update_start_idx - params.TARGET_LATENCY) %
                                 params.NUM_FRAMES:],
             self.max_tile_sizes[:(update_start_idx - params.TARGET_LATENCY) %
                                 params.NUM_FRAMES]),
            axis=0)
        max_rates_cur_step = max_rates_cur_step_core.copy()

        for _ in range((params.TARGET_LATENCY - 1) // params.NUM_FRAMES):
            max_rates_cur_step = np.concatenate(
                (max_rates_cur_step, max_rates_cur_step_core), axis=0)

        # tiles' byte size that are already in buffer
        buffered_tiles_sizes = self.buffer.copy()
        for i in range(params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE *
                       params.FPS):
            buffered_tiles_sizes.pop(0)
        for i in range(params.UPDATE_FREQ * params.FPS):
            buffered_tiles_sizes.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
        buffered_tiles_sizes = np.array(buffered_tiles_sizes)

        # each tile has a 4 tuple location: (frame_idx, x, y, z)
        # tiles_rate_solution = params.MAX_TILE_SIZE * np.ones((num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        if params.PROGRESSIVE_DOWNLOADING:
            if params.SCALABLE_CODING:
                tiles_rate_solution = buffered_tiles_sizes.copy()
            else:
                tiles_rate_solution = np.zeros_like(buffered_tiles_sizes)
        else:
            # all zeros
            tiles_rate_solution = buffered_tiles_sizes[-params.FPS:].copy()

        # each tile_value has a 5 tuple location: (frame_idx, x, y, z, value_idx)
        tiles_values = np.zeros(
            (2, num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))

        # locations of all tiles:
        # for sorting purpose later
        # v1 at value_idx=1; v2 at value_idx=0
        tiles_values_locations = []

        z_weight_locations = []

        # typeIII_tiles_set = set()

        # sum_typeIII_z_weight = 0

        nonzero_zWeight_locs = None
        if params.PROGRESSIVE_DOWNLOADING:
            nonzero_zWeight_locs = np.where((z_weights != 0) & (
                buffered_tiles_sizes != max_rates_cur_step))
        else:
            # if non-progressive downloading, the new coming tiles are impossible to be max rates now.
            nonzero_zWeight_locs = np.where(z_weights != 0)

        tiles_rate_solution[nonzero_zWeight_locs] = max_rates_cur_step[
            nonzero_zWeight_locs]

        nonzero_zWeight_frame_idx = nonzero_zWeight_locs[0]
        nonzero_zWeight_x = nonzero_zWeight_locs[1]
        nonzero_zWeight_y = nonzero_zWeight_locs[2]
        nonzero_zWeight_z = nonzero_zWeight_locs[3]

        # pdb.set_trace()

        tiles_values[0][nonzero_zWeight_locs] = z_weights[
            nonzero_zWeight_locs] / (max_rates_cur_step[nonzero_zWeight_locs] +
                                     1 / b_updated_tiles[nonzero_zWeight_locs])

        # nonzero_zWeight_nonzero_r0_locs = np.where((z_weights != 0) & (buffered_tiles_sizes != self.max_tile_sizes) & (buffered_tiles_sizes != 0))
        r0 = np.zeros(
            (num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))

        if params.PROGRESSIVE_DOWNLOADING:
            # seems redundant between r0 and buffered_tiles_sizes
            # r0[nonzero_zWeight_locs] = buffered_tiles_sizes[
            #     nonzero_zWeight_locs].copy()
            if params.SCALABLE_CODING:
                r0 = buffered_tiles_sizes.copy()
            else:
                r0 = np.zeros_like(buffered_tiles_sizes)

            # if non-progressive downloading, all r0's =0

        tiles_values[1][nonzero_zWeight_locs] = z_weights[
            nonzero_zWeight_locs] / (r0[nonzero_zWeight_locs] +
                                     1 / b_updated_tiles[nonzero_zWeight_locs])

        for nonzero_zWeight_idx in range(len(nonzero_zWeight_frame_idx)):
            frame_idx = nonzero_zWeight_frame_idx[nonzero_zWeight_idx]
            x = nonzero_zWeight_x[nonzero_zWeight_idx]
            y = nonzero_zWeight_y[nonzero_zWeight_idx]
            z = nonzero_zWeight_z[nonzero_zWeight_idx]

            # r0 = buffered_tiles_sizes[frame_idx][x][y][z]
            # z_weight = z_weights[frame_idx][x][y][z]

            # if r0 == 0:
            # tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":0})
            # else:
            # tiles_values[frame_idx][x][y][z][1] = z_weight / r0

            tiles_values_locations.append({
                "frame_idx": frame_idx,
                "x": x,
                "y": y,
                "z": z,
                "value_idx": 0
            })
            tiles_values_locations.append({
                "frame_idx": frame_idx,
                "x": x,
                "y": y,
                "z": z,
                "value_idx": 1
            })

            z_weight_locations.append({
                "frame_idx": frame_idx,
                "x": x,
                "y": y,
                "z": z
            })
            # tiles_values[frame_idx][x][y][z][0] = z_weight / params.MAX_TILE_SIZE

        # for frame_idx in range(num_frames_to_update):
        # 	for x in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
        # 		for y in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
        # 			for z in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
        # 				z_weight = z_weights[frame_idx][x][y][z] # can be 0

        # 				# byte size that already in buffer for this tile
        # 				r0 = buffered_tiles_sizes[frame_idx][x][y][z] # can be 0

        # 				if r0 == params.MAX_TILE_SIZE:
        # 					tiles_values[frame_idx][x][y][z][0] = z_weight / params.MAX_TILE_SIZE
        # 					tiles_values[frame_idx][x][y][z][1] = z_weight / params.MAX_TILE_SIZE
        # 					continue

        # 				if z_weight == 0:
        # 					tiles_values[frame_idx][x][y][z][1] = 0

        # 					# if z_weight is 0, optimal size of this tile should be r0 (unchanged)
        # 					# tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
        # 				elif r0 == 0:
        # 					tiles_values[frame_idx][x][y][z][1] = float('inf')
        # 					# sum_typeIII_z_weight += z_weight
        # 					# store the tile value location with inf v1
        # 					# typeIII_tiles_set.add((frame_idx, x, y, z))
        # 					tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":0})
        # 					z_weight_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z})

        # 				else:
        # 					tiles_values[frame_idx][x][y][z][1] = z_weight / r0

        # 					tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":0})
        # 					tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":1})

        # 					z_weight_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z})

        # 				tiles_values[frame_idx][x][y][z][0] = z_weight / params.MAX_TILE_SIZE
        ##########################################################################################################
        print("tiles values--- ",
              time.time() - self.start_time, " seconds ---")

        sorted_z_weights = sorted(z_weight_locations, \
                key=lambda loc: z_weights[loc["frame_idx"]]\
                        [loc["x"]]\
                        [loc["y"]]\
                        [loc["z"]], reverse=True)

        print("sort z weights--- ",
              time.time() - self.start_time, " seconds ---")

        # this is total size when lambda is the least positive tile value
        total_size = np.sum(tiles_rate_solution)

        # final total_size should be equal to total_size_constraint
        total_size_constraint = bandwidth_budget + np.sum(r0)

        if total_size <= total_size_constraint:
            print("lambda is the minimal positive tile value!", total_size,
                  total_size_constraint)
            return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(
                r0), sorted_z_weights

        # get sorted locations of all tiles' values (ascending order)
        # O(n*log(n)), where n is # of visible tiles
        # pdb.set_trace()
        sorted_tiles_values = sorted(tiles_values_locations, \
                key=lambda loc: tiles_values[loc["value_idx"]]\
                        [loc["frame_idx"]]\
                        [loc["x"]]\
                        [loc["y"]]\
                        [loc["z"]])
        print("sort tiles values--- ",
              time.time() - self.start_time, " seconds ---")
        # pdb.set_trace()
        # compute total size when lagrange_lambda is largest finite tile value (v1)
        # if total_size > total_size_constraint, then we know closed-form solution
        # otherwise, we need to divide-and-conquer with O(n)

        # tiles_rate_solution *= 0

        # # r* = Rmax
        # typeI_tiles_set = set()
        # # r* = r0
        # typeII_tiles_set = set()
        # r* = z_weight / lambda - 1 / b_k
        typeIII_tiles_set = set()
        visited_typeI_or_II_tiles_set = set()

        left_idx = 0
        right_idx = len(sorted_tiles_values) - 1  # last value

        lagrange_lambda = tiles_values[sorted_tiles_values[right_idx]['value_idx']] \
                 [sorted_tiles_values[right_idx]['frame_idx']] \
                 [sorted_tiles_values[right_idx]['x']] \
                 [sorted_tiles_values[right_idx]['y']] \
                 [sorted_tiles_values[right_idx]['z']]

        ############# first process right_idx itself: #######
        # it's either type I or type II, cannot be type III.
        tile_value_loc = sorted_tiles_values[right_idx]
        value_idx = tile_value_loc["value_idx"]
        frame_idx = tile_value_loc["frame_idx"]
        x = tile_value_loc["x"]
        y = tile_value_loc["y"]
        z = tile_value_loc["z"]
        tile_loc_tuple = (frame_idx, tile_value_loc["x"], tile_value_loc["y"],
                          tile_value_loc["z"])
        if value_idx == 0:  # type I
            print("should not enter this if branch!!!!!!!!!!!!!!")
            visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
            tiles_rate_solution[frame_idx][x][y][z] = max_rates_cur_step[
                frame_idx][x][y][z]
        else:  # type II
            visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
            tiles_rate_solution[frame_idx][x][y][z] = r0[frame_idx][x][y][z]
        ###########################################################

        for value_loc_idx in range(right_idx - 1, -1, -1):
            tile_value_loc = sorted_tiles_values[value_loc_idx]
            value_idx = tile_value_loc["value_idx"]
            frame_idx = tile_value_loc["frame_idx"]
            x = tile_value_loc["x"]
            y = tile_value_loc["y"]
            z = tile_value_loc["z"]
            tile_loc_tuple = (frame_idx, tile_value_loc["x"],
                              tile_value_loc["y"], tile_value_loc["z"])
            # v2's index is 0, v1's index is 1
            if value_idx == 1:  # type II
                visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
                tiles_rate_solution[frame_idx][x][y][z] = r0[frame_idx][x][y][
                    z]
            else:
                if tile_loc_tuple not in visited_typeI_or_II_tiles_set:  # type III
                    typeIII_tiles_set.add(tile_loc_tuple)
                    tiles_rate_solution[frame_idx][x][y][
                        z] = z_weights[frame_idx][x][y][
                            z] / lagrange_lambda - 1 / b_updated_tiles[
                                frame_idx][x][y][z]

            # print(visited_typeI_or_II_tiles_set)
            # print(typeIII_tiles_set)
            # pdb.set_trace()
        total_size = np.sum(tiles_rate_solution)

        # pdb.set_trace()

        if total_size >= total_size_constraint:  # total budget cannot satisfy all tiles with lowest rate version
            # with lod-based qr function, this if branch is impossible
            print(
                "!!!! total budget cannot satisfy all tiles with lowest rate version !!!!!!!"
            )

            # pivot_loc = sorted_tiles_values[-1]
            # pivot_value_idx = pivot_loc["value_idx"]
            # pivot_frame_idx = pivot_loc["frame_idx"]
            # pivot_x = pivot_loc["x"]
            # pivot_y = pivot_loc["y"]
            # pivot_z = pivot_loc["z"]
            # pivot_loc_tuple = (pivot_frame_idx, pivot_x, pivot_y, pivot_z)

            # if pivot_value_idx == 0: # typeI to typeIII
            # 	visited_typeI_or_II_tiles_set.remove(pivot_loc_tuple)
            # 	typeIII_tiles_set.add(pivot_loc_tuple)

            # total_size_of_typeI_and_II_tiles = total_size
            # sum_typeIII_z_weight = 0
            # sum_bk_over_ak = 0
            # for tile_loc_tuple in typeIII_tiles_set:
            # 	frame_idx = tile_loc_tuple[0]
            # 	x = tile_loc_tuple[1]
            # 	y = tile_loc_tuple[2]
            # 	z = tile_loc_tuple[3]
            # 	z_weight = z_weights[frame_idx][x][y][z]
            # 	sum_typeIII_z_weight += z_weight
            # 	sum_bk_over_ak += self.tile_b[x][y][z] / self.tile_a[x][y][z]
            # 	total_size_of_typeI_and_II_tiles -= tiles_rate_solution[frame_idx][x][y][z]

            # byte_size_constraint_of_typeIII_tiles = total_size_constraint - total_size_of_typeI_and_II_tiles
            # lagrange_lambda = sum_typeIII_z_weight / (byte_size_constraint_of_typeIII_tiles + sum_bk_over_ak)

            # print("left, lambda, right: ", \
            # tiles_values[sorted_tiles_values[left_idx]['value_idx']][sorted_tiles_values[left_idx]['frame_idx']][sorted_tiles_values[left_idx]['x']][sorted_tiles_values[left_idx]['y']][sorted_tiles_values[left_idx]['z']], \
            # lagrange_lambda, \
            # tiles_values[sorted_tiles_values[right_idx]['value_idx']][sorted_tiles_values[right_idx]['frame_idx']][sorted_tiles_values[right_idx]['x']][sorted_tiles_values[right_idx]['y']][sorted_tiles_values[right_idx]['z']])

            # # pdb.set_trace()

            # for tile_loc_tuple in typeIII_tiles_set:
            # 	frame_idx = tile_loc_tuple[0]
            # 	x = tile_loc_tuple[1]
            # 	y = tile_loc_tuple[2]
            # 	z = tile_loc_tuple[3]
            # 	z_weight = z_weights[frame_idx][x][y][z]

            # 	tiles_rate_solution[frame_idx][x][y][z] = z_weight / lagrange_lambda - self.tile_b[x][y][z] / self.tile_a[x][y][z]
            # 	if tiles_rate_solution[frame_idx][x][y][z] <= 1:
            # 		print("tiles rate*: %f <= 1" %(tiles_rate_solution[frame_idx][x][y][z]))
            # 	assert (tiles_rate_solution[frame_idx][x][y][z] < self.max_tile_sizes[x][y][z]), "!!!!!!! tile size: %f too large, reaches params.MAX_TILE_SIZE %f (before divide-and-conquer) !!!!!!!" %(tiles_rate_solution[frame_idx][x][y][z], self.max_tile_sizes[x][y][z])
            # 	assert (tiles_rate_solution[frame_idx][x][y][z] > buffered_tiles_sizes[frame_idx][x][y][z]), "!!!!!!! tile size: %f too small, reaches r0: %f (before divide-and-conquer) !!!!!!!" %(tiles_rate_solution[frame_idx][x][y][z], buffered_tiles_sizes[frame_idx][x][y][z])
            # print("lambda is larger than maximal finite tile value!", np.sum(tiles_rate_solution), total_size_constraint)

            # #### quantize ####
            # if params.QUANTIZE_TILE_SIZE:
            # 	tiles_rate_solution = self.quantize_tile_size(tiles_rate_solution.copy())
            # ##################

            #### quantize ####
            if params.ROUND_TILE_SIZE:
                tiles_rate_solution = self.round_tile_size(
                    tiles_rate_solution.copy(), z_weights,
                    total_size_constraint, z_weight_locations,
                    sorted_z_weights)
            ##################

            total_size = np.sum(tiles_rate_solution)
            return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(
                r0), sorted_z_weights

        middle_idx = (left_idx + right_idx) // 2
        # mark previous lambda is at right_idx or left_idx:
        # this impacts how to update typeIII tile set.
        prev_lambda_at_right = True
        while middle_idx != left_idx:
            # calculate total size when lambda=tile_value[middle_idx]
            # if new total size < total budget, right_idx=middle_idx; otherwise, left_idx=middle_idx

            lagrange_lambda = tiles_values[sorted_tiles_values[middle_idx]['value_idx']] \
                     [sorted_tiles_values[middle_idx]['frame_idx']] \
                     [sorted_tiles_values[middle_idx]['x']] \
                     [sorted_tiles_values[middle_idx]['y']] \
                     [sorted_tiles_values[middle_idx]['z']]

            visited_typeI_or_II_tiles_set = set()

            ############# first process middle_idx itself: #######
            # it's either type I or type II, cannot be type III.
            tile_value_loc = sorted_tiles_values[middle_idx]
            value_idx = tile_value_loc["value_idx"]
            frame_idx = tile_value_loc["frame_idx"]
            x = tile_value_loc["x"]
            y = tile_value_loc["y"]
            z = tile_value_loc["z"]
            tile_loc_tuple = (frame_idx, tile_value_loc["x"],
                              tile_value_loc["y"], tile_value_loc["z"])
            if tile_loc_tuple in typeIII_tiles_set:
                typeIII_tiles_set.remove(tile_loc_tuple)
            if value_idx == 0:  # type I
                visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
                tiles_rate_solution[frame_idx][x][y][z] = max_rates_cur_step[
                    frame_idx][x][y][z]
            else:  # type II
                visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
                tiles_rate_solution[frame_idx][x][y][z] = r0[frame_idx][x][y][
                    z]
            ###########################################################

            if prev_lambda_at_right:
                for value_loc_idx in range(middle_idx + 1, right_idx + 1):
                    tile_value_loc = sorted_tiles_values[value_loc_idx]
                    value_idx = tile_value_loc["value_idx"]
                    frame_idx = tile_value_loc["frame_idx"]
                    x = tile_value_loc["x"]
                    y = tile_value_loc["y"]
                    z = tile_value_loc["z"]
                    tile_loc_tuple = (frame_idx, tile_value_loc["x"],
                                      tile_value_loc["y"], tile_value_loc["z"])
                    # v2's index is 0, v1's index is 1
                    if value_idx == 0:  # type I
                        if tile_loc_tuple in typeIII_tiles_set:
                            typeIII_tiles_set.remove(tile_loc_tuple)
                        visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
                        tiles_rate_solution[frame_idx][x][y][
                            z] = max_rates_cur_step[frame_idx][x][y][z]
                    else:
                        if tile_loc_tuple not in visited_typeI_or_II_tiles_set:  # type III
                            typeIII_tiles_set.add(tile_loc_tuple)
                            # tiles_rate_solution[frame_idx][x][y][z] = z_weights[frame_idx][x][y][z] / lagrange_lambda

            else:
                for value_loc_idx in range(middle_idx - 1, left_idx - 1, -1):
                    tile_value_loc = sorted_tiles_values[value_loc_idx]
                    value_idx = tile_value_loc["value_idx"]
                    frame_idx = tile_value_loc["frame_idx"]
                    x = tile_value_loc["x"]
                    y = tile_value_loc["y"]
                    z = tile_value_loc["z"]
                    tile_loc_tuple = (frame_idx, tile_value_loc["x"],
                                      tile_value_loc["y"], tile_value_loc["z"])
                    # v2's index is 0, v1's index is 1
                    if value_idx == 1:  # type II
                        if tile_loc_tuple in typeIII_tiles_set:
                            typeIII_tiles_set.remove(tile_loc_tuple)
                        visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
                        tiles_rate_solution[frame_idx][x][y][z] = r0[
                            frame_idx][x][y][z]
                    else:
                        if tile_loc_tuple not in visited_typeI_or_II_tiles_set:  # type III
                            typeIII_tiles_set.add(tile_loc_tuple)
                            # tiles_rate_solution[frame_idx][x][y][z] = z_weights[frame_idx][x][y][z] / lagrange_lambda

            ######## update tile size for all typeIII tiles, due to the new lambda at middle_idx #######
            for tile_loc_tuple in typeIII_tiles_set:
                frame_idx = tile_loc_tuple[0]
                x = tile_loc_tuple[1]
                y = tile_loc_tuple[2]
                z = tile_loc_tuple[3]
                z_weight = z_weights[frame_idx][x][y][z]

                tiles_rate_solution[frame_idx][x][y][
                    z] = z_weight / lagrange_lambda - 1 / b_updated_tiles[
                        frame_idx][x][y][z]
            ##############################################################################################

            total_size = np.sum(tiles_rate_solution)

            if total_size > total_size_constraint:  # lambda should be bigger
                left_idx = middle_idx
                prev_lambda_at_right = False

            elif total_size == total_size_constraint:
                #### quantize ####
                if params.QUANTIZE_TILE_SIZE:
                    tiles_rate_solution = self.quantize_tile_size(
                        tiles_rate_solution.copy())
                    total_size = np.sum(tiles_rate_solution)
                ##################

                #### quantize ####
                if params.ROUND_TILE_SIZE:
                    tiles_rate_solution = self.round_tile_size(
                        tiles_rate_solution.copy(), z_weights,
                        total_size_constraint, z_weight_locations,
                        sorted_z_weights)
                ##################
                return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(
                    r0), sorted_z_weights
            else:
                right_idx = middle_idx
                prev_lambda_at_right = True

            middle_idx = (left_idx + right_idx) // 2

        # lambda is between (left_idx, right_idx)
        assert (tiles_values[sorted_tiles_values[left_idx]["value_idx"]][sorted_tiles_values[left_idx]["frame_idx"]][sorted_tiles_values[left_idx]["x"]][sorted_tiles_values[left_idx]["y"]][sorted_tiles_values[left_idx]["z"]] != \
          tiles_values[sorted_tiles_values[right_idx]["value_idx"]][sorted_tiles_values[right_idx]["frame_idx"]][sorted_tiles_values[right_idx]["x"]][sorted_tiles_values[right_idx]["y"]][sorted_tiles_values[right_idx]["z"]]), \
          "!!!!!!!!!!!! left tile value = right tile value !!!!!!"

        if prev_lambda_at_right:  # lambda should be between left_idx and previous_middle_idx
            # update previous_middle_idx (right_idx)
            pivot_loc = sorted_tiles_values[right_idx]
            pivot_value_idx = pivot_loc["value_idx"]
            pivot_frame_idx = pivot_loc["frame_idx"]
            pivot_x = pivot_loc["x"]
            pivot_y = pivot_loc["y"]
            pivot_z = pivot_loc["z"]
            pivot_loc_tuple = (pivot_frame_idx, pivot_x, pivot_y, pivot_z)

            if pivot_value_idx == 1:  # typeII to typeIII
                visited_typeI_or_II_tiles_set.remove(pivot_loc_tuple)
                typeIII_tiles_set.add(pivot_loc_tuple)
        else:  # lambda should be between previous_middle_idx and right_idx
            # update previous_middle_idx (right_idx)
            pivot_loc = sorted_tiles_values[left_idx]
            pivot_value_idx = pivot_loc["value_idx"]
            pivot_frame_idx = pivot_loc["frame_idx"]
            pivot_x = pivot_loc["x"]
            pivot_y = pivot_loc["y"]
            pivot_z = pivot_loc["z"]
            pivot_loc_tuple = (pivot_frame_idx, pivot_x, pivot_y, pivot_z)

            if pivot_value_idx == 0:  # typeI to typeIII
                visited_typeI_or_II_tiles_set.remove(pivot_loc_tuple)
                typeIII_tiles_set.add(pivot_loc_tuple)

        total_size_of_typeI_and_II_tiles = total_size
        sum_typeIII_z_weight = 0
        sum_bk_inverse = 0
        for tile_loc_tuple in typeIII_tiles_set:
            frame_idx = tile_loc_tuple[0]
            x = tile_loc_tuple[1]
            y = tile_loc_tuple[2]
            z = tile_loc_tuple[3]
            z_weight = z_weights[frame_idx][x][y][z]
            sum_typeIII_z_weight += z_weight
            sum_bk_inverse += 1 / b_updated_tiles[frame_idx][x][y][z]
            total_size_of_typeI_and_II_tiles -= tiles_rate_solution[frame_idx][
                x][y][z]

        byte_size_constraint_of_typeIII_tiles = total_size_constraint - total_size_of_typeI_and_II_tiles
        lagrange_lambda = sum_typeIII_z_weight / (
            byte_size_constraint_of_typeIII_tiles + sum_bk_inverse)

        # lambda is between (left_idx, right_idx)
        print("left_idx, right_idx: ", left_idx, right_idx)
        print("left, lambda, right: ", \
         tiles_values[sorted_tiles_values[left_idx]['value_idx']][sorted_tiles_values[left_idx]['frame_idx']][sorted_tiles_values[left_idx]['x']][sorted_tiles_values[left_idx]['y']][sorted_tiles_values[left_idx]['z']], \
         lagrange_lambda, \
         tiles_values[sorted_tiles_values[right_idx]['value_idx']][sorted_tiles_values[right_idx]['frame_idx']][sorted_tiles_values[right_idx]['x']][sorted_tiles_values[right_idx]['y']][sorted_tiles_values[right_idx]['z']])

        print("search lambda--- ",
              time.time() - self.start_time, " seconds ---")

        for tile_loc_tuple in typeIII_tiles_set:
            frame_idx = tile_loc_tuple[0]
            x = tile_loc_tuple[1]
            y = tile_loc_tuple[2]
            z = tile_loc_tuple[3]
            z_weight = z_weights[frame_idx][x][y][z]

            tiles_rate_solution[frame_idx][x][y][
                z] = z_weight / lagrange_lambda - 1 / b_updated_tiles[
                    frame_idx][x][y][z]

            if tiles_rate_solution[frame_idx][x][y][z] >= max_rates_cur_step[
                    frame_idx][x][y][z]:
                print("!!!!!!!! tile size: %f,  MAX size: %f" %
                      (tiles_rate_solution[frame_idx][x][y][z],
                       self.max_tile_sizes[x][y][z]))
                tiles_rate_solution[frame_idx][x][y][z] = max_rates_cur_step[
                    frame_idx][x][y][z]
                # pdb.set_trace()
            if tiles_rate_solution[frame_idx][x][y][z] <= r0[frame_idx][x][y][
                    z]:
                print("!!!!!!!! tile size: %f,  r0: %f" %
                      (tiles_rate_solution[frame_idx][x][y][z],
                       r0[frame_idx][x][y][z]))
                tiles_rate_solution[frame_idx][x][y][z] = r0[frame_idx][x][y][
                    z]
                # pdb.set_trace()
            assert (
                tiles_rate_solution[frame_idx][x][y][z] <=
                max_rates_cur_step[frame_idx][x][y][z]
            ), "!!!!!!! tile size: %f too large, reaches params.MAX_TILE_SIZE %f (during divide-and-conquer) !!!!!!!" % (
                tiles_rate_solution[frame_idx][x][y][z],
                max_rates_cur_step[frame_idx][x][y][z])
            assert (
                tiles_rate_solution[frame_idx][x][y][z] >=
                r0[frame_idx][x][y][z]
            ), "!!!!!!! tile size: %f too small, reaches r0: %f (during divide-and-conquer) !!!!!!!" % (
                tiles_rate_solution[frame_idx][x][y][z],
                r0[frame_idx][x][y][z])

        print("calculate typeIII tiles sizes--- ",
              time.time() - self.start_time, " seconds ---")

        #### quantize ####
        if params.QUANTIZE_TILE_SIZE:
            tiles_rate_solution = self.quantize_tile_size(
                tiles_rate_solution.copy())
        ##################

        #### quantize ####
        if params.ROUND_TILE_SIZE:
            tiles_rate_solution = self.round_tile_size(
                tiles_rate_solution.copy(), z_weights, total_size_constraint,
                z_weight_locations, sorted_z_weights)
            ##################

            print("rounding--- ",
                  time.time() - self.start_time, " seconds ---")

        total_size = np.sum(tiles_rate_solution)

        print("calculate summation--- ",
              time.time() - self.start_time, " seconds ---")

        assert (total_size - total_size_constraint <
                1e-4), "!!!! total size != total bw budget %f, %f!!!!!" % (
                    total_size, total_size_constraint)
        return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(
            r0), sorted_z_weights

    def hybrid_tiling(self, z_weights, bandwidth_budget, update_start_idx,
                      update_end_idx):
        ##################### get v1 and v2 for each tile: ################################
        # v1 = z_weight / r0; v2 = z_weight / params.MAX_TILE_SIZE.
        num_frames_to_update = update_end_idx - update_start_idx + 1
        # pdb.set_trace()

        # tiles' byte size that are already in buffer
        buffered_tiles_sizes = self.buffer.copy()
        for i in range(params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE *
                       params.FPS):
            buffered_tiles_sizes.pop(0)
        for i in range(params.UPDATE_FREQ * params.FPS):
            buffered_tiles_sizes.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
        buffered_tiles_sizes = np.array(buffered_tiles_sizes)

        # each tile has a 4 tuple location: (frame_idx, x, y, z)
        tiles_rate_solution = buffered_tiles_sizes.copy()

        # each tile_value has a 5 tuple location: (frame_idx, x, y, z, value_idx)
        tiles_values = np.zeros(
            (num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME, 2))

        # locations of all tiles:
        # for sorting purpose later
        # v1 at value_idx=1; v2 at value_idx=0
        tiles_values_locations = []

        # typeIII_tiles_set = set()

        # sum_typeIII_z_weight = 0

        frames, tile_xs, tile_ys, tile_zs = z_weights.nonzero()
        num_utility_tiles = len(frames)
        for point_idx in range(num_utility_tiles):
            tiles_values_locations.append({
                "frame_idx": frames[point_idx],
                "x": tile_xs[point_idx],
                "y": tile_ys[point_idx],
                "z": tile_zs[point_idx]
            })

        # for frame_idx in range(num_frames_to_update):
        # 	for x in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
        # 		for y in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
        # 			for z in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
        # 				z_weight = z_weights[frame_idx][x][y][z] # can be 0

        # 				# byte size that already in buffer for this tile
        # 				r0 = buffered_tiles_sizes[frame_idx][x][y][z] # can be 0

        # 				if r0 == params.MAX_TILE_SIZE:
        # 					tiles_values[frame_idx][x][y][z][0] = z_weight / params.MAX_TILE_SIZE
        # 					tiles_values[frame_idx][x][y][z][1] = z_weight / params.MAX_TILE_SIZE
        # 					continue

        # 				if z_weight == 0:
        # 					tiles_values[frame_idx][x][y][z][1] = 0

        # 					# if z_weight is 0, optimal size of this tile should be r0 (unchanged)
        # 					tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
        # 				elif r0 == 0:
        # 					tiles_values[frame_idx][x][y][z][1] = float('inf')
        # 					# sum_typeIII_z_weight += z_weight
        # 					# store the tile value location with inf v1
        # 					# typeIII_tiles_set.add((frame_idx, x, y, z))
        # 					tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":0})

        # 				else:
        # 					tiles_values[frame_idx][x][y][z][1] = z_weight / r0

        # 					tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":0})
        # 					tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":1})

        # 				tiles_values[frame_idx][x][y][z][0] = z_weight / params.MAX_TILE_SIZE
        ##########################################################################################################

        # this is total size when lambda is the least positive tile value
        total_size = 0

        # final total_size should be equal to total_size_constraint
        total_size_constraint = bandwidth_budget

        # if total_size <= total_size_constraint:
        # 	print("all tiles are max rate", total_size, total_size_constraint)
        # 	return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes)

        # get sorted locations of all tiles' values (ascending order)
        # O(n*log(n)), where n is # of visible tiles
        sorted_tiles_values = sorted(tiles_values_locations, \
                key=lambda loc: z_weights[loc["frame_idx"]]\
                        [loc["x"]]\
                        [loc["y"]]\
                        [loc["z"]], reverse=True)
        if params.BYTE_SIZES[0] * num_utility_tiles > total_size_constraint:
            total_size = 0
            for z_weight_idx in range(num_utility_tiles):
                tile_value_loc = sorted_tiles_values[z_weight_idx]
                frame_idx = tile_value_loc["frame_idx"]
                x = tile_value_loc["x"]
                y = tile_value_loc["y"]
                z = tile_value_loc["z"]
                tiles_rate_solution[frame_idx][x][y][z] = params.BYTE_SIZES[0]
                total_size += params.BYTE_SIZES[0]
                if total_size > total_size_constraint:
                    tiles_rate_solution[frame_idx][x][y][z] = 0
                    total_size -= params.BYTE_SIZES[0]
                    break
        elif params.BYTE_SIZES[1] * num_utility_tiles > total_size_constraint:
            total_size = params.BYTE_SIZES[0] * num_utility_tiles
            tiles_rate_solution[z_weights.nonzero()] = params.BYTE_SIZES[0]
            for z_weight_idx in range(num_utility_tiles):
                tile_value_loc = sorted_tiles_values[z_weight_idx]
                frame_idx = tile_value_loc["frame_idx"]
                x = tile_value_loc["x"]
                y = tile_value_loc["y"]
                z = tile_value_loc["z"]
                tiles_rate_solution[frame_idx][x][y][z] = params.BYTE_SIZES[1]
                total_size += (params.BYTE_SIZES[1] - params.BYTE_SIZES[0])
                if total_size > total_size_constraint:
                    tiles_rate_solution[frame_idx][x][y][
                        z] = params.BYTE_SIZES[0]
                    total_size -= (params.BYTE_SIZES[1] - params.BYTE_SIZES[0])
                    break
        elif params.BYTE_SIZES[2] * num_utility_tiles > total_size_constraint:
            total_size = params.BYTE_SIZES[1] * num_utility_tiles
            tiles_rate_solution[z_weights.nonzero()] = params.BYTE_SIZES[1]
            for z_weight_idx in range(num_utility_tiles):
                tile_value_loc = sorted_tiles_values[z_weight_idx]
                frame_idx = tile_value_loc["frame_idx"]
                x = tile_value_loc["x"]
                y = tile_value_loc["y"]
                z = tile_value_loc["z"]
                tiles_rate_solution[frame_idx][x][y][z] = params.BYTE_SIZES[2]
                total_size += (params.BYTE_SIZES[2] - params.BYTE_SIZES[1])
                if total_size > total_size_constraint:
                    tiles_rate_solution[frame_idx][x][y][
                        z] = params.BYTE_SIZES[1]
                    total_size -= (params.BYTE_SIZES[2] - params.BYTE_SIZES[1])
                    break
        else:
            total_size = params.BYTE_SIZES[2] * num_utility_tiles
            tiles_rate_solution[z_weights.nonzero()] = params.BYTE_SIZES[2]

        return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(
            buffered_tiles_sizes), sorted_tiles_values

    def quantize_tile_size(self, tiles_rate_solution):
        tiles_rate_solution[np.where(
            tiles_rate_solution < params.BYTE_SIZES[0])] = 0
        tiles_rate_solution[
            np.where((tiles_rate_solution < params.BYTE_SIZES[1])
                     & (tiles_rate_solution >= params.BYTE_SIZES[0])
                     )] = params.BYTE_SIZES[0]
        tiles_rate_solution[
            np.where((tiles_rate_solution < params.BYTE_SIZES[2])
                     & (tiles_rate_solution >= params.BYTE_SIZES[1])
                     )] = params.BYTE_SIZES[1]
        return tiles_rate_solution

    def round_tile_size(self, tiles_rate_solution, z_weights,
                        total_size_constraint, z_weight_locations,
                        sorted_z_weights):
        extra_budget = 0

        lowest_z_weight_idx = len(sorted_z_weights) - 1
        z_weight_idx = 0
        while z_weight_idx <= lowest_z_weight_idx:
            # for z_weight_idx in range(len(sorted_z_weights)):
            z_weight_loc = sorted_z_weights[z_weight_idx]
            frame_idx = z_weight_loc["frame_idx"]
            x = z_weight_loc["x"]
            y = z_weight_loc["y"]
            z = z_weight_loc["z"]
            current_rate = tiles_rate_solution[frame_idx][x][y][z]
            if current_rate == self.rate_versions[0][x][y][
                    z] or current_rate == 0 or current_rate == self.rate_versions[
                        1][x][y][z] or current_rate == self.rate_versions[2][
                            x][y][z]:
                z_weight_idx += 1
                continue
            current_version = self.decide_rate_version(
                current_rate, self.rate_versions[0][x][y][z],
                self.rate_versions[1][x][y][z],
                self.rate_versions[2][x][y][z])  # 0,1,2
            bandwidth_needed = self.rate_versions[params.NUM_RATE_VERSIONS -
                                                  (current_version +
                                                   1)][x][y][z] - current_rate

            # if abs(extra_budget - 85.85051597904624) < 1e-4:
            # 	pdb.set_trace()

            if extra_budget >= bandwidth_needed:
                # round up high-weight tile's rate
                tiles_rate_solution[frame_idx][x][y][z] = self.rate_versions[
                    params.NUM_RATE_VERSIONS - (current_version + 1)][x][y][z]
                extra_budget -= bandwidth_needed
                z_weight_idx += 1
                continue

            # reduce rates of tiles with lowest z_weights
            reduced_rates = 0

            tmp_lowest_z_weight_idx = lowest_z_weight_idx
            while tmp_lowest_z_weight_idx > z_weight_idx:
                low_z_weight_loc = sorted_z_weights[tmp_lowest_z_weight_idx]
                low_frame_idx = low_z_weight_loc["frame_idx"]
                low_x = low_z_weight_loc["x"]
                low_y = low_z_weight_loc["y"]
                low_z = low_z_weight_loc["z"]
                low_weight_tile_rate = tiles_rate_solution[low_frame_idx][
                    low_x][low_y][low_z]
                if low_weight_tile_rate == self.rate_versions[0][low_x][low_y][
                        low_z] or low_weight_tile_rate == 0 or low_weight_tile_rate == self.rate_versions[
                            1][low_x][low_y][
                                low_z] or low_weight_tile_rate == self.rate_versions[
                                    2][low_x][low_y][low_z]:
                    tmp_lowest_z_weight_idx -= 1
                    continue
                low_current_version = self.decide_rate_version(
                    low_weight_tile_rate,
                    self.rate_versions[0][low_x][low_y][low_z],
                    self.rate_versions[1][low_x][low_y][low_z],
                    self.rate_versions[2][low_x][low_y][low_z])  # 0,1,2
                # # quantize to this low_current_version
                # tiles_rate_solution[low_frame_idx][low_x][low_y][low_z] = params.MAP_VERSION_TO_SIZE[low_current_version]
                if low_current_version == 0:
                    reduced_rates += low_weight_tile_rate
                else:
                    reduced_rates += (
                        low_weight_tile_rate - self.rate_versions[
                            params.NUM_RATE_VERSIONS -
                            low_current_version][low_x][low_y][low_z])

                if reduced_rates + extra_budget >= bandwidth_needed:
                    # if abs(extra_budget - 85.85051597904624) < 1e-4:
                    # 	pdb.set_trace()
                    break

                tmp_lowest_z_weight_idx -= 1

            if reduced_rates + extra_budget >= bandwidth_needed:
                extra_budget = reduced_rates + extra_budget - bandwidth_needed

                # round up high-weight tile's rate
                tiles_rate_solution[frame_idx][x][y][z] = self.rate_versions[
                    params.NUM_RATE_VERSIONS - (current_version + 1)][x][y][z]

                # for low tiles, quantize to low_current_version
                new_lowest_z_weight_idx = tmp_lowest_z_weight_idx - 1
                while tmp_lowest_z_weight_idx <= lowest_z_weight_idx:
                    low_z_weight_loc = sorted_z_weights[
                        tmp_lowest_z_weight_idx]
                    low_frame_idx = low_z_weight_loc["frame_idx"]
                    low_x = low_z_weight_loc["x"]
                    low_y = low_z_weight_loc["y"]
                    low_z = low_z_weight_loc["z"]
                    low_weight_tile_rate = tiles_rate_solution[low_frame_idx][
                        low_x][low_y][low_z]
                    if low_weight_tile_rate == self.rate_versions[0][low_x][low_y][
                            low_z] or low_weight_tile_rate == 0 or low_weight_tile_rate == self.rate_versions[
                                1][low_x][low_y][
                                    low_z] or low_weight_tile_rate == self.rate_versions[
                                        2][low_x][low_y][low_z]:
                        tmp_lowest_z_weight_idx += 1
                        continue
                    low_current_version = self.decide_rate_version(
                        low_weight_tile_rate,
                        self.rate_versions[0][low_x][low_y][low_z],
                        self.rate_versions[1][low_x][low_y][low_z],
                        self.rate_versions[2][low_x][low_y][low_z])  # 0,1,2
                    if low_current_version == 0:
                        tiles_rate_solution[low_frame_idx][low_x][low_y][
                            low_z] = 0
                    else:
                        tiles_rate_solution[low_frame_idx][low_x][low_y][
                            low_z] = self.rate_versions[
                                params.NUM_RATE_VERSIONS -
                                low_current_version][low_x][low_y][low_z]

                    tmp_lowest_z_weight_idx += 1

                lowest_z_weight_idx = new_lowest_z_weight_idx

                # pdb.set_trace()
                # print(extra_budget)

            else:  # cannot round up, should round down instead
                if current_version == 0:
                    tiles_rate_solution[frame_idx][x][y][z] = 0
                    extra_budget += current_rate
                else:
                    tiles_rate_solution[frame_idx][x][y][
                        z] = self.rate_versions[params.NUM_RATE_VERSIONS -
                                                current_version][x][y][z]
                    extra_budget += (
                        current_rate -
                        self.rate_versions[params.NUM_RATE_VERSIONS -
                                           current_version][x][y][z])

                # print(extra_budget)

            z_weight_idx += 1

        print("extra_budget:", extra_budget)

        return tiles_rate_solution

    def decide_rate_version(self, rate, versions):
        if rate == versions[0]:
            return 6
        elif rate >= versions[1]:
            return 5
        elif rate >= versions[2]:
            return 4
        elif rate >= versions[3]:
            return 3
        elif rate >= versions[4]:
            return 2
        elif rate >= versions[5]:
            return 1
        else:
            return 0
