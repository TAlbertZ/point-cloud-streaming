import numpy as np
import pickle as pk
import pdb
import math
import time
import logging
import matplotlib.pyplot as plt

import params
from params import Algo
from params import ActionInitType
from hidden_points_removal import HiddenPointsRemoval
if params.SCALABLE_CODING:
    from dynamics import AutoDiffDynamics
    from cost import AutoDiffCost
else:
    from dynamics_nonscalable_coding import AutoDiffDynamics
    from cost_nonscalable_coding import AutoDiffCost

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
                 N,
                 max_reg=1e10,
                 hessians=False):

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

        self.fov_traces_obj = fov_traces_obj
        self.valid_tiles_obj = valid_tiles_obj
        self.bw_traces_obj = bw_traces_obj
        self.qr_weights_obj = qr_weights_obj

        # a, b and distance_weight are waiting to be fit
        self.tile_a = self.qr_weights_obj.qr_weights[
            "a"]  # for each tile: [300x16x16x16]
        self.tile_b = self.qr_weights_obj.qr_weights[
            "b"]  # for each tile: [300x16x16x16]
        self.min_rates = self.qr_weights_obj.min_rates  # for each tile: [300x16x16x16]
        # tmp = np.expand_dims(self.tile_a.copy(), axis=0)
        # self.tile_a_frames = np.repeat(tmp.copy(), self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS, axis=0)
        # tmp = np.expand_dims(self.tile_b.copy(), axis=0)
        # self.tile_b_frames = np.repeat(tmp.copy(), self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS, axis=0)
        # tmp = np.expand_dims(self.min_rates.copy(), axis=0)
        # self.min_rates_frames = np.repeat(tmp.copy(), self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS, axis=0)

        self.rate_versions = self.qr_weights_obj.rate_versions

        # sigmoid coefficient c: (1 + exp(-c*d))^(-1)
        self.distance_weight = 1

        # linearly increasing from 0 to 1
        self.frame_weights = None

        # initialize according to fov dataset H1, assume the initial viewpoint is always like this:
        # {x, y, z, roll, yaw, pitch} = {0.05, 1.7868, -1.0947, 6.9163, 350.8206, 359.9912}
        # z-x plane is floor
        self.history_viewpoints = {
            "x": [0.05] * params.FOV_PREDICTION_HISTORY_WIN_LENGTH,
            "y": [1.7868] * params.FOV_PREDICTION_HISTORY_WIN_LENGTH,
            "z": [-1.0947] * params.FOV_PREDICTION_HISTORY_WIN_LENGTH,
            "pitch": [6.9163 + 360] *
            params.FOV_PREDICTION_HISTORY_WIN_LENGTH,  # rotate around x axis
            "yaw": [350.8206] *
            params.FOV_PREDICTION_HISTORY_WIN_LENGTH,  # rotate around y axis
            "roll": [359.9912] * params.FOV_PREDICTION_HISTORY_WIN_LENGTH
        }  # rotate around z axis

        self.history_viewpoints = {
            "x": [0.05],
            "y": [1.7868],
            "z": [-1.0947],
            "pitch": [6.9163 + 360],  # rotate around x axis
            "yaw": [350.8206],  # rotate around y axis
            "roll": [359.9912]
        }  # rotate around z axis

        # initialize bandwidth history according to '../bw_traces/100ms_loss1'
        self.history_bandwidths = [2.7
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
        self.frame_quality_var = []
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

        if params.ALGO == Algo.ILQR:
            overlap_ratio_hist_len = self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS + (
                params.ILQR_HORIZON - 1) * params.FPS
        else:
            overlap_ratio_hist_len = self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS
        for frame_idx in range(overlap_ratio_hist_len):
            self.overlap_ratio_history.append([])
            for _ in range(params.OVERLAP_RATIO_HISTORY_WIN_LENGTH):
                self.overlap_ratio_history[frame_idx].append(1.0)

        self.bw_predict_accuracy_trace = []
        self.success_download_rate_trace = []
        self.frame_size_list = []
        self.num_valid_tiles_per_frame = []

        self.num_max_tiles_per_frame = []
        self.mean_size_over_tiles_per_frame = []

        self.tile_sizes_sol = []

        self.num_intersect_visible_tiles_trace = []

        self.mean_size_over_tiles_per_fov = []
        self.effective_rate = []

        self.start_time = 0

        ### sun's ilqr #########################
        # For new traces
        self.w1 = 1.5
        self.w2 = 1
        self.w3 = 1.0  # Freeze
        self.w4 = 0.05  # Latency
        self.w5 = 15  # Speed unnormal, due to **2
        self.w6 = 15  # Speed change, due to **2
        self.barrier_1 = 1
        self.barrier_2 = 1
        self.barrier_3 = 1
        self.barrier_4 = 1

        self.delta = 0.2  # 0.2s
        self.n_step = params.TARGET_LATENCY // params.FPS
        self.predicted_bw = None
        # self.predicted_rtt = predicted_rtt
        self.predicted_rtt = None
        # self.n_iteration = 10
        self.Bu = None
        self.b0 = None
        self.l0 = None
        self.pu0 = None
        self.ps0 = None
        # self.med_latency = 6

        self.kt_step = 1.
        self.KT_step = 1.
        self.step_size = 0.1
        self.decay = 0.99
        self.bw_ratio = 1.0

        self.eta1 = 0.9
        self.eta2 = 1.1
        ########################

        #### from my ilqr #####
        self.N = N
        self._use_hessians = hessians
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._k = None
        self._K = None

        ## added by me ###
        self.dynamics = AutoDiffDynamics()
        self.cost = AutoDiffCost()

        ##################

        #########################

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
                                       frame_idx_within_video):
        if self.current_viewing_frame_idx + 1 < params.TARGET_LATENCY:
            return 0, 0
        tiles_byte_sizes = self.buffer[0]  # front of buffer: cubic array

        ### quantize / quantization #######
        # tiles_byte_sizes[np.where(tiles_byte_sizes < params.BYTE_SIZES[0])] = 0
        # tiles_byte_sizes[np.where((tiles_byte_sizes < params.BYTE_SIZES[1]) & (tiles_byte_sizes >= params.BYTE_SIZES[0]))] = params.BYTE_SIZES[0]
        # tiles_byte_sizes[np.where((tiles_byte_sizes < params.BYTE_SIZES[2]) & (tiles_byte_sizes >= params.BYTE_SIZES[1]))] = params.BYTE_SIZES[1]
        ####################################

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

        # total_span_degree = 0
        # if len(viewing_probability.nonzero()[0]):
        #     total_span_degree = np.sum(
        #         1 / distances[viewing_probability.nonzero()]
        #     ) * params.TILE_SIDE_LEN * 180 / np.pi

        # overlap tiles between viewing_probability and tiles_byte_sizes
        # are those contributing toward quality
        mask_downloaded_tiles = np.zeros_like(tiles_byte_sizes)
        mask_downloaded_tiles[tiles_byte_sizes.nonzero()] = 1
        mask_visible_tiles = mask_downloaded_tiles * viewing_probability
        # mask_visible_tiles = viewing_probability.copy()
        visible_tiles_pos = mask_visible_tiles.nonzero()
        total_visible_size = np.sum(tiles_byte_sizes[visible_tiles_pos])
        distance_of_interest = distances[visible_tiles_pos]

        theta_of_interest = params.TILE_SIDE_LEN / distance_of_interest * params.RADIAN_TO_DEGREE

        a_tiles_of_interest = self.tile_a[frame_idx_within_video][
            visible_tiles_pos]
        b_tiles_of_interest = self.tile_b[frame_idx_within_video][
            visible_tiles_pos]
        tiles_rates_to_be_watched = tiles_byte_sizes[visible_tiles_pos]

        # num_points = a_tiles_of_interest * tiles_rates_to_be_watched + b_tiles_of_interest
        # num_points_per_degree = num_points**0.5 / theta_of_interest  # f_ang
        lod = a_tiles_of_interest * np.log(b_tiles_of_interest *
                                           tiles_rates_to_be_watched + 1)
        num_points_per_degree = 2**lod / theta_of_interest
        quality_tiles = self.cost.get_quality_for_tiles(
            num_points_per_degree)  # update quality_tiles

        quality_tiles_in_mat = np.zeros_like(tiles_byte_sizes)
        quality_tiles_in_mat[visible_tiles_pos] = quality_tiles


        if params.SHOW_ANGULAR_RESOL:
            if params.WEIGHTED_AVERAGE_ANG_RESOL:
                frame_quality = np.sum(num_points_per_degree * theta_of_interest) / np.sum(params.TILE_SIDE_LEN / distances[viewing_probability.nonzero()] * params.RADIAN_TO_DEGREE)
                # frame_quality = np.sum(num_points_per_degree * theta_of_interest) / np.sum(theta_of_interest)
                # frame_quality = np.sum(num_points_per_degree * theta_of_interest)
            else:
                frame_quality = np.sum(num_points_per_degree)
        else:
            frame_quality = np.sum(quality_tiles)

        # if frame_idx_within_video > 200:
        #     pdb.set_trace()

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

        # # return 0 if np.sum(z_weights) == 0 else frame_quality / np.sum(z_weights)
        # if total_span_degree == 0:
        #     frame_quality_per_degree = 0
        # else:
        #     frame_quality_per_degree = frame_quality / total_span_degree
        # pdb.set_trace()

        return frame_quality, frame_quality_var

    def emit_buffer(self):
        '''
			emit params.UPDATE_FREQ*params.FPS frames from front of buffer;;
			Based on their true viewpoint, calculate their HPR, distance, and quality;
			update pointers: buffer, current_viewing_frame_idx, history_viewpoints, history_bandwidths, current_bandwidth_idx
		'''

        previous_visible_tiles_set = set()

        for frame_idx in range(
                self.current_viewing_frame_idx + 1,
                self.current_viewing_frame_idx +
                params.FPS * params.UPDATE_FREQ + 1):
            frame_idx_within_video = (
                frame_idx - params.TARGET_LATENCY) % params.NUM_FRAMES
            current_visible_tiles_set = set()
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

            viewing_probability, distances = self.calculate_probability_to_be_viewed(
                viewpoint, frame_idx, frame_idx, emitting_buffer=True)
            # z_weights = self.calculate_z(viewing_probability, distances, frame_idx, frame_idx, evaluation_flag=True)

            for tile_idx in range(len(viewing_probability[0].nonzero()[0])):
                x = viewing_probability[0].nonzero()[0][tile_idx]
                y = viewing_probability[0].nonzero()[1][tile_idx]
                z = viewing_probability[0].nonzero()[2][tile_idx]
                current_visible_tiles_set.add((x, y, z))
            if frame_idx >= 1:
                intersect = current_visible_tiles_set.intersection(
                    previous_visible_tiles_set)
                self.num_intersect_visible_tiles_trace.append(len(intersect))
            previous_visible_tiles_set = current_visible_tiles_set.copy()

            if params.SHOW_ANGULAR_RESOL and params.WEIGHTED_AVERAGE_ANG_RESOL:
                # calculate total span of fov (how many degrees)
                if len(viewing_probability[0].nonzero()[0]):
                    self.num_valid_tiles_per_frame.append(
                        np.sum(1 / distances[0][viewing_probability[0].nonzero()])
                        * params.TILE_SIDE_LEN * 180 / np.pi)
                else:
                    self.num_valid_tiles_per_frame.append(0)
            else:
                self.num_valid_tiles_per_frame.append(len(viewing_probability[0].nonzero()[0]))

            # if self.update_step > params.BUFFER_LENGTH // (params.UPDATE_FREQ * params.FPS) and frame_idx >= 1:
            # 	print(self.num_valid_tiles_per_frame[-1], self.num_intersect_visible_tiles_trace[-1], self.num_intersect_visible_tiles_trace[-1] / self.num_valid_tiles_per_frame[-1] * 100)
            # pdb.set_trace()

            true_frame_quality, true_frame_quality_var = self.true_frame_quality_sum_and_var(
                viewing_probability[0], distances[0], frame_idx_within_video)
            self.frame_quality.append(true_frame_quality)
            self.frame_quality_var.append(true_frame_quality_var)

            # if true_quality < 7.4 and self.update_step > 10:
            # 	print(frame_idx)
            # 	pdb.set_trace()

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

        if params.ALGO == Algo.ILQR:
            predict_end_idx = update_end_idx + (params.ILQR_HORIZON -
                                                1) * params.FPS
        else:
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

        viewing_probability, distances = self.calculate_probability_to_be_viewed(
            predicted_viewpoints, update_start_idx, update_end_idx)

        logging.info("viewing_probability--- %f seconds ---",
                     time.time() - self.start_time)

        # calculate distance only for viewable valid tiles
        # distances = self.calculate_distance(predicted_viewpoints)
        if params.ALGO != Algo.ILQR:
            z_weights = self.calculate_z(viewing_probability, distances,
                                         update_start_idx, update_end_idx)

            logging.info("calculate_z--- %f seconds ---",
                         time.time() - self.start_time)

        # predict bandwidth of future 1s
        predicted_bandwidth_budget = self.predict_bandwidth(
        ) * params.SCALE_BW  # Mbps

        logging.info("predict_bandwidth--- %f seconds ---",
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
        elif params.ALGO == Algo.ILQR:
            tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0 = self.iterate_LQR(
                distances,
                viewing_probability,
                predicted_bandwidth_budget * params.Mbps_TO_Bps,
                update_start_idx,
                update_end_idx,
                n_iterations=params.NUM_LQR_ITERATIONS)
        else:
            pass

        if params.ALGO == Algo.KKT:
            logging.info("kkt--- %f seconds ---",
                         time.time() - self.start_time)
        elif params.ALGO == Algo.ILQR:
            logging.info("ilqr--- %f seconds ---",
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

        if params.USING_RUMA and params.RUMA_SCALABLE_CODING == False:
            locs = np.where(tiles_rate_solution > buffered_tiles_sizes)
            # pdb.set_trace()
            consumed_bandwidth = np.sum(tiles_rate_solution[locs])
        if consumed_bandwidth != 0:
            success_download_rate = min(
                1, true_bandwidth_budget * params.Mbps_TO_Bps /
                consumed_bandwidth)
        else:
            logging.info("!!!!!!!!! nothing download !!!")
        # pdb.set_trace()
        if params.ALGO != Algo.ILQR:
            if success_download_rate < 1 - 1e-4:  # 1e-4 is noise error term
                print("success download rate:", success_download_rate)
                # tiles_rate_solution = (tiles_rate_solution - buffered_tiles_sizes) * success_download_rate + buffered_tiles_sizes
                download_end_bool = False
                new_consumed_bandwidth = 0

                # higher z_weight has higher priority to be download / fetched
                for z_weight_idx in range(len(sorted_z_weights)):
                    z_weight_loc = sorted_z_weights[z_weight_idx]
                    frame_idx = z_weight_loc["frame_idx"]
                    x = z_weight_loc["x"]
                    y = z_weight_loc["y"]
                    z = z_weight_loc["z"]

                    if download_end_bool:  # already consumed as much bandwidth as possible
                        tiles_rate_solution[frame_idx][x][y][
                            z] = buffered_tiles_sizes[frame_idx][x][y][z]

                    download_size = tiles_rate_solution[frame_idx][x][y][
                        z] - buffered_tiles_sizes[frame_idx][x][y][z]
                    if params.USING_RUMA and params.RUMA_SCALABLE_CODING == False:
                        download_size = tiles_rate_solution[frame_idx][x][y][
                            z] if download_size != 0 else 0
                    new_consumed_bandwidth += download_size
                    if new_consumed_bandwidth > true_bandwidth_budget * params.Mbps_TO_Bps:  # cannot download more
                        new_consumed_bandwidth -= download_size
                        tiles_rate_solution[frame_idx][x][y][
                            z] = buffered_tiles_sizes[frame_idx][x][y][z]
                        download_end_bool = True
                        # break

                success_download_rate = new_consumed_bandwidth / consumed_bandwidth
                # if params.USING_RUMA and params.RUMA_SCALABLE_CODING == False:
                # 	locs = np.where(tiles_rate_solution > buffered_tiles_sizes)
                # 	success_download_rate = new_consumed_bandwidth / consumed_bandwidth
                # 	pdb.set_trace()
                sum_solution_rate = np.sum(tiles_rate_solution)

        logging.info("start updating buffer--- %f seconds ---",
                     time.time() - self.start_time)

        for frame_idx in range(update_start_idx, update_end_idx + 1):
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

        logging.info("update buffer ends--- %f seconds ---",
                     time.time() - self.start_time)

    def calculate_z(self,
                    viewing_probability,
                    distances,
                    update_start_idx,
                    update_end_idx,
                    evaluation_flag=False):
        '''
            also need self.frame_weights, self.tile_a and self.distance_weight
        '''
        z_weights = []

        a_updated_tiles = np.concatenate(
            (self.tile_a[(update_start_idx - params.TARGET_LATENCY) %
                         params.NUM_FRAMES:],
             self.tile_a[:(update_start_idx - params.TARGET_LATENCY) %
                         params.NUM_FRAMES]),
            axis=0)

        # if evaluation_flag:
        #     z_weights.append(
        #         np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
        #                   params.NUM_TILES_PER_SIDE_IN_A_FRAME,
        #                   params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
        #     valid_locs = viewing_probability[frame_idx -
        #                                      update_start_idx].nonzero()
        #     # z_weights[frame_idx - update_start_idx][valid_locs] = viewing_probability[frame_idx - update_start_idx][valid_locs] \
        #     #               * params.TILE_SIDE_LEN / 2 / distances[frame_idx - update_start_idx][valid_locs] * 180 / np.pi
        #     z_weights[frame_idx - update_start_idx][valid_locs] = viewing_probability[frame_idx - update_start_idx][valid_locs] \
        #                   * params.TILE_SIDE_LEN / 2 / distances[frame_idx - update_start_idx][valid_locs] * 180 / np.pi
        #     return z_weights

        # frame weight is linear wrt. frame_idx: w_j = a * frame_idx + b
        frame_weight_decrease_speed = 0
        if not params.FOV_ORACLE_KNOW:
        	frame_weight_decrease_speed = -(params.MAX_FRAME_WEIGHT - params.MIN_FRAME_WEIGHT) / (update_end_idx - update_start_idx)
        frame_weight = 1

        for frame_idx in range(update_start_idx, update_end_idx + 1):
            if params.FRAME_WEIGHT_TYPE == params.FrameWeightType.LINEAR_DECREASE:
                # maximal frame_weight = 1, minimal frame_weight is 0.1
                frame_weight = frame_weight_decrease_speed * (
                    frame_idx - update_start_idx) + params.MAX_FRAME_WEIGHT
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.CONST:
                frame_weight = 1
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.FOV_PRED_ACCURACY:  # based on fov prediction accuracy: overlap ratio
                frame_weight = np.mean(
                    self.overlap_ratio_history[frame_idx - update_start_idx])
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.STEP_FUNC:
                frame_weight = 0
                if frame_idx - update_start_idx < params.FRAME_WEIGHT_STEP_IDX:
                    frame_weight = 1
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.EXP_DECREASE:
                frame_weight = np.power(params.FRAME_WEIGHT_EXP_BOTTOM, params.FRAME_WEIGHT_EXP_FACTOR*(1-(frame_idx-update_start_idx)/(update_end_idx-update_start_idx)))
            else:
                frame_weight = 1

            if params.MMSYS_HYBRID_TILING:
                if update_end_idx - frame_idx < params.UPDATE_FREQ * params.FPS:
                    frame_weight = 1
                else:
                    frame_weight = 0

            # # weight = 1 only for first 1-s content to be updated: should be more variational
            # if frame_idx - update_start_idx < params.UPDATE_FREQ * params.FPS:
            # 	frame_weight = 1
            # else:
            # 	frame_weight = 0

            z_weights.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            valid_locs = viewing_probability[frame_idx -
                                             update_start_idx].nonzero()
            z_weights[frame_idx - update_start_idx][valid_locs] = frame_weight * viewing_probability[frame_idx - update_start_idx][valid_locs] \
                          * a_updated_tiles[frame_idx - update_start_idx][valid_locs] * np.log(2)

            # for x in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
            # 	for y in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
            # 		for z in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
            # 			if viewing_probability[frame_idx - update_start_idx][x][y][z] == 0:
            # 				continue

            # 			assert (self.tile_a[x][y][z] > 0), "!!!!! qr weight a non-positive when calculating z: (%d, %d, %d, %f) !!!!!!" %(x, y, z, self.tile_a[x][y][z])
            # 			# quality = (tile_a * np.log(rate) + tile_b) / (1 + np.exp(-dist_c * distance))
            # 			# z_weights[frame_idx - update_start_idx][x][y][z] = frame_weight * viewing_probability[frame_idx - update_start_idx][x][y][z] * self.tile_a[x][y][z] / (1 + np.exp(-self.distance_weight * distances[frame_idx - update_start_idx][x][y][z]))
            # 			# quality = tile_a * np.log(20 / distance) * np.log(rate) + tile_b * np.log(distance) * 5
            # 			distance  = distances[frame_idx - update_start_idx][x][y][z]
            # 			z_weights[frame_idx - update_start_idx][x][y][z] = frame_weight * viewing_probability[frame_idx - update_start_idx][x][y][z] * self.tile_a[x][y][z] * np.log(20 / distance)
            # 			assert (z_weights[frame_idx - update_start_idx][x][y][z] >= 0),"!!!!!!!!!!!! Negative weights !!!!!!"

            # if frame_idx >= 383:
            # 	pdb.set_trace()
        return z_weights

    def calculate_probability_to_be_viewed(self,
                                           viewpoints,
                                           update_start_idx,
                                           update_end_idx,
                                           emitting_buffer=False):
        # probability can be represented by overlap ratio

        # HPR
        viewing_probability = []
        distances = []

        if params.ALGO == Algo.ILQR and emitting_buffer == False:
            update_end_idx += (params.ILQR_HORIZON - 1) * params.FPS

        for frame_idx in range(update_start_idx, update_end_idx + 1):
            viewing_probability.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            distances.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
            tile_center_points = []
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

            tile_xs, tile_ys, tile_zs = valid_tiles.nonzero()
            for point_idx in range(len(tile_xs)):
                tile_center_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate(
                    tile_xs[point_idx], tile_ys[point_idx], tile_zs[point_idx])
                tile_center_points.append(tile_center_coordinate)

            # modify object coordinate: origin at obj's bottom center
            tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(
                self.origin, tile_center_points)

            # mirror x and z axis to let obj face the user start view orientation
            tile_center_points = np.array(tile_center_points)
            tile_center_points[:, 0] = -tile_center_points[:, 0]
            tile_center_points[:, 2] = -tile_center_points[:, 2]

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

            # modify object coordinate: origin at viewpoint
            # tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(viewpoint_position, tile_center_points)

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
            flippedPoints = HPR_obj.sphericalFlip(
                viewpoint_position, math.pi
            )  # Reflect the point cloud about a sphere centered at viewpoint_position
            myHull = HPR_obj.convexHull(
                flippedPoints
            )  # Take the convex hull of the center of the sphere and the deformed point cloud

            true_flippedPoints = HPR_obj.sphericalFlip(
                true_position, math.pi
            )  # Reflect the point cloud about a sphere centered at viewpoint_position
            true_myHull = HPR_obj.convexHull(
                true_flippedPoints
            )  # Take the convex hull of the center of the sphere and the deformed point cloud

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
                pitch = viewpoint["pitch"] * np.pi / 180
                yaw = viewpoint["yaw"] * np.pi / 180
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
                    viewing_probability[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = 1
                    distances[frame_idx - update_start_idx][
                        viewable_tile_idx[0]][viewable_tile_idx[1]][
                            viewable_tile_idx[2]] = self.calculate_distance(
                                vertex_coordinate, viewpoint_position)

                    predicted_visible_tiles_set.add(viewable_tile_idx)

            true_visible_tiles_set = set()
            for vertex in true_myHull.vertices[:-1]:
                vertex_coordinate = np.array([
                    tile_center_points[vertex, 0],
                    tile_center_points[vertex, 1], tile_center_points[vertex,
                                                                      2]
                ])
                vector_from_viewpoint_to_tilecenter = vertex_coordinate - true_position
                pitch = true_viewpoint[
                    params.MAP_6DOF_TO_HMD_DATA["pitch"]] * np.pi / 180
                yaw = true_viewpoint[
                    params.MAP_6DOF_TO_HMD_DATA["yaw"]] * np.pi / 180
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
                    # viewing_probability[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = 1
                    # distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)

                    true_visible_tiles_set.add(viewable_tile_idx)

            ########################################################################

            # update overlap_ratio history
            overlap_tiles_set = true_visible_tiles_set.intersection(
                predicted_visible_tiles_set)
            overlap_ratio = len(overlap_tiles_set) / len(
                true_visible_tiles_set)
            self.overlap_ratio_history[frame_idx -
                                       update_start_idx].append(overlap_ratio)
            self.overlap_ratio_history[frame_idx - update_start_idx].pop(0)
            # if overlap_ratio < 1:
            # 	pdb.set_trace()

        return viewing_probability, distances

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

    def calculate_unweighted_tile_quality(self, rate, distance, a, b):
        if rate == 0:
            return 0
        tmp = a * rate + b
        assert (
            tmp > 0
        ), "!!!! RUMA->calculate_unweighted_tile_quality->rate is too small !!!!!!!!!"
        return np.log(distance / params.TILE_SIDE_LEN * np.power(tmp, 0.5) /
                      180 * np.pi)

    def _control(self,
                 xs,
                 us,
                 k,
                 K,
                 bandwidth_budget,
                 viewing_probability,
                 distances,
                 update_start_idx,
                 update_end_idx,
                 alpha=1.0):
        """Applies the controls for a given trajectory.

		Args:
			xs: Nominal state path [N+1, state_size].
			us: Nominal control path [N, action_size].
			k: Feedforward gains [N, action_size].
			K: Feedback gains [N, action_size, state_size].
			alpha: Line search coefficient.

		Returns:
			Tuple of
				xs: state path [N+1, state_size].
				us: control path [N, action_size].
		"""
        # xs_new = np.zeros_like(xs)
        # us_new = np.zeros_like(us)
        xs_new = [0.0] * len(xs)
        us_new = [0.0] * len(us)
        xs_new[0] = xs[0].copy()
        J = [0.0] * self.N

        for i in range(self.N):
            # Eq (12).
            us_new[i] = us[i] + alpha * k[i] + K[i].dot(xs_new[i] - xs[i])

            # Eq (8c).
            # us_new[i] = np.clip(us_new[i], -ctrl_limit, ctrl_limit)
            xs_new[i + 1] = self.dynamics.f(xs_new[i], us_new[i], i,
                                            viewing_probability)
            J[i] = self.cost.l(xs_new[i],
                               us_new[i],
                               i,
                               bandwidth_budget,
                               viewing_probability,
                               distances,
                               self.tile_a,
                               self.tile_b,
                               update_start_idx,
                               update_end_idx,
                               self.dynamics,
                               self.buffer,
                               self.rate_versions,
                               terminal=False)
            # pdb.set_trace()

        return xs_new, us_new, sum(J)

    def _trajectory_cost(self, xs, us, bandwidth_budget, viewing_probability,
                         distances, update_start_idx, update_end_idx):
        """Computes the given trajectory's cost.

		Args:
			xs: State path [N+1, state_size].
			us: Control path [N, action_size].

		Returns:
			Trajectory's total cost.
		"""
        # J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
        #                                              range(self.N)))

        J = [0.0] * self.N
        for i in range(self.N):
            J[i] = self.cost.l(xs[i],
                               us[i],
                               i,
                               bandwidth_budget,
                               viewing_probability,
                               distances,
                               self.tile_a,
                               self.tile_b,
                               update_start_idx,
                               update_end_idx,
                               self.dynamics,
                               self.buffer,
                               self.rate_versions,
                               terminal=False)
        return sum(J)
        # do not have cost-to-go value for point cloud streaming
        # return sum(J) + self.cost.l(
        #     xs[-1], None, self.N, terminal=True, get_Q=get_Q)

    def _forward_rollout(self, x0, us, state_size, action_size,
                         bandwidth_budget, viewing_probability, distances,
                         update_start_idx, update_end_idx):
        """Apply the forward dynamics to have a trajectory from the starting
		state x0 by applying the control path us.

		Args:
			x0: Initial state [state_size].
			us: Control path [N, action_size].
			bandwidth_budget: [N,]

		Returns:
			Tuple of:
				xs: State path [N+1, state_size].
				F_x: Jacobian of state path w.r.t. x
					[N, state_size, state_size].
				F_u: Jacobian of state path w.r.t. u
					[N, state_size, action_size].
				L: Cost path [N+1].
				L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
				L_u: Jacobian of cost path w.r.t. u [N, action_size].
				L_xx: Hessian of cost path w.r.t. x, x
					[N+1, state_size, state_size].
				L_ux: Hessian of cost path w.r.t. u, x
					[N, action_size, state_size].
				L_uu: Hessian of cost path w.r.t. u, u
					[N, action_size, action_size].
				F_xx: Hessian of state path w.r.t. x, x if Hessians are used
					[N, state_size, state_size, state_size].
				F_ux: Hessian of state path w.r.t. u, x if Hessians are used
					[N, state_size, action_size, state_size].
				F_uu: Hessian of state path w.r.t. u, u if Hessians are used
					[N, state_size, action_size, action_size].
		"""
        # state_size = self.dynamics.state_size
        # action_size = self.dynamics.action_size
        N = params.ILQR_HORIZON

        # xs = np.empty((N + 1, state_size))
        # F_x = np.empty((N, state_size, state_size))
        # F_u = np.empty((N, state_size, action_size))

        # L = np.empty(N + 1)
        # L_x = np.empty((N + 1, state_size))
        # L_u = np.empty((N, action_size))
        # L_xx = np.empty((N + 1, state_size, state_size))
        # L_ux = np.empty((N, action_size, state_size))
        # L_uu = np.empty((N, action_size, action_size))

        xs = [0.0] * (N + 1)
        F_x = [0.0] * N
        F_u = [0.0] * N

        L = [0.0] * (N + 1)
        L_x = [0.0] * (N + 1)
        L_u = [0.0] * N
        L_xx = [0.0] * (N + 1)
        L_ux = [0.0] * N
        L_uu = [0.0] * N

        xs[0] = x0
        for i in range(N):
            logging.info('%dth time step of iLQR forward rollout', i)
            x = xs[i]
            # pdb.set_trace()

            # state/action sizes are different in every iteration
            # if us[i] is 0.0 (float), it needs initialization
            if isinstance(us[i], float):
                # tiles of interest are different in every iteration
                tile_of_interest_pos = viewing_probability[
                    i * params.FPS:i * params.FPS +
                    params.TARGET_LATENCY].nonzero()

                # new tiles are from last params.FPS frames in this forward round
                new_tiles_into_buf_pos = viewing_probability[
                    (i - 1) * params.FPS +
                    params.TARGET_LATENCY:i * params.FPS +
                    params.TARGET_LATENCY]

                # TODO by Tongyu: correct rate_version_start_idx
                rate_version_start_idx = (
                    update_start_idx + i * params.FPS) % params.BUFFER_LENGTH

                update_time_step = update_start_idx // params.FPS

                # # buffer is not fully occupied yet
                # if update_time_step + i < params.TARGET_LATENCY // params.FPS:
                #     max_rates_cur_step = np.concatenate(
                #         (np.zeros(
                #             (params.TARGET_LATENCY - rate_version_start_idx,
                #              params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                #              params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                #              params.NUM_TILES_PER_SIDE_IN_A_FRAME)),
                #          self.max_tile_sizes[:rate_version_start_idx]),
                #         axis=0)
                #     min_rates_cur_step = np.concatenate(
                #         (np.zeros(
                #             (params.TARGET_LATENCY - rate_version_start_idx,
                #              params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                #              params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                #              params.NUM_TILES_PER_SIDE_IN_A_FRAME)),
                #          self.min_rates[:rate_version_start_idx]),
                #         axis=0)

                # else:
                #     # looks wierd here, that's because the video is played repeatedly, so
                #     # every 1s the 'self.max_tile_sizes' and 'self.min_rates' should be circularly shifted!
                #     max_rates_cur_step = np.concatenate(
                #         (self.max_tile_sizes[
                #             (update_start_idx + i * params.FPS -
                #              params.TARGET_LATENCY) % params.NUM_FRAMES:],
                #          self.max_tile_sizes[:(update_start_idx +
                #                                i * params.FPS -
                #                                params.TARGET_LATENCY) %
                #                              params.NUM_FRAMES]),
                #         axis=0)

                #     min_rates_cur_step = np.concatenate(
                #         (self.min_rates[(update_start_idx + i * params.FPS -
                #                          params.TARGET_LATENCY) %
                #                         params.NUM_FRAMES:],
                #          self.min_rates[:(update_start_idx + i * params.FPS -
                #                           params.TARGET_LATENCY) %
                #                         params.NUM_FRAMES]),
                #         axis=0)

                ######## initialize action #########
                # TODO by Tongyu: add a function for action initialization

                # # 1e-3 is to avoid reaching upperbound
                # u_upper_bound = max_rates_cur_step[
                #     tile_of_interest_pos] - 1e-3 - x

                # num_new_tiles_into_buffer = len(new_tiles_into_buf_pos[0])
                # num_old_tiles_in_buf = len(x) - num_new_tiles_into_buffer

                # u_lower_bound = np.zeros_like(x)
                # diff_old_states_with_minrate = min_rates_cur_step[
                #     tile_of_interest_pos][:
                #                           num_old_tiles_in_buf] - x[:
                #                                                     num_old_tiles_in_buf]
                # diff_old_states_with_minrate[
                #     diff_old_states_with_minrate < 0] = 0

                # # action lower bound for old tiles in buffer
                # u_lower_bound[:num_old_tiles_in_buf] += (
                #     diff_old_states_with_minrate + 1e-3)
                # if params.SCALABLE_CODING == False:
                #     u_lower_bound[:num_old_tiles_in_buf] = min_rates_cur_step[
                #     tile_of_interest_pos][:num_old_tiles_in_buf] + 1e-3

                # # action lower bound for new tiles coming into buffer
                # u_lower_bound[num_old_tiles_in_buf:] = min_rates_cur_step[
                #     tile_of_interest_pos][num_old_tiles_in_buf:] + 1e-3

                # if params.ACTION_INIT_TYPE == ActionInitType.LOW:
                #     us[i] = u_lower_bound.copy()
                # elif params.ACTION_INIT_TYPE == ActionInitType.RANDOM_UNIFORM:
                #     us[i] = np.random.uniform(u_lower_bound, u_upper_bound,
                #                               (len(xs[i]), ))
                # else:
                #     pass
                ######################################
                # us[i] = np.random.uniform(0.1, 1, (len(xs[i]), ))
                us[i] = np.full(len(xs[i], ), 1e-3)

            u = us[i]

            start_time = time.time()
            xs[i + 1] = self.dynamics.f(x, u, i, viewing_probability)

            # print("f--- ", time.time() - start_time, " seconds ---")

            F_x[i] = self.dynamics.f_x(x, u, i)
            # print("f_x--- ", time.time() - start_time, " seconds ---")

            F_u[i] = self.dynamics.f_u(x, u, i)
            # print("f_u--- ", time.time() - start_time, " seconds ---")

            L[i] = self.cost.l(x,
                               u,
                               i,
                               bandwidth_budget,
                               viewing_probability,
                               distances,
                               self.tile_a,
                               self.tile_b,
                               update_start_idx,
                               update_end_idx,
                               self.dynamics,
                               self.buffer,
                               self.rate_versions,
                               terminal=False)
            # print("l--- ", time.time() - start_time, " seconds ---")

            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            # print("l_x--- ", time.time() - start_time, " seconds ---")

            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            # print("l_u--- ", time.time() - start_time, " seconds ---")

            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            # print("l_xx--- ", time.time() - start_time, " seconds ---")

            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            # print("l_ux--- ", time.time() - start_time, " seconds ---")

            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)
            # print("l_uu--- ", time.time() - start_time, " seconds ---")

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        next_state_size = len(x)

        # TODO by Tongyu: learn cost-to-go values in future
        # cost-to-go values are 0's for point cloud video streaming
        L[-1] = 0
        L_x[-1] = np.zeros((next_state_size, ))
        L_xx[-1] = np.zeros((next_state_size, next_state_size))
        # L[-1] = self.cost.l(x, None, N, terminal=True, get_Q=get_Q)
        # L_x[-1] = self.cost.l_x(x, None, N, terminal=True, get_Qx=get_Qx)
        # L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True, get_Qxx=get_Qxx)

        return xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

		Args:
			F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
			F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
			L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
			L_u: Jacobian of cost path w.r.t. u [N, action_size].
			L_xx: Hessian of cost path w.r.t. x, x
				[N+1, state_size, state_size].
			L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
			L_uu: Hessian of cost path w.r.t. u, u
				[N, action_size, action_size].
			F_xx: Hessian of state path w.r.t. x, x if Hessians are used
				[N, state_size, state_size, state_size].
			F_ux: Hessian of state path w.r.t. u, x if Hessians are used
				[N, state_size, action_size, state_size].
			F_uu: Hessian of state path w.r.t. u, u if Hessians are used
				[N, state_size, action_size, action_size].

		Returns:
			Tuple of
				k: feedforward gains [N, action_size].
				K: feedback gains [N, action_size, state_size].
		"""
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        # k = np.empty_like(self._k)
        # K = np.empty_like(self._K)

        # every element of k (or K) has different shapes,
        # so cannot define them as arrays
        k = [0.0] * self.N
        K = [0.0] * self.N

        for i in range(self.N - 1, -1, -1):
            if self._use_hessians:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx,
                                                     F_xx[i], F_ux[i], F_uu[i])
            else:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx)

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return k, K

        # each element of k (or K) has different dimension,
        # so impossible to be converted to array.
        # return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

		Args:
			F_x: Jacobian of state w.r.t. x [state_size, state_size].
			F_u: Jacobian of state w.r.t. u [state_size, action_size].
			L_x: Jacobian of cost w.r.t. x [state_size].
			L_u: Jacobian of cost w.r.t. u [action_size].
			L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
			L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
			L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
			V_x: Jacobian of the value function at the next time step
				[state_size].
			V_xx: Hessian of the value function at the next time step w.r.t.
				x, x [state_size, state_size].
			F_xx: Hessian of state w.r.t. x, x if Hessians are used
				[state_size, state_size, state_size].
			F_ux: Hessian of state w.r.t. u, x if Hessians are used
				[state_size, action_size, state_size].
			F_uu: Hessian of state w.r.t. u, u if Hessians are used
				[state_size, action_size, action_size].

		Returns:
			Tuple of
				Q_x: [state_size].
				Q_u: [action_size].
				Q_xx: [state_size, state_size].
				Q_ux: [action_size, state_size].
				Q_uu: [action_size, action_size].
		"""
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(len(V_x))
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def iterate_LQR(self,
                    distances,
                    viewing_probability,
                    bandwidth_budget,
                    update_start_idx,
                    update_end_idx,
                    n_iterations=100,
                    tol=1e-6,
                    on_iteration=None):
        # tiles' byte size that are already in buffer
        self.buffered_tiles_sizes = self.buffer.copy()
        for i in range(params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE *
                       params.FPS):
            self.buffered_tiles_sizes.pop(0)
        for i in range(params.UPDATE_FREQ * params.FPS):
            self.buffered_tiles_sizes.append(
                np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                          params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
        self.buffered_tiles_sizes = np.array(
            self.buffered_tiles_sizes)  # current state
        viewing_probability = np.array(viewing_probability)
        tile_of_interest_pos = viewing_probability[:params.
                                                   TARGET_LATENCY].nonzero()
        new_tiles_into_buf_pos = viewing_probability[(
            params.TARGET_LATENCY -
            params.FPS):params.TARGET_LATENCY].nonzero()
        x0 = self.buffered_tiles_sizes[tile_of_interest_pos]

        # each tile has a 4 tuple location: (frame_idx, x, y, z)
        # tiles_rate_solution = params.MAX_TILE_SIZE * np.ones((num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        tiles_rate_solution = self.buffered_tiles_sizes.copy()

        action_size = x0.shape[0]
        state_size = action_size

        # self._k = np.zeros((N, action_size))
        # self._K = np.zeros((N, action_size, state_size))
        self._k = [0.0] * self.N
        self._K = [0.0] * self.N
        # self.N = self.n_step
        self._mu = 1.0
        self._delta = self._delta_0

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        update_time_step = update_start_idx // params.FPS

        # buffer is not fully occupied yet
        # if update_time_step < params.TARGET_LATENCY // params.FPS:
        #     self.max_rates_cur_step = np.concatenate(
        #         (np.zeros((params.TARGET_LATENCY - update_start_idx,
        #                    params.NUM_TILES_PER_SIDE_IN_A_FRAME,
        #                    params.NUM_TILES_PER_SIDE_IN_A_FRAME,
        #                    params.NUM_TILES_PER_SIDE_IN_A_FRAME)),
        #          self.max_tile_sizes[:update_start_idx]),
        #         axis=0)
        #     self.min_rates_cur_step = np.concatenate(
        #         (np.zeros((params.TARGET_LATENCY - update_start_idx,
        #                    params.NUM_TILES_PER_SIDE_IN_A_FRAME,
        #                    params.NUM_TILES_PER_SIDE_IN_A_FRAME,
        #                    params.NUM_TILES_PER_SIDE_IN_A_FRAME)),
        #          self.min_rates[:update_start_idx]),
        #         axis=0)

        # else:
        #     # looks wierd here, that's because the video is played repeatedly, so
        #     # every 1s the 'self.max_tile_sizes' and 'self.min_rates' should be circularly shifted!
        #     self.max_rates_cur_step = np.concatenate(
        #         (self.max_tile_sizes[(update_start_idx -
        #                               params.TARGET_LATENCY) %
        #                              params.NUM_FRAMES:],
        #          self.max_tile_sizes[:(update_start_idx -
        #                                params.TARGET_LATENCY) %
        #                              params.NUM_FRAMES]),
        #         axis=0)

        #     self.min_rates_cur_step = np.concatenate(
        #         (self.min_rates[(update_start_idx - params.TARGET_LATENCY) %
        #                         params.NUM_FRAMES:],
        #          self.min_rates[:(update_start_idx - params.TARGET_LATENCY) %
        #                         params.NUM_FRAMES]),
        #         axis=0)

        # us = np.random.uniform(0.1, 1, (self.n_step, action_size)) # initialize actions
        us = [0.0] * self.N

        ######## initialize action #########
        # TODO by Tongyu: add a function for action initialization
        # TODO by Tongyu: try another kind of initialization: only last params.FPS frames are initialized as min_rates+1e-3, other tiles are just (1e-3)

        # # 1e-3 is to avoid reaching upperbound
        # u_upper_bound = self.max_rates_cur_step[
        #     tile_of_interest_pos] - 1e-3 - x0

        # num_new_tiles_into_buffer = len(new_tiles_into_buf_pos[0])
        # num_old_tiles_in_buf = len(x0) - num_new_tiles_into_buffer

        # u_lower_bound = np.zeros_like(x0)
        # diff_old_states_with_minrate = self.min_rates_cur_step[
        #     tile_of_interest_pos][:
        #                           num_old_tiles_in_buf] - x0[:
        #                                                      num_old_tiles_in_buf]
        # diff_old_states_with_minrate[diff_old_states_with_minrate < 0] = 0

        # # action lower bound for old tiles in buffer
        # u_lower_bound[:num_old_tiles_in_buf] += (diff_old_states_with_minrate +
        #                                          1e-3)
        # if params.SCALABLE_CODING == False:
        #     u_lower_bound[:num_old_tiles_in_buf] = self.min_rates_cur_step[
        #             tile_of_interest_pos][:num_old_tiles_in_buf] + 1e-3

        # # action lower bound for new tiles coming into buffer
        # u_lower_bound[num_old_tiles_in_buf:] = self.min_rates_cur_step[
        #     tile_of_interest_pos][num_old_tiles_in_buf:] + 1e-3

        # if params.ACTION_INIT_TYPE == ActionInitType.LOW:
        #     us[0] = u_lower_bound.copy()
        # elif params.ACTION_INIT_TYPE == ActionInitType.RANDOM_UNIFORM:
        #     us[0] = np.random.uniform(u_lower_bound, u_upper_bound,
        #                               (action_size, ))
        # else:
        #     pass

        # us[0] = np.random.uniform(0.1, 1,
        #                           (action_size, ))  # initialize actions
        us[0] = np.full((action_size, ), 1e-3)
        ######################################

        # pdb.set_trace()
        # self.logger.debug('action size: %d; length of tile_of_interest_pos: %d', action_size, len(tile_of_interest_pos[0]))
        # us = us_init.copy()
        # print(us)
        k = self._k
        K = self._K

        changed = True
        converged = False
        for iteration in range(n_iterations):
            logging.info('\n###### %dth iteration of iLQR ######', iteration)
            accepted = False

            # Forward rollout only if it needs to be recomputed.
            if changed:
                start_time = time.time()
                (xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux,
                 L_uu) = self._forward_rollout(x0, us, state_size, action_size,
                                               bandwidth_budget,
                                               viewing_probability, distances,
                                               update_start_idx,
                                               update_end_idx)
                print("_forward_rollout--- ",
                      time.time() - start_time, " seconds ---")
                J_opt = sum(L)
                changed = False

            try:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux,
                                           L_uu)
                print("_backward_pass--- ",
                      time.time() - start_time, " seconds ---")

                # Backtracking line search.
                for alpha in alphas:
                    self.logger.debug('%dth alpha: %f',
                                      np.where(alphas == alpha)[0][0], alpha)
                    xs_new, us_new, J_new = self._control(
                        xs, us, k, K, bandwidth_budget, viewing_probability,
                        distances, update_start_idx, update_end_idx, alpha)
                    print("_control--- ",
                          time.time() - start_time, " seconds ---")

                    # J_new = self._trajectory_cost(xs_new, us_new,
                    #                               bandwidth_budget,
                    #                               viewing_probability,
                    #                               distances, update_start_idx,
                    #                               update_end_idx)
                    # print("_trajectory_cost--- ",
                    #       time.time() - start_time, " seconds ---")

                    # TODO by Tongyu: log different parts of cost value, including quality_sum and quality_var, etc.
                    self.logger.debug(
                        'J_new = %f, J_opt = %f. J_new < J_opt is %s', J_new,
                        J_opt, str(J_new < J_opt))

                    if J_new < J_opt:
                        self.logger.debug('improvement is %f',
                                          np.abs((J_opt - J_new) / J_opt))
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0
                        self.logger.debug('[accepted]. delta = %f, mu = %f',
                                          self._delta, self._mu)

                        # Accept this.
                        accepted = True
                        self.logger.debug('accepted is %s', str(accepted))
                        self.logger.debug('converged is %s', str(converged))
                        break
                self.logger.debug('final accepted in this iter is %s',
                                  str(accepted))

            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                # warnings.warn(str(e))
                self.logger.warning(str(e))

            # tried all alpha's, but still find no better solution than initialized one
            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                self.logger.debug('[NOT accepted]. delta = %f, mu = %f',
                                  self._delta, self._mu)
                if self._mu_max and self._mu >= self._mu_max:
                    # warnings.warn("exceeded max regularization term")
                    self.logger.warning(
                        'iLQR: exceeded max regularization term')
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged)

            if converged:
                break

            # print(us)
            # pdb.set_trace()

        self.logger.debug('converged is %s for %dth buf update',
                          str(converged), update_time_step)

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        # tiles_rate_solution is not delta_rates, it the updated rates!!
        tiles_rate_solution = self.buffered_tiles_sizes.copy()
        if params.SCALABLE_CODING:
            tiles_rate_solution[tile_of_interest_pos] += us[0]
        else:
            tiles_rate_solution[tile_of_interest_pos] = us[0]

        total_size = np.sum(tiles_rate_solution)
        return tiles_rate_solution, self.buffered_tiles_sizes, total_size, np.sum(
            self.buffered_tiles_sizes)
        # return xs, us

        # VT = 0
        # vt = 0
        # for ite_i in range(self.n_iteration):
        # 	converge = True
        # 	KT_list = [0.0] * self.n_step
        # 	kt_list = [0.0] * self.n_step
        # 	VT_list = [0.0] * self.n_step
        # 	vt_list = [0.0] * self.n_step
        # 	pre_xt_list = [0.0] * self.n_step
        # 	new_xt_list = [0.0] * self.n_step
        # 	pre_ut_list  = [0.0] * self.n_step

        # 	# Backward pass
        # 	for step_i in reversed(range(self.n_step)):
        # 		self.update_matrix(step_i, action_size)
        # 		xt = np.array([[self.states[step_i][0]],[self.states[step_i][1]], [self.states[step_i][2]], [self.states[step_i][3]]])	  #2*1
        # 		ut = np.array([[self.rates[step_i]], [self.speeds[step_i]]])
        # 		pre_xt_list[step_i] = xt
        # 		pre_ut_list[step_i] = ut
        # 		if step_i == self.n_step-1:
        # 			Qt = self.CT
        # 			qt = self.ct
        # 		else:
        # 			# To be modified
        # 			Qt = self.CT + np.dot(np.dot(self.ft.T, VT), self.ft)	# 3*3
        # 			qt = self.ct + np.dot(self.ft.T, vt)					 # 3*1. self.ft is FT in equation, and ft in this equation is zeor (no constant)
        # 			if LQR_DEBUG:
        # 				print("vt: ", vt)
        # 				print("qt: ", qt)

        # 		# Origin
        # 		Q_xx = Qt[:4,:4]		 #4*4
        # 		Q_xu = Qt[:4,4:]		 #4*2
        # 		Q_ux = Qt[4:,:4]		 #2*4
        # 		Q_uu = Qt[4:,4:]		 #2*2
        # 		q_x = qt[:4]			 #4*1
        # 		q_u = qt[4:]			 #2*1
        # 		# print(q_x)
        # 		# print(q_u)

        # 		# print(Qt)
        # 		# Q_xx = Qt[:4,:4]		 #4*4
        # 		# Q_xu = Qt[:4,4:]		 #4*2
        # 		# Q_ux = Qt[4:,:4]		 #2*4
        # 		# Q_uu = Qt[4:,4:]		 #2*2
        # 		# q_x = qt[:4]			 #4*1
        # 		# q_u = qt[4:]			 #2*1

        # 		KT = np.dot(-1, np.dot(np.linalg.inv(Q_uu), Q_ux))
        # 		kt = np.dot(-1, np.dot(np.linalg.inv(Q_uu), q_u))

        # 		if iLQR_SHOW:
        # 			print("Ct: ", self.CT)
        # 			print("   ")
        # 			print("self.ft.T: ", self.ft.T)
        # 			print("VT: ", VT)
        # 			print("self.ft: ", self.ft)
        # 			print("Q_ux: ", Q_ux)
        # 			print("Q_uu: ", Q_uu)
        # 			print("KT: ", KT)
        # 			print("kt: ", kt)
        # 			print("Step: ", step_i)
        # 			print("<======>")

        # 		VT = Q_xx + np.dot(Q_xu, KT) + np.dot(KT.T, Q_ux) + np.dot(np.dot(KT.T, Q_uu), KT)  #2*2
        # 		vt = q_x + np.dot(Q_xu, kt) + np.dot(KT.T, q_u) + np.dot(np.dot(KT.T, Q_uu), kt)	#2*1
        # 		KT_list[step_i] = KT
        # 		kt_list[step_i] = kt
        # 		VT_list[step_i] = self.decay*VT
        # 		vt_list[step_i] = self.decay*vt

        # 		if iLQR_SHOW:
        # 			print(VT)
        # 			print(",,,")
        # 			print(Q_xx)
        # 			print(",,,")
        # 			print(np.dot(Q_xu, KT))
        # 			print("...")
        # 			print(np.dot(KT.T, Q_ux))
        # 			print("last")
        # 			print(np.dot(np.dot(KT.T, Q_uu), KT))
        # 			print("end!!")

        # 	if LQR_DEBUG:
        # 		print("!!!!!! Backward done!!!!!!!!")
        # 		print("pre xt: ", pre_xt_list)
        # 		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 	# Forward pass
        # 	new_xt_list[0] = pre_xt_list[0]
        # 	for step_i in range(self.n_step):
        # 		if LQR_DEBUG:
        # 			print("<=========================>")
        # 			print("forward pass, step: ", step_i)
        # 			print("new xt: ", new_xt_list[step_i])
        # 			print("pre xt: ", pre_xt_list[step_i])
        # 			print("kt matrix is: ", kt_list[step_i])
        # 		d_x = new_xt_list[step_i] - pre_xt_list[step_i]
        # 		k_t = self.kt_step*kt_list[step_i]
        # 		K_T = self.KT_step*KT_list[step_i]

        # 		d_u = np.dot(K_T, d_x) + k_t
        # 		# new_u = pre_ut_list[step_i] + self.step_size*d_u	   # New action
        # 		new_u = [0,0]
        # 		new_u[0] = max(0.75*pre_ut_list[step_i][0], min(1.1*pre_ut_list[step_i][0], pre_ut_list[step_i][0] + self.step_size*d_u[0]))
        # 		new_u[1] = max(0.75*pre_ut_list[step_i][1], min(1.1*pre_ut_list[step_i][1], pre_ut_list[step_i][1] + self.step_size*d_u[1]))
        # 		if iLQR_SHOW:
        # 			print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # 			print("Step: ", step_i)
        # 			print("Dx is: ", d_x)
        # 			print("kt: ", k_t)
        # 			print("KT: ", K_T)
        # 			print("Du: ", d_u)
        # 			print("Ut: ", pre_ut_list[step_i])
        # 			print("New action: ", new_u)
        # 			print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # 			if new_u[0] >= 2*self.predicted_bw[step_i]:
        # 				input()
        # 		# n_rate = np.round(new_u[0][0], 2)
        # 		n_rate =  min(max(np.round(new_u[0][0], 2), 0.3), 6.0)
        # 		n_speed = min(max(np.round(new_u[1][0], 2), 0.9), 1.1)
        # 		# Check converge
        # 		if converge and (np.round(n_rate,1) != np.round(self.rates[step_i],2) or np.round(n_speed,2) != np.round(self.speeds[step_i], 1)):
        # 			converge = False
        # 		self.rates[step_i] = n_rate
        # 		self.speeds[step_i] = n_speed
        # 		new_x = new_xt_list[step_i]			 # Get new state
        # 		rtt = self.predicted_rtt[step_i]
        # 		bw = self.predicted_bw[step_i]

        # 		new_next_b, new_next_l = self.sim_fetch(new_x[0][0], new_x[1][0], n_rate, n_speed, rtt, bw)			   # Simulate to get new next state
        # 		if LQR_DEBUG:
        # 			print("new x: ", new_x)
        # 			print("new b: ", new_next_b)
        # 			print("bew l: ", new_next_l)
        # 		if step_i < self.n_step - 1:
        # 			new_xt_list[step_i+1] = [[new_next_b], [new_next_l], [n_rate], [n_speed]]
        # 			self.states[step_i+1] = [np.round(new_next_b, 2), np.round(new_next_l, 2), self.rates[step_i], self.speeds[step_i]]
        # 		else:
        # 			self.states[step_i+1] = [np.round(new_next_b, 2), np.round(new_next_l, 2), self.rates[step_i], self.speeds[step_i]]

        # 	# Check converge
        # 	if converge:
        # 		break

        # 	if LQR_DEBUG:
        # 		print("New states: ", self.states)
        # 		print("New actions: ", self.rates)

        # 	# Check convergence
        # 	if iLQR_SHOW:
        # 		print("Iteration ", ite_i, ", previous rate: ", self.states[0][1])
        # 		print("Iteration ", ite_i, ", buffer is: ", [x[0] for x in self.states])
        # 		print("Iteration ", ite_i, ", latency is: ", [x[1] for x in self.states])
        # 		print("Iteration ", ite_i, ", pre bw is: ", self.predicted_bw)
        # 		print("Iteration ", ite_i, ", action is: ", self.rates)
        # 		print("Iteration ", ite_i, ", action is: ", self.speeds)

        # 		print("<===============================================>")

        # r_idx = self.translate_to_rate_idx()
        # # s_idx = self.translate_to_speed_idx()
        # s_idx = self.translate_to_speed_idx_accu()
        # # s_idx = self.translate_to_speed_idx_accu_new()
        # return r_idx, s_idx

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

        z_weight_locations = []
        nonzero_zWeight_locs = np.where(
            (z_weights != 0) & (buffered_tiles_sizes != self.max_tile_sizes))
        nonzero_zWeight_frame_idx = nonzero_zWeight_locs[0]
        nonzero_zWeight_x = nonzero_zWeight_locs[1]
        nonzero_zWeight_y = nonzero_zWeight_locs[2]
        nonzero_zWeight_z = nonzero_zWeight_locs[3]

        r0 = np.zeros(
            (num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        r0[nonzero_zWeight_locs] = np.maximum(
            buffered_tiles_sizes[nonzero_zWeight_locs],
            self.min_rates_frames[nonzero_zWeight_locs])

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
                current_rate, self.rate_versions[0][x][y][z],
                self.rate_versions[1][x][y][z], self.rate_versions[2][x][y][z])
            if current_version == params.NUM_RATE_VERSIONS:  # max rate version
                utility_rate_slopes[frame_idx][x][y][z] = 0
                continue
            next_version_rate = self.rate_versions[params.NUM_RATE_VERSIONS -
                                                   (current_version +
                                                    1)][x][y][z]
            next_version_rates[frame_idx][x][y][z] = next_version_rate
            assert (next_version_rate - current_rate >
                    0), "!!! same rate between 2 levels (before loop) !!!!!"
            utility_rate_slopes[frame_idx][x][y][z] = z_weights[frame_idx][x][y][z] * 2 * \
                        (self.calculate_unweighted_tile_quality(next_version_rate, distances[frame_idx][x][y][z], self.tile_a[x][y][z], self.tile_b[x][y][z]) \
                       - self.calculate_unweighted_tile_quality(tiles_rate_solution[frame_idx][x][y][z], distances[frame_idx][x][y][z], self.tile_a[x][y][z], self.tile_b[x][y][z])) \
                       / (next_version_rate - current_rate)

            # print(self.calculate_unweighted_tile_quality(next_version_rate, distances[frame_idx][x][y][z], self.tile_a[x][y][z], self.tile_b[x][y][z]))
            # print(self.calculate_unweighted_tile_quality(tiles_rate_solution[frame_idx][x][y][z], distances[frame_idx][x][y][z], self.tile_a[x][y][z], self.tile_b[x][y][z]))
            # print(next_version_rate)
            # print(current_rate)
            # pdb.set_trace()

        sorted_z_weights = sorted(z_weight_locations, \
                key=lambda loc: z_weights[loc["frame_idx"]]\
                        [loc["x"]]\
                        [loc["y"]]\
                        [loc["z"]], reverse=True)

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

            if params.RUMA_SCALABLE_CODING:
                consumed_bandwidth += (next_version_rate - current_rate)
            else:
                # locs = np.where(tiles_rate_solution > buffered_tiles_sizes)
                # consumed_bandwidth = np.sum(tiles_rate_solution[locs])
                # consumed_bandwidth += next_version_rate
                if current_rate == buffered_tiles_sizes[max_slope_frame_idx][
                        max_slope_x][max_slope_y][max_slope_z]:
                    consumed_bandwidth += next_version_rate
                else:
                    consumed_bandwidth += (next_version_rate - current_rate)

            if consumed_bandwidth > total_size_constraint:
                tiles_rate_solution[max_slope_frame_idx][max_slope_x][
                    max_slope_y][max_slope_z] = current_rate
                break

            if next_version_rate == self.rate_versions[0][max_slope_x][
                    max_slope_y][max_slope_z]:
                utility_rate_slopes[max_slope_frame_idx][max_slope_x][
                    max_slope_y][max_slope_z] = 0
            else:
                current_version = self.decide_rate_version(
                    next_version_rate, self.rate_versions[0][max_slope_x]
                    [max_slope_y][max_slope_z], self.rate_versions[1]
                    [max_slope_x][max_slope_y][max_slope_z],
                    self.rate_versions[2][max_slope_x][max_slope_y]
                    [max_slope_z])
                next_version_rate = self.rate_versions[
                    params.NUM_RATE_VERSIONS -
                    (current_version +
                     1)][max_slope_x][max_slope_y][max_slope_z]
                next_version_rates[max_slope_frame_idx][max_slope_x][
                    max_slope_y][max_slope_z] = next_version_rate
                assert (next_version_rate -
                        tiles_rate_solution[max_slope_frame_idx][max_slope_x]
                        [max_slope_y][max_slope_z] >
                        0), "!!! same rate between 2 levels (in loop) !!!!!"
                utility_rate_slopes[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] = z_weights[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] * 2 * \
                           (self.calculate_unweighted_tile_quality(next_version_rate, distances[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], self.tile_a[max_slope_x][max_slope_y][max_slope_z], self.tile_b[max_slope_x][max_slope_y][max_slope_z]) \
                          - self.calculate_unweighted_tile_quality(tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], distances[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], self.tile_a[max_slope_x][max_slope_y][max_slope_z], self.tile_b[max_slope_x][max_slope_y][max_slope_z])) \
                          / (next_version_rate - tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z])

        return tiles_rate_solution, buffered_tiles_sizes, np.sum(
            tiles_rate_solution), np.sum(
                buffered_tiles_sizes), sorted_z_weights

    def kkt(self, z_weights, bandwidth_budget, update_start_idx,
            update_end_idx):
        ##################### get v1 and v2 for each tile: ################################
        # v1 = z_weight / r0; v2 = z_weight / params.MAX_TILE_SIZE.
        num_frames_to_update = update_end_idx - update_start_idx + 1
        # pdb.set_trace()

        b_updated_tiles = np.concatenate(
            (self.tile_b[(update_start_idx - params.TARGET_LATENCY) %
                         params.NUM_FRAMES:],
             self.tile_b[:(update_start_idx - params.TARGET_LATENCY) %
                         params.NUM_FRAMES]),
            axis=0)

        # TODO by Tongyu: wherever using max_rates_cur_step, assert max_rates_cur_step[.] != 0 to make sure it matches right tile positions.
        # also for a_update_tiles and b_update_tiles.
        max_rates_cur_step = np.concatenate(
            (self.max_tile_sizes[(update_start_idx - params.TARGET_LATENCY) %
                                 params.NUM_FRAMES:],
             self.max_tile_sizes[:(update_start_idx - params.TARGET_LATENCY) %
                                 params.NUM_FRAMES]),
            axis=0)

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
            tiles_rate_solution = buffered_tiles_sizes.copy()
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
            nonzero_zWeight_locs = np.where(
                (z_weights != 0) & (buffered_tiles_sizes != max_rates_cur_step))
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
            r0 = buffered_tiles_sizes.copy()

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
            tiles_rate_solution[frame_idx][x][y][z] = max_rates_cur_step[frame_idx][x][y][z]
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
                        z] = z_weights[frame_idx][x][y][z] / lagrange_lambda - 1 / b_updated_tiles[frame_idx][x][y][z]

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
                tiles_rate_solution[frame_idx][x][y][z] = max_rates_cur_step[frame_idx][
                    x][y][z]
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
                    z] = z_weight / lagrange_lambda - 1/b_updated_tiles[frame_idx][x][y][z]
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
            sum_bk_inverse += 1/b_updated_tiles[frame_idx][x][y][z]
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
                z] = z_weight / lagrange_lambda - 1/b_updated_tiles[frame_idx][x][y][z]

            if tiles_rate_solution[frame_idx][x][y][z] >= max_rates_cur_step[frame_idx][
                    x][y][z]:
                print("!!!!!!!! tile size: %f,  MAX size: %f" %
                      (tiles_rate_solution[frame_idx][x][y][z],
                       self.max_tile_sizes[x][y][z]))
                tiles_rate_solution[frame_idx][x][y][z] = max_rates_cur_step[frame_idx][
                    x][y][z]
                # pdb.set_trace()
            if tiles_rate_solution[frame_idx][x][y][z] <= r0[
                    frame_idx][x][y][z]:
                print("!!!!!!!! tile size: %f,  r0: %f" %
                      (tiles_rate_solution[frame_idx][x][y][z],
                       r0[frame_idx][x][y][z]))
                tiles_rate_solution[frame_idx][x][y][z] = r0[
                    frame_idx][x][y][z]
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

            print("rounding--- ", time.time() - self.start_time, " seconds ---")

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

    def decide_rate_version(self, rate, max_version, mid_version, min_version):
        if rate == max_version:
            return 3
        elif rate >= mid_version:
            return 2
        elif rate >= min_version:
            return 1
        else:
            return 0
