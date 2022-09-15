import numpy as np
import pickle as pk
import pdb
import math
import time

import params
from params import Algo
from hidden_points_removal import HiddenPointsRemoval


np.random.seed(7)

class Buffer():
	'''
		buffer controlling
		include buffer initialization and update
	'''

	def __init__(self, fov_traces_obj, bw_traces_obj, valid_tiles_obj, qr_weights_obj):
		self.buffer_length = params.BUFFER_LENGTH # frame

		self.update_step = 0

		# self.buffer stores 3d numpy arrays,
		# each 3d array represents one frame,
		# which contains byte sizes of all tiles
		self.buffer = []

		self.max_tile_sizes = qr_weights_obj.rate_versions[0] # 16x16x16
		tmp = np.expand_dims(self.max_tile_sizes.copy(), axis=0)
		self.max_tile_sizes_frames = np.repeat(tmp.copy(), self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS, axis=0)

		self.min_tile_sizes = qr_weights_obj.rate_versions[2] # 16x16x16
		tmp = np.expand_dims(self.min_tile_sizes.copy(), axis=0)
		self.min_tile_sizes_frames = np.repeat(tmp.copy(), self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS, axis=0)

		self.fov_traces_obj = fov_traces_obj
		self.valid_tiles_obj = valid_tiles_obj
		self.bw_traces_obj = bw_traces_obj
		self.qr_weights_obj = qr_weights_obj

		# a, b and distance_weight are waiting to be fit
		self.tile_a = self.qr_weights_obj.qr_weights["a"] # cubic array for each tile
		self.tile_b = self.qr_weights_obj.qr_weights["b"] # cubic array for each tile
		self.min_rates = self.qr_weights_obj.min_rates
		tmp = np.expand_dims(self.tile_a.copy(), axis=0)
		self.tile_a_frames = np.repeat(tmp.copy(), self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS, axis=0)
		tmp = np.expand_dims(self.tile_b.copy(), axis=0)
		self.tile_b_frames = np.repeat(tmp.copy(), self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS, axis=0)
		tmp = np.expand_dims(self.min_rates.copy(), axis=0)
		self.min_rates_frames = np.repeat(tmp.copy(), self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS, axis=0)

		self.rate_versions = self.qr_weights_obj.rate_versions

		# sigmoid coefficient c: (1 + exp(-c*d))^(-1)
		self.distance_weight = 1

		# linearly increasing from 0 to 1
		self.frame_weights = None

		# initialize according to fov dataset H1, assume the initial viewpoint is always like this:
		# {x, y, z, roll, yaw, pitch} = {0.05, 1.7868, -1.0947, 6.9163, 350.8206, 359.9912}
		# z-x plane is floor
		self.history_viewpoints = {"x":[0.05]*params.FOV_PREDICTION_HISTORY_WIN_LENGTH, 
								   "y":[1.7868]*params.FOV_PREDICTION_HISTORY_WIN_LENGTH, 
								   "z":[-1.0947]*params.FOV_PREDICTION_HISTORY_WIN_LENGTH, 
								   "pitch":[6.9163 + 360]*params.FOV_PREDICTION_HISTORY_WIN_LENGTH, # rotate around x axis
								   "yaw":[350.8206]*params.FOV_PREDICTION_HISTORY_WIN_LENGTH, # rotate around y axis
								   "roll":[359.9912]*params.FOV_PREDICTION_HISTORY_WIN_LENGTH} # rotate around z axis

		self.history_viewpoints = {"x":[0.05], 
								   "y":[1.7868], 
								   "z":[-1.0947], 
								   "pitch":[6.9163 + 360], # rotate around x axis
								   "yaw":[350.8206], # rotate around y axis
								   "roll":[359.9912]} # rotate around z axis
		
		# initialize bandwidth history according to '../bw_traces/100ms_loss1'
		self.history_bandwidths = [2.7] * params.BW_PREDICTION_HISTORY_WIN_LENGTH

		self.current_viewing_frame_idx = -1

		self.current_bandwidth_idx = -1

		self.origin = self.calculate_origin(self.valid_tiles_obj.valid_tiles[0]) # calculate origin according to the first frame

		# # r* = Rmax
		# self.typeI_tiles_set = set()
		# # r* = r0
		# self.typeII_tiles_set = set()
		# # r* = z_weight / lambda
		# self.typeIII_tiles_set = set()
		self.frame_quality = []
		self.plot_bw_trace = []
		self.plot_predicted_bw_trace = []
		self.buffer_size_trace = []
		self.delta_buffer_size_trace = []
		self.true_viewpoints = {"x":[], 'y':[], "z":[], "pitch":[], "yaw":[], "roll":[]}
		self.fov_predict_accuracy_trace = {"x":[], 'y':[], "z":[], "pitch":[], "yaw":[], "roll":[]}
		self.overlap_ratio_history = []

		for key in self.true_viewpoints.keys():
			for frame_idx in range(self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS):
				self.true_viewpoints[key].append([])
				self.fov_predict_accuracy_trace[key].append([])

		for frame_idx in range(self.buffer_length + params.FPS * params.UPDATE_FREQ - params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS):
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


	def calculate_origin(self, valid_tiles):
		tile_xs, tile_ys, tile_zs = valid_tiles.nonzero()
		x_list = []
		y_list = []
		z_list = []
		for point_idx in range(len(tile_xs)):
			tile_center_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate(tile_xs[point_idx], tile_ys[point_idx], tile_zs[point_idx])
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
			self.buffer.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
			# if frame_idx < params.FPS:
			# 	valid_tiles_coordinates = self.valid_tiles_obj.valid_tiles[frame_idx].nonzero()
			# 	self.buffer[frame_idx][valid_tiles_coordinates] = params.MAX_TILE_SIZE

	def true_frame_quality(self, viewing_probability, distances):
		tiles_byte_sizes = self.buffer[0] # front of buffer: cubic array

		### quantize / quantization #######
		# tiles_byte_sizes[np.where(tiles_byte_sizes < params.BYTE_SIZES[0])] = 0
		# tiles_byte_sizes[np.where((tiles_byte_sizes < params.BYTE_SIZES[1]) & (tiles_byte_sizes >= params.BYTE_SIZES[0]))] = params.BYTE_SIZES[0]
		# tiles_byte_sizes[np.where((tiles_byte_sizes < params.BYTE_SIZES[2]) & (tiles_byte_sizes >= params.BYTE_SIZES[1]))] = params.BYTE_SIZES[1]
		####################################


		self.tile_sizes_sol.append(tiles_byte_sizes)
		self.frame_size_list.append(np.sum(tiles_byte_sizes))
		self.num_max_tiles_per_frame.append(len(np.where(tiles_byte_sizes==self.max_tile_sizes)[0]))
		if len(viewing_probability.nonzero()[0]) == 0:
			self.mean_size_over_tiles_per_frame.append(float("inf"))
			# return 0
		else:
			if len(tiles_byte_sizes.nonzero()[0]) == 0:
				self.mean_size_over_tiles_per_frame.append(0)
			else:
				self.mean_size_over_tiles_per_frame.append(np.sum(tiles_byte_sizes) / len(tiles_byte_sizes.nonzero()[0]))
		total_visible_size = 0
		tile_xs, tile_ys, tile_zs = tiles_byte_sizes.nonzero()
		frame_quality = 0
		total_span_degree = 0
		if len(viewing_probability.nonzero()[0]):
			total_span_degree = np.sum(1 / distances[viewing_probability.nonzero()]) * params.TILE_SIDE_LEN * 180 / np.pi
		for point_idx in range(len(tile_xs)):
			x = tile_xs[point_idx]
			y = tile_ys[point_idx]
			z = tile_zs[point_idx]

			if viewing_probability[x][y][z] == 0:
				continue

			# total_span_degree += params.TILE_SIDE_LEN / distances[x][y][z] * 180 / np.pi
			if tiles_byte_sizes[x][y][z] == 0:
				continue

			total_visible_size += tiles_byte_sizes[x][y][z]

			assert (self.tile_a[x][y][z] > 0), "!!!!! qr weight a non-positive when calculating true frame quality: (%d, %d, %d, %f) !!!!!!" %(x, y, z, self.tile_a[x][y][z])
			########## quality = (tile_a * np.log(rate) + tile_b) / (1 + np.exp(-dist_c * distance)) ############
			# tile_psnr = self.tile_a[x][y][z] * np.log(tiles_byte_sizes[x][y][z]) + self.tile_b[x][y][z]
			# tile_quality = viewing_probability[x][y][z] * tile_psnr / (1 + np.exp(-self.distance_weight * distances[x][y][z]))
			########################################################################################################

			########## quality = tile_a * np.log(20 / distance) * np.log(rate) + tile_b * np.log(distance) * 5 ############
			# tile_quality = viewing_probability[x][y][z] * self.tile_a[x][y][z] * np.log(20 / distances[x][y][z]) * np.log(tiles_byte_sizes[x][y][z]) + self.tile_b[x][y][z] * np.log(distances[x][y][z]) * 5
			num_points_per_degree = distances[x][y][z] / params.TILE_SIDE_LEN / 180 * np.pi * np.power(self.tile_a[x][y][z] * tiles_byte_sizes[x][y][z] + self.tile_b[x][y][z], 0.5)
			assert (num_points_per_degree > 0), "!!!!! non-positive number of points per degree: %f !!!!!!" %(num_points_per_degree)
			tile_quality = params.TILE_SIDE_LEN / distances[x][y][z] * 180 / np.pi * np.log(num_points_per_degree)
			# print(distances[x][y][z])
			########################################################################################################

			frame_quality += tile_quality

		if len(viewing_probability.nonzero()[0]) == 0:
			self.mean_size_over_tiles_per_fov.append(float("inf"))
		else:
			self.mean_size_over_tiles_per_fov.append(total_visible_size / len(viewing_probability.nonzero()[0]))
		# visible size / frame size
		if np.sum(tiles_byte_sizes) == 0:
			self.effective_rate.append(float("inf"))
		else:
			self.effective_rate.append(total_visible_size / np.sum(tiles_byte_sizes))

		# return 0 if np.sum(z_weights) == 0 else frame_quality / np.sum(z_weights)
		if total_span_degree == 0:
			frame_quality_per_degree = 0
		else:
			frame_quality_per_degree = frame_quality / total_span_degree
		return frame_quality_per_degree

	def emit_buffer(self):
		'''
			emit params.UPDATE_FREQ*params.FPS frames from front of buffer;;
			Based on their true viewpoint, calculate their HPR, distance, and quality;
			update pointers: buffer, current_viewing_frame_idx, history_viewpoints, history_bandwidths, current_bandwidth_idx
		'''

		previous_visible_tiles_set = set()

		for frame_idx in range(self.current_viewing_frame_idx + 1, self.current_viewing_frame_idx + params.FPS * params.UPDATE_FREQ + 1):
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

			viewing_probability, distances = self.calculate_probability_to_be_viewed(viewpoint, frame_idx, frame_idx)
			# z_weights = self.calculate_z(viewing_probability, distances, frame_idx, frame_idx, evaluation_flag=True)


			for tile_idx in range(len(viewing_probability[0].nonzero()[0])):
				x = viewing_probability[0].nonzero()[0][tile_idx]
				y = viewing_probability[0].nonzero()[1][tile_idx]
				z = viewing_probability[0].nonzero()[2][tile_idx]
				current_visible_tiles_set.add((x, y, z))
			if frame_idx >= 1:
				intersect = current_visible_tiles_set.intersection(previous_visible_tiles_set)
				self.num_intersect_visible_tiles_trace.append(len(intersect))
			previous_visible_tiles_set = current_visible_tiles_set.copy()

			# self.num_valid_tiles_per_frame.append(len(viewing_probability[0].nonzero()[0]))
			# calculate total span of fov (how many degrees)
			if len(viewing_probability[0].nonzero()[0]):
				self.num_valid_tiles_per_frame.append(np.sum(1 / distances[0][viewing_probability[0].nonzero()]) * params.TILE_SIDE_LEN * 180 / np.pi)
			else:
				self.num_valid_tiles_per_frame.append(0)

			# if self.update_step > params.BUFFER_LENGTH // (params.UPDATE_FREQ * params.FPS) and frame_idx >= 1:
			# 	print(self.num_valid_tiles_per_frame[-1], self.num_intersect_visible_tiles_trace[-1], self.num_intersect_visible_tiles_trace[-1] / self.num_valid_tiles_per_frame[-1] * 100)
				# pdb.set_trace()

			true_quality = self.true_frame_quality(viewing_probability[0], distances[0])
			self.frame_quality.append(true_quality)

			# if true_quality < 7.4 and self.update_step > 10:
			# 	print(frame_idx)
			# 	pdb.set_trace()

			# pop processed frame
			self.buffer.pop(0)

			# update history_viewpoints
			for key in self.history_viewpoints.keys():
				if len(self.history_viewpoints[key]) == params.FOV_PREDICTION_HISTORY_WIN_LENGTH:
					self.history_viewpoints[key].pop(0)
				viewpoint_dof = viewpoint[key][0]
				if key == 'pitch' or key == 'yaw' or key == 'roll':
					if viewpoint_dof < 90: # user always move from 358 to 20
						viewpoint_dof += 360

				if self.update_step > params.BUFFER_LENGTH // (params.UPDATE_FREQ * params.FPS):
					self.history_viewpoints[key].append(viewpoint_dof)

		# update current_viewing_frame_idx, history_bandwidths, current_bandwidth_idx
		self.current_viewing_frame_idx = frame_idx
		self.current_bandwidth_idx += 1
		self.history_bandwidths.pop(0)
		self.history_bandwidths.append(self.bw_traces_obj.bw_trace[self.current_bandwidth_idx])

		print("finish emitting buffer--- ", time.time() - self.start_time, " seconds ---")

	def update_tile_size_in_buffer(self):
		self.update_step += 1
		##################### predict viewpoint #######################
		update_start_idx = self.current_viewing_frame_idx + params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS + 1
		update_end_idx = self.current_viewing_frame_idx + self.buffer_length + params.FPS * params.UPDATE_FREQ

		self.start_time = time.time()

		predicted_viewpoints = self.predict_viewpoint(predict_start_idx=update_start_idx, predict_end_idx=update_end_idx)

		print("predict_viewpoint--- ", time.time() - self.start_time, " seconds ---")

		for key in self.true_viewpoints.keys():
			for frame_idx in range(update_end_idx - update_start_idx + 1):
				# if using csv fov file
				# true_viewpoint_at_this_dof = self.fov_traces_obj.fov_traces.at[frame_idx + update_start_idx, params.MAP_6DOF_TO_HMD_DATA[key]]
				# if using txt fov file
				true_viewpoint_at_this_dof = self.fov_traces_obj.fov_traces[frame_idx + update_start_idx][params.MAP_6DOF_TO_HMD_DATA[key]]
				predicted_viewpoint_at_this_dof = predicted_viewpoints[key][frame_idx]
				self.true_viewpoints[key][frame_idx].append(true_viewpoint_at_this_dof)
				if key == 'pitch' or key == 'yaw' or key == 'roll':
					predicted_viewpoint_at_this_dof = predicted_viewpoint_at_this_dof if 360 - predicted_viewpoint_at_this_dof >= predicted_viewpoint_at_this_dof else predicted_viewpoint_at_this_dof - 360
					true_viewpoint_at_this_dof = true_viewpoint_at_this_dof if 360 - true_viewpoint_at_this_dof >= true_viewpoint_at_this_dof else true_viewpoint_at_this_dof - 360
				if self.update_step > params.BUFFER_LENGTH // (params.UPDATE_FREQ * params.FPS):
					self.fov_predict_accuracy_trace[key][frame_idx].append(abs(predicted_viewpoint_at_this_dof - true_viewpoint_at_this_dof))
		# pdb.set_trace()
		print("fov_predict_accuracy_trace--- ", time.time() - self.start_time, " seconds ---")
		#################################################################

		viewing_probability, distances = self.calculate_probability_to_be_viewed(predicted_viewpoints, update_start_idx, update_end_idx)

		print("viewing_probability--- ", time.time() - self.start_time, " seconds ---")
		
		# calculate distance only for viewable valid tiles
		# distances = self.calculate_distance(predicted_viewpoints)

		z_weights = self.calculate_z(viewing_probability, distances, update_start_idx, update_end_idx)

		print("calculate_z--- ", time.time() - self.start_time, " seconds ---")

		# predict bandwidth of future 1s
		predicted_bandwidth_budget = self.predict_bandwidth() * params.SCALE_BW # Mbps

		print("predict_bandwidth--- ", time.time() - self.start_time, " seconds ---")

		if params.ALGO == Algo.MMSYS_HYBRID_TILING:
			tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0, sorted_z_weights = self.hybrid_tiling(np.array(z_weights), predicted_bandwidth_budget * params.Mbps_TO_Bps, update_start_idx, update_end_idx)
		elif params.ALGO == Algo.RUMA_SCALABLE or params.ALGO == params.Algo.RUMA_NONSCALABLE:
			tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0, sorted_z_weights = self.RUMA(distances, np.array(z_weights), predicted_bandwidth_budget * params.Mbps_TO_Bps, update_start_idx, update_end_idx)
		elif params.ALGO == Algo.KKT:
			tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0, sorted_z_weights = self.kkt(np.array(z_weights), predicted_bandwidth_budget * params.Mbps_TO_Bps, update_start_idx, update_end_idx)
		else:
			pass
		
		print("kkt--- ", time.time() - self.start_time, " seconds ---")

		# update buffer following kkt output
		for _ in range(params.FPS * params.UPDATE_FREQ):
			self.buffer.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))

		true_bandwidth_budget = self.bw_traces_obj.bw_trace[self.current_bandwidth_idx + 1] * params.SCALE_BW # Mbps
		success_download_rate = 1
		consumed_bandwidth = sum_solution_rate - sum_r0
		if params.USING_RUMA and params.RUMA_SCALABLE_CODING == False:
			locs = np.where(tiles_rate_solution > buffered_tiles_sizes)
			# pdb.set_trace()
			consumed_bandwidth = np.sum(tiles_rate_solution[locs])
		if consumed_bandwidth != 0:
			success_download_rate = min(1, true_bandwidth_budget * params.Mbps_TO_Bps / consumed_bandwidth)
		else:
			print("!!!!!!!!! nothing download !!!")
		# pdb.set_trace()
		if success_download_rate < 1 - 1e-4: # 1e-4 is noise error term
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

				if download_end_bool: # already consumed as much bandwidth as possible
					tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]

				download_size = tiles_rate_solution[frame_idx][x][y][z] - buffered_tiles_sizes[frame_idx][x][y][z]
				if params.USING_RUMA and params.RUMA_SCALABLE_CODING == False:
					download_size = tiles_rate_solution[frame_idx][x][y][z] if download_size != 0 else 0
				new_consumed_bandwidth += download_size
				if new_consumed_bandwidth > true_bandwidth_budget * params.Mbps_TO_Bps: # cannot download more
					new_consumed_bandwidth -= download_size
					tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
					download_end_bool = True
					# break

			success_download_rate = new_consumed_bandwidth / consumed_bandwidth
			# if params.USING_RUMA and params.RUMA_SCALABLE_CODING == False:
			# 	locs = np.where(tiles_rate_solution > buffered_tiles_sizes)
			# 	success_download_rate = new_consumed_bandwidth / consumed_bandwidth
			# 	pdb.set_trace()
			sum_solution_rate = np.sum(tiles_rate_solution)

		print("update buffer--- ", time.time() - self.start_time, " seconds ---")

		for frame_idx in range(update_start_idx, update_end_idx + 1):
			self.buffer[frame_idx - self.current_viewing_frame_idx - 1] = tiles_rate_solution[frame_idx - update_start_idx].copy()
		self.plot_bw_trace.append(true_bandwidth_budget) # Mbps
		self.plot_predicted_bw_trace.append(predicted_bandwidth_budget) # Mbps
		self.buffer_size_trace.append(sum_solution_rate) # byte
		self.delta_buffer_size_trace.append(sum_solution_rate - buffered_tiles_sizes) # byte
		self.success_download_rate_trace.append(success_download_rate)
		self.bw_predict_accuracy_trace.append(predicted_bandwidth_budget - true_bandwidth_budget) # Mbps
		# pdb.set_trace()

		print("update buffer ends--- ", time.time() - self.start_time, " seconds ---")

	def calculate_z(self, viewing_probability, distances, update_start_idx, update_end_idx, evaluation_flag=False):
		'''
			also need self.frame_weights, self.tile_a and self.distance_weight
		'''
		z_weights = []


		if evaluation_flag:
			z_weights.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
			valid_locs = viewing_probability[frame_idx - update_start_idx].nonzero()
			z_weights[frame_idx - update_start_idx][valid_locs] = viewing_probability[frame_idx - update_start_idx][valid_locs] \
																	* params.TILE_SIDE_LEN / 2 / distances[frame_idx - update_start_idx][valid_locs] * 180 / np.pi
			return z_weights

		# frame weight is linear wrt. frame_idx: w_j = a * frame_idx + b
		frame_weight_decrease_speed = 0
		# if update_end_idx != update_start_idx:
		# 	frame_weight_decrease_speed = -(1 - 0.1) / (update_end_idx - update_start_idx)
		frame_weight = 1
		
		for frame_idx in range(update_start_idx, update_end_idx + 1):
			if params.FRAME_WEIGHT_TYPE == 1:
			# maximal frame_weight = 1, minimal frame_weight is 0.1
				frame_weight = frame_weight_decrease_speed * (frame_idx - update_start_idx) + 1
			elif params.FRAME_WEIGHT_TYPE == 0:
				frame_weight = 1
			elif params.FRAME_WEIGHT_TYPE == 2:
				frame_weight = -(10 - 0.1) / (update_end_idx - update_start_idx) * (frame_idx - update_start_idx) + 10
			elif params.FRAME_WEIGHT_TYPE == 3: # based on fov prediction accuracy: overlap ratio
				frame_weight = np.mean(self.overlap_ratio_history[frame_idx - update_start_idx])
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


			z_weights.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
			valid_locs = viewing_probability[frame_idx - update_start_idx].nonzero()
			z_weights[frame_idx - update_start_idx][valid_locs] = frame_weight * viewing_probability[frame_idx - update_start_idx][valid_locs] \
																	* params.TILE_SIDE_LEN / 2 / distances[frame_idx - update_start_idx][valid_locs] * 180 / np.pi
																	
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

	def calculate_probability_to_be_viewed(self, viewpoints, update_start_idx, update_end_idx):
		# probability can be represented by overlap ratio

		# HPR
		viewing_probability = []
		distances = []

		for frame_idx in range(update_start_idx, update_end_idx + 1):
			viewing_probability.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
			distances.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
			tile_center_points = []
			viewpoint = {"x":0, 'y':0, "z":0, "pitch":0, "yaw":0, "roll":0}
			for key in viewpoint.keys():
				viewpoint[key] = viewpoints[key][frame_idx - update_start_idx]

			if frame_idx < params.BUFFER_LENGTH:
				continue
			valid_tiles = self.valid_tiles_obj.valid_tiles[(frame_idx - params.BUFFER_LENGTH) % params.NUM_FRAMES] # cubic array denotes whether a tile is empty or not

			### fixed / constant obj ############
			# valid_tiles = self.valid_tiles_obj.valid_tiles[0]
			#####################################

			tile_xs, tile_ys, tile_zs = valid_tiles.nonzero()
			for point_idx in range(len(tile_xs)):
				tile_center_coordinate = self.valid_tiles_obj.convert_pointIdx_to_coordinate(tile_xs[point_idx], tile_ys[point_idx], tile_zs[point_idx])
				tile_center_points.append(tile_center_coordinate)

			# modify object coordinate: origin at obj's bottom center
			tile_center_points = self.valid_tiles_obj.change_tile_coordinates_origin(self.origin, tile_center_points)

			# mirror x and z axis to let obj face the user start view orientation
			tile_center_points = np.array(tile_center_points)
			tile_center_points[:, 0] = -tile_center_points[:, 0]
			tile_center_points[:, 2] = -tile_center_points[:, 2]

			viewpoint_position = np.array([viewpoint["x"], viewpoint["y"], viewpoint["z"]])
			viewpoint_position = np.expand_dims(viewpoint_position, axis=0)

			true_viewpoint = self.fov_traces_obj.fov_traces[frame_idx]
			true_position = np.array([true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["x"]], true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["y"]], true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["z"]]])
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
			flippedPoints = HPR_obj.sphericalFlip(viewpoint_position, math.pi) # Reflect the point cloud about a sphere centered at viewpoint_position
			myHull = HPR_obj.convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud

			true_flippedPoints = HPR_obj.sphericalFlip(true_position, math.pi) # Reflect the point cloud about a sphere centered at viewpoint_position
			true_myHull = HPR_obj.convexHull(true_flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud

			# fig = plt.figure(figsize = plt.figaspect(0.5))
			# plt.title('Test Case With A Sphere (Left) and Visible Sphere Viewed From Well Above (Right)')
			# ax = fig.add_subplot(1,2,1, projection = '3d')
			# ax.scatter(flippedPoints[:, 0], flippedPoints[:, 1], flippedPoints[:, 2], c='r', marker='^') # Plot all points
			# ax.set_xlabel('X Axis')
			# ax.set_ylabel('Y Axis')
			# ax.set_zlabel('Z Axis')
			# plt.show()

			# HPR_obj.plot(visible_hull_points=myHull)
			# pdb.set_trace()

			### to do: use gradient descent to optimize radius of HPR ####

			###############################################################

			############ check which visible points are within fov #############
			predicted_visible_tiles_set = set()
			for vertex in myHull.vertices[:-1]:
				vertex_coordinate = np.array([tile_center_points[vertex, 0], tile_center_points[vertex, 1], tile_center_points[vertex, 2]])
				vector_from_viewpoint_to_tilecenter = vertex_coordinate - viewpoint_position
				pitch = viewpoint["pitch"] * np.pi / 180
				yaw = viewpoint["yaw"] * np.pi / 180
				viewing_ray_unit_vector = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
				intersection_angle = np.arccos(np.dot(vector_from_viewpoint_to_tilecenter, viewing_ray_unit_vector) / np.linalg.norm(vector_from_viewpoint_to_tilecenter))
				if intersection_angle <= params.FOV_DEGREE_SPAN:
					# viewable => viewing probability = 1
					viewable_tile_idx = (tile_xs[vertex], tile_ys[vertex], tile_zs[vertex]) # position among all tiles
					# as long as the tile is visiblle, the viewing probability is 1 (which means the overlap ratio is 100%)
					viewing_probability[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = 1
					distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)

					predicted_visible_tiles_set.add(viewable_tile_idx)

			true_visible_tiles_set = set()
			for vertex in true_myHull.vertices[:-1]:
				vertex_coordinate = np.array([tile_center_points[vertex, 0], tile_center_points[vertex, 1], tile_center_points[vertex, 2]])
				vector_from_viewpoint_to_tilecenter = vertex_coordinate - true_position
				pitch = true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["pitch"]] * np.pi / 180
				yaw = true_viewpoint[params.MAP_6DOF_TO_HMD_DATA["yaw"]] * np.pi / 180
				viewing_ray_unit_vector = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
				intersection_angle = np.arccos(np.dot(vector_from_viewpoint_to_tilecenter, viewing_ray_unit_vector) / np.linalg.norm(vector_from_viewpoint_to_tilecenter))
				if intersection_angle <= params.FOV_DEGREE_SPAN:
					# viewable => viewing probability = 1
					viewable_tile_idx = (tile_xs[vertex], tile_ys[vertex], tile_zs[vertex]) # position among all tiles
					# as long as the tile is visiblle, the viewing probability is 1 (which means the overlap ratio is 100%)
					# viewing_probability[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = 1
					# distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)

					true_visible_tiles_set.add(viewable_tile_idx)

			########################################################################

			# update overlap_ratio history
			overlap_tiles_set = true_visible_tiles_set.intersection(predicted_visible_tiles_set)
			overlap_ratio = len(overlap_tiles_set) / len(true_visible_tiles_set)
			self.overlap_ratio_history[frame_idx - update_start_idx].append(overlap_ratio)
			self.overlap_ratio_history[frame_idx - update_start_idx].pop(0)
			# if overlap_ratio < 1:
			# 	pdb.set_trace()

		return viewing_probability, distances


	def calculate_distance(self, point1, point2):
		distance = np.linalg.norm(point1 - point2)
		# return 0
		return distance

	def predict_viewpoint(self, predict_start_idx, predict_end_idx):
		predicted_viewpoints = self.fov_traces_obj.predict_6dof(self.current_viewing_frame_idx, predict_start_idx, predict_end_idx, self.history_viewpoints)
		return predicted_viewpoints

	def predict_bandwidth(self):
		bandwidth = self.bw_traces_obj.predict_bw(self.current_bandwidth_idx, self.history_bandwidths)
		return bandwidth

	def calculate_unweighted_tile_quality(self, rate, distance, a, b):
		if rate == 0:
			return 0
		tmp = a * rate + b
		assert (tmp > 0), "!!!! RUMA->calculate_unweighted_tile_quality->rate is too small !!!!!!!!!"
		return np.log(distance / params.TILE_SIDE_LEN * np.power(tmp, 0.5) / 180 * np.pi)

	def RUMA(self, distances, z_weights, bandwidth_budget, update_start_idx, update_end_idx):
		num_frames_to_update = update_end_idx - update_start_idx + 1

		buffered_tiles_sizes = self.buffer.copy()
		for i in range(params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS):
			buffered_tiles_sizes.pop(0)
		for i in range(params.UPDATE_FREQ * params.FPS):
			buffered_tiles_sizes.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
		buffered_tiles_sizes = np.array(buffered_tiles_sizes)

		tiles_rate_solution = buffered_tiles_sizes.copy()

		z_weight_locations = []
		nonzero_zWeight_locs = np.where((z_weights != 0) & (buffered_tiles_sizes != self.max_tile_sizes_frames))
		nonzero_zWeight_frame_idx = nonzero_zWeight_locs[0]
		nonzero_zWeight_x = nonzero_zWeight_locs[1]
		nonzero_zWeight_y = nonzero_zWeight_locs[2]
		nonzero_zWeight_z = nonzero_zWeight_locs[3]

		r0 = np.zeros((num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))
		r0[nonzero_zWeight_locs] = np.maximum(buffered_tiles_sizes[nonzero_zWeight_locs], self.min_rates_frames[nonzero_zWeight_locs])

		# tiles_rate_solution[nonzero_zWeight_locs] = np.maximum(buffered_tiles_sizes[nonzero_zWeight_locs], self.min_rates_frames[nonzero_zWeight_locs])
		tiles_rate_solution[nonzero_zWeight_locs] = buffered_tiles_sizes[nonzero_zWeight_locs].copy()

		utility_rate_slopes = np.zeros((num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))
		next_version_rates = np.zeros((num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))

		for nonzero_zWeight_idx in range(len(nonzero_zWeight_frame_idx)):
			frame_idx = nonzero_zWeight_frame_idx[nonzero_zWeight_idx]
			x = nonzero_zWeight_x[nonzero_zWeight_idx]
			y = nonzero_zWeight_y[nonzero_zWeight_idx]
			z = nonzero_zWeight_z[nonzero_zWeight_idx]
			z_weight_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z})
			current_rate = tiles_rate_solution[frame_idx][x][y][z]
			current_version = self.decide_rate_version(current_rate, self.rate_versions[0][x][y][z], self.rate_versions[1][x][y][z], self.rate_versions[2][x][y][z])
			if current_version == params.NUM_RATE_VERSIONS: # max rate version
				utility_rate_slopes[frame_idx][x][y][z] = 0
				continue
			next_version_rate = self.rate_versions[params.NUM_RATE_VERSIONS - (current_version + 1)][x][y][z]
			next_version_rates[frame_idx][x][y][z] = next_version_rate
			assert (next_version_rate - current_rate > 0), "!!! same rate between 2 levels (before loop) !!!!!"
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

		if consumed_bandwidth >= total_size_constraint: # total budget cannot satisfy all tiles with lowest rate version
			print("!!!! total budget cannot satisfy all tiles with lowest rate version !!!!!!!")
			return tiles_rate_solution, buffered_tiles_sizes, np.sum(tiles_rate_solution), np.sum(buffered_tiles_sizes), sorted_z_weights

		while consumed_bandwidth < total_size_constraint:
			max_slope_frame_idx, max_slope_x, max_slope_y, max_slope_z = np.unravel_index(np.argmax(utility_rate_slopes, axis=None), utility_rate_slopes.shape)
			max_slope = utility_rate_slopes[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z]
			if max_slope == 0:
				break
			current_rate = tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z]
			next_version_rate = next_version_rates[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z]
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

			tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] = next_version_rate

			if params.RUMA_SCALABLE_CODING:
				consumed_bandwidth += (next_version_rate - current_rate)
			else:
				# locs = np.where(tiles_rate_solution > buffered_tiles_sizes)
				# consumed_bandwidth = np.sum(tiles_rate_solution[locs])
				# consumed_bandwidth += next_version_rate
				if current_rate == buffered_tiles_sizes[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z]:
					consumed_bandwidth += next_version_rate
				else:
					consumed_bandwidth += (next_version_rate - current_rate)

			if consumed_bandwidth > total_size_constraint:
				tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] = current_rate
				break

			if next_version_rate == self.rate_versions[0][max_slope_x][max_slope_y][max_slope_z]:
				utility_rate_slopes[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] = 0
			else:
				current_version = self.decide_rate_version(next_version_rate, self.rate_versions[0][max_slope_x][max_slope_y][max_slope_z], self.rate_versions[1][max_slope_x][max_slope_y][max_slope_z], self.rate_versions[2][max_slope_x][max_slope_y][max_slope_z])
				next_version_rate = self.rate_versions[params.NUM_RATE_VERSIONS - (current_version + 1)][max_slope_x][max_slope_y][max_slope_z]
				next_version_rates[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] = next_version_rate
				assert (next_version_rate - tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] > 0), "!!! same rate between 2 levels (in loop) !!!!!"
				utility_rate_slopes[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] = z_weights[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z] * 2 * \
													  (self.calculate_unweighted_tile_quality(next_version_rate, distances[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], self.tile_a[max_slope_x][max_slope_y][max_slope_z], self.tile_b[max_slope_x][max_slope_y][max_slope_z]) \
													 - self.calculate_unweighted_tile_quality(tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], distances[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z], self.tile_a[max_slope_x][max_slope_y][max_slope_z], self.tile_b[max_slope_x][max_slope_y][max_slope_z])) \
													 / (next_version_rate - tiles_rate_solution[max_slope_frame_idx][max_slope_x][max_slope_y][max_slope_z])



		return tiles_rate_solution, buffered_tiles_sizes, np.sum(tiles_rate_solution), np.sum(buffered_tiles_sizes), sorted_z_weights

	def kkt(self, z_weights, bandwidth_budget, update_start_idx, update_end_idx):
		##################### get v1 and v2 for each tile: ################################
		# v1 = z_weight / r0; v2 = z_weight / params.MAX_TILE_SIZE.
		num_frames_to_update = update_end_idx - update_start_idx + 1
		# pdb.set_trace()

		# tiles' byte size that are already in buffer
		buffered_tiles_sizes = self.buffer.copy()
		for i in range(params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS):
			buffered_tiles_sizes.pop(0)
		for i in range(params.UPDATE_FREQ * params.FPS):
			buffered_tiles_sizes.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
		buffered_tiles_sizes = np.array(buffered_tiles_sizes)

		# each tile has a 4 tuple location: (frame_idx, x, y, z)
		# tiles_rate_solution = params.MAX_TILE_SIZE * np.ones((num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))
		tiles_rate_solution = buffered_tiles_sizes.copy()

		# each tile_value has a 5 tuple location: (frame_idx, x, y, z, value_idx)
		tiles_values = np.zeros((2, num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))

		# locations of all tiles:
		# for sorting purpose later
		# v1 at value_idx=1; v2 at value_idx=0
		tiles_values_locations = []

		z_weight_locations = []

		# typeIII_tiles_set = set()

		# sum_typeIII_z_weight = 0

		nonzero_zWeight_locs = np.where((z_weights != 0) & (buffered_tiles_sizes != self.max_tile_sizes_frames))
		tiles_rate_solution[nonzero_zWeight_locs] = self.max_tile_sizes_frames[nonzero_zWeight_locs]

		nonzero_zWeight_frame_idx = nonzero_zWeight_locs[0]
		nonzero_zWeight_x = nonzero_zWeight_locs[1]
		nonzero_zWeight_y = nonzero_zWeight_locs[2]
		nonzero_zWeight_z = nonzero_zWeight_locs[3]

		# pdb.set_trace()

		tiles_values[0][nonzero_zWeight_locs] = z_weights[nonzero_zWeight_locs] / (self.max_tile_sizes_frames[nonzero_zWeight_locs] + self.tile_b_frames[nonzero_zWeight_locs] / self.tile_a_frames[nonzero_zWeight_locs])

		# nonzero_zWeight_nonzero_r0_locs = np.where((z_weights != 0) & (buffered_tiles_sizes != self.max_tile_sizes) & (buffered_tiles_sizes != 0))
		r0 = np.zeros((num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))
		r0[nonzero_zWeight_locs] = np.maximum(buffered_tiles_sizes[nonzero_zWeight_locs], self.min_rates_frames[nonzero_zWeight_locs])
		tiles_values[1][nonzero_zWeight_locs] = z_weights[nonzero_zWeight_locs] / (r0[nonzero_zWeight_locs] + self.tile_b_frames[nonzero_zWeight_locs] / self.tile_a_frames[nonzero_zWeight_locs])

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

			tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":0})
			tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":1})

			z_weight_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z})
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
		print("tiles values--- ", time.time() - self.start_time, " seconds ---")

		sorted_z_weights = sorted(z_weight_locations, \
									 key=lambda loc: z_weights[loc["frame_idx"]]\
									 							 [loc["x"]]\
									 							 [loc["y"]]\
									 							 [loc["z"]], reverse=True)

		print("sort z weights--- ", time.time() - self.start_time, " seconds ---")

		# this is total size when lambda is the least positive tile value
		total_size = np.sum(tiles_rate_solution)

		# final total_size should be equal to total_size_constraint
		total_size_constraint = bandwidth_budget + np.sum(buffered_tiles_sizes)

		if total_size <= total_size_constraint:
			print("lambda is the minimal positive tile value!", total_size, total_size_constraint)
			return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes), sorted_z_weights

		# get sorted locations of all tiles' values (ascending order)
		# O(n*log(n)), where n is # of visible tiles
		# pdb.set_trace()
		sorted_tiles_values = sorted(tiles_values_locations, \
									 key=lambda loc: tiles_values[loc["value_idx"]]\
									 							 [loc["frame_idx"]]\
									 							 [loc["x"]]\
									 							 [loc["y"]]\
									 							 [loc["z"]])
		print("sort tiles values--- ", time.time() - self.start_time, " seconds ---")
		# pdb.set_trace()
		# compute total size when lagrange_lambda is largest finite tile value (v1)
		# if total_size > total_size_constraint, then we know closed-form solution
		# otherwise, we need to divide-and-conquer with O(n)

		# tiles_rate_solution *= 0

		# # r* = Rmax
		# typeI_tiles_set = set()
		# # r* = r0
		# typeII_tiles_set = set()
		# r* = z_weight / lambda
		typeIII_tiles_set = set()
		visited_typeI_or_II_tiles_set = set()

		left_idx = 0
		right_idx = len(sorted_tiles_values) - 1 # last value

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
		tile_loc_tuple = (frame_idx, tile_value_loc["x"], tile_value_loc["y"], tile_value_loc["z"])
		if value_idx == 0: # type I
			visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
			tiles_rate_solution[frame_idx][x][y][z] = self.max_tile_sizes[x][y][z]
		else: # type II
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
			tile_loc_tuple = (frame_idx, tile_value_loc["x"], tile_value_loc["y"], tile_value_loc["z"])
			# v2's index is 0, v1's index is 1
			if value_idx == 1: # type II
				visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
				tiles_rate_solution[frame_idx][x][y][z] = r0[frame_idx][x][y][z]
			else:
				if tile_loc_tuple not in visited_typeI_or_II_tiles_set: # type III
					typeIII_tiles_set.add(tile_loc_tuple)
					tiles_rate_solution[frame_idx][x][y][z] = z_weights[frame_idx][x][y][z] / lagrange_lambda

			# print(visited_typeI_or_II_tiles_set)
			# print(typeIII_tiles_set)
			# pdb.set_trace()
		total_size = np.sum(tiles_rate_solution)

		# pdb.set_trace()

		if total_size >= total_size_constraint: # total budget cannot satisfy all tiles with lowest rate version
			print("!!!! total budget cannot satisfy all tiles with lowest rate version !!!!!!!")

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
				tiles_rate_solution = self.round_tile_size(tiles_rate_solution.copy(), z_weights, total_size_constraint, z_weight_locations, sorted_z_weights)
			##################

			total_size = np.sum(tiles_rate_solution)
			return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes), sorted_z_weights

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
			tile_loc_tuple = (frame_idx, tile_value_loc["x"], tile_value_loc["y"], tile_value_loc["z"])
			if tile_loc_tuple in typeIII_tiles_set:
				typeIII_tiles_set.remove(tile_loc_tuple)
			if value_idx == 0: # type I
				visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
				tiles_rate_solution[frame_idx][x][y][z] = self.max_tile_sizes[x][y][z]
			else: # type II
				visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
				tiles_rate_solution[frame_idx][x][y][z] = r0[frame_idx][x][y][z]
			###########################################################

			if prev_lambda_at_right:
				for value_loc_idx in range(middle_idx + 1, right_idx + 1):
					tile_value_loc = sorted_tiles_values[value_loc_idx]
					value_idx = tile_value_loc["value_idx"]
					frame_idx = tile_value_loc["frame_idx"]
					x = tile_value_loc["x"]
					y = tile_value_loc["y"]
					z = tile_value_loc["z"]
					tile_loc_tuple = (frame_idx, tile_value_loc["x"], tile_value_loc["y"], tile_value_loc["z"])
					# v2's index is 0, v1's index is 1
					if value_idx == 0: # type I
						if tile_loc_tuple in typeIII_tiles_set:
							typeIII_tiles_set.remove(tile_loc_tuple)
						visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
						tiles_rate_solution[frame_idx][x][y][z] = self.max_tile_sizes[x][y][z]
					else:
						if tile_loc_tuple not in visited_typeI_or_II_tiles_set: # type III
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
					tile_loc_tuple = (frame_idx, tile_value_loc["x"], tile_value_loc["y"], tile_value_loc["z"])
					# v2's index is 0, v1's index is 1
					if value_idx == 1: # type II
						if tile_loc_tuple in typeIII_tiles_set:
							typeIII_tiles_set.remove(tile_loc_tuple)
						visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
						tiles_rate_solution[frame_idx][x][y][z] = r0[frame_idx][x][y][z]
					else:
						if tile_loc_tuple not in visited_typeI_or_II_tiles_set: # type III
							typeIII_tiles_set.add(tile_loc_tuple)
							# tiles_rate_solution[frame_idx][x][y][z] = z_weights[frame_idx][x][y][z] / lagrange_lambda

			######## update tile size for all typeIII tiles, due to the new lambda at middle_idx #######
			for tile_loc_tuple in typeIII_tiles_set:
				frame_idx = tile_loc_tuple[0]
				x = tile_loc_tuple[1]
				y = tile_loc_tuple[2]
				z = tile_loc_tuple[3]
				z_weight = z_weights[frame_idx][x][y][z]

				tiles_rate_solution[frame_idx][x][y][z] = z_weight / lagrange_lambda - self.tile_b[x][y][z] / self.tile_a[x][y][z]
			##############################################################################################

			total_size = np.sum(tiles_rate_solution)

			if total_size > total_size_constraint: # lambda should be bigger
				left_idx = middle_idx
				prev_lambda_at_right = False

			elif total_size == total_size_constraint:
				#### quantize ####
				if params.QUANTIZE_TILE_SIZE:
					tiles_rate_solution = self.quantize_tile_size(tiles_rate_solution.copy())
					total_size = np.sum(tiles_rate_solution)
				##################

				#### quantize ####
				if params.ROUND_TILE_SIZE:
					tiles_rate_solution = self.round_tile_size(tiles_rate_solution.copy(), z_weights, total_size_constraint, z_weight_locations, sorted_z_weights)
				##################
				return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes), sorted_z_weights
			else:
				right_idx = middle_idx
				prev_lambda_at_right = True

			middle_idx = (left_idx + right_idx) // 2

		# lambda is between (left_idx, right_idx)
		assert (tiles_values[sorted_tiles_values[left_idx]["value_idx"]][sorted_tiles_values[left_idx]["frame_idx"]][sorted_tiles_values[left_idx]["x"]][sorted_tiles_values[left_idx]["y"]][sorted_tiles_values[left_idx]["z"]] != \
				tiles_values[sorted_tiles_values[right_idx]["value_idx"]][sorted_tiles_values[right_idx]["frame_idx"]][sorted_tiles_values[right_idx]["x"]][sorted_tiles_values[right_idx]["y"]][sorted_tiles_values[right_idx]["z"]]), \
				"!!!!!!!!!!!! left tile value = right tile value !!!!!!"

		if prev_lambda_at_right: # lambda should be between left_idx and previous_middle_idx
			# update previous_middle_idx (right_idx)
			pivot_loc = sorted_tiles_values[right_idx]
			pivot_value_idx = pivot_loc["value_idx"]
			pivot_frame_idx = pivot_loc["frame_idx"]
			pivot_x = pivot_loc["x"]
			pivot_y = pivot_loc["y"]
			pivot_z = pivot_loc["z"]
			pivot_loc_tuple = (pivot_frame_idx, pivot_x, pivot_y, pivot_z)

			if pivot_value_idx == 1: # typeII to typeIII
				visited_typeI_or_II_tiles_set.remove(pivot_loc_tuple)
				typeIII_tiles_set.add(pivot_loc_tuple)
		else: # lambda should be between previous_middle_idx and right_idx
			# update previous_middle_idx (right_idx)
			pivot_loc = sorted_tiles_values[left_idx]
			pivot_value_idx = pivot_loc["value_idx"]
			pivot_frame_idx = pivot_loc["frame_idx"]
			pivot_x = pivot_loc["x"]
			pivot_y = pivot_loc["y"]
			pivot_z = pivot_loc["z"]
			pivot_loc_tuple = (pivot_frame_idx, pivot_x, pivot_y, pivot_z)

			if pivot_value_idx == 0: # typeI to typeIII
				visited_typeI_or_II_tiles_set.remove(pivot_loc_tuple)
				typeIII_tiles_set.add(pivot_loc_tuple)


		total_size_of_typeI_and_II_tiles = total_size
		sum_typeIII_z_weight = 0
		sum_bk_over_ak = 0
		for tile_loc_tuple in typeIII_tiles_set:
			frame_idx = tile_loc_tuple[0]
			x = tile_loc_tuple[1]
			y = tile_loc_tuple[2]
			z = tile_loc_tuple[3]
			z_weight = z_weights[frame_idx][x][y][z]
			sum_typeIII_z_weight += z_weight
			sum_bk_over_ak += self.tile_b[x][y][z] / self.tile_a[x][y][z]
			total_size_of_typeI_and_II_tiles -= tiles_rate_solution[frame_idx][x][y][z]

		byte_size_constraint_of_typeIII_tiles = total_size_constraint - total_size_of_typeI_and_II_tiles
		lagrange_lambda = sum_typeIII_z_weight / (byte_size_constraint_of_typeIII_tiles + sum_bk_over_ak)

		# lambda is between (left_idx, right_idx)
		print("left_idx, right_idx: ", left_idx, right_idx)
		print("left, lambda, right: ", \
			tiles_values[sorted_tiles_values[left_idx]['value_idx']][sorted_tiles_values[left_idx]['frame_idx']][sorted_tiles_values[left_idx]['x']][sorted_tiles_values[left_idx]['y']][sorted_tiles_values[left_idx]['z']], \
			lagrange_lambda, \
			tiles_values[sorted_tiles_values[right_idx]['value_idx']][sorted_tiles_values[right_idx]['frame_idx']][sorted_tiles_values[right_idx]['x']][sorted_tiles_values[right_idx]['y']][sorted_tiles_values[right_idx]['z']])

		print("search lambda--- ", time.time() - self.start_time, " seconds ---")

		for tile_loc_tuple in typeIII_tiles_set:
			frame_idx = tile_loc_tuple[0]
			x = tile_loc_tuple[1]
			y = tile_loc_tuple[2]
			z = tile_loc_tuple[3]
			z_weight = z_weights[frame_idx][x][y][z]

			tiles_rate_solution[frame_idx][x][y][z] = z_weight / lagrange_lambda - self.tile_b[x][y][z] / self.tile_a[x][y][z]

			if tiles_rate_solution[frame_idx][x][y][z] >= self.max_tile_sizes[x][y][z]:
				print("!!!!!!!! tile size: %f,  MAX size: %f" %(tiles_rate_solution[frame_idx][x][y][z], self.max_tile_sizes[x][y][z]))
				tiles_rate_solution[frame_idx][x][y][z] = self.max_tile_sizes[x][y][z]
				# pdb.set_trace()
			if tiles_rate_solution[frame_idx][x][y][z] <= buffered_tiles_sizes[frame_idx][x][y][z]:
				print("!!!!!!!! tile size: %f,  r0: %f" %(tiles_rate_solution[frame_idx][x][y][z], buffered_tiles_sizes[frame_idx][x][y][z]))
				tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
				pdb.set_trace()
			assert (tiles_rate_solution[frame_idx][x][y][z] <= self.max_tile_sizes[x][y][z]), "!!!!!!! tile size: %f too large, reaches params.MAX_TILE_SIZE %f (during divide-and-conquer) !!!!!!!" %(tiles_rate_solution[frame_idx][x][y][z], self.max_tile_sizes[x][y][z])
			assert (tiles_rate_solution[frame_idx][x][y][z] >= buffered_tiles_sizes[frame_idx][x][y][z]), "!!!!!!! tile size: %f too small, reaches r0: %f (during divide-and-conquer) !!!!!!!" %(tiles_rate_solution[frame_idx][x][y][z], buffered_tiles_sizes[frame_idx][x][y][z])
		
		print("calculate typeIII tiles sizes--- ", time.time() - self.start_time, " seconds ---")

		#### quantize ####
		if params.QUANTIZE_TILE_SIZE:
			tiles_rate_solution = self.quantize_tile_size(tiles_rate_solution.copy())
		##################

		#### quantize ####
		if params.ROUND_TILE_SIZE:
			tiles_rate_solution = self.round_tile_size(tiles_rate_solution.copy(), z_weights, total_size_constraint, z_weight_locations, sorted_z_weights)
		##################

		print("rounding--- ", time.time() - self.start_time, " seconds ---")

		total_size = np.sum(tiles_rate_solution)

		print("calculate summation--- ", time.time() - self.start_time, " seconds ---")

		assert (total_size - total_size_constraint < 1e-4), "!!!! total size != total bw budget %f, %f!!!!!" %(total_size, total_size_constraint)
		return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes), sorted_z_weights

	def hybrid_tiling(self, z_weights, bandwidth_budget, update_start_idx, update_end_idx):
		##################### get v1 and v2 for each tile: ################################
		# v1 = z_weight / r0; v2 = z_weight / params.MAX_TILE_SIZE.
		num_frames_to_update = update_end_idx - update_start_idx + 1
		# pdb.set_trace()

		# tiles' byte size that are already in buffer
		buffered_tiles_sizes = self.buffer.copy()
		for i in range(params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS):
			buffered_tiles_sizes.pop(0)
		for i in range(params.UPDATE_FREQ * params.FPS):
			buffered_tiles_sizes.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))
		buffered_tiles_sizes = np.array(buffered_tiles_sizes)

		# each tile has a 4 tuple location: (frame_idx, x, y, z)
		tiles_rate_solution = buffered_tiles_sizes.copy()

		# each tile_value has a 5 tuple location: (frame_idx, x, y, z, value_idx)
		tiles_values = np.zeros((num_frames_to_update, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, 2))

		# locations of all tiles:
		# for sorting purpose later
		# v1 at value_idx=1; v2 at value_idx=0
		tiles_values_locations = []

		# typeIII_tiles_set = set()

		# sum_typeIII_z_weight = 0

		frames, tile_xs, tile_ys, tile_zs = z_weights.nonzero()
		num_utility_tiles = len(frames)
		for point_idx in range(num_utility_tiles):
			tiles_values_locations.append({"frame_idx":frames[point_idx], "x":tile_xs[point_idx], "y":tile_ys[point_idx], "z":tile_zs[point_idx]})

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
					tiles_rate_solution[frame_idx][x][y][z] = params.BYTE_SIZES[0]
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
					tiles_rate_solution[frame_idx][x][y][z] = params.BYTE_SIZES[1]
					total_size -= (params.BYTE_SIZES[2] - params.BYTE_SIZES[1])
					break
		else:
			total_size = params.BYTE_SIZES[2] * num_utility_tiles
			tiles_rate_solution[z_weights.nonzero()] = params.BYTE_SIZES[2]


		return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes), sorted_tiles_values

	def quantize_tile_size(self, tiles_rate_solution):
		tiles_rate_solution[np.where(tiles_rate_solution < params.BYTE_SIZES[0])] = 0
		tiles_rate_solution[np.where((tiles_rate_solution < params.BYTE_SIZES[1]) & (tiles_rate_solution >= params.BYTE_SIZES[0]))] = params.BYTE_SIZES[0]
		tiles_rate_solution[np.where((tiles_rate_solution < params.BYTE_SIZES[2]) & (tiles_rate_solution >= params.BYTE_SIZES[1]))] = params.BYTE_SIZES[1]
		return tiles_rate_solution

	def round_tile_size(self, tiles_rate_solution, z_weights, total_size_constraint, z_weight_locations, sorted_z_weights):
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
			if current_rate == self.rate_versions[0][x][y][z] or current_rate == 0 or current_rate == self.rate_versions[1][x][y][z] or current_rate == self.rate_versions[2][x][y][z]:
				z_weight_idx += 1
				continue
			current_version = self.decide_rate_version(current_rate, self.rate_versions[0][x][y][z], self.rate_versions[1][x][y][z], self.rate_versions[2][x][y][z]) # 0,1,2
			bandwidth_needed = self.rate_versions[params.NUM_RATE_VERSIONS - (current_version + 1)][x][y][z] - current_rate

			# if abs(extra_budget - 85.85051597904624) < 1e-4:
			# 	pdb.set_trace()

			if extra_budget >= bandwidth_needed:
				# round up high-weight tile's rate
				tiles_rate_solution[frame_idx][x][y][z] = self.rate_versions[params.NUM_RATE_VERSIONS - (current_version + 1)][x][y][z]
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
				low_weight_tile_rate = tiles_rate_solution[low_frame_idx][low_x][low_y][low_z]
				if low_weight_tile_rate == self.rate_versions[0][low_x][low_y][low_z] or low_weight_tile_rate == 0 or low_weight_tile_rate == self.rate_versions[1][low_x][low_y][low_z] or low_weight_tile_rate == self.rate_versions[2][low_x][low_y][low_z]:
					tmp_lowest_z_weight_idx -= 1
					continue
				low_current_version = self.decide_rate_version(low_weight_tile_rate, self.rate_versions[0][low_x][low_y][low_z], self.rate_versions[1][low_x][low_y][low_z], self.rate_versions[2][low_x][low_y][low_z]) # 0,1,2
				# # quantize to this low_current_version
				# tiles_rate_solution[low_frame_idx][low_x][low_y][low_z] = params.MAP_VERSION_TO_SIZE[low_current_version]
				if low_current_version == 0:
					reduced_rates += low_weight_tile_rate
				else:
					reduced_rates += (low_weight_tile_rate - self.rate_versions[params.NUM_RATE_VERSIONS - low_current_version][low_x][low_y][low_z])

				if reduced_rates + extra_budget >= bandwidth_needed:
					# if abs(extra_budget - 85.85051597904624) < 1e-4:
					# 	pdb.set_trace()
					break

				tmp_lowest_z_weight_idx -= 1

			if reduced_rates + extra_budget >= bandwidth_needed:
				extra_budget = reduced_rates + extra_budget - bandwidth_needed

				# round up high-weight tile's rate
				tiles_rate_solution[frame_idx][x][y][z] = self.rate_versions[params.NUM_RATE_VERSIONS - (current_version + 1)][x][y][z]

				# for low tiles, quantize to low_current_version
				new_lowest_z_weight_idx = tmp_lowest_z_weight_idx - 1
				while tmp_lowest_z_weight_idx <= lowest_z_weight_idx:
					low_z_weight_loc = sorted_z_weights[tmp_lowest_z_weight_idx]
					low_frame_idx = low_z_weight_loc["frame_idx"]
					low_x = low_z_weight_loc["x"]
					low_y = low_z_weight_loc["y"]
					low_z = low_z_weight_loc["z"]
					low_weight_tile_rate = tiles_rate_solution[low_frame_idx][low_x][low_y][low_z]
					if low_weight_tile_rate == self.rate_versions[0][low_x][low_y][low_z] or low_weight_tile_rate == 0 or low_weight_tile_rate == self.rate_versions[1][low_x][low_y][low_z] or low_weight_tile_rate == self.rate_versions[2][low_x][low_y][low_z]:
						tmp_lowest_z_weight_idx += 1
						continue
					low_current_version = self.decide_rate_version(low_weight_tile_rate, self.rate_versions[0][low_x][low_y][low_z], self.rate_versions[1][low_x][low_y][low_z], self.rate_versions[2][low_x][low_y][low_z]) # 0,1,2
					if low_current_version == 0:
						tiles_rate_solution[low_frame_idx][low_x][low_y][low_z] = 0
					else:
						tiles_rate_solution[low_frame_idx][low_x][low_y][low_z] = self.rate_versions[params.NUM_RATE_VERSIONS - low_current_version][low_x][low_y][low_z]

					tmp_lowest_z_weight_idx += 1

				lowest_z_weight_idx = new_lowest_z_weight_idx

				# pdb.set_trace()
				# print(extra_budget)

			else: # cannot round up, should round down instead
				if current_version == 0:
					tiles_rate_solution[frame_idx][x][y][z] = 0
					extra_budget += current_rate
				else:
					tiles_rate_solution[frame_idx][x][y][z] = self.rate_versions[params.NUM_RATE_VERSIONS - current_version][x][y][z]
					extra_budget += (current_rate - self.rate_versions[params.NUM_RATE_VERSIONS - current_version][x][y][z])

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