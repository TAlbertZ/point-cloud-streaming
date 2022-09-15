'''
	simulate KKT, an analytical approach to periodically updating buffer.

	observe:
		- rate selection every 1s
		- rendered quality
		- prediction accuracy (BW & FoV) of every frame in buffer
'''

import numpy as np
import pickle as pk
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
import scipy as sp
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from statistics import harmonic_mean

np.random.seed(7)

FPS = 30
TARGET_LATENCY = 300 # in frame

# Assume frame is independently encoded/decoded
BUFFER_LENGTH = TARGET_LATENCY - 1 # in frame

# cannot update for some contents in front of buffer,
# because cannot they finish updating before being played
TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE = 1 # in second

UPDATE_FREQ = 1 # in second

# assume max tile size is the same across all tiles,
# later should be modified
MAX_TILE_SIZE = 1532 # bytes

# quality parameters a and b
# generated randomly accoding to normal distribution
QUALITY_PARA = []

# each frame in buffer has a weight for update,
# here assume ther're fixed, later should be modified as:
# adapt to dynamic conditions (bandwidth and fov evolution)
FRAME_WEIGHT_IN_BUFFER = []

# 1 / (1 + exp(-c * distance)),
# distance is between viewpoint and each tile center
DISTANCE_WEIGHT = 1

TILE_LEVEL_FROM_ROOT = 4
NUM_TILES_PER_FRAME = 8**TILE_LEVEL_FROM_ROOT

# side length in num_tiles
NUM_TILES_PER_SIDE_IN_A_FRAME = 2**TILE_LEVEL_FROM_ROOT

NUM_FRAMES = 300

# video: H1 user: P01_V1
NUM_FRAMES_VIEWED = 549 # watching looply
NUM_UPDATES = NUM_FRAMES_VIEWED // FPS
# NUM_UPDATES = 3

VIDEO_NAME = 'longdress'
VALID_TILES_PATH = '../valid_tiles/' + VIDEO_NAME

if VIDEO_NAME == 'longdress':
	fov_traces_file = 'H1_nav.csv'
elif VIDEO_NAME == 'loot':
	fov_traces_file = 'H2_nav.csv'
elif VIDEO_NAME == 'readandblack':
	fov_traces_file = 'H3_nav.csv'
elif VIDEO_NAME == 'soldier':
	fov_traces_file = 'H4_nav.csv'
else:
	pass
FOV_TRACES_PATH = '../fov_traces/6DoF-HMD-UserNavigationData/NavigationData/' + fov_traces_file

BANDWIDTH_TRACES_PATH = '../bw_traces/100ms_loss1.txt'

FOV_PREDICTION_HISTORY_WIN_LENGTH = (BUFFER_LENGTH + UPDATE_FREQ * FPS) // 2 # frame; according to vivo paper, best use half of prediciton window
BW_PREDICTION_HISTORY_WIN_LENGTH = 5 # in second

OBJECT_SIDE_LEN = 1.8 # meter, according to http://plenodb.jpeg.org/pc/8ilabs
TILE_SIDE_LEN = OBJECT_SIDE_LEN / NUM_TILES_PER_SIDE_IN_A_FRAME # meter

FOV_DEGREE_SPAN = np.pi / 2 # 90 degrees, a circle fov

QR_WEIGHTS_PATH_A = '../psnr_weights/a_16x16x16.pkl'
QR_WEIGHTS_PATH_B = '../psnr_weights/b_16x16x16.pkl'

BANDWIDTH_ORACLE_KNOWN = 1

Mbps_TO_Bps = 1e6 / 8

MAP_6DOF_TO_HMD_DATA = {'x':'HMDPX', 'y':'HMDPY', 'z':'HMDPZ', 'pitch':'HMDRX', 'yaw':'HMDRY', 'roll':'HMDRZ'}

MAP_DOF_TO_PLOT_POS = {'x':[0, 0], 'y':[0, 1], 'z':[0, 2], 'pitch':[1, 0], 'yaw':[1, 1], 'roll':[1, 2]}

FRAME_WEIGHT_TYPE = 0

SCALE_BW = 10 # know bw oracle, otherwise not correct

class HiddenPointsRemoval():
	'''
		reference: 
			Katz, Sagi, Ayellet Tal, and Ronen Basri. "Direct Visibility of Point Sets." 2007. 
		github:
			https://github.com/williamsea/Hidden_Points_Removal_HPR/blob/master/MyHPR_HaiTang.py
	'''

	def __init__(self, points):
		self.points = points

	'''
	Function used to Perform Spherical Flip on the Original Point Cloud
	'''
	def sphericalFlip(self, center, param):

		n = len(self.points) # total n points
		points = self.points - np.repeat(center, n, axis = 0) # Move C to the origin
		normPoints = np.linalg.norm(points, axis = 1) # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
		# print(normPoints)
		# print(max(normPoints))
		R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis = 0) # Radius of Sphere
		# print(R)
		# pdb.set_trace()
		R = np.repeat(1.5, n, axis = 0) # Radius of Sphere
		
		flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints).reshape(n,1), len(points[0]), axis = 1), points) 
		flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
		flippedPoints += points 

		return flippedPoints

	'''
	Function used to Obtain the Convex hull
	'''
	def convexHull(self, points):

		points = np.append(points, [[0,0,0]], axis = 0) # All points plus origin
		hull = ConvexHull(points) # Visibal points plus possible origin. Use its vertices property.

		return hull

	def plot(self, visible_hull_points):
		fig = plt.figure(figsize = plt.figaspect(0.5))
		plt.title('Test Case With A Sphere (Left) and Visible Sphere Viewed From Well Above (Right)')

		# First subplot
		ax = fig.add_subplot(1,2,1, projection = '3d')
		ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='r', marker='^') # Plot all points
		ax.set_xlabel('X Axis')
		ax.set_ylabel('Y Axis')
		ax.set_zlabel('Z Axis')

		# Second subplot
		ax = fig.add_subplot(1,2,2, projection = '3d')
		for vertex in visible_hull_points.vertices[:-1]:
			ax.scatter(self.points[vertex, 0], self.points[vertex, 1], self.points[vertex, 2], c='b', marker='o') # Plot visible points
		# ax.set_zlim3d(-1.5, 1.5)
		ax.set_xlabel('X Axis')
		ax.set_ylabel('Y Axis')
		ax.set_zlabel('Z Axis')

		plt.show()

class Buffer():
	'''
		buffer controlling
		include buffer initialization and update
	'''

	def __init__(self, fov_traces_obj, bw_traces_obj, valid_tiles_obj, qr_weights_obj):
		self.buffer_length = BUFFER_LENGTH # frame

		# self.buffer stores 3d numpy arrays,
		# each 3d array represents one frame,
		# which contains byte sizes of all tiles
		self.buffer = []

		self.fov_traces_obj = fov_traces_obj
		self.valid_tiles_obj = valid_tiles_obj
		self.bw_traces_obj = bw_traces_obj
		self.qr_weights_obj = qr_weights_obj

		# a, b and distance_weight are waiting to be fit
		self.tile_a = self.qr_weights_obj.qr_weights["a"] # cubic array for each tile
		self.tile_b = self.qr_weights_obj.qr_weights["b"] # cubic array for each tile

		# sigmoid coefficient c: (1 + exp(-c*d))^(-1)
		self.distance_weight = 1

		# linearly increasing from 0 to 1
		self.frame_weights = None

		# initialize according to fov dataset H1, assume the initial viewpoint is always like this:
		# {x, y, z, roll, yaw, pitch} = {0.05, 1.7868, -1.0947, 6.9163, 350.8206, 359.9912}
		# z-x plane is floor

		# self.history_viewpoints = {"x":[0.05]*FOV_PREDICTION_HISTORY_WIN_LENGTH, 
		# 						   "y":[1.7868]*FOV_PREDICTION_HISTORY_WIN_LENGTH, 
		# 						   "z":[-1.0947]*FOV_PREDICTION_HISTORY_WIN_LENGTH, 
		# 						   "pitch":[6.9163 + 360]*FOV_PREDICTION_HISTORY_WIN_LENGTH, # rotate around x axis
		# 						   "yaw":[350.8206]*FOV_PREDICTION_HISTORY_WIN_LENGTH, # rotate around y axis
		# 						   "roll":[359.9912]*FOV_PREDICTION_HISTORY_WIN_LENGTH} # rotate around z axis

		self.history_viewpoints = {"x":[0.05], 
								   "y":[1.7868], 
								   "z":[-1.0947], 
								   "pitch":[6.9163 + 360], # rotate around x axis
								   "yaw":[350.8206], # rotate around y axis
								   "roll":[359.9912]} # rotate around z axis
		
		# initialize bandwidth history according to '../bw_traces/100ms_loss1'
		self.history_bandwidths = [2.7] * BW_PREDICTION_HISTORY_WIN_LENGTH

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
		for key in self.true_viewpoints.keys():
			for frame_idx in range(self.buffer_length + FPS * UPDATE_FREQ - TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * FPS):
				self.true_viewpoints[key].append([])
				self.fov_predict_accuracy_trace[key].append([])
		self.bw_predict_accuracy_trace = []
		self.success_download_rate_trace = []


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
			self.buffer.append(np.zeros((NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME)))
			# if frame_idx < FPS:
			# 	valid_tiles_coordinates = self.valid_tiles_obj.valid_tiles[frame_idx].nonzero()
			# 	self.buffer[frame_idx][valid_tiles_coordinates] = MAX_TILE_SIZE

	def true_frame_quality(self, viewing_probability, distances):
		tiles_byte_sizes = self.buffer[0] # front of buffer: cubic array
		tile_xs, tile_ys, tile_zs = tiles_byte_sizes.nonzero()
		tile_xs, tile_ys, tile_zs = viewing_probability.nonzero()
		print(len(tile_xs))
		if len(tile_xs) == 0:
			return 0
		frame_quality = 0
		for point_idx in range(len(tile_xs)):
			x = tile_xs[point_idx]
			y = tile_ys[point_idx]
			z = tile_zs[point_idx]

			assert (self.tile_a[x][y][z] > 0), "!!!!! qr weight a non-positive when calculating true frame quality: (%d, %d, %d, %f) !!!!!!" %(x, y, z, self.tile_a[x][y][z])
			tile_psnr = 0
			# if tiles_byte_sizes[x][y][z] < 467:
			# 	tile_psnr = 0
			# elif tiles_byte_sizes[x][y][z] < 1532:
			# 	tile_psnr = self.tile_a[x][y][z] * np.log(467) + self.tile_b[x][y][z]
			# elif tiles_byte_sizes[x][y][z] == 1532:
			# 	tile_psnr = self.tile_a[x][y][z] * np.log(1532) + self.tile_b[x][y][z]
			# tile_psnr = self.tile_a[x][y][z] * np.log(tiles_byte_sizes[x][y][z]) + self.tile_b[x][y][z]
			tile_psnr = self.tile_a[x][y][z] * np.log(MAX_TILE_SIZE) + self.tile_b[x][y][z]
			tile_quality = viewing_probability[x][y][z] * tile_psnr / (1 + np.exp(-self.distance_weight * distances[x][y][z]))
			frame_quality += tile_quality
		return frame_quality / len(tile_xs)

	def emit_buffer(self):
		'''
			emit UPDATE_FREQ*FPS frames from front of buffer;;
			Based on their true viewpoint, calculate their HPR, distance, and quality;
			update pointers: buffer, current_viewing_frame_idx, history_viewpoints, history_bandwidths, current_bandwidth_idx
		'''
		for frame_idx in range(self.current_viewing_frame_idx + 1, self.current_viewing_frame_idx + FPS * UPDATE_FREQ + 1):
			# viewpoint = {"x":[self.fov_traces_obj.fov_traces.at[frame_idx, 'HMDPX']], 'y':[self.fov_traces_obj.fov_traces.at[frame_idx, 'HMDPY']], "z":[self.fov_traces_obj.fov_traces.at[frame_idx, 'HMDPZ']], \
						 # "pitch":[self.fov_traces_obj.fov_traces.at[frame_idx, 'HMDRX']], "yaw":[self.fov_traces_obj.fov_traces.at[frame_idx, 'HMDRY']], "roll":[self.fov_traces_obj.fov_traces.at[frame_idx, 'HMDRZ']]}
			
			viewpoint = {"x":[0.05],
						 "y":[1.7868],
						 "z":[-1.0947],
						 "pitch":[6.9163],
						 "yaw":[350.8206],
						 "roll":[359.9912]}
			viewing_probability, distances = self.calculate_probability_to_be_viewed(viewpoint, frame_idx, frame_idx)

			self.frame_quality.append(self.true_frame_quality(viewing_probability[0], distances[0]))

			# pop processed frame
			self.buffer.pop(0)

			# update history_viewpoints
			for key in self.history_viewpoints.keys():
				if len(self.history_viewpoints[key]) == FOV_PREDICTION_HISTORY_WIN_LENGTH:
					self.history_viewpoints[key].pop(0)
				viewpoint_dof = viewpoint[key][0]
				if key == 'pitch' or key == 'yaw' or key == 'roll':
					if viewpoint_dof < 90: # user always move from 358 to 20
						viewpoint_dof += 360
				self.history_viewpoints[key].append(viewpoint_dof)

		# update current_viewing_frame_idx, history_bandwidths, current_bandwidth_idx
		self.current_viewing_frame_idx = frame_idx
		self.current_bandwidth_idx += 1
		self.history_bandwidths.pop(0)
		self.history_bandwidths.append(self.bw_traces_obj.bw_trace[self.current_bandwidth_idx])

	def update_tile_size_in_buffer(self):
		##################### predict viewpoint #######################
		update_start_idx = self.current_viewing_frame_idx + TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * FPS + 1
		update_end_idx = self.current_viewing_frame_idx + self.buffer_length + FPS * UPDATE_FREQ
		predicted_viewpoints = self.predict_viewpoint(predict_start_idx=update_start_idx, predict_end_idx=update_end_idx)
		for key in self.true_viewpoints.keys():
			for frame_idx in range(update_end_idx - update_start_idx + 1):
				predicted_viewpoint_at_this_dof = predicted_viewpoints[key][frame_idx]
				true_viewpoint_at_this_dof = predicted_viewpoint_at_this_dof
				self.true_viewpoints[key][frame_idx].append(true_viewpoint_at_this_dof)
				if key == 'pitch' or key == 'yaw' or key == 'roll':
					predicted_viewpoint_at_this_dof = predicted_viewpoint_at_this_dof if 360 - predicted_viewpoint_at_this_dof >= predicted_viewpoint_at_this_dof else predicted_viewpoint_at_this_dof - 360
					true_viewpoint_at_this_dof = true_viewpoint_at_this_dof if 360 - true_viewpoint_at_this_dof >= true_viewpoint_at_this_dof else true_viewpoint_at_this_dof - 360
				self.fov_predict_accuracy_trace[key][frame_idx].append(abs(predicted_viewpoint_at_this_dof - true_viewpoint_at_this_dof))
		# pdb.set_trace()
		#################################################################

		viewing_probability, distances = self.calculate_probability_to_be_viewed(predicted_viewpoints, update_start_idx, update_end_idx)
		
		# calculate distance only for viewable valid tiles
		# distances = self.calculate_distance(predicted_viewpoints)

		z_weights = self.calculate_z(viewing_probability, distances, update_start_idx, update_end_idx)

		# predict bandwidth of future 1s
		predicted_bandwidth_budget = self.predict_bandwidth() * SCALE_BW # Mbps

		tiles_rate_solution, buffered_tiles_sizes, sum_solution_rate, sum_r0 = self.kkt(z_weights, predicted_bandwidth_budget * Mbps_TO_Bps, update_start_idx, update_end_idx)
		# update buffer following kkt output
		for _ in range(FPS * UPDATE_FREQ):
			self.buffer.append(np.zeros((NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME)))

		true_bandwidth_budget = self.bw_traces_obj.bw_trace[self.current_bandwidth_idx + 1] * SCALE_BW # Mbps
		success_download_rate = min(1, true_bandwidth_budget * Mbps_TO_Bps / (sum_solution_rate - sum_r0))
		if success_download_rate < 1 - 1e-4: # 1e-4 is noise error term
			tiles_rate_solution = (tiles_rate_solution - buffered_tiles_sizes) * success_download_rate + buffered_tiles_sizes
			sum_solution_rate = np.sum(tiles_rate_solution)
		for frame_idx in range(update_start_idx, update_end_idx + 1):
			self.buffer[frame_idx - self.current_viewing_frame_idx - 1] = tiles_rate_solution[frame_idx - update_start_idx].copy()
		self.plot_bw_trace.append(true_bandwidth_budget) # Mbps
		self.plot_predicted_bw_trace.append(predicted_bandwidth_budget) # Mbps
		self.buffer_size_trace.append(sum_solution_rate) # byte
		self.delta_buffer_size_trace.append(sum_solution_rate - buffered_tiles_sizes) # byte
		self.success_download_rate_trace.append(success_download_rate)
		self.bw_predict_accuracy_trace.append(predicted_bandwidth_budget - true_bandwidth_budget) # Mbps
		# pdb.set_trace()

	def calculate_z(self, viewing_probability, distances, update_start_idx, update_end_idx):
		'''
			also need self.frame_weights, self.tile_a and self.distance_weight
		'''
		z_weights = []
		# frame weight is linear wrt. frame_idx: w_j = a * frame_idx + b
		frame_weight_decrease_speed = -(1 - 0.1) / (update_end_idx - update_start_idx)
		frame_weight = 1
		
		for frame_idx in range(update_start_idx, update_end_idx + 1):
			if FRAME_WEIGHT_TYPE == 1:
			# maximal frame_weight = 1, minimal frame_weight is 0.1
				frame_weight = frame_weight_decrease_speed * (frame_idx - update_start_idx) + 1
			elif FRAME_WEIGHT_TYPE == 0:
				frame_weight = 1
			else:
				frame_weight = -(10 - 0.1) / (update_end_idx - update_start_idx) * (frame_idx - update_start_idx) + 10

			# # weight = 1 only for first 1-s content to be updated: should be more variational
			# if frame_idx - update_start_idx < UPDATE_FREQ * FPS:
			# 	frame_weight = 1
			# else:
			# 	frame_weight = 0

			tile_xs, tile_ys, tile_zs = viewing_probability[frame_idx - update_start_idx].nonzero()
			z_weights.append(np.zeros((NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME)))
			for x in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
				for y in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
					for z in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
						if viewing_probability[frame_idx - update_start_idx][x][y][z] == 0:
							continue

						assert (self.tile_a[x][y][z] > 0), "!!!!! qr weight a non-positive when calculating z: (%d, %d, %d, %f) !!!!!!" %(x, y, z, self.tile_a[x][y][z])
						z_weights[frame_idx - update_start_idx][x][y][z] = frame_weight * viewing_probability[frame_idx - update_start_idx][x][y][z] * self.tile_a[x][y][z] / (1 + np.exp(-self.distance_weight * distances[frame_idx - update_start_idx][x][y][z]))
						assert (z_weights[frame_idx - update_start_idx][x][y][z] >= 0),"!!!!!!!!!!!! Negative weights !!!!!!"
		return z_weights

	def calculate_probability_to_be_viewed(self, viewpoints, update_start_idx, update_end_idx):
		# probability can be represented by overlap ratio

		# HPR
		viewing_probability = []
		distances = []

		for frame_idx in range(update_start_idx, update_end_idx + 1):
			viewing_probability.append(np.zeros((NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME)))
			distances.append(np.zeros((NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME)))
			tile_center_points = []
			viewpoint = {"x":0, 'y':0, "z":0, "pitch":0, "yaw":0, "roll":0}
			for key in viewpoint.keys():
				viewpoint[key] = viewpoints[key][frame_idx - update_start_idx]

			valid_tiles = self.valid_tiles_obj.valid_tiles[frame_idx % NUM_FRAMES] # cubic array denotes whether a tile is empty or not
			# valid_tiles = self.valid_tiles_obj.valid_tiles[0]
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
			for vertex in myHull.vertices[:-1]:
				vertex_coordinate = np.array([tile_center_points[vertex, 0], tile_center_points[vertex, 1], tile_center_points[vertex, 2]])
				vector_from_viewpoint_to_tilecenter = vertex_coordinate - viewpoint_position
				pitch = viewpoint["pitch"] * np.pi / 180
				yaw = viewpoint["yaw"] * np.pi / 180
				viewing_ray_unit_vector = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
				intersection_angle = np.arccos(np.dot(vector_from_viewpoint_to_tilecenter, viewing_ray_unit_vector) / np.linalg.norm(vector_from_viewpoint_to_tilecenter))
				if intersection_angle <= FOV_DEGREE_SPAN:
					# viewable => viewing probability = 1
					viewable_tile_idx = [tile_xs[vertex], tile_ys[vertex], tile_zs[vertex]] # position among all tiles
					# as long as the tile is visiblle, the viewing probability is 1 (which means the overlap ratio is 100%)
					viewing_probability[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = 1
					distances[frame_idx - update_start_idx][viewable_tile_idx[0]][viewable_tile_idx[1]][viewable_tile_idx[2]] = self.calculate_distance(vertex_coordinate, viewpoint_position)

			########################################################################

		return viewing_probability, distances


	def calculate_distance(self, point1, point2):
		distance = np.linalg.norm(point1 - point2)
		return distance

	def predict_viewpoint(self, predict_start_idx, predict_end_idx):
		predicted_viewpoints = self.fov_traces_obj.predict_6dof(self.current_viewing_frame_idx, predict_start_idx, predict_end_idx, self.history_viewpoints)
		return predicted_viewpoints

	def predict_bandwidth(self):
		bandwidth = self.bw_traces_obj.predict_bw(self.current_bandwidth_idx, self.history_bandwidths)
		return bandwidth

	def kkt(self, z_weights, bandwidth_budget, update_start_idx, update_end_idx):
		##################### get v1 and v2 for each tile: ################################
		# v1 = z_weight / r0; v2 = z_weight / MAX_TILE_SIZE.
		num_frames_to_update = update_end_idx - update_start_idx + 1
		# pdb.set_trace()

		# tiles' byte size that are already in buffer
		buffered_tiles_sizes = self.buffer.copy()
		for i in range(TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * FPS):
			buffered_tiles_sizes.pop(0)
		for i in range(UPDATE_FREQ * FPS):
			buffered_tiles_sizes.append(np.zeros((NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME)))
		buffered_tiles_sizes = np.array(buffered_tiles_sizes)

		# each tile has a 4 tuple location: (frame_idx, x, y, z)
		tiles_rate_solution = MAX_TILE_SIZE * np.ones((num_frames_to_update, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME))

		# each tile_value has a 5 tuple location: (frame_idx, x, y, z, value_idx)
		tiles_values = np.zeros((num_frames_to_update, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, 2))

		# locations of all tiles:
		# for sorting purpose later
		# v1 at value_idx=1; v2 at value_idx=0
		tiles_values_locations = []

		# typeIII_tiles_set = set()

		# sum_typeIII_z_weight = 0

		for frame_idx in range(num_frames_to_update):
			for x in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
				for y in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
					for z in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
						z_weight = z_weights[frame_idx][x][y][z] # can be 0

						# byte size that already in buffer for this tile
						r0 = buffered_tiles_sizes[frame_idx][x][y][z] # can be 0

						if z_weight == 0:
							tiles_values[frame_idx][x][y][z][1] = 0

							# if z_weight is 0, optimal size of this tile should be r0 (unchanged)
							tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
						elif r0 == 0:
							tiles_values[frame_idx][x][y][z][1] = float('inf')
							# sum_typeIII_z_weight += z_weight
							# store the tile value location with inf v1
							# typeIII_tiles_set.add((frame_idx, x, y, z))
							tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":0})

						else:
							tiles_values[frame_idx][x][y][z][1] = z_weight / r0

							tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":0})
							tiles_values_locations.append({"frame_idx":frame_idx, "x":x, "y":y, "z":z, "value_idx":1})


						tiles_values[frame_idx][x][y][z][0] = z_weight / MAX_TILE_SIZE
		##########################################################################################################

		# this is total size when lambda is the least positive tile value
		total_size = np.sum(tiles_rate_solution)

		# final total_size should be equal to total_size_constraint
		total_size_constraint = bandwidth_budget + np.sum(buffered_tiles_sizes)

		if total_size <= total_size_constraint:
			print("lambda is the minimal positive tile value!", total_size, total_size_constraint)
			return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes)

		# get sorted locations of all tiles' values (ascending order)
		# O(n*log(n)), where n is # of visible tiles
		sorted_tiles_values = sorted(tiles_values_locations, \
									 key=lambda loc: tiles_values[loc["frame_idx"]]\
									 							 [loc["x"]]\
									 							 [loc["y"]]\
									 							 [loc["z"]]\
									 							 [loc["value_idx"]])

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

		lagrange_lambda = tiles_values[sorted_tiles_values[right_idx]['frame_idx']] \
									  [sorted_tiles_values[right_idx]['x']] \
									  [sorted_tiles_values[right_idx]['y']] \
									  [sorted_tiles_values[right_idx]['z']] \
									  [sorted_tiles_values[right_idx]['value_idx']]

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
			tiles_rate_solution[frame_idx][x][y][z] = MAX_TILE_SIZE
		else: # type II
			visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
			tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
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
				tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
			else:
				if tile_loc_tuple not in visited_typeI_or_II_tiles_set: # type III
					typeIII_tiles_set.add(tile_loc_tuple)
					tiles_rate_solution[frame_idx][x][y][z] = z_weights[frame_idx][x][y][z] / lagrange_lambda

			# print(visited_typeI_or_II_tiles_set)
			# print(typeIII_tiles_set)
			# pdb.set_trace()
		total_size = np.sum(tiles_rate_solution)

		# pdb.set_trace()

		if total_size > total_size_constraint:
			total_size_of_typeI_and_II_tiles = total_size
			sum_typeIII_z_weight = 0
			for tile_loc_tuple in typeIII_tiles_set:
				frame_idx = tile_loc_tuple[0]
				x = tile_loc_tuple[1]
				y = tile_loc_tuple[2]
				z = tile_loc_tuple[3]
				z_weight = z_weights[frame_idx][x][y][z]
				sum_typeIII_z_weight += z_weight
				total_size_of_typeI_and_II_tiles -= tiles_rate_solution[frame_idx][x][y][z]

			byte_size_constraint_of_typeIII_tiles = total_size_constraint - total_size_of_typeI_and_II_tiles
			lagrange_lambda = sum_typeIII_z_weight / byte_size_constraint_of_typeIII_tiles

			print("left, lambda, right: ", \
			tiles_values[sorted_tiles_values[left_idx]['frame_idx']][sorted_tiles_values[left_idx]['x']][sorted_tiles_values[left_idx]['y']][sorted_tiles_values[left_idx]['z']][sorted_tiles_values[left_idx]['value_idx']], \
			lagrange_lambda, \
			tiles_values[sorted_tiles_values[right_idx]['frame_idx']][sorted_tiles_values[right_idx]['x']][sorted_tiles_values[right_idx]['y']][sorted_tiles_values[right_idx]['z']][sorted_tiles_values[right_idx]['value_idx']])

			# pdb.set_trace()

			for tile_loc_tuple in typeIII_tiles_set:
				frame_idx = tile_loc_tuple[0]
				x = tile_loc_tuple[1]
				y = tile_loc_tuple[2]
				z = tile_loc_tuple[3]
				z_weight = z_weights[frame_idx][x][y][z]

				tiles_rate_solution[frame_idx][x][y][z] = byte_size_constraint_of_typeIII_tiles * z_weight / sum_typeIII_z_weight
				if tiles_rate_solution[frame_idx][x][y][z] <= 1:
					print("tiles rate*: %f <= 1" %(tiles_rate_solution[frame_idx][x][y][z]))
				assert (tiles_rate_solution[frame_idx][x][y][z] < MAX_TILE_SIZE), "!!!!!!! tile size: %f too large, reaches MAX_TILE_SIZE (before divide-and-conquer) !!!!!!!" %(tiles_rate_solution[frame_idx][x][y][z])
				assert (tiles_rate_solution[frame_idx][x][y][z] > buffered_tiles_sizes[frame_idx][x][y][z]), "!!!!!!! tile size: %f too small, reaches r0: %f (before divide-and-conquer) !!!!!!!" %(tiles_rate_solution[frame_idx][x][y][z], buffered_tiles_sizes[frame_idx][x][y][z])
			print("lambda is larger than maximal finite tile value!", np.sum(tiles_rate_solution), total_size_constraint)
			total_size = np.sum(tiles_rate_solution)
			return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes)

		middle_idx = (left_idx + right_idx) // 2
		# mark previous lambda is at right_idx or left_idx:
		# this impacts how to update typeIII tile set.
		prev_lambda_at_right = True
		while middle_idx != left_idx:
			# calculate total size when lambda=tile_value[middle_idx]
			# if new total size < total budget, right_idx=middle_idx; otherwise, left_idx=middle_idx

			lagrange_lambda = tiles_values[sorted_tiles_values[middle_idx]['frame_idx']] \
										  [sorted_tiles_values[middle_idx]['x']] \
										  [sorted_tiles_values[middle_idx]['y']] \
										  [sorted_tiles_values[middle_idx]['z']] \
										  [sorted_tiles_values[middle_idx]['value_idx']]

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
				tiles_rate_solution[frame_idx][x][y][z] = MAX_TILE_SIZE
			else: # type II
				visited_typeI_or_II_tiles_set.add(tile_loc_tuple)
				tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
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
						tiles_rate_solution[frame_idx][x][y][z] = MAX_TILE_SIZE
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
						tiles_rate_solution[frame_idx][x][y][z] = buffered_tiles_sizes[frame_idx][x][y][z]
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

				tiles_rate_solution[frame_idx][x][y][z] = z_weight / lagrange_lambda
			##############################################################################################

			total_size = np.sum(tiles_rate_solution)

			if total_size > total_size_constraint: # lambda should be bigger
				left_idx = middle_idx
				prev_lambda_at_right = False

			elif total_size == total_size_constraint:
				return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes)
			else:
				right_idx = middle_idx
				prev_lambda_at_right = True

			middle_idx = (left_idx + right_idx) // 2

		# lambda is between (left_idx, right_idx)
		assert (tiles_values[sorted_tiles_values[left_idx]["frame_idx"]][sorted_tiles_values[left_idx]["x"]][sorted_tiles_values[left_idx]["y"]][sorted_tiles_values[left_idx]["z"]][sorted_tiles_values[left_idx]["value_idx"]] != \
				tiles_values[sorted_tiles_values[right_idx]["frame_idx"]][sorted_tiles_values[right_idx]["x"]][sorted_tiles_values[right_idx]["y"]][sorted_tiles_values[right_idx]["z"]][sorted_tiles_values[right_idx]["value_idx"]]), \
				"!!!!!!!!!!!! left tile value = right tile value !!!!!!"

		total_size_of_typeI_and_II_tiles = total_size
		sum_typeIII_z_weight = 0
		for tile_loc_tuple in typeIII_tiles_set:
			frame_idx = tile_loc_tuple[0]
			x = tile_loc_tuple[1]
			y = tile_loc_tuple[2]
			z = tile_loc_tuple[3]
			z_weight = z_weights[frame_idx][x][y][z]
			sum_typeIII_z_weight += z_weight
			total_size_of_typeI_and_II_tiles -= tiles_rate_solution[frame_idx][x][y][z]

		byte_size_constraint_of_typeIII_tiles = total_size_constraint - total_size_of_typeI_and_II_tiles
		lagrange_lambda = sum_typeIII_z_weight / byte_size_constraint_of_typeIII_tiles

		# lambda is between (left_idx, right_idx)
		print("left_idx, right_idx: ", left_idx, right_idx)
		print("left, lambda, right: ", \
			tiles_values[sorted_tiles_values[left_idx]['frame_idx']][sorted_tiles_values[left_idx]['x']][sorted_tiles_values[left_idx]['y']][sorted_tiles_values[left_idx]['z']][sorted_tiles_values[left_idx]['value_idx']], \
			lagrange_lambda, \
			tiles_values[sorted_tiles_values[right_idx]['frame_idx']][sorted_tiles_values[right_idx]['x']][sorted_tiles_values[right_idx]['y']][sorted_tiles_values[right_idx]['z']][sorted_tiles_values[right_idx]['value_idx']])

		for tile_loc_tuple in typeIII_tiles_set:
			frame_idx = tile_loc_tuple[0]
			x = tile_loc_tuple[1]
			y = tile_loc_tuple[2]
			z = tile_loc_tuple[3]
			z_weight = z_weights[frame_idx][x][y][z]

			tiles_rate_solution[frame_idx][x][y][z] = byte_size_constraint_of_typeIII_tiles * z_weight / sum_typeIII_z_weight

			if tiles_rate_solution[frame_idx][x][y][z] >= MAX_TILE_SIZE:
				pdb.set_trace()
			if tiles_rate_solution[frame_idx][x][y][z] <= buffered_tiles_sizes[frame_idx][x][y][z]:
				pdb.set_trace()
			assert (tiles_rate_solution[frame_idx][x][y][z] < MAX_TILE_SIZE), "!!!!!!! tile size: %f too large, reaches MAX_TILE_SIZE (during divide-and-conquer) !!!!!!!" %(tiles_rate_solution[frame_idx][x][y][z])
			assert (tiles_rate_solution[frame_idx][x][y][z] > buffered_tiles_sizes[frame_idx][x][y][z]), "!!!!!!! tile size: %f too small, reaches r0: %f (during divide-and-conquer) !!!!!!!" %(tiles_rate_solution[frame_idx][x][y][z], buffered_tiles_sizes[frame_idx][x][y][z])
		
		total_size = np.sum(tiles_rate_solution)
		assert (total_size - total_size_constraint < 1e-4), "!!!! total size != total bw budget %f, %f!!!!!" %(total_size, total_size_constraint)
		return tiles_rate_solution, buffered_tiles_sizes, total_size, np.sum(buffered_tiles_sizes)

class ValidTiles():
	'''
		read valid_tiles of every frame
	'''

	def __init__(self):
		self.valid_tiles = []
		self.tiles_coordinates = []

		# convert index to real-world coordinates,
		# another option is to do the conversion just for valid(non-empty) tiles
		for x in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
			self.tiles_coordinates.append([])
			for y in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
				self.tiles_coordinates[x].append([])
				for z in range(NUM_TILES_PER_SIDE_IN_A_FRAME):
					self.tiles_coordinates[x][y].append([x + TILE_SIDE_LEN / 2 - TILE_SIDE_LEN * NUM_TILES_PER_SIDE_IN_A_FRAME / 2, \
														 y + TILE_SIDE_LEN / 2, \
														 z + TILE_SIDE_LEN / 2 - TILE_SIDE_LEN * NUM_TILES_PER_SIDE_IN_A_FRAME / 2])

	def read_valid_tiles(self, path_prefix):
		'''
			read valid_tiles of every frame
		'''

		start_frame_idx = 1051 # according to the file names Yixiang provided
		for frame_idx in range(NUM_FRAMES):
			path = path_prefix + VIDEO_NAME + '_vox10_' + str(start_frame_idx + frame_idx) + '.ply.p'
			with open(path, 'rb') as file:
				# 16*16*16 numpy array indicating which tiles are non-empty
				valid_tile_this_frame = pk.load(file)

				# swap x and z axis to comform with coordinate system in dataset
				# valid_tile_this_frame = np.swapaxes(valid_tile_this_frame, 0, 2)

				self.valid_tiles.append(valid_tile_this_frame)
			file.close()

	def convert_pointIdx_to_coordinate(self, x, y, z):
		x = x * TILE_SIDE_LEN + TILE_SIDE_LEN / 2
		y = y * TILE_SIDE_LEN + TILE_SIDE_LEN / 2
		z = z * TILE_SIDE_LEN + TILE_SIDE_LEN / 2

		return [x, y, z]

	def change_tile_coordinates_origin(self, origin, tile_center_points):
		shifted_tile_center_points = []
		for point in tile_center_points:
			shifted_tile_center_points.append([point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]])

		return shifted_tile_center_points

class FovTraces():
	'''
		read and operate 6dof trace
	'''

	def __init__(self):
		self.fov_traces = None

	def read_fov_traces(self, path):
		'''
			read 6dof of every frame
		'''

		self.fov_traces = pd.read_csv(path)
		# pdb.set_trace()

	def predict_6dof(self, current_viewing_frame_idx, predict_start_idx, predict_end_idx, history_viewpoints):
		# ARMA-style prediction
		# record prediction accuracy (difference from ground truth) for all frames

		# each dof maintains a list including all predicted frames from [predict_start_idx, predict_end_idx]
		predicted_viewpoints = {"x":[], 'y':[], "z":[], "pitch":[], "yaw":[], "roll":[]}

		for frame_idx in range(predict_start_idx, predict_end_idx + 1):
			prediction_win = frame_idx - current_viewing_frame_idx
			history_win = prediction_win // 2 # according to vivo paper, best use half of prediciton window
			history_win = history_win if len(history_viewpoints['x']) >= history_win else len(history_viewpoints['x'])
			x_list = np.arange(history_win).reshape(-1, 1)
			# print("index: ", predict_start_idx, " ", predict_end_idx)
			# print("win: ", prediction_win, " ", history_win)
			# print("x: ", x_list)
			for key in predicted_viewpoints.keys():
				# print("key: ", key)
				y_list = np.array(history_viewpoints[key][-history_win:]).reshape(-1, 1)
				# print("y: ", y_list)
				reg = LinearRegression().fit(x_list, y_list)
				predicted_dof = reg.predict([[history_win + prediction_win - 1]])[0][0]
				if key == 'pitch' or key == 'yaw' or key == 'roll':
					if predicted_dof >= 360:
						predicted_dof -= 360
				predicted_viewpoints[key].append(predicted_dof)
				# print("result: ", predicted_viewpoints)
		# pdb.set_trace()

		return predicted_viewpoints


class BandwidthTraces():
	'''
		read and operate bandwidth trace
	'''

	def __init__(self):
		self.bw_trace = []

	def read_bw_traces(self, path):
		'''
			read 6dof of every frame
		'''
		with open(path) as f:
			line = f.readline()
			while line:
				self.bw_trace.append(float(line))
				line = f.readline()
		f.close()

	def predict_bw(self, current_bandwidth_idx, history_bandwidths):
		if BANDWIDTH_ORACLE_KNOWN:
			return self.bw_trace[current_bandwidth_idx + 1]
		else: 
			return harmonic_mean(history_bandwidths[-BW_PREDICTION_HISTORY_WIN_LENGTH:])

class QRWeights():
	'''
		read weights of quality-rate function
	'''

	def __init__(self):
		self.qr_weights = {"a":np.zeros((NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME)), \
						   "b":np.zeros((NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME, NUM_TILES_PER_SIDE_IN_A_FRAME))}
		self.mean_a = 0
		self.mean_b = 0

	def read_weights(self, path_a, path_b):
		with open(path_a, 'rb') as file:
			self.qr_weights["a"] = pk.load(file)
		file.close()
		with open(path_b, 'rb') as file:
			self.qr_weights["b"] = pk.load(file)
		file.close()

		# pdb.set_trace()

	def give_every_tile_valid_weights(self):
		# remove nan value
		self.qr_weights["a"][np.where(np.isnan(self.qr_weights["a"])==True)] = 0
		self.qr_weights["b"][np.where(np.isnan(self.qr_weights["b"])==True)] = 0
		# calculate mean of a and b
		self.mean_a = np.mean(self.qr_weights["a"][self.qr_weights["a"].nonzero()])
		self.mean_b = np.mean(self.qr_weights["b"][self.qr_weights["b"].nonzero()])
		# assign
		self.qr_weights["a"][np.where(self.qr_weights["a"]==0)] = self.mean_a
		self.qr_weights["b"][np.where(self.qr_weights["b"]==0)] = self.mean_b

		# self.qr_weights["a"] = np.ones(self.qr_weights["a"].shape) * self.mean_a
		# self.qr_weights["b"] = np.ones(self.qr_weights["b"].shape) * self.mean_b

		# pdb.set_trace()

def plot_point_cloud(frame):
	'''
		plot 3d points

		args:
		@frame: 3d numpy array
	'''
	plt.rcParams["figure.figsize"] = [10, 10]
	# plt.rcParams["figure.autolayout"] = True
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x, y, z = frame.nonzero()
	# print(x)
	# print(z)
	# print(y)
	ax.scatter(x, y, z, alpha=1, s=1)
	plt.show()

def plot_frame_quality_trace(frame_quality_lists):
	frame_indexes = range(len(frame_quality_lists) - TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * FPS)
	legend = []
	plt.plot(frame_indexes, frame_quality_lists[TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * FPS:], linewidth=2, color='red')
	legend.append('constant Wj, know BW')
	# plt.plot(frame_indexes, frame_quality_lists[1][TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * FPS:], linewidth=2, color='blue')
	# legend.append('linear (flat) Wj, know BW')
	# plt.plot(frame_indexes, frame_quality_lists[2][TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * FPS:], linewidth=2, color='green')
	# legend.append('linear (steep) Wj, know BW')
	plt.legend(legend, fontsize=30, loc='best', ncol=1)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.title('Frame Quality of Longdress, Latency=%ds' %(TARGET_LATENCY // FPS), fontsize=40, fontweight='bold')
	
	plt.xlabel('frame idx', fontsize=40, fontweight='bold')
	plt.ylabel('quality', fontsize=40, fontweight='bold')

	plt.xticks(fontsize=30)
	# plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
	plt.yticks(fontsize=30)
	# plt.tight_layout()
	plt.show()

def plot_bandwidth_trace(plot_bw_trace, plot_predicted_bw_trace, bw_predict_accuracy_trace):
	timestamps_in_sec = range(len(plot_bw_trace))
	legend = []

	plt.plot(timestamps_in_sec, plot_bw_trace, linewidth=2, color='red')
	legend.append('true bw')
	plt.plot(timestamps_in_sec, plot_predicted_bw_trace, linewidth=2, color='blue')
	legend.append('predicted bw')
	plt.plot(timestamps_in_sec, bw_predict_accuracy_trace, linewidth=2, color='green')
	legend.append('difference')

	plt.legend(legend, fontsize=30, loc='best', ncol=1)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.title('Predicted/True Bandwidth Trace', fontsize=40, fontweight='bold')
	
	plt.xlabel('time/s', fontsize=40, fontweight='bold')
	plt.ylabel('Mbps', fontsize=40, fontweight='bold')

	plt.xticks(fontsize=30)
	# plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
	plt.yticks(fontsize=30)
	# plt.tight_layout()
	plt.show()

def plot_buffer_size_trace(buffer_size_trace_lists):
	timestamps_in_sec = range(len(buffer_size_trace_lists))
	legend = []
	plt.plot(timestamps_in_sec, np.array(buffer_size_trace_lists) / 1000, linewidth=2, color='red')
	legend.append('constant Wj, know BW')
	# plt.plot(timestamps_in_sec, np.array(buffer_size_trace_lists[1]) / 1000, linewidth=2, color='blue')
	# legend.append('linear (flat) Wj, know BW')
	# plt.plot(timestamps_in_sec, np.array(buffer_size_trace_lists[2]) / 1000, linewidth=2, color='green')
	# legend.append('linear (steep) Wj, know BW')
	# plt.legend(legend, fontsize=30, loc='best', ncol=1)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.title('Buffer Size Evolution of Longdress, Latency=%ds' %(TARGET_LATENCY // FPS), fontsize=40, fontweight='bold')
	
	plt.xlabel('time/s', fontsize=40, fontweight='bold')
	plt.ylabel('buffer size / KB', fontsize=40, fontweight='bold')

	plt.xticks(fontsize=30)
	# plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
	plt.yticks(fontsize=30)
	# plt.tight_layout()
	plt.show()

def plot_success_download_rate_trace(success_download_rate_trace_lists):
	timestamps_in_sec = range(len(success_download_rate_trace_lists[0]))
	legend = []

	plt.plot(timestamps_in_sec, success_download_rate_trace_lists[0], linewidth=2, color='red')
	legend.append('constant Wj')
	plt.plot(timestamps_in_sec, success_download_rate_trace_lists[1], linewidth=2, color='blue')
	legend.append('linear (flat) Wj')
	plt.plot(timestamps_in_sec, success_download_rate_trace_lists[2], linewidth=2, color='green')
	legend.append('linear (steep) Wj')

	plt.legend(legend, fontsize=30, loc='best', ncol=1)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.title('Download Rate Trace', fontsize=40, fontweight='bold')
	
	plt.xlabel('time/s', fontsize=40, fontweight='bold')
	plt.ylabel('', fontsize=40, fontweight='bold')

	plt.xticks(fontsize=30)
	plt.yticks([0, 0.5, 1, 1.5], fontsize=30)
	# plt.tight_layout()
	plt.show()

def plot_fov_prediction_accuracy_series(fov_predict_accuracy_trace):
	'''
		6 subplots for 6dof
	'''
	# pdb.set_trace()
	num_frames  = len(fov_predict_accuracy_trace['x'])
	num_updates = len(fov_predict_accuracy_trace['x'][0])
	timestamps_in_sec = range(num_updates)
	num_valid_accuracy = (NUM_FRAMES_VIEWED - BUFFER_LENGTH - UPDATE_FREQ * FPS) // (UPDATE_FREQ * FPS)
	###########################
	# x 	  y 	  z
	# pitch   yaw     roll
	###########################
	map_from_pos_in_buffer_to_legend = {0:"buffer front", num_frames // 2:"buffer middle", num_frames - 1:"buffer end"}
	fig, axs = plt.subplots(2, 3)
	fig.suptitle("FoV prediction Accuracy", fontsize=30, fontweight='bold')
	for key in fov_predict_accuracy_trace:
		pos_row = MAP_DOF_TO_PLOT_POS[key][0]
		pos_col = MAP_DOF_TO_PLOT_POS[key][1]
		legend = []
		# smaller pos_in_buffer gets higher fov prediction accuracy
		for pos_in_buffer in [0, num_frames // 2, num_frames - 1]:
			axs[pos_row, pos_col].plot(timestamps_in_sec[:num_valid_accuracy], fov_predict_accuracy_trace[key][pos_in_buffer][:num_valid_accuracy], linewidth=2)
			legend.append(map_from_pos_in_buffer_to_legend[pos_in_buffer])
		axs[pos_row, pos_col].legend(legend, fontsize=15, prop={'weight':'bold'}, loc='best', ncol=1)
		axs[pos_row, pos_col].set_title(key, fontsize=15, fontweight='bold')
		axs[pos_row, pos_col].set_ylabel('Absolute Error', fontsize = 20.0) # Y label
		axs[pos_row, pos_col].set_xlabel('time/s', fontsize = 20) # X label
		axs[pos_row, pos_col].tick_params(axis='both', labelsize=15)
		axs[pos_row, pos_col].grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
		# axs[pos_row, pos_col].label_outer()

	plt.show()

def plot_mean_fov_prediction_accuracy_for_every_buffer_pos(fov_predict_accuracy_trace):
	'''
		6 subplots for 6dof
	'''
	# pdb.set_trace()
	num_frames  = len(fov_predict_accuracy_trace['x'])
	frame_indexes = range(num_frames)
	num_updates = len(fov_predict_accuracy_trace['x'][0])
	num_valid_accuracy = (NUM_FRAMES_VIEWED - BUFFER_LENGTH - UPDATE_FREQ * FPS) // (UPDATE_FREQ * FPS)
	###########################
	# x 	  y 	  z
	# pitch   yaw     roll
	###########################
	fig, axs = plt.subplots(2, 3)
	fig.suptitle("Mean FoV prediction Accuracy (MAE)", fontsize=30, fontweight='bold')
	for key in fov_predict_accuracy_trace:

		pos_row = MAP_DOF_TO_PLOT_POS[key][0]
		pos_col = MAP_DOF_TO_PLOT_POS[key][1]
		# legend = []
		# smaller pos_in_buffer gets higher fov prediction accuracy
		axs[pos_row, pos_col].plot(frame_indexes, np.array(fov_predict_accuracy_trace[key])[:, :num_valid_accuracy].mean(axis=1), linewidth=2)
		# legend.append(map_from_pos_in_buffer_to_legend[pos_in_buffer])
		# axs[pos_row, pos_col].legend(legend, fontsize=15, prop={'weight':'bold'}, loc='best', ncol=1)
		axs[pos_row, pos_col].set_title(key, fontsize=15, fontweight='bold')
		axs[pos_row, pos_col].set_ylabel('Mean Absolute Error', fontsize = 20.0) # Y label
		axs[pos_row, pos_col].set_xlabel('frame idx', fontsize = 20) # X label
		axs[pos_row, pos_col].tick_params(axis='both', labelsize=15)
		axs[pos_row, pos_col].grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
		# axs[pos_row, pos_col].label_outer()

	plt.show()


def main():
	global FRAME_WEIGHT_TYPE

	valid_tiles_obj = ValidTiles()
	valid_tiles_obj.read_valid_tiles(VALID_TILES_PATH + '/')
	# plot_point_cloud(valid_tiles_obj.valid_tiles[0])
	# pdb.set_trace()

	fov_traces_obj = FovTraces()
	fov_traces_obj.read_fov_traces(FOV_TRACES_PATH)

	bw_traces_obj = BandwidthTraces()
	bw_traces_obj.read_bw_traces(BANDWIDTH_TRACES_PATH)

	qr_weights_obj = QRWeights()
	qr_weights_obj.read_weights(QR_WEIGHTS_PATH_A, QR_WEIGHTS_PATH_B)
	qr_weights_obj.give_every_tile_valid_weights()

	# frame_quality_lists = []
	# buffer_size_trace_lists = []
	# success_download_rate_trace_lists = []
	# for FRAME_WEIGHT_TYPE in range(3):
	# 	buffer_obj = Buffer(fov_traces_obj, bw_traces_obj, valid_tiles_obj, qr_weights_obj)
	# 	buffer_obj.initialize_buffer()

	# 	for update_time_step in range(NUM_UPDATES):
	# 		print(str(update_time_step + 1) + "th update step")
	# 		buffer_obj.update_tile_size_in_buffer()
	# 		buffer_obj.emit_buffer()

	# 	frame_quality_lists.append(buffer_obj.frame_quality)
	# 	buffer_size_trace_lists.append(buffer_obj.buffer_size_trace)
	# 	success_download_rate_trace_lists.append(buffer_obj.success_download_rate_trace)

	buffer_obj = Buffer(fov_traces_obj, bw_traces_obj, valid_tiles_obj, qr_weights_obj)
	buffer_obj.initialize_buffer()
	for update_time_step in range(NUM_UPDATES):
		print(str(update_time_step + 1) + "th update step")
		buffer_obj.update_tile_size_in_buffer()
		buffer_obj.emit_buffer()

	# print(buffer_obj.frame_quality)
	plot_frame_quality_trace(buffer_obj.frame_quality)
	plot_bandwidth_trace(buffer_obj.plot_bw_trace, buffer_obj.plot_predicted_bw_trace, buffer_obj.bw_predict_accuracy_trace)
	plot_buffer_size_trace(buffer_obj.buffer_size_trace)
	# plot_success_download_rate_trace(buffer_obj.success_download_rate_trace)
	# plot_fov_prediction_accuracy_series(buffer_obj.fov_predict_accuracy_trace)
	# plot_mean_fov_prediction_accuracy_for_every_buffer_pos(buffer_obj.fov_predict_accuracy_trace)

if __name__ == '__main__':
	main()