import numpy as np
import pickle as pk
import pdb

import params
import logging

np.random.seed(7)


class ValidTiles():
    '''
		read valid_tiles of every frame
	'''

    def __init__(self):
        self.valid_tiles = []
        self.tiles_coordinates = []
        self.num_valid_tiles_per_frame = []
        self.intersect_trace = []

        # convert index to real-world coordinates,
        # another option is to do the conversion just for valid(non-empty) tiles
        for x in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
            self.tiles_coordinates.append([])
            for y in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
                self.tiles_coordinates[x].append([])
                for z in range(params.NUM_TILES_PER_SIDE_IN_A_FRAME):
                    self.tiles_coordinates[x][y].append([x + params.TILE_SIDE_LEN / 2 - params.TILE_SIDE_LEN * params.NUM_TILES_PER_SIDE_IN_A_FRAME / 2, \
                              y + params.TILE_SIDE_LEN / 2, \
                              z + params.TILE_SIDE_LEN / 2 - params.TILE_SIDE_LEN * params.NUM_TILES_PER_SIDE_IN_A_FRAME / 2])

    def read_valid_tiles(self, path_prefix):
        '''
			read valid_tiles of every frame
		'''

        start_frame_idx = 1051  # according to the file names Yixiang provided
        previous_valid_tiles_set = set()
        # current_valid_tiles_set = set()

        for frame_idx in range(params.NUM_FRAMES):
            current_valid_tiles_set = set()
            path = path_prefix + params.VIDEO_NAME + '_vox10_' + str(
                start_frame_idx + frame_idx) + '.ply.p'
            with open(path, 'rb') as file:
                # 16*16*16 numpy array indicating which tiles are non-empty
                valid_tile_this_frame = pk.load(file)

                # swap x and z axis to comform with coordinate system in dataset
                # valid_tile_this_frame = np.swapaxes(valid_tile_this_frame, 0, 2)

                self.valid_tiles.append(valid_tile_this_frame)
                self.num_valid_tiles_per_frame.append(
                    len(valid_tile_this_frame.nonzero()[0]))
                for tile_idx in range(len(valid_tile_this_frame.nonzero()[0])):
                    x = valid_tile_this_frame.nonzero()[0][tile_idx]
                    y = valid_tile_this_frame.nonzero()[1][tile_idx]
                    z = valid_tile_this_frame.nonzero()[2][tile_idx]
                    current_valid_tiles_set.add((x, y, z))
                if frame_idx >= 1:
                    intersect = current_valid_tiles_set.intersection(
                        previous_valid_tiles_set)
                    self.intersect_trace.append(
                        len(intersect) /
                        self.num_valid_tiles_per_frame[frame_idx - 1])
                previous_valid_tiles_set = current_valid_tiles_set.copy()
            file.close()
        # plt.plot(range(len(self.num_valid_tiles_per_frame)), self.num_valid_tiles_per_frame, linewidth=2)
        # plt.plot(range(len(self.intersect_trace)), self.intersect_trace, linewidth=2)
        # plt.show()
        # pdb.set_trace()

    # def padding(self):
    # 	for _ in range(params.BUFFER_LENGTH):
    # 		self.valid_tiles.append(np.zeros((params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)))

    def read_valid_tiles_from_num_points(self, path):
        '''
			read valid_tiles of every frame
		'''

        previous_valid_tiles_set = set()
        with open(path, 'rb') as file:
            num_points_versions = pk.load(file) # 3x300x16x16x16
        file.close()
        sum_num_points_versions = np.sum(num_points_versions, axis=0) # 300x16x16x16

        for frame_idx in range(params.NUM_FRAMES):
            current_valid_tiles_set = set()
            # 16*16*16 numpy array indicating which tiles are non-empty
            valid_tile_this_frame = sum_num_points_versions[frame_idx]
            valid_pos = valid_tile_this_frame.nonzero()
            valid_tile_this_frame[valid_pos] = 1

            # swap x and z axis to comform with coordinate system in dataset
            # valid_tile_this_frame = np.swapaxes(valid_tile_this_frame, 0, 2)

            self.valid_tiles.append(valid_tile_this_frame)
            self.num_valid_tiles_per_frame.append(
                len(valid_tile_this_frame.nonzero()[0]))
            for tile_idx in range(len(valid_pos[0])):
                x = valid_pos[0][tile_idx]
                y = valid_pos[1][tile_idx]
                z = valid_pos[2][tile_idx]
                current_valid_tiles_set.add((x, y, z))
            if frame_idx >= 1:
                intersect = current_valid_tiles_set.intersection(
                    previous_valid_tiles_set)
                self.intersect_trace.append(
                    len(intersect) /
                    self.num_valid_tiles_per_frame[frame_idx - 1])
            previous_valid_tiles_set = current_valid_tiles_set.copy()

        # self.valid_tiles = np.array(self.valid_tiles)
        # pdb.set_trace()

    def convert_pointIdx_to_coordinate(self, x, y, z):
        x = x * params.TILE_SIDE_LEN + params.TILE_SIDE_LEN / 2
        y = y * params.TILE_SIDE_LEN + params.TILE_SIDE_LEN / 2
        z = z * params.TILE_SIDE_LEN + params.TILE_SIDE_LEN / 2

        return [x, y, z]

    def convert_pointIdx_to_coordinate_4_ply(self, x, y, z):
        x = x * params.VOXEL_SIDE_LEN + params.VOXEL_SIDE_LEN / 2
        y = y * params.VOXEL_SIDE_LEN + params.VOXEL_SIDE_LEN / 2
        z = z * params.VOXEL_SIDE_LEN + params.VOXEL_SIDE_LEN / 2

        return [x, y, z]
    def change_tile_coordinates_origin(self, origin, tile_center_points):
        shifted_tile_center_points = []
        for point in tile_center_points:
            shifted_tile_center_points.append([
                point[0] - origin[0], point[1] - origin[1],
                point[2] - origin[2]
            ])

        return shifted_tile_center_points
