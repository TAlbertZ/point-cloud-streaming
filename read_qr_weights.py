import numpy as np
import pickle as pk
import pdb

import params

np.random.seed(7)


class QRWeights():
    '''
		read weights of quality-rate function
	'''

    # TODO by Tongyu: look at same tile index of any two consecutive frames,
    # their weights a and b should be similar
    def __init__(self):
        self.qr_weights = {"a":np.zeros((params.NUM_FRAMES, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME)), \
               "b":np.zeros((params.NUM_FRAMES, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME, params.NUM_TILES_PER_SIDE_IN_A_FRAME))}
        self.mean_a = 0
        self.mean_b = 0
        self.rate_versions = np.zeros(
            (params.NUM_RATE_VERSIONS, params.NUM_FRAMES,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        self.num_pts_versions = np.zeros(
            (params.NUM_RATE_VERSIONS, params.NUM_FRAMES,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        self.min_rates = np.zeros(
            (params.NUM_FRAMES, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))

    def read_weights(self, path_a, path_b, path_rate_versions,
                     path_num_pts_versions):
        with open(path_a, 'rb') as file:
            self.qr_weights["a"] = pk.load(file)
        file.close()
        with open(path_b, 'rb') as file:
            self.qr_weights["b"] = pk.load(file)
        file.close()

        with open(path_rate_versions, 'rb') as file:
            self.rate_versions = pk.load(file)
        file.close()

        with open(path_num_pts_versions, 'rb') as file:
            self.num_pts_versions = pk.load(file)
        file.close()

        # pdb.set_trace()

    def give_every_tile_valid_weights(self):
        # remove nan value
        # self.qr_weights["a"][np.where(np.isnan(self.qr_weights["a"])==True)] = 0
        # self.qr_weights["b"][np.where(np.isnan(self.qr_weights["b"])==True)] = 0
        # calculate mean of a and b
        # self.mean_a = np.mean(self.qr_weights["a"][self.qr_weights["a"].nonzero()])
        # self.mean_b = np.mean(self.qr_weights["b"][self.qr_weights["b"].nonzero()])
        # assign
        # self.qr_weights["a"][np.where(self.qr_weights["a"]==0)] = self.mean_a
        # self.qr_weights["b"][np.where(self.qr_weights["b"]==0)] = self.mean_b

        # pdb.set_trace()

        # self.qr_weights["b"][np.where(self.qr_weights["a"]>10)] = self.mean_b
        # self.qr_weights["a"][np.where(self.qr_weights["a"]>10)] = self.mean_a

        # self.qr_weights["a"] = np.ones(self.qr_weights["a"].shape) * self.mean_a
        # self.qr_weights["b"] = np.ones(self.qr_weights["b"].shape) * self.mean_b

        # self.qr_weights["a"][6][3][9] = self.mean_a
        # self.qr_weights["b"][6][3][9] = self.mean_b

        # self.qr_weights["a"][6][4][9] = self.mean_a
        # self.qr_weights["b"][6][4][9] = self.mean_b

        # pdb.set_trace()

        # mean_rate_versions = np.zeros((params.NUM_RATE_VERSIONS, ))
        # for rate_version_idx in range(params.NUM_RATE_VERSIONS):
        # 	mean_rate_versions[rate_version_idx] = np.mean(self.rate_versions[rate_version_idx][self.rate_versions[rate_version_idx].nonzero()])
        # 	# pdb.set_trace()
        # 	print("min:", np.min(self.rate_versions[rate_version_idx][self.rate_versions[rate_version_idx].nonzero()]))
        # 	print("max:", np.max(self.rate_versions[rate_version_idx][self.rate_versions[rate_version_idx].nonzero()]))
        # 	self.rate_versions[rate_version_idx][np.where(self.rate_versions[rate_version_idx]==0)] = mean_rate_versions[rate_version_idx]
        nonzero_pos = self.qr_weights["a"].nonzero()
        # self.min_rates[nonzero_pos] = -self.qr_weights["b"][
        #     nonzero_pos] / self.qr_weights["a"][nonzero_pos]
        # self.min_rates[self.min_rates < 0] = 0
        # self.min_rates[nonzero_pos] += 1e-3
