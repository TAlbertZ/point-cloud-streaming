import numpy as np
import pdb
from statistics import harmonic_mean

import params
from params import Algo

np.random.seed(7)

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
		if params.BANDWIDTH_ORACLE_KNOWN:
			if params.ALGO == Algo.ILQR:
				return np.array(self.bw_trace[current_bandwidth_idx + 1 : current_bandwidth_idx + params.ILQR_HORIZON + 1])
			return self.bw_trace[current_bandwidth_idx + 1]
		else: 
			return harmonic_mean(history_bandwidths[-params.BW_PREDICTION_HISTORY_WIN_LENGTH:])