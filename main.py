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
import math
from mpl_toolkits.mplot3d import Axes3D
import time
import logging

import params
from buffer_control_ilqr import Buffer
from read_valid_tiles import ValidTiles
from read_fov_traces import FovTraces
from read_bw_traces import BandwidthTraces
from read_qr_weights import QRWeights
from plot_figs import PlotFigs

np.random.seed(7)


def set_loggers():
    # root logger
    logging.basicConfig(format='%(message)s - %(lineno)d - %(asctime)s',
                        level=logging.INFO)


# FIXME by Tongyu: video frames should also be back-forward like fov trace
# TODO by Tongyu: use HPC for multiple experiments, e.g., set different buf length
# TODO [URGENT] by Tongyu: find more reasonable and calculation-friendly utility model for frame, tile and segment.
# TODO [URGENT] by Tongyu: how to calculate frame-level quality variance between adjacent frames? currently is average quality over tiles.
# TODO by Tongyu: compare total number of points, number of nonempty tiles, and average number of points over these tiles, between two ajacent frames;
#                   also, find if there exists a shift between two frames.
def main():
    set_loggers()
    valid_tiles_obj = ValidTiles()
    # valid_tiles_obj.read_valid_tiles(params.VALID_TILES_PATH + '/')
    valid_tiles_obj.read_valid_tiles_from_num_points(
        params.VALID_TILES_PATH_FROM_NUM_PTS)
    # valid_tiles_obj.padding()
    # plot_figs_obj.plot_point_cloud(valid_tiles_obj.valid_tiles[0])
    # pdb.set_trace()

    fov_traces_obj = FovTraces()
    num_frames_load_from_fov_trace = params.NUM_FRAMES_VIEWED + params.TARGET_LATENCY
    if params.ALGO == params.Algo.ILQR:
        num_frames_load_from_fov_trace = params.NUM_FRAMES_VIEWED + (
            params.ILQR_HORIZON - 1) * params.FPS
    fov_traces_obj.read_fov_traces_txt(params.FOV_TRACES_PATH,
                                       num_frames_load_from_fov_trace)
    fov_traces_obj.padding_txt()

    bw_traces_obj = BandwidthTraces()
    bw_traces_obj.read_bw_traces(params.BANDWIDTH_TRACES_PATH)

    qr_weights_obj = QRWeights()
    qr_weights_obj.read_weights(params.QR_WEIGHTS_PATH_A,
                                params.QR_WEIGHTS_PATH_B,
                                params.PATH_RATE_VERSIONS,
                                params.PATH_NUM_PTS_VERSIONS)
    qr_weights_obj.give_every_tile_valid_weights()

    plot_figs_obj = PlotFigs()

    # frame_quality_lists = []
    # buffer_size_trace_lists = []
    # success_download_rate_trace_lists = []
    # for FRAME_WEIGHT_TYPE in range(3):
    # 	buffer_obj = Buffer(fov_traces_obj, bw_traces_obj, valid_tiles_obj, qr_weights_obj)
    # 	buffer_obj.initialize_buffer()

    # 	for update_time_step in range(params.NUM_UPDATES):
    # 		print(str(update_time_step + 1) + "th update step")
    # 		buffer_obj.update_tile_size_in_buffer()
    # 		buffer_obj.emit_buffer()

    # 	frame_quality_lists.append(buffer_obj.frame_quality)
    # 	buffer_size_trace_lists.append(buffer_obj.buffer_size_trace)
    # 	success_download_rate_trace_lists.append(buffer_obj.success_download_rate_trace)

    buffer_obj = Buffer(fov_traces_obj,
                        bw_traces_obj,
                        valid_tiles_obj,
                        qr_weights_obj,
                        N=params.ILQR_HORIZON)
    buffer_obj.initialize_buffer()
    for update_time_step in range(params.NUM_UPDATES + params.TARGET_LATENCY // params.FPS):
        logging.info(
            '\n######################## %dth update step #######################',
            update_time_step + 1)
        buffer_obj.update_tile_size_in_buffer()
        buffer_obj.emit_buffer()

    # print(buffer_obj.frame_quality)

    # with open('./tile_sizes_sol_trace.data', 'wb+') as filehandle:
    # 	pk.dump(buffer_obj.tile_sizes_sol, filehandle)
    # filehandle.close()

    # with open('./tile_sizes_sol_trace.data', 'rb') as infile:
    # 	tile_sizes_sol = pk.load(infile)
    # infile.close()

    # plot_figs_obj.plot_tile_size_trace(tile_sizes_sol)
    # plot_figs_obj.plot_num_max_tiles_per_frame(buffer_obj.num_max_tiles_per_frame)

    # plot_figs_obj.plot_mean_size_over_tiles_per_frame(
    #     buffer_obj.mean_size_over_tiles_per_frame)
    # plot_figs_obj.plot_mean_size_over_tiles_per_fov(
    #     buffer_obj.mean_size_over_tiles_per_fov)
    # plot_figs_obj.plot_effective_rate_per_frame(buffer_obj.effective_rate)
    # # print(buffer_obj.num_valid_tiles_per_frame[params.BUFFER_LENGTH:])
    # plot_figs_obj.plot_num_valid_tiles_per_frame(
    #     buffer_obj.num_valid_tiles_per_frame)

    # plot_figs_obj.plot_mean_tile_quality_per_frame_trace(buffer_obj.frame_quality, buffer_obj.num_valid_tiles_per_frame)
    # plot_figs_obj.plot_per_degree_quality_per_frame_trace(buffer_obj.frame_quality, buffer_obj.total_span_per_frame)
    plot_figs_obj.plot_frame_quality_trace(buffer_obj.frame_quality)
    # plot_figs_obj.plot_frame_quality_var_trace(buffer_obj.frame_quality_var)

    # plot_figs_obj.plot_frame_size_trace(buffer_obj.frame_size_list)
    # # plot_figs_obj.plot_bandwidth_trace(buffer_obj.plot_bw_trace, buffer_obj.plot_predicted_bw_trace, buffer_obj.bw_predict_accuracy_trace)
    # # plot_figs_obj.plot_filtered_bandwidth_trace(buffer_obj.plot_bw_trace, buffer_obj.plot_predicted_bw_trace)
    # plot_figs_obj.plot_buffer_size_trace(buffer_obj.buffer_size_trace)
    # plot_figs_obj.plot_success_download_rate_trace(
    #     buffer_obj.success_download_rate_trace)

    # plot_figs_obj.plot_fov_prediction_accuracy_series(buffer_obj.fov_predict_accuracy_trace)
    # plot_figs_obj.plot_mean_fov_prediction_accuracy_for_every_buffer_pos(buffer_obj.fov_predict_accuracy_trace)


# def main():
# 	with open('./tile_sizes_sol_trace.data', 'rb') as infile:
# 		tile_sizes_sol = pk.load(infile)
# 	infile.close()

# 	# while True:
# 	plot_figs_obj.plot_tile_size_trace(tile_sizes_sol)

if __name__ == '__main__':
    main()
