import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import params

np.random.seed(7)


class PlotFigs():

    def __init__(self):
        self.directory = './ilqr_results/buf' + str(
            params.TARGET_LATENCY // params.FPS) + 's'

        # If the directory does not exist, create it
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def plot_point_cloud(self, frame):
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

    def plot_frame_quality_trace(self, frame_quality_lists):
        frame_indexes = range(len(frame_quality_lists) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 frame_quality_lists[params.BUFFER_LENGTH:],
                 linewidth=1,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1, color='gray')
        plt.title('Frame Quality of Longdress, Latency=%ds' %
                  (params.TARGET_LATENCY // params.FPS),
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')
        plt.ylabel('quality', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=15)
        # plt.tight_layout()
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'frame_quality.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_frame_quality_var_trace(self, frame_quality_var_lists):
        frame_indexes = range(
            len(frame_quality_var_lists) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 frame_quality_var_lists[params.BUFFER_LENGTH:],
                 linewidth=1,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_var_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, frame_quality_var_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1, color='gray')
        plt.title('Frame Quality Variance of Longdress, Latency=%ds' %
                  (params.TARGET_LATENCY // params.FPS),
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')
        plt.ylabel('quality', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=15)
        # plt.tight_layout()
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'frame_quality_var.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_mean_tile_quality_per_frame_trace(self, frame_quality_lists,
                                               num_valid_tiles_per_frame):
        frame_indexes = range(len(frame_quality_lists) - params.BUFFER_LENGTH)
        legend = []
        # print(frame_quality_lists[params.BUFFER_LENGTH:])
        # print(num_valid_tiles_per_frame[params.BUFFER_LENGTH:])
        # print(np.array(frame_quality_lists[params.BUFFER_LENGTH:]) / np.array(num_valid_tiles_per_frame[params.BUFFER_LENGTH:]))
        plt.plot(frame_indexes,
                 np.array(frame_quality_lists[params.BUFFER_LENGTH:]) /
                 np.array(num_valid_tiles_per_frame[params.BUFFER_LENGTH:]),
                 linewidth=2,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('Mean Tile Quality per Frame of Longdress, Latency=%ds' %
                  (params.TARGET_LATENCY // params.FPS),
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=40, fontweight='bold')
        plt.ylabel('mean tile quality', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

        # pdb.set_trace()

    def plot_frame_size_trace(self, frame_size_list):
        frame_indexes = range(len(frame_size_list) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 np.array(frame_size_list[params.BUFFER_LENGTH:]) / 1000,
                 linewidth=2,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('Frame Size of Longdress, Latency=%ds' %
                  (params.TARGET_LATENCY // params.FPS),
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=40, fontweight='bold')
        plt.ylabel('size/KB', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_bandwidth_trace(self, plot_bw_trace, plot_predicted_bw_trace,
                             bw_predict_accuracy_trace):
        timestamps_in_sec = range(len(plot_bw_trace))
        legend = []

        plt.plot(timestamps_in_sec, plot_bw_trace, linewidth=2, color='red')
        legend.append('true bw')
        plt.plot(timestamps_in_sec,
                 plot_predicted_bw_trace,
                 linewidth=2,
                 color='blue')
        legend.append('predicted bw')
        plt.plot(timestamps_in_sec,
                 bw_predict_accuracy_trace,
                 linewidth=2,
                 color='green')
        legend.append('difference')

        plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('Predicted/True Bandwidth Trace',
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('time/s', fontsize=40, fontweight='bold')
        plt.ylabel('Mbps', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_filtered_bandwidth_trace(self, plot_bw_trace,
                                      plot_predicted_bw_trace):
        timestamps_in_sec = range(
            len(plot_bw_trace) - params.TARGET_LATENCY // params.FPS + 1)
        legend = []

        # plot_bw_trace = np.array(plot_bw_trace)
        # plot_predicted_bw_trace = np.array(plot_predicted_bw_trace)

        filtered_bw_trace = []
        filtered_predicted_bw_trace = []
        for i in range(len(timestamps_in_sec)):
            filtered_bw_trace.append(
                np.mean(plot_bw_trace[i:i +
                                      params.TARGET_LATENCY // params.FPS]))
            filtered_predicted_bw_trace.append(
                np.mean(plot_predicted_bw_trace[i:i + params.TARGET_LATENCY //
                                                params.FPS]))

        plt.plot(timestamps_in_sec,
                 filtered_bw_trace,
                 linewidth=2,
                 color='red')
        legend.append('true bw')
        plt.plot(timestamps_in_sec,
                 filtered_predicted_bw_trace,
                 linewidth=2,
                 color='blue')
        legend.append('predicted bw')
        # plt.plot(timestamps_in_sec, bw_predict_accuracy_trace, linewidth=2, color='green')
        # legend.append('difference')

        plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('Predicted/True %ds-Average Bandwidth Trace' %
                  (params.TARGET_LATENCY // params.FPS),
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('time/s', fontsize=40, fontweight='bold')
        plt.ylabel('Mbps', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_buffer_size_trace(self, buffer_size_trace_lists):
        timestamps_in_sec = range(len(buffer_size_trace_lists))
        legend = []
        plt.plot(timestamps_in_sec,
                 np.array(buffer_size_trace_lists) / 1000,
                 linewidth=2,
                 color='red')
        legend.append('know BW')
        # plt.plot(timestamps_in_sec, np.array(buffer_size_trace_lists[1]) / 1000, linewidth=2, color='blue')
        # legend.append('linear (flat) Wj, know BW')
        # plt.plot(timestamps_in_sec, np.array(buffer_size_trace_lists[2]) / 1000, linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('Buffer Size Evolution of Longdress, Latency=%ds' %
                  (params.TARGET_LATENCY // params.FPS),
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('time/s', fontsize=40, fontweight='bold')
        plt.ylabel('buffer size / KB', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_num_valid_tiles_per_frame(self, num_valid_tiles_per_frame):
        frame_indexes = range(
            len(num_valid_tiles_per_frame) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 num_valid_tiles_per_frame[params.BUFFER_LENGTH:],
                 linewidth=2,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        # legend.append('linear (flat) Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('# Visible Tiles Per Frame of Longdress',
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=40, fontweight='bold')
        plt.ylabel('', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_num_max_tiles_per_frame(self, num_max_tiles_per_frame):
        frame_indexes = range(
            len(num_max_tiles_per_frame) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 num_max_tiles_per_frame[params.BUFFER_LENGTH:],
                 linewidth=2,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        # legend.append('linear (flat) Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('# MAX Tiles Per Frame of Longdress',
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=40, fontweight='bold')
        plt.ylabel('', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_mean_size_over_tiles_per_frame(self,
                                            mean_size_over_tiles_per_frame):
        frame_indexes = range(
            len(mean_size_over_tiles_per_frame) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 mean_size_over_tiles_per_frame[params.BUFFER_LENGTH:],
                 linewidth=2,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        # legend.append('linear (flat) Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('# Mean Size over Tiles Per Frame of Longdress',
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=40, fontweight='bold')
        plt.ylabel('Mean Size / Byte', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_mean_size_over_tiles_per_fov(self, mean_size_over_tiles_per_fov):
        frame_indexes = range(
            len(mean_size_over_tiles_per_fov) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 mean_size_over_tiles_per_fov[params.BUFFER_LENGTH:],
                 linewidth=2)
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        # legend.append('linear (flat) Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('# Mean Size over Tiles Per FoV of Longdress',
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=40, fontweight='bold')
        plt.ylabel('Mean Size / Byte', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_effective_rate_per_frame(self, effective_rate):
        frame_indexes = range(len(effective_rate) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 effective_rate[params.BUFFER_LENGTH:],
                 linewidth=2)
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        # legend.append('linear (flat) Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('# Effective Ratio Per Frame of Longdress',
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=40, fontweight='bold')
        plt.ylabel('', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_tile_size_trace(self, tile_sizes_sol):
        frame_indexes = range(len(tile_sizes_sol) - params.BUFFER_LENGTH)
        legend = []
        tile_sizes = tile_sizes_sol[params.BUFFER_LENGTH]
        nonzero_pos = tile_sizes.nonzero()
        num_nonzero_elm = len(nonzero_pos[0])
        print(num_nonzero_elm)
        tile_idx = np.random.randint(num_nonzero_elm)
        x = nonzero_pos[0][tile_idx]
        y = nonzero_pos[1][tile_idx]
        z = nonzero_pos[2][tile_idx]
        print(x, y, z)
        print(
            np.array(tile_sizes_sol[params.BUFFER_LENGTH:])[:, 1, 12,
                                                            1].nonzero())
        plt.plot(frame_indexes,
                 np.array(tile_sizes_sol[params.BUFFER_LENGTH:])[:, 1, 12, 1],
                 linewidth=2,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        # legend.append('linear (flat) Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('Tile Size Evolution of Longdress',
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=40, fontweight='bold')
        plt.ylabel('Size / Byte', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_success_download_rate_trace(self,
                                         success_download_rate_trace_lists):
        timestamps_in_sec = range(len(success_download_rate_trace_lists))
        legend = []

        plt.plot(timestamps_in_sec,
                 success_download_rate_trace_lists,
                 linewidth=2,
                 color='red')
        # legend.append('constant Wj')
        # plt.plot(timestamps_in_sec, success_download_rate_trace_lists[1], linewidth=2, color='blue')
        # legend.append('linear (flat) Wj')
        # plt.plot(timestamps_in_sec, success_download_rate_trace_lists[2], linewidth=2, color='green')
        # legend.append('linear (steep) Wj')

        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('Download Rate Trace', fontsize=40, fontweight='bold')

        plt.xlabel('time/s', fontsize=40, fontweight='bold')
        plt.ylabel('', fontsize=40, fontweight='bold')

        plt.xticks(fontsize=30)
        plt.yticks([0, 0.5, 1, 1.5], fontsize=30)
        # plt.tight_layout()
        plt.show()

    def plot_fov_prediction_accuracy_series(self, fov_predict_accuracy_trace):
        '''
			6 subplots for 6dof
		'''
        # pdb.set_trace()
        num_frames = len(fov_predict_accuracy_trace['x'])
        num_updates = len(fov_predict_accuracy_trace['x'][0])
        timestamps_in_sec = range(num_updates)
        num_valid_accuracy = (params.NUM_FRAMES_VIEWED - params.BUFFER_LENGTH
                              ) // (params.UPDATE_FREQ * params.FPS)
        ###########################
        # x 	  y 	  z
        # pitch   yaw	 roll
        ###########################
        map_from_pos_in_buffer_to_legend = {
            0: "buffer front",
            num_frames // 2: "buffer middle",
            num_frames - 1: "buffer end"
        }
        fig, axs = plt.subplots(2, 3)
        fig.suptitle("FoV prediction Accuracy", fontsize=30, fontweight='bold')
        for key in fov_predict_accuracy_trace:
            pos_row = params.MAP_DOF_TO_PLOT_POS[key][0]
            pos_col = params.MAP_DOF_TO_PLOT_POS[key][1]
            legend = []
            # smaller pos_in_buffer gets higher fov prediction accuracy
            for pos_in_buffer in [0, num_frames // 2, num_frames - 1]:
                axs[pos_row,
                    pos_col].plot(timestamps_in_sec[:num_valid_accuracy],
                                  fov_predict_accuracy_trace[key]
                                  [pos_in_buffer][:num_valid_accuracy],
                                  linewidth=2)
                legend.append(map_from_pos_in_buffer_to_legend[pos_in_buffer])
            axs[pos_row, pos_col].legend(legend,
                                         fontsize=15,
                                         prop={'weight': 'bold'},
                                         loc='best',
                                         ncol=1)
            axs[pos_row, pos_col].set_title(key,
                                            fontsize=15,
                                            fontweight='bold')
            axs[pos_row, pos_col].set_ylabel('Absolute Error',
                                             fontsize=20.0)  # Y label
            axs[pos_row, pos_col].set_xlabel('time/s', fontsize=20)  # X label
            axs[pos_row, pos_col].tick_params(axis='both', labelsize=15)
            axs[pos_row, pos_col].grid(linestyle='dashed',
                                       axis='y',
                                       linewidth=1.5,
                                       color='gray')
            # axs[pos_row, pos_col].label_outer()

        plt.show()

    def plot_mean_fov_prediction_accuracy_for_every_buffer_pos(
            self, fov_predict_accuracy_trace):
        '''
			6 subplots for 6dof
		'''
        # pdb.set_trace()
        num_frames = len(fov_predict_accuracy_trace['x'])
        frame_indexes = range(num_frames)
        num_updates = len(fov_predict_accuracy_trace['x'][0])
        num_valid_accuracy = (params.NUM_FRAMES_VIEWED - params.BUFFER_LENGTH
                              ) // (params.UPDATE_FREQ * params.FPS)
        ###########################
        # x 	  y 	  z
        # pitch   yaw	 roll
        ###########################
        fig, axs = plt.subplots(2, 3)
        fig.suptitle("Mean FoV prediction Accuracy (MAE)",
                     fontsize=30,
                     fontweight='bold')
        for key in fov_predict_accuracy_trace:

            pos_row = params.MAP_DOF_TO_PLOT_POS[key][0]
            pos_col = params.MAP_DOF_TO_PLOT_POS[key][1]
            # legend = []
            # smaller pos_in_buffer gets higher fov prediction accuracy
            axs[pos_row,
                pos_col].plot(frame_indexes,
                              np.array(fov_predict_accuracy_trace[key])
                              [:, :num_valid_accuracy].mean(axis=1),
                              linewidth=2)
            # legend.append(map_from_pos_in_buffer_to_legend[pos_in_buffer])
            # axs[pos_row, pos_col].legend(legend, fontsize=15, prop={'weight':'bold'}, loc='best', ncol=1)
            axs[pos_row, pos_col].set_title(key,
                                            fontsize=15,
                                            fontweight='bold')
            axs[pos_row, pos_col].set_ylabel('Mean Absolute Error',
                                             fontsize=20.0)  # Y label
            axs[pos_row, pos_col].set_xlabel('frame idx',
                                             fontsize=20)  # X label
            axs[pos_row, pos_col].tick_params(axis='both', labelsize=15)
            axs[pos_row, pos_col].grid(linestyle='dashed',
                                       axis='y',
                                       linewidth=1.5,
                                       color='gray')
            # axs[pos_row, pos_col].label_outer()

        plt.show()
