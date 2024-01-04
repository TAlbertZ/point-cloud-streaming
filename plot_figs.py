import numpy as np
import pickle as pk
import pdb
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import params

np.random.seed(7)


class PlotFigs():

    def __init__(self):
        results_path = None
        if params.ALGO == params.Algo.ILQR:
            results_path= 'ilqr_results'
        elif params.ALGO == params.Algo.KKT:
            results_path = 'kkt_results'
        elif params.ALGO == params.Algo.AVERAGE:
            results_path = 'ave_results'
        elif params.ALGO == params.Algo.RUMA_SCALABLE:
            results_path = 'ruma_results'
        else:
            pass

        if not params.PROGRESSIVE_DOWNLOADING:
            results_path = os.path.join(results_path, 'non_progress')

        if not params.SCALABLE_CODING:
            results_path = os.path.join(results_path, 'non_scal')

        if params.HOMO_TILES:
            results_path = os.path.join(results_path, 'homo_tile')

        if params.UNIFORM_DISTANCE:
            results_path = os.path.join(results_path, 'uni_dist')

        if not params.DISTANCE_BASED_UTILITY:
            results_path = os.path.join(results_path, 'no_dist_util')

        self.directory = os.path.join(results_path, 'buf' + str(
            params.TARGET_LATENCY // params.FPS) + 's')

        if not params.BANDWIDTH_ORACLE_KNOWN:
            self.directory = os.path.join(self.directory, 'wBwErr_har')
        else:
            self.directory = os.path.join(self.directory, 'noBwErr')
        if not params.FOV_ORACLE_KNOW:
            self.directory = os.path.join(self.directory, 'wFovErr_lr')

            if params.FRAME_WEIGHT_TYPE == params.FrameWeightType.LINEAR_DECREASE:
                self.directory = os.path.join(self.directory, 'frameWeight_linear')
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.FOV_PRED_ACCURACY:
                self.directory = os.path.join(self.directory, 'frameWeight_fovErr')
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.DYNAMIC:
                self.directory = os.path.join(self.directory, 'frameWeight_dyn')
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.STEP_FUNC:
                self.directory = os.path.join(self.directory, 'frameWeight_step')
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.EXP_DECREASE:
                self.directory = os.path.join(self.directory, 'frameWeight_exp')
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.BELL_SHAPE:
                self.directory = os.path.join(self.directory, 'frameWeight_bell')
            elif params.FRAME_WEIGHT_TYPE == params.FrameWeightType.CONST:
                self.directory = os.path.join(self.directory, 'frameWeight_const')
            else:
                pass
        else:
            self.directory = os.path.join(self.directory, 'noFovErr')

        self.render_directory = os.path.join(params.RENDER_RESULTS_PATH, self.directory)

        # If the directory does not exist, create it
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        if not os.path.exists(self.render_directory):
            os.makedirs(self.render_directory)

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

    def plot_ang_resol_trace(self, ang_resol_list):
        ##################### save as pickle ########################
        pkl_file_name = 'ang_resol_bw' + str(params.SCALE_BW) + '.pkl'
        save_path = os.path.join(self.directory, pkl_file_name)

        with open(save_path, 'wb+') as file:
        	pk.dump(ang_resol_list, file)
        file.close()
        ##############################################################

        frame_indexes = range(len(ang_resol_list) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 ang_resol_list[params.BUFFER_LENGTH:],
                 linewidth=1,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, ang_resol_list[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, ang_resol_list[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1, color='gray')

        plt.title('Angular Resolution per Frame',
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')

        plt.ylabel('Angular Resol', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        # plt.yticks(range(0, 220, 20), fontsize=15)
 
        # plt.text(250, 20, 'mean={:f}, std={:f}'.format(np.mean(ang_resol_list[2 * params.BUFFER_LENGTH:]), np.std(ang_resol_list[2 * params.BUFFER_LENGTH:])), fontsize=15, fontweight='bold')

        # plt.tight_layout()
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'ang_resol_bw' + str(params.SCALE_BW) + '.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        # print("quality trace:", ang_resol_list[params.BUFFER_LENGTH:])
        # print("mean quality:", np.mean(ang_resol_list[2 * params.BUFFER_LENGTH:]))
        # print("var quality:", np.std(ang_resol_list[2 * params.BUFFER_LENGTH:]))

    def plot_frame_quality_per_degree_trace(self, frame_quality_per_degree_lists):
        ##################### save as pickle ########################
        pkl_file_name = 'frame_quality_per_degree_bw' + str(params.SCALE_BW) + '.pkl'
        save_path = os.path.join(self.directory, pkl_file_name)

        with open(save_path, 'wb+') as file:
        	pk.dump(frame_quality_per_degree_lists, file)
        file.close()
        ##############################################################

        frame_indexes = range(len(frame_quality_per_degree_lists) - params.BUFFER_LENGTH)
        legend = []
        plt.plot(frame_indexes,
                 frame_quality_per_degree_lists[params.BUFFER_LENGTH:],
                 linewidth=1,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_per_degree_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, frame_quality_per_degree_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1, color='gray')

        plt.title('Per-Degree Quality per Frame',
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')

        plt.ylabel('quality', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        # plt.yticks(range(0, 220, 20), fontsize=15)

        # plt.text(250, 20, 'mean={:f}, std={:f}'.format(np.mean(frame_quality_per_degree_lists[2 * params.BUFFER_LENGTH:]), np.std(frame_quality_per_degree_lists[2 * params.BUFFER_LENGTH:])), fontsize=15, fontweight='bold')

        # plt.tight_layout()
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'frame_quality_per_degree_bw' + str(params.SCALE_BW) + '.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        # print("quality trace:", frame_quality_per_degree_lists[params.BUFFER_LENGTH:])
        # print("mean quality:", np.mean(frame_quality_per_degree_lists[2 * params.BUFFER_LENGTH:]))
        # print("var quality:", np.std(frame_quality_per_degree_lists[2 * params.BUFFER_LENGTH:]))

    def plot_frame_quality_trace(self, frame_quality_lists):
        ##################### save as pickle ########################
        pkl_file_name = 'frame_quality_bw' + str(params.SCALE_BW) + '.pkl'
        save_path = os.path.join(self.directory, pkl_file_name)

        with open(save_path, 'wb+') as file:
        	pk.dump(frame_quality_lists, file)
        file.close()
        ##############################################################

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

        plt.title('Frame Quality',
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')

        plt.ylabel('quality', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        # plt.yticks(range(0, 220, 20), fontsize=15)

        # plt.text(250, 20, 'mean={:f}, std={:f}'.format(np.mean(frame_quality_lists[2 * params.BUFFER_LENGTH:]), np.std(frame_quality_lists[2 * params.BUFFER_LENGTH:])), fontsize=15, fontweight='bold')

        # plt.tight_layout()
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'frame_quality_bw' + str(params.SCALE_BW) + '.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        # print("quality trace:", frame_quality_lists[params.BUFFER_LENGTH:])
        # print("mean quality:", np.mean(frame_quality_lists[2 * params.BUFFER_LENGTH:]))
        # print("var quality:", np.std(frame_quality_lists[2 * params.BUFFER_LENGTH:]))

    def plot_wasted_rate_trace(self, wasted_rate, visible_rate):
        ##################### save as pickle ########################
        pkl_file_name = 'wasted_rates_bw' + str(params.SCALE_BW) + '.pkl'
        save_path = os.path.join(self.directory, pkl_file_name)

        with open(save_path, 'wb+') as file:
            pk.dump(wasted_rate, file)
        file.close()

        pkl_file_name = 'visible_rate_bw' + str(params.SCALE_BW) + '.pkl'
        save_path = os.path.join(self.directory, pkl_file_name)

        with open(save_path, 'wb+') as file:
            pk.dump(visible_rate, file)
        file.close()
        ##############################################################

        frame_indexes = range(len(wasted_rate))
        legend = []
        plt.plot(frame_indexes,
                 wasted_rate[:],
                 linewidth=1,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1, color='gray')
        plt.title('Wasted Rates per Frame',
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')
        plt.ylabel('wasted rates / byte', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        # plt.yticks(range(0, 220, 20), fontsize=15)
 
        # plt.text(250, 20, 'mean={:f}, std={:f}'.format(np.mean(wasted_rate[1 * params.BUFFER_LENGTH:]), np.std(wasted_rate[1 * params.BUFFER_LENGTH:])), fontsize=15, fontweight='bold')

        # plt.tight_layout()
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'wasted_rates_bw' + str(params.SCALE_BW) + '.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_wasted_ratio_trace(self, wasted_ratio):
        ##################### save as pickle ########################
        pkl_file_name = 'wasted_ratio_bw' + str(params.SCALE_BW) + '.pkl'
        save_path = os.path.join(self.directory, pkl_file_name)

        with open(save_path, 'wb+') as file:
            pk.dump(wasted_ratio, file)
        file.close()
        ##############################################################

        frame_indexes = range(len(wasted_ratio))
        legend = []
        plt.plot(frame_indexes,
                 wasted_ratio[:],
                 linewidth=1,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1, color='gray')
        plt.title('Wasted Ratio per Frame',
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')
        plt.ylabel('wasted ratio', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        # plt.yticks(range(0, 220, 20), fontsize=15)
 
        # plt.text(250, 20, 'mean={:f}, std={:f}'.format(np.mean(wasted_ratio[1 * params.BUFFER_LENGTH:]), np.std(wasted_ratio[1 * params.BUFFER_LENGTH:])), fontsize=15, fontweight='bold')

        # plt.tight_layout()
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'wasted_ratio_bw' + str(params.SCALE_BW) + '.eps'
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
        plt.ylabel('quality variance', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(range(0, 14, 2), fontsize=15)
        # plt.tight_layout()
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'frame_quality_var_bw' + str(params.SCALE_BW) + '.eps'
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
        mean_quality = np.array(frame_quality_lists[params.BUFFER_LENGTH:]) /np.array(num_valid_tiles_per_frame[params.BUFFER_LENGTH:])
        plt.plot(frame_indexes,
                 mean_quality,
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
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')
        plt.ylabel('mean tile quality', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        # plt.yticks(np.linspace(0, 3, 13), fontsize=15)
        plt.yticks(fontsize=15)
        # plt.tight_layout()
        plt.text(250, 0.5, 'mean={:f}, std={:f}'.format(np.mean(mean_quality[300:]), np.std(mean_quality[300:])), fontsize=15, fontweight='bold')
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'frame_mean_tile_quality_bw' + str(params.SCALE_BW) + '_constWave.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        # pdb.set_trace()

    def plot_per_degree_quality_per_frame_trace(self, frame_quality_lists,
                                               total_span_per_frame):
        frame_indexes = range(len(frame_quality_lists) - params.BUFFER_LENGTH)
        legend = []
        # print(frame_quality_lists[params.BUFFER_LENGTH:])
        # print(total_span_per_frame[params.BUFFER_LENGTH:])
        # print(np.array(frame_quality_lists[params.BUFFER_LENGTH:]) / np.array(total_span_per_frame[params.BUFFER_LENGTH:]))
        mean_quality = np.array(frame_quality_lists[params.BUFFER_LENGTH:]) /np.array(total_span_per_frame[params.BUFFER_LENGTH:])
        plt.plot(frame_indexes,
                 mean_quality,
                 linewidth=2,
                 color='red')
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        legend.append('know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('Per Degree Quality per Frame of Longdress, Latency=%ds' %
                  (params.TARGET_LATENCY // params.FPS),
                  fontsize=20,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')
        plt.ylabel('per degree quality', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        # plt.yticks(np.linspace(0, 3, 13), fontsize=15)
        plt.yticks(fontsize=15)
        # plt.tight_layout()
        plt.text(250, 1, 'mean={:f}, std={:f}'.format(np.mean(mean_quality[300:]), np.std(mean_quality[300:])), fontsize=15, fontweight='bold')
        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'frame_per_degree_quality_bw' + str(params.SCALE_BW) + '.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

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
        plt.title('Mean Size over Tiles Per Frame of Longdress',
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
            len(mean_size_over_tiles_per_fov))
        legend = []
        plt.plot(frame_indexes,
                 mean_size_over_tiles_per_fov[:],
                 linewidth=2)
        # legend.append('constant Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[1][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='blue')
        # legend.append('linear (flat) Wj, know BW')
        # plt.plot(frame_indexes, frame_quality_lists[2][params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE * params.FPS:], linewidth=2, color='green')
        # legend.append('linear (steep) Wj, know BW')
        # plt.legend(legend, fontsize=30, loc='best', ncol=1)
        plt.text(0, 700, 'mean={:f}, std={:f}'.format(np.mean(mean_size_over_tiles_per_fov[1 * params.BUFFER_LENGTH:]), np.std(mean_size_over_tiles_per_fov[1 * params.BUFFER_LENGTH:])), fontsize=15, fontweight='bold')
        plt.grid(linestyle='dashed', axis='y', linewidth=1.5, color='gray')
        plt.title('# Mean Visible Tile Size of Longdress',
                  fontsize=40,
                  fontweight='bold')

        plt.xlabel('frame idx', fontsize=15, fontweight='bold')
        plt.ylabel('Mean Size / Byte', fontsize=15, fontweight='bold')

        plt.xticks(fontsize=15)
        # plt.yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], fontsize=30)
        plt.yticks(fontsize=15)
        # plt.tight_layout()

        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'mean_visible_tile_size_bw' + str(params.SCALE_BW) + '.eps'
            save_path = os.path.join(self.directory, file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

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
        num_valid_accuracy = (params.NUM_FRAMES_VIEWED # - params.BUFFER_LENGTH
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
        fig.suptitle("FoV prediction Accuracy", fontsize=8, fontweight='bold')
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
                                         fontsize=8,
                                         prop={'weight': 'bold'},
                                         loc='best',
                                         ncol=1)
            axs[pos_row, pos_col].set_title(key,
                                            fontsize=8,
                                            fontweight='bold')
            axs[pos_row, pos_col].set_ylabel('Absolute Error',
                                             fontsize=8)  # Y label
            axs[pos_row, pos_col].set_xlabel('time/s', fontsize=8)  # X label
            axs[pos_row, pos_col].tick_params(axis='both', labelsize=8)
            axs[pos_row, pos_col].grid(linestyle='dashed',
                                       axis='y',
                                       linewidth=1.5,
                                       color='gray')
            # axs[pos_row, pos_col].label_outer()

        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'fov_pred_absError_vs_time_buf' + str(params.TARGET_LATENCY // params.FPS) + 's.eps'
            save_path = os.path.join('fov_pred', file_name)
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


    def plot_mean_fov_prediction_accuracy_for_every_buffer_pos(
            self, fov_predict_accuracy_trace):
        '''
			6 subplots for 6dof
		'''
        # pdb.set_trace()
        num_frames = len(fov_predict_accuracy_trace['x'])
        frame_indexes = range(num_frames)
        num_updates = len(fov_predict_accuracy_trace['x'][0])
        num_valid_accuracy = (params.NUM_FRAMES_VIEWED # - params.BUFFER_LENGTH
                              ) // (params.UPDATE_FREQ * params.FPS)
        ###########################
        # x 	  y 	  z
        # pitch   yaw	 roll
        ###########################
        fig, axs = plt.subplots(2, 3, sharex='col', sharey='row')
        fig.suptitle("Mean FoV prediction Accuracy (MAE)",
                     fontsize=8,
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

            # with open('./fov_pred/mae_vs_buflen_' + key + '.pkl', 'wb+') as filehandle:
            # 	pk.dump(np.array(fov_predict_accuracy_trace[key])[:, :num_valid_accuracy].mean(axis=1), filehandle)
            # filehandle.close()

            # pdb.set_trace()
            # legend.append(map_from_pos_in_buffer_to_legend[pos_in_buffer])
            # axs[pos_row, pos_col].legend(legend, fontsize=8, prop={'weight':'bold'}, loc='best', ncol=1)
            axs[pos_row, pos_col].set_title(key,
                                            fontsize=8,
                                            fontweight='bold')
            if key == 'x' or key == 'pitch':
                axs[pos_row, pos_col].set_ylabel('Mean Absolute Error',
                                                 fontsize=8)  # Y label
            if key == 'yaw' or key == 'pitch' or key == 'roll':
                axs[pos_row, pos_col].set_xlabel('frame idx',
                                                 fontsize=8)  # X label
            axs[pos_row, pos_col].tick_params(axis='both', labelsize=8)
            axs[pos_row, pos_col].grid(linestyle='dashed',
                                       axis='y',
                                       linewidth=1.5,
                                       color='gray')
            # axs[pos_row, pos_col].label_outer()

        if params.SAVE_WHEN_PLOTTING:
            # The final path to save to
            file_name = 'fov_pred_mae_vs_hor_buf' + str(params.TARGET_LATENCY // params.FPS) + 's.eps'
            save_path = os.path.join('fov_pred', file_name)
            plt.savefig(save_path, transparent=True) #, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def save_frame_lod_list(self, frame_lod_list):
        file_name = 'frame_lod_list_bw' + str(params.SCALE_BW) + '.pkl'
        save_path = os.path.join(self.directory, file_name)

        with open(save_path, 'wb+') as file:
        	pk.dump(frame_lod_list, file)
        file.close()
