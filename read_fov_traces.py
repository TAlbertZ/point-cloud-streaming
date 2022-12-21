import numpy as np
import pickle as pk
import pdb
from sklearn.linear_model import LinearRegression
import pandas as pd

import params

np.random.seed(7)


class FovTraces():
    '''
		read and operate 6dof trace
	'''

    def __init__(self):
        self.fov_traces = None
        self.fov_predict_mae_vs_buflen = {
            "x": [],
            'y': [],
            "z": [],
            "pitch": [],
            "yaw": [],
            "roll": []
        }
        for key in params.MAP_DOF_TO_PLOT_POS:
            file = open('./fov_pred/mae_vs_buflen_' + key + '.pkl', 'rb')
            self.fov_predict_mae_vs_buflen[key] = pk.load(file)
            file.close()

    def read_fov_traces(self, path):
        '''
			read 6dof of every frame
		'''

        self.fov_traces = pd.read_csv(path)
        # pdb.set_trace()

    def padding(self):
        hmdpx = self.fov_traces.at[0, 'HMDPX']
        hmdpy = self.fov_traces.at[0, 'HMDPY']
        hmdpz = self.fov_traces.at[0, 'HMDPZ']
        hmdrx = self.fov_traces.at[0, 'HMDRX']
        hmdry = self.fov_traces.at[0, 'HMDRY']
        hmdrz = self.fov_traces.at[0, 'HMDRZ']
        for _ in range(params.BUFFER_LENGTH):
            new_row = pd.DataFrame({'FrameNumber':-1, \
                  'HMDPX': hmdpx, \
                  'HMDPY': hmdpy, \
                  'HMDPZ': hmdpz, \
                  'HMDRX': hmdrx, \
                  'HMDRY': hmdry, \
                  'HMDRZ': hmdrz}, index=[0])
            self.fov_traces = pd.concat([new_row, self.fov_traces
                                         ]).reset_index(drop=True)
        # pdb.set_trace()

    def read_fov_traces_txt(self, path, num_lines):
        self.fov_traces = []
        line_idx = 0
        f = open(path, "r")
        for line in f:
            line_list = line.split()
            self.fov_traces.append([
                float(line_list[1]),
                float(line_list[2]),
                float(line_list[3]),
                float(line_list[4]),
                float(line_list[5]),
                float(line_list[6])
            ])

            line_idx += 1
            if line_idx == num_lines:
                break

    def padding_txt(self):
        viewpoint_6dof = self.fov_traces[0].copy()
        for _ in range(params.BUFFER_LENGTH):
            self.fov_traces.insert(0, viewpoint_6dof)

    def predict_6dof(self, current_viewing_frame_idx, predict_start_idx,
                     predict_end_idx, history_viewpoints):
        # ARMA-style prediction
        # record prediction accuracy (difference from ground truth) for all frames

        # each dof maintains a list including all predicted frames from [predict_start_idx, predict_end_idx]
        predicted_viewpoints = {
            "x": [],
            'y': [],
            "z": [],
            "pitch": [],
            "yaw": [],
            "roll": []
        }

        for frame_idx in range(predict_start_idx, predict_end_idx + 1):
            prediction_win = frame_idx - current_viewing_frame_idx
            history_win = prediction_win // 2  # according to vivo paper, best use half of prediciton window
            history_win = history_win if len(
                history_viewpoints['x']) >= history_win else len(
                    history_viewpoints['x'])
            # x_list = np.arange(history_win).reshape(-1, 1)
            # print("index: ", predict_start_idx, " ", predict_end_idx)
            # print("win: ", prediction_win, " ", history_win)
            # print("x: ", x_list)
            for key in predicted_viewpoints.keys():
                # print("key: ", key)
                truncated_idx = self.truncate_trace(
                    history_viewpoints[key][-history_win:])
                # if truncated_idx != 0:
                #     print("truncated_idx is: ", truncated_idx)
                #     print("key is: ", key, ", history win is: ", history_win, ", pred win is: ", prediction_win)
                #     print("history seq:\n", history_viewpoints[key][-history_win:])
                #     pdb.set_trace()
                x_list = np.arange(history_win - truncated_idx).reshape(-1, 1)
                y_list = np.array(
                    history_viewpoints[key][-history_win +
                                            truncated_idx:]).reshape(-1, 1)
                # print("y: ", y_list)
                reg = LinearRegression().fit(x_list, y_list)
                predicted_dof = reg.predict(
                    [[history_win + prediction_win - 1]])[0][0]
                if key == 'pitch' or key == 'yaw' or key == 'roll':
                    if predicted_dof >= 360:
                        predicted_dof -= 360

                if not params.FOV_ORACLE_KNOW:
                    if params.FAKE_MAE_ERROR:
                        sign = 1 if np.random.random() < 0.5 else -1
                        predicted_viewpoints[key].append(
                            self.fov_traces[frame_idx][
                                params.MAP_6DOF_TO_HMD_DATA[key]] +
                            sign * self.fov_predict_mae_vs_buflen[key]
                            [prediction_win - 1 - params.FPS * params.
                             TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE] * 0.01)
                        # print(prediction_win - 1 - params.FPS * params.
                        #      TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE)
                        # pdb.set_trace()
                    else:
                        predicted_viewpoints[key].append(predicted_dof)
                else:
                    ### know future fov oracle ##
                    predicted_viewpoints[key].append(
                        self.fov_traces[frame_idx][
                            params.MAP_6DOF_TO_HMD_DATA[key]])
                    ##########################

                # print("result: ", predicted_viewpoints)
        # pdb.set_trace()

        return predicted_viewpoints

    def truncate_trace(self, trace):
        tail_idx = len(trace) - 2
        # print(len(trace))
        # while tail_idx >=0 and trace[tail_idx] == trace[tail_idx+1]:
        # 	tail_idx -= 1
        # if tail_idx < 0 :
        # 	return time_trace, trace
        current_sign = np.sign(trace[tail_idx + 1] -
                               trace[tail_idx])  # Get real sign, in order

        # If 0 range is large, no linear
        # Truncate trace
        while tail_idx >= 0:
            # if np.sign(trace[tail_idx+1] - trace[tail_idx]) == current_sign or abs(trace[tail_idx+1] - trace[tail_idx]) <= 1:
            if np.sign(trace[tail_idx + 1] - trace[tail_idx]) == current_sign:
                tail_idx -= 1
            else:
                break
        # truncated_trace = trace[tail_idx+1:]
        return tail_idx + 1
