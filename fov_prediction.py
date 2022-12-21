
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


        for key in self.true_viewpoints.keys():
            for frame_idx in range(
                    self.buffer_length + params.FPS * params.UPDATE_FREQ -
                    params.TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE *
                    params.FPS):
                self.true_viewpoints[key].append([])
                self.fov_predict_accuracy_trace[key].append([])


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


    def predict_viewpoint(self, predict_start_idx, predict_end_idx):
        predicted_viewpoints = self.fov_traces_obj.predict_6dof(
            self.current_viewing_frame_idx, predict_start_idx, predict_end_idx,
            self.history_viewpoints)
        return predicted_viewpoints



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
