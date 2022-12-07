# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Instantaneous Cost Function."""

import numpy as np
import logging
import warnings
import pdb
from scipy.optimize import approx_fprime

import params
from params import FrameWeightType


# TODO by Tongyu: check value of quality_sum and quality_var and barriers,
# set scale params for them, and adjust shapes of barriers,
# also check frame_quality and frame_quality_var in evaluation module.
class AutoDiffCost():
    """Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self):
        """Constructs an AutoDiffCost.

        Args:
            l: Vector Theano tensor expression for instantaneous cost.
                This needs to be a function of x and u and must return a scalar.
            l_terminal: Vector Theano tensor expression for terminal cost.
                This needs to be a function of x only and must retunr a scalar.
            x_inputs: Theano state input variables [state_size].
            u_inputs: Theano action input variables [action_size].
            i: Theano tensor time step variable.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        # self._i = T.dscalar("i") if i is None else i
        # self._x_inputs = x_inputs
        # self._u_inputs = u_inputs

        # non_t_inputs = np.hstack([x_inputs, u_inputs]).tolist()
        # inputs = np.hstack([x_inputs, u_inputs, self._i]).tolist()
        # terminal_inputs = np.hstack([x_inputs, self._i]).tolist()

        # x_dim = len(x_inputs)
        # u_dim = len(u_inputs)

        # self._J = jacobian_scalar(l, non_t_inputs)
        # self._Q = hessian_scalar(l, non_t_inputs)

        # self._l = as_function(l, inputs, name="l", **kwargs)

        # self._l_x = as_function(self._J[:x_dim], inputs, name="l_x", **kwargs)
        # self._l_u = as_function(self._J[x_dim:], inputs, name="l_u", **kwargs)

        # self._l_xx = as_function(self._Q[:x_dim, :x_dim],
        #                          inputs,
        #                          name="l_xx",
        #                          **kwargs)
        # self._l_ux = as_function(self._Q[x_dim:, :x_dim],
        #                          inputs,
        #                          name="l_ux",
        #                          **kwargs)
        # self._l_uu = as_function(self._Q[x_dim:, x_dim:],
        #                          inputs,
        #                          name="l_uu",
        #                          **kwargs)
        self.weighted_quality_sum = 0.0
        self.quality_var = 0.0
        self.tile_utility_coef = params.TILE_UTILITY_COEF
        self.quality_mat_prev_unupdateable_fr = np.zeros(
            (params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))

        self.logger = self.set_logger(logger_level=logging.INFO)
        self.logger_check_cost = logging.getLogger(
            'buffer_control_ilqr.check_cost')

        # regard all runtime warnings as errors, for debug use
        # warnings.simplefilter('error')

    def set_logger(self,
                   logger_name=__name__,
                   logger_level=logging.DEBUG,
                   logger_propagate=False):
        logger = logging.getLogger(logger_name)
        logger.propagate = logger_propagate
        logger.setLevel(logger_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s - %(lineno)d - %(module)s\n%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def x(self):
        """The state variables."""
        return self._x_inputs

    def u(self):
        """The control variables."""
        return self._u_inputs

    def i(self):
        """The time step variable."""
        return self._i

    def get_quality_for_tiles(self, num_points_per_degree):
        quality_tiles = np.log(num_points_per_degree)
        return quality_tiles

    def l(self,
          x,
          u,
          i,
          bandwidth_budget,
          viewing_probability,
          distances,
          tile_a,
          tile_b,
          update_start_idx,
          update_end_idx,
          dynamics,
          buffer_,
          rate_versions,
          terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
            buffer_: real current buffered tiles, shape: [buffer_length,
                                                    NUM_TILES_PER_SIDE_IN_A_FRAME,
                                                    NUM_TILES_PER_SIDE_IN_A_FRAME,
                                                    NUM_TILES_PER_SIDE_IN_A_FRAME]

        Returns:
            Instantaneous cost (scalar).
        """
        # if terminal:
        #     z = np.hstack([x, i])
        #     return np.asscalar(self._l_terminal(*z))

        ############### quality_sum ##########
        self.max_rates = rate_versions[0]
        self.current_cost = 0
        self.current_state_size = x.shape[0]
        self.current_action_size = self.current_state_size
        self.dynamics = dynamics
        self.tile_a = tile_a
        self.tile_b = tile_b
        self.viewing_probability = viewing_probability
        self.distances = distances
        self.bandwidth_budget = bandwidth_budget
        self.update_start_idx = update_start_idx
        self.update_end_idx = update_end_idx
        self.weighted_quality_sum = 0.0
        self.quality_var = 0.0
        self.quality_tiles = np.zeros(
            (self.dynamics.num_old_tiles_removed_from_current_state, ))
        self.tiles_rates_to_be_watched = self.dynamics.updated_current_state[:
                                                                             self
                                                                             .
                                                                             dynamics
                                                                             .
                                                                             num_old_tiles_removed_from_current_state].copy(
                                                                             )

        # TODO by Tongyu: correct start_frame_idx_within_video, it should be exactly equal to self.tile_a_start_idx
        start_frame_idx_within_video = (self.update_start_idx +
                                        i * params.FPS) % params.TARGET_LATENCY

        # shape: [FPS, NUM_TILES_PER_SIDE_IN_A_FRAME, ..., ...]
        self.viewing_probability_of_interest = self.viewing_probability[
            i * params.FPS:(i + 1) * params.FPS].copy()

        # at the beginning of streaming, buffer is almost empty
        # quality = 0
        if not self.viewing_probability_of_interest.any():
            # only barrier functions play a role
            try:
                self.zero_bound_barrier = -np.sum(np.log(u))
            except Exception as err:
                # warnings.warn(str(err))
                self.logger.warning(str(err))
                pdb.set_trace()
            self.updated_tiles_pos = self.viewing_probability[
                i * params.FPS:i * params.FPS +
                params.TARGET_LATENCY].nonzero()
            # self.logger.debug('updated_tiles_pos: %s\nstart_frame_idx_within_video: %d', self.updated_tiles_pos, start_frame_idx_within_video)

            update_time_step = self.update_start_idx // params.FPS

            # buffer is not fully occupied yet
            if update_time_step + i < params.TARGET_LATENCY // params.FPS:
                self.max_rates_cur_step = np.concatenate(
                    (np.zeros(
                        (params.TARGET_LATENCY - start_frame_idx_within_video,
                         params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                         params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                         params.NUM_TILES_PER_SIDE_IN_A_FRAME)),
                     self.max_rates[:start_frame_idx_within_video]),
                    axis=0)[self.updated_tiles_pos]

            else:
                # looks wierd here, that's because the video is played repeatedly, so
                # every 1s the 'self.max_rates' should be circularly shifted!
                self.max_rates_cur_step = np.concatenate(
                    (self.max_rates[(update_start_idx + i * params.FPS -
                                     params.TARGET_LATENCY) %
                                    params.NUM_FRAMES:],
                     self.max_rates[:(update_start_idx + i * params.FPS -
                                      params.TARGET_LATENCY) %
                                    params.NUM_FRAMES]),
                    axis=0)[self.updated_tiles_pos]

            try:
                self.upper_bound_barrier = -np.sum(
                    np.log(self.max_rates_cur_step -
                           self.dynamics.updated_current_state))
            except Exception as err:
                # warnings.warn(str(err))
                self.logger.warning(str(err))
                pdb.set_trace()

            self.sum_u = np.sum(u)

            self.bw_bound_barrier = -np.log(self.bandwidth_budget[i] -
                                            self.sum_u)

            self.current_cost = (self.zero_bound_barrier +
                                 self.bw_bound_barrier) * params.BARRIER_WEIGHT

            return self.current_cost

        # tiles that finish updating, i.e., the first params.FPS frames
        self.tile_of_interest_pos = self.viewing_probability_of_interest.nonzero(
        )

        # update_start_idx+i*FPS should be >= TARGET_LATENCY here,
        # because viewing_probability_of_interest.any() is true here (according to previous if(.))
        assert (self.update_start_idx + i * params.FPS >= params.TARGET_LATENCY
                ), "update_start_idx + i * FPS should be >= TARGET_LATENCY"
        self.tile_a_start_idx = (self.update_start_idx + i * params.FPS -
                                 params.TARGET_LATENCY) % params.NUM_FRAMES
        self.a_updated_tiles = self.tile_a[
            self.tile_a_start_idx:self.tile_a_start_idx +
            params.FPS][self.tile_of_interest_pos]
        self.b_updated_tiles = self.tile_b[
            self.tile_a_start_idx:self.tile_a_start_idx +
            params.FPS][self.tile_of_interest_pos]

        self.distances = np.array(self.distances)
        self.distance_of_interest = self.distances[i * params.FPS:(
            i + 1) * params.FPS][self.tile_of_interest_pos]
        self.frame_weights = np.ones_like(self.tiles_rates_to_be_watched)
        if params.FRAME_WEIGHT_TYPE == FrameWeightType.CONST:
            self.frame_weights = np.ones_like(self.tiles_rates_to_be_watched) * params.QUALITY_SUM_WEIGHT
        ### TODO by Tongyu: add more frame weight types

        self.theta_of_interest = params.TILE_SIDE_LEN / self.distance_of_interest * params.RADIAN_TO_DEGREE
        # self.num_points = self.a_updated_tiles * self.tiles_rates_to_be_watched + self.b_updated_tiles
        # self.num_points_per_degree = self.num_points**0.5 / self.theta_of_interest  # f_ang
        try:
            self.lod = self.a_updated_tiles * np.log(
                self.b_updated_tiles * self.tiles_rates_to_be_watched + 1)
            self.num_points_per_degree = 2**self.lod / self.theta_of_interest
        except Exception as err:
            pdb.set_trace()
        self.quality_tiles = self.get_quality_for_tiles(
            self.num_points_per_degree)  # update self.quality_tiles

        self.weighted_quality_sum = np.dot(self.quality_tiles,
                                           self.frame_weights.T)

        ################# quality_var ####################

        # mark as 1 for overlap tiles for each pair of adjacent frames,
        # from 1st frame to FPS_th frame in this update round
        self.mask_overlap_tiles_in_mat = np.zeros(
            (params.FPS, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        self.quality_diff = np.zeros(
            (params.FPS, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        self.num_overlap_tiles_adjacent_frames = np.zeros((params.FPS, ))
        self.quality_var_adjacent_frames = np.zeros((params.FPS, ))

        # convenient for calculating quality_var and quality_var_x
        self.quality_tiles_in_mat = self.viewing_probability_of_interest.copy()
        self.quality_tiles_in_mat[
            self.tile_of_interest_pos] = self.quality_tiles

        # index of tiles; shape: [FPS, NUM_TILES_PER_SIDE_IN_A_FRAME, ..., ...]
        # start from 0.
        self.order_nonzero_view_prob_of_interest = self.viewing_probability_of_interest.copy(
        )
        self.order_nonzero_view_prob_of_interest[
            self.tile_of_interest_pos] = np.arange(
                self.dynamics.num_old_tiles_removed_from_current_state)
        self.order_nonzero_view_prob_of_interest = self.order_nonzero_view_prob_of_interest.astype(
            int)

        # FIXME [URGENT] by Tongyu: not buffer_ here, should be quality of previous frame
        # if i==0: recalculate quality matrix of previous frame, elif i>0:refer to self.prev_fr_quality_tiles_in_mat
        if i == 0:
            tile_of_interest_pos_prev_unupdateable_fr = buffer_[params.FPS -
                                                                1].nonzero()
            viewing_probability_prev_unupdateable_fr = np.zeros_like(
                buffer_[params.FPS - 1])
            viewing_probability_prev_unupdateable_fr[
                tile_of_interest_pos_prev_unupdateable_fr] = 1

            self.mask_overlap_tiles_in_mat[
                0] = viewing_probability_prev_unupdateable_fr * self.viewing_probability_of_interest[
                    0]
            self.num_overlap_tiles_adjacent_frames[0] = np.count_nonzero(
                self.mask_overlap_tiles_in_mat[0])
            # if self.num_overlap_tiles_adjacent_frames[0] != 0:

            #     # calculate tile_quality_mat of previous frame
            #     distance_of_interest_last_frame = distances[visible_tiles_pos]

            #     theta_of_interest_last_fr = params.TILE_SIDE_LEN / distance_of_interest * params.RADIAN_TO_DEGREE

            #     a_tiles_of_interest_last_fr = self.tile_a[frame_idx_within_video][
            #         visible_tiles_pos]
            #     b_tiles_of_interest_last_fr = self.tile_b[frame_idx_within_video][
            #         visible_tiles_pos]
            #     tiles_rates_to_be_watched_last_fr = tiles_byte_sizes[visible_tiles_pos]

            #     lod_last_fr = a_tiles_of_interest_last_fr*np.log(b_tiles_of_interest_last_fr * tiles_rates_to_be_watched_last_fr+1)
            #     num_points_per_degree_last_fr = 2**lod_last_fr / theta_of_interest_last_fr  # f_ang
            #     quality_tiles_last_fr = self.cost.get_quality_for_tiles(
            #         num_points_per_degree_last_fr)  # update quality_tiles

            #     quality_tiles_in_mat_last_fr = np.zeros_like(tiles_byte_sizes)
            #     quality_tiles_in_mat_last_fr[visible_tiles_pos] = quality_tiles_last_fr

            #     self.quality_diff[0] = (
            #         quality_tiles_in_mat_last_fr -
            #         self.quality_tiles_in_mat[0]) * self.mask_overlap_tiles_in_mat[0]
        else:
            tile_of_interest_pos_prev_unupdateable_fr = self.quality_mat_prev_unupdateable_fr.nonzero(
            )
            viewing_probability_prev_unupdateable_fr = np.zeros_like(
                self.quality_mat_prev_unupdateable_fr)
            viewing_probability_prev_unupdateable_fr[
                tile_of_interest_pos_prev_unupdateable_fr] = 1

            self.mask_overlap_tiles_in_mat[
                0] = viewing_probability_prev_unupdateable_fr * self.viewing_probability_of_interest[
                    0]
            self.num_overlap_tiles_adjacent_frames[0] = np.count_nonzero(
                self.mask_overlap_tiles_in_mat[0])
        self.quality_mat_prev_unupdateable_fr = self.quality_tiles_in_mat[
            params.FPS - 1].copy()

        if self.num_overlap_tiles_adjacent_frames[0] != 0:
            self.quality_var_adjacent_frames[0] = np.sum(
                self.quality_diff[0]**
                2) / self.num_overlap_tiles_adjacent_frames[0]

        for frame_idx in range(1, params.FPS):
            # a mask for overlapped tiles in each pair of adjacent frames
            self.mask_overlap_tiles_in_mat[
                frame_idx] = self.viewing_probability_of_interest[
                    frame_idx -
                    1] * self.viewing_probability_of_interest[frame_idx]

            self.quality_diff[frame_idx] = (
                self.quality_tiles_in_mat[frame_idx - 1] -
                self.quality_tiles_in_mat[frame_idx]
            ) * self.mask_overlap_tiles_in_mat[frame_idx]

            self.num_overlap_tiles_adjacent_frames[
                frame_idx] = np.count_nonzero(self.quality_diff[frame_idx])
            self.quality_var_adjacent_frames[frame_idx] = np.sum(
                self.quality_diff[frame_idx]**
                2) / self.num_overlap_tiles_adjacent_frames[frame_idx]

        self.sum_quality_var = np.sum(self.quality_var_adjacent_frames)

        ##########################################################

        ################# barrier functions ##########################
        try:
            self.zero_bound_barrier = -np.sum(np.log(u))
        except Exception as err:
            # warnings.warn(str(err))
            self.logger.warning(str(err))
            pdb.set_trace()
        self.updated_tiles_pos = self.viewing_probability[
            i * params.FPS:i * params.FPS + params.TARGET_LATENCY].nonzero()

        update_time_step = self.update_start_idx // params.FPS

        # buffer is not fully occupied yet
        if update_time_step + i < params.TARGET_LATENCY // params.FPS:
            self.max_rates_cur_step = np.concatenate(
                (np.zeros(
                    (params.TARGET_LATENCY - start_frame_idx_within_video,
                     params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                     params.NUM_TILES_PER_SIDE_IN_A_FRAME,
                     params.NUM_TILES_PER_SIDE_IN_A_FRAME)),
                 self.max_rates[:start_frame_idx_within_video]),
                axis=0)[self.updated_tiles_pos]

        else:
            # looks wierd here, that's because the video is played repeatedly, so
            # every 1s the 'self.max_rates' should be circularly shifted!
            self.max_rates_cur_step = np.concatenate(
                (self.max_rates[(update_start_idx + i * params.FPS -
                                 params.TARGET_LATENCY) % params.NUM_FRAMES:],
                 self.max_rates[:(update_start_idx + i * params.FPS -
                                  params.TARGET_LATENCY) % params.NUM_FRAMES]),
                axis=0)[self.updated_tiles_pos]

        try:
            self.upper_bound_barrier = -np.sum(
                np.log(self.max_rates_cur_step -
                       self.dynamics.updated_current_state))
        except Exception as err:
            # warnings.warn(str(err))
            self.logger.warning(str(err))
            pdb.set_trace()

        # only care about the first params.FPS frames among all updated tiles,
        # because only these tiles finish being updated and to be involved in
        # final visual quality.
        # self.lower_bound_barrier = -np.sum(
        #     np.log(
        #         self.dynamics.
        #         updated_current_state[:self.dynamics.
        #                               num_old_tiles_removed_from_current_state]
        #         + self.b_updated_tiles / self.a_updated_tiles))

        self.sum_u = np.sum(u)
        self.bw_bound_barrier = -np.log(self.bandwidth_budget[i] - self.sum_u)

        # TODO by Tongyu: weights of barrier function should be as small as possible! maybe no need to adjust inner weights of them. e.g., 1e-6 * zero_bound_barrier
        # if do so, all the derivatives also need to be modified with weights
        self.current_cost = -self.weighted_quality_sum + self.sum_quality_var * params.WEIGHT_VARIANCE + (
            self.zero_bound_barrier + 
            self.bw_bound_barrier) * params.BARRIER_WEIGHT

        if not np.isnan(self.current_cost):
            self.logger_check_cost.info(
                'weighted quality sum: %f\nsum quality var: %f\nzero bound barrier: %f\nupper_bound barrier: %f\nbw bound barrier: %f\ntile_rates: %s\nmax_rates: %s\nsum_u: %f\nbw_budget: %f',
                -self.weighted_quality_sum, self.sum_quality_var,
                self.zero_bound_barrier, self.upper_bound_barrier,
                self.bw_bound_barrier, self.dynamics.updated_current_state[:100], self.max_rates_cur_step[:100], self.sum_u, self.bandwidth_budget[i])

            # if self.sum_quality_var > 1e4:
            #     pdb.set_trace()

        return self.current_cost

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """

        ################# initialization ####################

        # cost of current step wrt. current state (non-updated)
        self.current_cost_x = np.zeros((self.current_state_size, ))
        self.quality_sum_x = np.zeros((self.current_state_size, ))
        self.quality_var_x = np.zeros((self.current_state_size, ))

        # zero_bound_barrier_x and bw_bound_barrier_x are all 0 since it only depends on actions u
        # self.upper_bound_barrier_x = np.zeros((self.current_state_size, ))
        # self.lower_bound_barrier_x = np.zeros((self.current_state_size, ))

        # at the beginning of streaming, buffer is almost empty
        # quality = 0
        if not self.viewing_probability_of_interest.any():
            # only barrier functions play a role
            self.upper_bound_barrier_x = 1 / (
                self.max_rates_cur_step - self.dynamics.updated_current_state)

            self.current_cost_x = np.zeros((self.current_state_size, ))

            return self.current_cost_x

        ################# quality_sum_x ####################
        self.quality_sum_x[:self.dynamics.
                           num_old_tiles_removed_from_current_state] = np.log(
                               2
                           ) * self.a_updated_tiles * self.b_updated_tiles / (
                               self.b_updated_tiles *
                               self.tiles_rates_to_be_watched + 1)
        self.weighted_quality_sum_x = self.quality_sum_x.copy()
        self.weighted_quality_sum_x[:self.dynamics.
                                    num_old_tiles_removed_from_current_state] *= self.frame_weights

        ################# quality_var_x ####################

        # TODO by Tongyu: assign equation number in the paper for "normalized_mask_overlap_tiles_in_mat"
        # 2nd addition term of equation (?) in the paper
        # for calculation of quality_var_xx
        self.normalized_mask_overlap_tiles_in_mat = np.zeros(
            (params.FPS, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))

        # TODO by Tongyu: assign equation number in the paper for "normalized_quality_diff_in_mat "
        # 2nd multiplication factor of equation (?) in the paper
        self.normalized_quality_diff_in_mat = np.zeros(
            (params.FPS, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        for frame_idx in range(params.FPS - 1):
            if self.num_overlap_tiles_adjacent_frames[frame_idx] == 0:
                quality_diff_first_part = 0
                mask_first_part = 0
            else:
                quality_diff_first_part = -self.quality_diff[
                    frame_idx] * 2 / self.num_overlap_tiles_adjacent_frames[
                        frame_idx]
                mask_first_part = self.mask_overlap_tiles_in_mat[
                    frame_idx] * 2 / self.num_overlap_tiles_adjacent_frames[
                        frame_idx]

            if self.num_overlap_tiles_adjacent_frames[frame_idx + 1] == 0:
                quality_diff_second_part = 0
                mask_second_part = 0
            else:
                quality_diff_second_part = self.quality_diff[
                    frame_idx +
                    1] * 2 / self.num_overlap_tiles_adjacent_frames[frame_idx +
                                                                    1]
                mask_second_part = self.mask_overlap_tiles_in_mat[
                    frame_idx +
                    1] * 2 / self.num_overlap_tiles_adjacent_frames[frame_idx +
                                                                    1]

            self.normalized_quality_diff_in_mat[
                frame_idx] = quality_diff_first_part + quality_diff_second_part

            self.normalized_mask_overlap_tiles_in_mat[
                frame_idx] = mask_first_part + mask_second_part

        if self.num_overlap_tiles_adjacent_frames[frame_idx] == 0:
            self.normalized_quality_diff_in_mat[frame_idx] = 0
            self.normalized_mask_overlap_tiles_in_mat[frame_idx] = 0
        else:
            self.normalized_quality_diff_in_mat[
                frame_idx] = -self.quality_diff[
                    frame_idx] * 2 / self.num_overlap_tiles_adjacent_frames[
                        frame_idx]

            self.normalized_mask_overlap_tiles_in_mat[
                frame_idx] = self.mask_overlap_tiles_in_mat[
                    frame_idx] * 2 / self.num_overlap_tiles_adjacent_frames[
                        frame_idx]

        # map "normalized_quality_diff_in_mat" from matrix to a vector,
        # which corresponds with "quality_sum_x" for multiplication.
        self.normalized_quality_diff_in_vec = self.normalized_quality_diff_in_mat[
            self.tile_of_interest_pos]

        # map "normalized_mask_overlap_tiles_in_mat" from matrix to a vector,
        # which corresponds with "quality_var_xx" for multiplication.
        self.normalized_mask_overlap_tiles_in_vec = self.normalized_quality_diff_in_mat[
            self.tile_of_interest_pos]

        self.quality_var_x[:self.dynamics.
                           num_old_tiles_removed_from_current_state] = self.quality_sum_x[:self
                                                                                          .
                                                                                          dynamics
                                                                                          .
                                                                                          num_old_tiles_removed_from_current_state] * self.normalized_quality_diff_in_vec

        ################# barrier functions ##########################
        self.upper_bound_barrier_x = 1 / (self.max_rates_cur_step -
                                          self.dynamics.updated_current_state)
        # self.lower_bound_barrier_x[:self.dynamics.
        #                            num_old_tiles_removed_from_current_state] = -1 / (
        #                                self.dynamics.
        #                                updated_current_state[:self.dynamics.
        #                                                      num_old_tiles_removed_from_current_state]
        #                                + self.b_updated_tiles /
        #                                self.a_updated_tiles)

        self.current_cost_x = -self.weighted_quality_sum_x + self.quality_var_x * params.WEIGHT_VARIANCE

        return self.current_cost_x

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [current_state_size].
            u: Current control [current_action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [current_action_size].
        """

        ################# initialization ####################
        self.current_cost_u = np.zeros((self.current_state_size, ))

        # at the beginning of streaming, buffer is almost empty
        # quality = 0
        if not self.viewing_probability_of_interest.any():
            # only barrier functions play a role
            self.zero_bound_barrier_u = -1 / u
            self.upper_bound_barrier_u = self.upper_bound_barrier_x.copy()
            self.bw_bound_barrier_u = 1 / (self.bandwidth_budget[i] -
                                           self.sum_u)

            self.current_cost_u = (
                self.zero_bound_barrier_u + 
                self.bw_bound_barrier_u) * params.BARRIER_WEIGHT

            return self.current_cost_u

        ################# quality_sum_u ####################
        self.quality_sum_u = self.quality_sum_x.copy()
        self.weighted_quality_sum_u = self.weighted_quality_sum_x.copy()

        ################# quality_var_u ####################
        self.quality_var_u = self.quality_var_x.copy()

        ################# barrier functions ##########################
        self.zero_bound_barrier_u = -1 / u
        self.upper_bound_barrier_u = self.upper_bound_barrier_x.copy()
        # self.lower_bound_barrier_u = self.lower_bound_barrier_x.copy()
        self.bw_bound_barrier_u = 1 / (self.bandwidth_budget[i] - self.sum_u)

        self.current_cost_u = -self.weighted_quality_sum_u + self.quality_var_u * params.WEIGHT_VARIANCE + (
            self.zero_bound_barrier_u + 
            self.bw_bound_barrier_u) * params.BARRIER_WEIGHT

        return self.current_cost_u

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [current_state_size].
            u: Current control [current_action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        ################# initialization ####################

        # zero_bound_barrier_xx and bw_bound_barrier_xx are all 0 since it only depends on actions u
        self.upper_bound_barrier_xx = np.zeros(
            (self.current_state_size, self.current_state_size))
        # self.lower_bound_barrier_xx = np.zeros(
        #     (self.current_state_size, self.current_state_size))

        # at the beginning of streaming, buffer is almost empty
        # quality = 0
        if not self.viewing_probability_of_interest.any():
            # only barrier functions play a role
            np.fill_diagonal(
                self.upper_bound_barrier_xx,
                -1 / (self.max_rates_cur_step -
                      self.dynamics.updated_current_state)**2)

            self.current_cost_xx = np.zeros((self.current_state_size, self.current_state_size))

            return self.current_cost_xx

        self.current_cost_xx = np.zeros(
            (self.current_state_size, self.current_state_size))
        self.weighted_quality_sum_xx = np.zeros(
            (self.current_state_size, self.current_state_size))
        self.quality_sum_xx_diagnl_elems = np.zeros(
            (self.dynamics.num_old_tiles_removed_from_current_state, ))
        self.quality_var_xx = np.zeros(
            (self.current_state_size, self.current_state_size))
        self.frame_weights_diagnl = np.diag(self.frame_weights)

        ################# quality_sum_xx ####################
        # a vector of diagnal elements of "quality_sum_xx"
        self.quality_sum_xx_diagnl_elems = -self.quality_sum_x[:self.dynamics.
                                                               num_old_tiles_removed_from_current_state] * self.b_updated_tiles / (
                                                                   self.
                                                                   b_updated_tiles
                                                                   * self.
                                                                   tiles_rates_to_be_watched
                                                                   + 1)
        self.weighted_quality_sum_xx_diagnl_elems = self.quality_sum_xx_diagnl_elems * self.frame_weights_diagnl

        np.fill_diagonal(
            self.
            weighted_quality_sum_xx[:self.dynamics.
                                    num_old_tiles_removed_from_current_state, :
                                    self.dynamics.
                                    num_old_tiles_removed_from_current_state],
            self.weighted_quality_sum_xx_diagnl_elems)

        ################# quality_var_xx ####################

        # ------------ 2nd-order partial derivatives for single variable --------------

        # a vector of diagnal elements of "quality_var_xx"
        self.quality_var_xx_diagnl_elems = self.quality_sum_xx_diagnl_elems * self.normalized_quality_diff_in_vec + self.quality_sum_x[:self
                                                                                                                                       .dynamics
                                                                                                                                       .
                                                                                                                                       num_old_tiles_removed_from_current_state]**2 * self.normalized_mask_overlap_tiles_in_vec

        np.fill_diagonal(
            self.
            quality_var_xx[:self.dynamics.
                           num_old_tiles_removed_from_current_state, :self.
                           dynamics.num_old_tiles_removed_from_current_state],
            self.quality_var_xx_diagnl_elems)

        # ------------ mixed 2nd-order partial derivatives --------------
        for frame_idx in range(1, params.FPS - 1):
            overlap_pos = self.quality_diff[frame_idx].nonzero()
            small_indexes = self.order_nonzero_view_prob_of_interest[
                frame_idx - 1][overlap_pos]
            large_indexes = self.order_nonzero_view_prob_of_interest[
                frame_idx][overlap_pos]
            num_overlap_tiles = self.num_overlap_tiles_adjacent_frames[
                frame_idx]

            # symmetric quality_var_xx
            self.quality_var_xx[
                small_indexes,
                large_indexes] = -2 / num_overlap_tiles * self.quality_sum_x[
                    small_indexes] * self.quality_sum_x[large_indexes]
            self.quality_var_xx[
                large_indexes,
                small_indexes] = -2 / num_overlap_tiles * self.quality_sum_x[
                    large_indexes] * self.quality_sum_x[small_indexes]

        ################# barrier functions ##########################
        np.fill_diagonal(
            self.upper_bound_barrier_xx, -1 /
            (self.max_rates_cur_step - self.dynamics.updated_current_state)**2)

        # np.fill_diagonal(
        #     self.
        #     lower_bound_barrier_xx[:self.dynamics.
        #                            num_old_tiles_removed_from_current_state, :
        #                            self.dynamics.
        #                            num_old_tiles_removed_from_current_state],
        #     1 /
        #     (self.dynamics.
        #      updated_current_state[:self.dynamics.
        #                            num_old_tiles_removed_from_current_state] +
        #      self.b_updated_tiles / self.a_updated_tiles)**2)
        # pdb.set_trace()

        self.current_cost_xx = -self.weighted_quality_sum_xx + self.quality_var_xx * params.WEIGHT_VARIANCE

        return self.current_cost_xx

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [current_state_size].
            u: Current control [current_action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [current_action_size, current_state_size].
        """
        # at the beginning of streaming, buffer is almost empty
        # quality = 0
        if not self.viewing_probability_of_interest.any():
            # only barrier functions play a role
            self.upper_bound_barrier_ux = self.upper_bound_barrier_xx.copy()

            self.current_cost_ux = np.zeros((self.current_state_size, self.current_state_size))
            return self.current_cost_ux

        ################# quality_sum_ux ####################
        self.weighted_quality_sum_ux = self.weighted_quality_sum_xx.copy()

        ################# quality_var_ux ####################
        self.quality_var_ux = self.quality_var_xx.copy()

        ################# barrier functions ##########################
        self.upper_bound_barrier_ux = self.upper_bound_barrier_xx.copy()
        # self.lower_bound_barrier_ux = self.lower_bound_barrier_xx.copy()

        self.current_cost_ux = -self.weighted_quality_sum_ux + self.quality_var_ux * params.WEIGHT_VARIANCE

        return self.current_cost_ux

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [current_state_size].
            u: Current control [current_action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [current_action_size, current_action_size].
        """
        # at the beginning of streaming, buffer is almost empty
        # quality = 0
        if not self.viewing_probability_of_interest.any():
            # only barrier functions play a role
            self.zero_bound_barrier_uu = np.eye(
                self.current_action_size) / u**2
            self.upper_bound_barrier_uu = self.upper_bound_barrier_xx.copy()

            # there exists non-zero mixed 2nd-order partial derivatives for bw_bound_barrier function!!
            self.bw_bound_barrier_uu = np.full(
                (self.current_action_size, self.current_action_size),
                -1 / (self.bandwidth_budget[i] - self.sum_u)**2)

            self.current_cost_uu = (
                self.zero_bound_barrier_uu + 
                self.bw_bound_barrier_uu) * params.BARRIER_WEIGHT

            return self.current_cost_uu

        ################# quality_sum_uu ####################
        self.weighted_quality_sum_uu = self.weighted_quality_sum_xx.copy()

        ################# quality_var_uu ####################
        self.quality_var_uu = self.quality_var_xx.copy()

        ################# barrier functions ##########################
        self.zero_bound_barrier_uu = np.eye(self.current_action_size) / u**2
        self.upper_bound_barrier_uu = self.upper_bound_barrier_xx.copy()
        # self.lower_bound_barrier_uu = self.lower_bound_barrier_xx.copy()

        # there exists non-zero mixed 2nd-order partial derivatives for bw_bound_barrier function!!
        self.bw_bound_barrier_uu = np.full(
            (self.current_action_size, self.current_action_size),
            -1 / (self.bandwidth_budget[i] - self.sum_u)**2)

        self.current_cost_uu = -self.weighted_quality_sum_uu + self.quality_var_uu * params.WEIGHT_VARIANCE + (
            self.zero_bound_barrier_uu +
            self.bw_bound_barrier_uu) * params.BARRIER_WEIGHT

        return self.current_cost_uu
