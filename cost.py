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
from scipy.optimize import approx_fprime

import params
from params import FrameWeightType


class AutoDiffCost(Cost):
    """Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self, l, l_terminal, x_inputs, u_inputs, i=None, **kwargs):
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
        self.quality_sum = 0.0
        self.quality_var = 0.0
        self.tile_utility_coef = params.TILE_UTILITY_COEF

    def x(self):
        """The state variables."""
        return self._x_inputs

    def u(self):
        """The control variables."""
        return self._u_inputs

    def i(self):
        """The time step variable."""
        return self._i

    def get_quality_for_tiles(self):
        self.log_fang = np.log10(self.num_points_per_degree)
        # tmp = np.array([1, np.log10(1 / self.theta_of_interest), self.log_fang, self.log_fang**2, self.log_fang**3])
        self.quality_tiles = self.tile_utility_coef[
            0] + self.tile_utility_coef[1] * np.log10(
                1 / self.theta_of_interest) + self.tile_utility_coef[
                    2] * self.log_fang + self.tile_utility_coef[
                        3] * self.log_fang**2 + self.tile_utility_coef[
                            4] * self.log_fang**3

    def l(self,
          x,
          u,
          i,
          bandwidth_budget,
          viewing_probability,
          distances,
          overlap_ratio_history,
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
        Self.max_rates = rate_versions[0]
        self.current_cost = 0
        self.current_state_size = x.shape[0]
        self.dynamics = dynamics
        self.tile_a = tile_a
        self.tile_b = tile_b
        self.overlap_ratio_history = overlap_ratio_history
        self.viewing_probability = viewing_probability
        self.distances = distances
        self.bandwidth_budget = bandwidth_budget
        self.update_start_idx = update_start_idx
        self.update_end_idx = update_end_idx
        self.quality_sum = 0.0
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
        self.start_frame_idx_within_video = (
            self.update_start_idx + i * params.FPS) % params.TARGET_LATENCY

        # shape: [FPS, NUM_TILES_PER_SIDE_IN_A_FRAME, ..., ...]
        self.viewing_probability_of_interest = self.viewing_probability[
            i * params.FPS:(i + 1) * params.FPS].copy()

        # tiles that finish updating, i.e., the first params.FPS framess
        self.tile_of_interest_pos = self.viewing_probability[i * params.FPS:(
            i + 1) * params.FPS].nonzero()
        self.tile_a_cur_step = self.tile_a[self.start_frame_idx_within_video:
                                           self.start_frame_idx_within_video +
                                           params.FPS][
                                               self.tile_of_interest_pos]
        self.tile_b_cur_step = self.tile_b[self.start_frame_idx_within_video:
                                           self.start_frame_idx_within_video +
                                           params.FPS][
                                               self.tile_of_interest_pos]

        self.distance_of_interest = self.distances[i * params.FPS:(
            i + 1) * params.FPS][self.tile_of_interest_pos]
        self.frame_weights = np.ones_like(self.tiles_rates_to_be_watched)
        if params.FRAME_WEIGHT_TYPE == FrameWeightType.CONST:
            self.frame_weights = np.ones_like(self.tiles_rates_to_be_watched)
        ### todo by Tongyu: add more frame weight types

        self.theta_of_interest = params.TILE_SIDE_LEN / self.distance_of_interest
        self.num_points = self.tile_a_cur_step * self.tiles_rates_to_be_watched + self.tile_b_cur_step
        self.num_points_per_degree = self.num_points**0.5 / self.theta_of_interest  # f_ang
        self.get_quality_for_tiles()  # update self.quality_tiles

        self.quality_sum = np.dot(self.quality_tiles, self.frame_weights.T)

        ################# quality_var ####################
        self.quality_diff = np.zeros(
            (params.FPS, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        self.num_overlap_tiles_adjacent_frames = np.zeros((params.FPS, ))
        self.quality_var_adjacent_frames = np.zeros((params.FPS, ))

        # convenient for calculating quality_var and quality_var_x
        self.quality_tiles_in_matrix = self.viewing_probability_of_interest.copy(
        )
        self.quality_tiles_in_matrix[
            self.tile_of_interest_pos] = self.quality_tiles

        # index of tiles; shape: [FPS, NUM_TILES_PER_SIDE_IN_A_FRAME, ..., ...]
        # start from 0.
        self.order_nonzero_view_prob_of_interest = self.viewing_probability_of_interest.copy(
        )
        self.order_nonzero_view_prob_of_interest[
            self.tile_of_interest_pos] = np.arange(
                self.dynamics.num_old_tiles_removed_from_current_state)

        # params.FPS elements, each of which is a list having two arrays
        # [0] for smaller-index frame, [1] for larger-index frame
        # this stores indexes for indexing self.quality_tiles!!
        # Then variance can be calculated for batch of tiles!
        self.overlap_tiles_index_adjacent_frames = []

        # overlap is the tiles that are not empty in both current and previous
        # frames. Here is the overlap between the 1st frame_of_interest (31st
        # frame in current buffer) and 30th frame in buffer.
        # each element of overlap_pos has shape: [NUM_TILES_PER_SIDE_IN_A_FRAME, ..., ...]
        overlap_pos = []
        overlap_pos.append(
            np.where((buffer_[params.FPS - 1] != 0)
                     & (self.order_nonzero_view_prob_of_interest[0] != 0)))

        # this stores indexes for self.quality_tiles to index!! Then variance
        # can be calculated for batch of tiles!
        self.overlap_tiles_index_adjacent_frames.append(
            self.order_nonzero_view_prob_of_interest[0][overlap_pos[0]])
        self.quality_diff[0][overlap_pos[0]] = buffer_[params.FPS - 1][
            overlap_pos[0]] - self.quality_tiles[
                self.overlap_tiles_index_adjacent_frames[0]]
        self.num_overlap_tiles_adjacent_frames[0] = len(overlap_pos[0][0])
        self.quality_var_adjacent_frames[0] = np.sum(
            self.quality_diff[0]**
            2) / self.num_overlap_tiles_adjacent_frames[0]

        for frame_idx in range(1, params.FPS):
            overlap_pos.append(
                np.where((self.order_nonzero_view_prob_of_interest[frame_idx -
                                                                   1] != 0)
                         & (self.order_nonzero_view_prob_of_interest[frame_idx]
                            != 0)))

            self.quality_diff[frame_idx] = (
                self.quality_tiles_in_matrix[frame_idx - 1] -
                self.quality_tiles_in_matrix[frame_idx]
            ) * self.viewing_probability_of_interest[
                frame_idx -
                1] * self.viewing_probability_of_interest[frame_idx]

            self.num_overlap_tiles_adjacent_frames[
                frame_idx] = np.count_nonzero(self.quality_diff[frame_idx])
            self.quality_var_adjacent_frames[frame_idx] = np.sum(
                self.quality_diff[frame_idx]**
                2) / self.num_overlap_tiles_adjacent_frames[frame_idx]

        self.sum_quality_var = np.sum(
            self.quality_var_adjacent_frames) * params.WEIGHT_VARIANCE

        ##########################################################

        ################# barrier functions ##########################
        self.zero_bound_barrier = -np.sum(np.log(u))
        self.updated_tiles_pos = self.viewing_probability[
            i * params.FPS:i * params.FPS + params.TARGET_LATENCY].nonzero()

        # looks wierd here, that's because the video is played repeatedly, so
        # every 1s the 'self.max_rates' should be circularly shifted!
        self.max_rates_cur_step = np.concatenate(
            (self.max_rates[self.start_frame_idx_within_video:],
             self.max_rates[:self.start_frame_idx_within_video]),
            axis=0)[self.updated_tiles_pos]
        self.upper_bound_barrier = -np.sum(
            np.log(self.max_rates_cur_step -
                   self.dynamics.updated_current_state))

        self.a_updated_tiles = np.concatenate(
            (self.tile_a[self.start_frame_idx_within_video:],
             self.tile_a[:self.start_frame_idx_within_video]),
            axis=0)[self.tile_of_interest_pos]  # stores params.FPS frames
        self.b_updated_tiles = np.concatenate(
            (self.tile_b[self.start_frame_idx_within_video:],
             self.tile_b[:self.start_frame_idx_within_video]),
            axis=0)[self.tile_of_interest_pos]  # stores params.FPS frames

        # only care about the first params.FPS frames among all updated tiles,
        # because only these tiles finish being updated and to be involved in
        # final visual quality.
        self.lower_bound_barrier = -np.sum(
            np.log(self.dynamics.updated_current_state[:params.FPS] +
                   self.b_updated_tiles / self.a_updated_tiles))

        self.sum_u = np.sum(u)
        self.bw_bound_barrier = -np.log(self.bandwidth_budget[i] - self.sum_u)

        self.current_cost = -self.quality_sum + self.sum_quality_var + self.zero_bound_barrier + self.upper_bound_barrier + self.lower_bound_barrier + self.bw_bound_barrier

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

        # zero_bound_barrier_x is all 0 since it only depends on actions u
        self.upper_bound_barrier_x = np.zeros((self.current_state_size, ))
        self.lower_bound_barrier_x = np.zeros((self.current_state_size, ))

        ################# quality_sum_x ####################
        self.quality_tiles_wrt_log_fang = self.tile_utility_coef[
            2] + 2 * self.tile_utility_coef[
                3] * self.log_fang + 3 * self.tile_utility_coef[
                    4] * self.log_fang**2
        self.log_fang_wrt_x = self.tile_a_cur_step / (
            2 * np.log(10) * self.tile_a_cur_step * self.num_points)
        self.quality_sum_x[:self.dynamics.
                           num_old_tiles_removed_from_current_state] = self.quality_tiles_wrt_log_fang * self.log_fang_wrt_x * self.frame_weights

        ################# quality_var_x ####################

        # toDo by Tongyu: assign equation number in the paper for "normalized_quality_diff_in_matrix "
        # 2nd factor of equation (?) in the paper
        self.normalized_quality_diff_in_matrix = np.zeros(
            (params.FPS, params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME,
             params.NUM_TILES_PER_SIDE_IN_A_FRAME))
        for frame_idx in range(params.FPS - 1):
            self.normalized_quality_diff_in_matrix[
                frame_idx] = -self.quality_diff[
                    frame_idx] * 2 / self.num_overlap_tiles_adjacent_frames[
                        frame_idx] + self.quality_diff[
                            frame_idx +
                            1] * 2 / self.num_overlap_tiles_adjacent_frames[
                                frame_idx + 1]
        self.num_overlap_tiles_adjacent_frames[frame_idx] = -self.quality_diff[
            frame_idx] * 2 / self.num_overlap_tiles_adjacent_frames[frame_idx]

        # map "num_overlap_tiles_adjacent_frames" from matrix to a vector,
        # which corresponds with "quality_sum_x" for multiplication.
        self.normalized_quality_diff_in_vector = self.normalized_quality_diff_in_matrix[
            self.tile_of_interest_pos]
        self.quality_var_x[:self.dynamics.
                           num_old_tiles_removed_from_current_state] = self.quality_sum_x[:self
                                                                                          .
                                                                                          dynamics
                                                                                          .
                                                                                          num_old_tiles_removed_from_current_state] * self.normalized_quality_diff_in_vector

        ################# barrier functions ##########################
        self.upper_bound_barrier_x = 1 / (self.max_rates_cur_step -
                                          self.dynamics.updated_current_state)
        self.lower_bound_barrier_x[self.tile_of_interest_pos] = -1 / (
            self.dynamics.updated_current_state[:params.FPS] +
            self.b_updated_tiles / self.a_updated_tiles)

        self.current_cost_x = -self.quality_sum_x + self.quality_var_x + self.upper_bound_barrier_x + self.lower_bound_barrier_x

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

        ################# quality_sum_u ####################
        self.quality_sum_u = self.quality_sum_x.copy()

        ################# quality_var_u ####################
        self.quality_var_u = self.quality_sum_x.copy()

        ################# barrier functions ##########################
        self.zero_bound_barrier_u = -1 / u
        self.upper_bound_barrier_u = self.upper_bound_barrier_x.copy()
        self.lower_bound_barrier_u = self.lower_bound_barrier_x.copy()
        self.bw_bound_barrier_u = 1 / (self.bandwidth_budget[i] - self.sum_u)

        self.current_cost_u = -self.quality_sum_u + self.quality_var_u + self.zero_bound_barrier_u + self.upper_bound_barrier_u + self.lower_bound_barrier_u + self.bw_bound_barrier_u

        return self.current_cost_u

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            z = np.hstack([x, i])
            return np.array(self._l_xx_terminal(*z))

        z = np.hstack([x, u, i])
        return np.array(self._l_xx(*z))

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        z = np.hstack([x, u, i])
        return np.array(self._l_ux(*z))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        z = np.hstack([x, u, i])
        return np.array(self._l_uu(*z))
