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
"""Dynamics model."""

import numpy as np
from scipy.optimize import approx_fprime

import params


class AutoDiffDynamics():

    """Auto-differentiated Dynamics Model."""

    def __init__(self):
        self.sum_u = None
        self.zm = None
        self.min_m = None
        self.exp_zm = None
        self.bandwidth_budget = None
        
        

    def f(self, x, u, i, bandwidth_budget, viewing_probability):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.
            bandwidth_budget: [N,]

        Returns:
            Next state [state_size].
        """
        self.current_state_size = x.shape[0]
        self.current_action_size = self.current_state_size
        self.bandwidth_budget = bandwidth_budget
        self.sum_u = np.sum(u)
        self.zm = self.bandwidth_budget[i] / params.SVC_OVERHEAD / self.sum_u
        self.exp_zm = np.exp(params.SMOOTH_MIN_PARAM * zm)
        exp_tmp = np.exp(params.SMOOTH_MIN_PARAM) / self.exp_zm
        self.min_m = (1 + zm * exp_tmp) / (1 + exp_tmp)
        self.next_state_start_frame_idx = (i + 1) * params.FPS
        self.next_tile_of_interest_pos = viewing_probability[self.next_state_start_frame_idx : self.next_state_start_frame_idx + params.TARGET_LATENCY].nonzero()
        self.next_state_size = len(self.next_tile_of_interest_pos[0])
        self.next_x = np.zeros((self.next_state_size,))
        self.num_new_tiles_for_next_state = np.count_nonzero(viewing_probability[params.TARGET_LATENCY + i * params.FPS : self.next_state_start_frame_idx + params.TARGET_LATENCY])
        self.num_old_tiles_removed_from_current_state = np.count_nonzero(viewing_probability[i * params.FPS : self.next_state_start_frame_idx])
        self.updated_current_state = x + u * self.min_m
        self.next_x[:(self.next_state_size - self.num_new_tiles_for_next_state)] = self.updated_current_state[self.num_old_tiles_removed_from_current_state:].copy()

        return self.next_x

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [next_state_size, current_state_size].
        """
        self.f_x = np.eye(N=self.next_state_size, M=self.current_state_size, k=self.num_old_tiles_removed_from_current_state)
        return self.f_x

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        self.zm_u_one_value = -self.bandwidth_budget[i] / params.SVC_OVERHEAD / self.sum_u**2
        self.zm_uu_one_value = 2 * self.bandwidth_budget[i] / params.SVC_OVERHEAD / self.sum_u**3
        self.zm_u = np.full(u.shape, -self.bandwidth_budget[i] / params.SVC_OVERHEAD / self.sum_u**2)
        # self.zm_uu = np.full((u.shape[0], u.shape[0]), 2 * self.bandwidth_budget[i] / params.SVC_OVERHEAD / self.sum_u**3)

        self.min_wrt_zm = (np.exp(100) + np.exp(50) * self.exp_zm * (51 - 50 * self.zm)) \
                            / np.power(self.exp_zm + np.exp(50), 2) # 1*1
        self.min_wrt_u = self.min_wrt_zm * self.zm_u # [current action_size]
        tmp = u * self.min_wrt_u # [current action_size]
        ## tmp.reshape((action_size, 1))
        self.f_u = np.eye(N=self.next_state_size, M=self.current_action_size, k=self.num_old_tiles_removed_from_current_state) * self.min_m
        num_overlapTiles_between_current_act_and_next_state = self.current_action_size - self.num_old_tiles_removed_from_current_state
        self.f_u[:num_overlapTiles_between_current_act_and_next_state, self.num_old_tiles_removed_from_current_state:] \
                += tmp[self.num_old_tiles_removed_from_current_state:].reshape((num_overlapTiles_between_current_act_and_next_state, 1))
        return self.f_u

