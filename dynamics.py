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

import logging
import pdb
import numpy as np
from scipy.optimize import approx_fprime

import params


class AutoDiffDynamics():
    """Auto-differentiated Dynamics Model."""

    def __init__(self):
        self.sum_u = None
        self.logger = self.set_logger()

    def set_logger(self):
        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s - %(lineno)d - %(module)s\n%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def f(self, x, u, i, viewing_probability):
        """Dynamics model.

        Args:
            x: Current state [current_state_size].
            u: Current control [current_action_size].
            i: Current time step.
            viewing_probability:

        Returns:
            Next state [next_state_size].
        """
        self.current_state_size = x.shape[0]
        self.current_action_size = self.current_state_size
        self.sum_u = np.sum(u)
        self.next_state_start_frame_idx = (i + 1) * params.FPS
        self.next_tile_of_interest_pos = viewing_probability[
            self.next_state_start_frame_idx:self.next_state_start_frame_idx +
            params.TARGET_LATENCY].nonzero()
        self.next_state_size = len(self.next_tile_of_interest_pos[0])
        self.next_x = np.zeros((self.next_state_size, ))
        self.num_new_tiles_for_next_state = np.count_nonzero(
            viewing_probability[params.TARGET_LATENCY + i *
                                params.FPS:self.next_state_start_frame_idx +
                                params.TARGET_LATENCY])
        self.num_old_tiles_removed_from_current_state = np.count_nonzero(
            viewing_probability[i *
                                params.FPS:self.next_state_start_frame_idx])
        self.updated_current_state = x + u
        self.next_x[:(
            self.next_state_size -
            self.num_new_tiles_for_next_state)] = self.updated_current_state[
                self.num_old_tiles_removed_from_current_state:].copy()

        return self.next_x

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [current_state_size].
            u: Current control [current_action_size].
            i: Current time step.

        Returns:
            df/dx [next_state_size, current_state_size].
        """
        self.dynamic_x = np.eye(
            N=self.next_state_size,
            M=self.current_state_size,
            k=self.num_old_tiles_removed_from_current_state)

        return self.dynamic_x

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [current_state_size].
            u: Current control [current_action_size].
            i: Current time step.

        Returns:
            df/du [next_state_size, current_action_size].
        """
        self.dynamic_u = np.eye(
            N=self.next_state_size,
            M=self.current_action_size,
            k=self.num_old_tiles_removed_from_current_state)

        return self.dynamic_u
