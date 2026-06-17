"""Catch-Point environment from HyAR benchmark (Li et al. 2022).

State (4D): [x, y, vx, vy]
  - (x, y): agent position in 2D plane
  - (vx, vy): agent velocity

Action (2D): [heading, move_flag]
  - heading in [0, 2*pi]: continuous heading direction
  - move_flag in {0, 1}: binary stop/move decision

Target is at origin.  Agent directly controls velocity direction and
whether to move, matching the paper description: "control over its
heading direction and the option to either stop or move."
"""

import torch


class CatchPoint:
    def __init__(
        self,
        dt: float = 0.01,
        speed: float = 0.05,
        w_goal: float = 1e3,
        w_action: float = 1e1,
        pos_max: float = 1.0,
        device: str = "cpu",
    ):
        self.dt = dt
        self.speed = speed
        self.w_goal = w_goal
        self.w_action = w_action
        self.device = device

        self.dim_state = 4
        self.dim_action = 2

    def forward_simulate(self, state, action):
        heading = action[:, 0]
        move_flag = action[:, 1]

        vx = self.speed * torch.cos(heading) * move_flag
        vy = self.speed * torch.sin(heading) * move_flag

        x_new = state[:, 0] + vx * self.dt
        y_new = state[:, 1] + vy * self.dt

        return torch.stack([x_new, y_new, vx, vy], dim=-1)

    def reward_state_action(self, state, action):
        d_goal = torch.linalg.norm(state[:, :2], dim=-1)
        r_goal = -1 * d_goal ** 2

        move_flag = action[:, 1]
        r_action = -self.w_action * move_flag

        return self.w_goal * r_goal + r_action

    def reward_action(self, action):
        return -self.w_action * action[:, 1]
