import numpy as np
from pid import PID

class Vehicle:
    '''
    2D vehicle controller.
    '''
    def __init__(self, kp_vel, ki_vel, kd_vel, cmd_max, cmd_min):
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel
        self.cmd_max = cmd_max
        self.cmd_min = cmd_min

        self.vel_controllers = [
            PID(
                self.kp_vel,
                self.ki_vel,
                self.kd_vel,
                self.cmd_max,
                self.cmd_min
            ),
            PID(
                self.kp_vel,
                self.ki_vel,
                self.kd_vel,
                self.cmd_max,
                self.cmd_min
            )
        ]

    def compute(self, desired_vel, current_vel):
        accel_cmds = np.array(
            [
                self.vel_controllers.compute(desired_vel[i], current_vel[i])
                for i in range(2)
            ]
        )
        return accel_cmds
