import numpy as np
from pid import PID

class VehicleController:
    '''
    2D vehicle controller.
    '''
    def __init__(
        self,
        kp_vel,
        ki_vel,
        kd_vel,
        acc_cmd_max,
        acc_cmd_min,
        speed_max,
        capture_radius
    ):
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel
        self.cmd_max = acc_cmd_max
        self.cmd_min = acc_cmd_min
        self.speed_max = speed_max

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

        self.wpt_cmd = np.zeros(2)
        self.capture_radius = capture_radius

    def computeAccelCommand(self, desired_vel, current_vel):
        accel_cmds = np.array(
            [
                self.vel_controllers.compute(desired_vel[i], current_vel[i])
                for i in range(2)
            ]
        )
        return accel_cmds

    def setWaypoint(self, new_wpt):
        self.wpt_cmd = np.array(new_wpt)

    def pursueWaypoint(self, current_vel, current_pos)
        wpt_vec = self.wpt_cmd - current_pos
        cmd_vel = self.max_speed*wpt_vec/(np.linalg.norm(wpt_vec) + 1e-6)
        return self.computeAccelCommand(cmd_vel, current_vel)
