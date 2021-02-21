import numpy as np

class PID:
    def __init__(self, kp, ki, kd, out_max, out_min):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_max = out_max
        self.out_min = out_min
        self.error_sum = 0
        self.prev_error = 0

    def compute(self, u, y):
        error = u - y
        self.error_sum += error
        i_term = self.ki*self.error_sum
        i_term = np.clip(i_term, self.out_min, self.out_max)
        d_term = self.kd*(error - self.prev_error)
        command = self.kp*error + i_term + d_term
        command = np.clip(command, self.out_min, self.out_max)
        self.prev_error = error

        return command