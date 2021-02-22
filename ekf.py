import numpy as np

class EKF:
   '''
   Tuned for sliding puck with acceleration measurements.
   '''
   def __init__(self, mass, drag, Q, R, dt):
      self.drag = drag
      self.mass = mass
      self.dt = dt
      self.Q_proc = Q
      self.R_meas = R

      self.F = np.array(
         [
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0]
            [0, 0, 0, 0, 0, 1]
         ]
      )

      self.G = np.array(
         [
            [0.5*dt*dt, 0],
            [0, 0.5*dt*dt],
            [dt, 0],
            [0, dt],
            [1, 0],
            [0, 1]
         ]
      )

      self.H = np.array(
         [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
         ]
      )

      self.x = np.zeros(6)
      self.P = 10*np.eye(6)
      self.K = np.eye(4)

   def propagateState(self, accel_input):
      x = np.dot(self.F, self.state_est_prev) + self.G*accel_input
      return x

   def update(self, accel_input, accel_meas):
      x_minus = self.propagateState(accel_input)
      P_minus = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q_proc
      gain_fac = np.linalg.inv(np.dot(np.dot(self.H, P_minus), self.H.T) + self.R_meas)
      self.K = np.dot(np.dot(P_minus, self.H.T), gain_fac)
      self.x = x_minus + np.dot(self.K, (accel_meas - np.dot(self.H, x_minus)))
      self.P = np.dot(np.eye(6) - np.dot(self.K, self.H), P_minus)
      return self.x