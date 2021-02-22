import numpy as np

class EKF:
   '''
   Tuned for sliding puck with acceleration measurements.
   '''
   def __init__(self, mass, drag, Q, R, dt=0.001):
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
            [0, 0, 0, 0, 1, 0],
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

      # Observation matrix changes based on when observations are available.
      self.H = np.zeros((6, 6))

      self.x = np.zeros(6)
      self.P = 0.3*np.eye(6)
      self.K = np.eye(6)

   def propagateState(self, accel_input):
      # print("F shape;", self.F.shape)
      # print("G shape;", self.G.shape)
      # print("x shape;", self.x.shape)
      x = np.dot(self.F, self.x) + np.dot(self.G, accel_input)
      return x

   def update(self, accel_input, pos_meas=None, accel_meas=None):
      self.H = np.zeros((6, 6), dtype=np.float32)
      meas_vec = np.zeros(6)
      if pos_meas is not None:
         self.H[0, 0] = 1
         self.H[1, 1] = 1
         meas_vec[0:2] = pos_meas
      if accel_meas is not None:
         self.H[4, 4] = 1
         self.H[5, 5] = 1
         meas_vec[4:] = accel_meas
      x_minus = self.propagateState(accel_input)
      P_minus = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q_proc
      gain_fac = np.linalg.inv(np.dot(np.dot(self.H, P_minus), self.H.T) + self.R_meas)
      self.K = np.dot(np.dot(P_minus, self.H.T), gain_fac)
      self.x = x_minus + np.dot(self.K, (meas_vec - np.dot(self.H, x_minus)))
      self.P = np.dot(np.eye(6) - np.dot(self.K, self.H), P_minus)
      return self.x
