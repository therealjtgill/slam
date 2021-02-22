import numpy as np

class EKFSlam:
   '''
   Tuned for sliding puck with acceleration measurements.
   '''
   def __init__(self, mass, drag, Q_proc, R_meas, max_num_landmarks, dt=0.001):
      '''
      R_meas is the covariance matrix for all measurements: accelerometer and
      range-bearing sensor.
      '''
      self.drag = drag
      self.mass = mass
      self.dt = dt
      self.Q_proc = np.array(Q_proc)
      self.R_meas = np.array(R_meas)
      self.num_landmarks = max_num_landmarks
      self.state_size = 6 + 2*self.num_landmarks

      F = np.array(
         [
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
         ]
      )
      self.F_map = np.eye(self.state_size)
      self.F_map[0:6, 0:6] = F

      G = np.array(
         [
            [0.5*dt*dt, 0],
            [0, 0.5*dt*dt],
            [dt, 0],
            [0, dt],
            [1, 0],
            [0, 1]
         ]
      )
      self.G_map = np.zeros((self.state_size, 2))
      self.G_map[0:6, 0:2] = G

      # Observation matrix changes based on when observations are available.
      # self.H = np.zeros((6, 6))

      # self.x = np.zeros(6)
      # self.P = 0.3*np.eye(6)
      # self.K = np.eye(6)

      self.H_map = np.zeros(
         (self.state_size, self.state_size),
         dtype=np.float32
      )

      # self.x = np.random.rand(self.state_size
      self.x = np.zeros(self.state_size, dtype=np.float32)
      self.P = 0.3*np.eye(self.state_size, dtype=np.float32)
      # Set uncertainties of landmark positions to big-ish values.
      self.P[6:, 6:] = 10000*np.eye(self.state_size - 6)
      self.K = np.eye(self.state_size, dtype=np.float32)

   def propagateState(self, accel_input):
      x = np.dot(self.F_map, self.x) + np.dot(self.G_map, accel_input)
      return x

   def update(
      self,
      accel_input,
      pos_meas=None,
      accel_meas=None,
      landmark_meas=None
   ):
      '''
      landmark_meas is a dictionary of landmark IDs to measured landmark
      positions.
      Assume something external provides the association between landmark
      measurements and landmark IDs.
      Landmark IDs are zero-indexed. Their indices correspond to their offset
      position in the state vector.
      E.g. landmark 0's features appears at index ID 6, 7 in the state vector.
      '''
      self.H_map[:, :] = 0.0
      meas_vec = np.zeros(self.state_size, dtype=np.float32)
      # Change observation matrix and observation vector based on the
      # availability of measurements.
      if pos_meas is not None:
         self.H_map[0, 0] = 1
         self.H_map[1, 1] = 1
         meas_vec[0:2] = pos_meas
      if accel_meas is not None:
         self.H_map[4, 4] = 1
         self.H_map[5, 5] = 1
         meas_vec[4:6] = accel_meas
      if landmark_meas is not None:
         for lid, pos in landmark_meas.items():
            meas_vec[(6 + 2*lid):(6 + 2*lid + 2)] = pos
            self.H_map[
               (6 + 2*lid):(6 + 2*lid + 2),
               (6 + 2*lid):(6 + 2*lid + 2)
            ] = np.eye(2)
            self.H_map[6 + 2*lid, 0] = -1
            self.H_map[6 + 2*lid + 1, 1] = -1
      x_minus = self.propagateState(accel_input)
      P_minus = np.dot(np.dot(self.F_map, self.P), self.F_map.T) + self.Q_proc
      gain_fac = np.linalg.inv(np.dot(np.dot(self.H_map, P_minus), self.H_map.T) + self.R_meas)
      self.K = np.dot(np.dot(P_minus, self.H_map.T), gain_fac)
      innovations = meas_vec - np.dot(self.H_map, x_minus)
      # print("innovations:", innovations, "\n", meas_vec)
      self.x = x_minus + np.dot(self.K, innovations)
      self.P = np.dot(np.eye(self.state_size) - np.dot(self.K, self.H_map), P_minus)
      return self.x
