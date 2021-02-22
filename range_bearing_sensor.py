import numpy as np

class RangeBearingSensor:
   '''
   Generates fake range/bearing measurements out of true vehicle position
   and true landmark positions.
   '''
   def __init__(
      self,
      min_range,
      max_range,
      measurement_covariance
   ):
      self.R_min = min_range
      self.R_max = max_range
      self.Q_meas = np.array(measurement_covariance)

   def getMeasurements(self, true_sensor_pos, true_landmarks):
      meas_landmarks = {
         i: np.random.multivariate_normal(l, self.Q_meas)
         for i, l in enumerate(true_landmarks)
         if ((np.linalg.norm(true_sensor_pos - l) <= self.R_max) and 
            (np.linalg.norm(true_sensor_pos - l) >= self.R_min))
      }
      return meas_landmarks
