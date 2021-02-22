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
      self.num_angular_samples = num_angular_samples
      self.Q_meas = measurement_covariance

   def getMeasurements(self, true_sensor_pos, true_landmarks):
      # for landmark in true_landmarks:
      #    l = landmark[0:2]
      #    if (np.linalg.norm(true_sensor_pos - l) <= self.R_max) and 
      #       (np.linalg.norm(true_sensor_pos - l) >= self.R_min):
      #       measured_landmark_pos = np.random.normal(l, self.Q_meas)
      meas_landmarks = [
         np.random.normal(l, self.Q_meas) for l in true_landmarks
         if ((np.linalg.norm(true_sensor_pos - l) <= self.R_max) and 
            (np.linalg.norm(true_sensor_pos - l) >= self.R_min))
      ]
      return meas_landmarks
