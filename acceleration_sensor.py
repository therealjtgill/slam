import numpy as np

class AccelerationSensor:
   def __init__(self, measurement_covariance):
      self.R_meas = measurement_covariance

   def getMeasurements(self, true_accel):
      return np.random.multivariate_normal(true_accel, self.R_meas)
