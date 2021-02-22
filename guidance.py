import numpy as np

class Guidance:
   def __init__(self, x_min, x_max, y_min, y_max, capture_radius):
      self.x_min = x_min
      self.x_max = x_max
      self.y_min = y_min
      self.y_max = y_max
      self.r_min = np.array([self.x_min, self.y_min])
      self.r_max = np.array([self.x_max, self.y_max])
      self.current_wpt = np.zeros(2)
      self.capture_radius = capture_radius
      
   def update(self, pos):
      new_wpt = False
      if np.linalg.norm(self.current_wpt - pos) <= self.capture_radius:
         self.current_wpt = np.random.rand(2)*(self.r_max - self.r_min) + self.r_min
         new_wpt = True
      return new_wpt, self.current_wpt
