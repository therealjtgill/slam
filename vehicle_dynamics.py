import numpy as np

class VehicleDynamics:
    def __init__(self, pos_init, vel_init, drag, mass, dt=0.001):
        self.pos = np.array(pos_init)
        self.vel = np.array(vel_init)
        self.drag = drag
        self.mass = mass
        self.dt = dt
        self.force = np.zeros(2)

    def update(self, ext_force):
        self.force = -1*self.drag*self.vel + ext_force

        self.accel = self.force/self.mass
        self.pos = self.pos + self.vel*self.dt + 0.5*self.accel*self.dt*self.dt
        self.vel = self.vel + self.accel*self.dt
