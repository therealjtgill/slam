import numpy as np

class VehicleDynamics:
    def __init__(self, pos_init, vel_init, drag, mass, dt=0.001):
        self.pos = pos_init
        self.vel = vel_init
        self.drag = drag
        self.mass = mass

    def propagate(self, ext_force):
        self.force = -1*drag*self.vel + ext_force

        self.accel = self.force/self.mass
        self.pos = self.pos + self.vel*dt + 0.5*self.accel*dt*dt
        self.vel = self.vel + self.accel*dt
