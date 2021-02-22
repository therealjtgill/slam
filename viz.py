import numpy as np
import pygame
import sys
from acceleration_sensor import AccelerationSensor
from ekf import EKF
from ekf_slam import EKFSlam
from vehicle_controller import VehicleController
from vehicle_dynamics import VehicleDynamics

def linear_params(in_min, in_max, out_min, out_max):
    m = (out_max - out_min)/(in_max - in_min)
    b = out_min - m*in_min
    return m, b

class WorldToScreen:
    def __init__(
        self,
        disp_x_max,
        disp_y_max,
        world_x_min,
        world_x_max,
        world_y_min,
        world_y_max
    ):
        self.disp_x_max = disp_x_max
        self.disp_y_max = disp_y_max
        self.world_x_max = world_x_max
        self.world_x_min = world_x_min
        self.world_y_max = world_y_max
        self.world_y_min = world_y_min

        self.m_x_w2s, self.b_x_w2s = linear_params(
            world_x_min, world_x_max, 0, disp_x_max
        )
        self.m_y_w2s, self.b_y_w2s = linear_params(
            world_y_min, world_y_max, disp_y_max, 0
        )
        self.m_x_s2w, self.b_x_s2w = linear_params(
            0, disp_x_max, world_x_min, world_x_max
        )
        self.m_y_s2w, self.b_y_s2w = linear_params(
            0, disp_y_max, world_y_min, world_y_max
        )
        print(self.m_x_w2s, self.b_x_w2s)
        print(self.m_y_w2s, self.b_y_w2s)

    def convertWorldToScreen(self, x_w, y_w):
        return np.array(
            [
                int(self.m_x_w2s*x_w + self.b_x_w2s),
                int(self.m_y_w2s*y_w + self.b_y_w2s)
            ]
        )
    
    def convertScreenToWorld(self, x_s, y_s):
        return np.array(
            [
                self.m_x_s2w*x_s + self.b_x_s2w,
                self.m_y_s2w*y_s + self.b_y_s2w
            ]
        )

class VehicleViz:
    def __init__(
        self,
        color,
        pos_init,
        pos_converter
    ):
        self.color = color
        self.pos = pos_init
        self.pos_converter = pos_converter
        
    def updateWorldPosition(self, new_pos_world):
        self.pos = new_pos_world

    def getScreenPosition(self):
        # print(self.pos)
        return self.pos_converter.convertWorldToScreen(*self.pos)

def main():
    (width, height) = (800, 600)
    screen = pygame.display.set_mode((width, height))
    pygame.display.flip()

    screen_converter = WorldToScreen(width, height, -100, 100, -100, 100)
    # sys.exit()
    mass = 1
    drag = 0.5
    vehicle_viz = VehicleViz((0, 0, 255), (0, 0), screen_converter)
    vehicle_dynamics = VehicleDynamics([0.0, 0.0], [0.0, 0.0], drag, mass)
    vehicle_controller = VehicleController(1, 0.001, 0.1, -5, 5)
    vehicle_controller.setWaypoint([30, 10])

    Q = 0.2*np.eye(6)
    R = 0.4*np.eye(6)
    accelerometer = AccelerationSensor(0.1*np.eye(2))
    kf = EKF(1, 0.5, Q, R)
    num_landmarks = 10
    Q_map = 0.2*np.eye(6 + num_landmarks)
    Q_map[6:, 6:] = 0
    R_map = np.zeros((6 + num_landmarks, 6 + num_landmarks))
    R_map[0:6, 0:6] = 0.4*np.eye(6)
    R_map[6:, 6:] = 0.5*np.eye(num_landmarks)
    ekf_slam = EKFSlam(mass, drag, Q_map, R_map, num_landmarks)

    running = True
    counter = 0
    while running:
        counter += 1
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        accel_cmd = vehicle_controller.pursueWaypoint(vehicle_dynamics.pos, vehicle_dynamics.vel)
        vehicle_dynamics.update(accel_cmd)
        force = vehicle_dynamics.force/vehicle_dynamics.mass - accel_cmd
        if counter % 100 == 0:
            pos_meas = np.random.multivariate_normal(vehicle_dynamics.pos, 0.3*np.eye(2))
        else:
            pos_meas = None
        accel_meas = accelerometer.getMeasurements(force)
        # print("accel meas:", accel_meas)
        predicted_state = kf.update(accel_cmd, pos_meas, accel_meas)
        pos_hat = predicted_state[0:2]
        # pos_cov = kf.P[0:2, 0:2]
        # vel_cov = kf.P[2:4, 2:4]
        # acc_cov = kf.P[4:6, 4:6]
        # print(pos_cov)
        # print(counter*0.001)
        screen_pos_hat = screen_converter.convertWorldToScreen(*pos_hat)
        pygame.draw.circle(screen, (255, 0, 0), screen_pos_hat, 6, 1)
        vehicle_viz.updateWorldPosition(vehicle_dynamics.pos)
        pygame.draw.circle(screen, vehicle_viz.color, vehicle_viz.getScreenPosition(), 10, 1)
        pygame.display.flip()

if __name__ == "__main__":
    main()