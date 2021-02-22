import numpy as np
import pygame
import sys
from acceleration_sensor import AccelerationSensor
from ekf import EKF
from ekf_slam import EKFSlam
from guidance import Guidance
from range_bearing_sensor import RangeBearingSensor
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

class DotViz:
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
    world_x_bounds = [-100, 100]
    world_y_bounds = [-100, 100]

    screen_converter = WorldToScreen(
        width,
        height,
        world_x_bounds[0],
        world_x_bounds[1],
        world_y_bounds[0],
        world_y_bounds[1]
    )

    mass = 1
    drag = 0.5
    num_landmarks = 10
    vehicle_viz = DotViz((0, 0, 255), (0, 0), screen_converter)
    vehicle_dynamics = VehicleDynamics([0.0, 0.0], [0.0, 0.0], drag, mass)
    vehicle_controller = VehicleController(1, 0.001, 0.1, -3, 3)
    vehicle_controller.setWaypoint([-30, -10])
    guidance = Guidance(*world_x_bounds, *world_y_bounds, 0.5)

    landmark_vizs = [
        DotViz(
            (0, 127, 180),
            ((np.random.rand()*2 - 1)*100, (np.random.rand()*2 - 1)*100),
            screen_converter
        ) for _ in range(num_landmarks)
    ]

    Q = 0.2*np.eye(6)
    R = 0.4*np.eye(6)
    accelerometer = AccelerationSensor(0.1*np.eye(2))
    kf = EKF(1, 0.5, Q, R)
    Q_map = 0.2*np.eye(6 + 2*num_landmarks)
    Q_map[6:, 6:] = 0
    R_map = np.zeros((6 + 2*num_landmarks, 6 + 2*num_landmarks))
    R_map[0:6, 0:6] = 0.4*np.eye(6)
    R_map[6:, 6:] = 0.5*np.eye(2*num_landmarks)
    ekf_slam = EKFSlam(mass, drag, Q_map, R_map, num_landmarks)
    landmark_sensor = RangeBearingSensor(0.5, 50, 0.9*np.eye(2))

    pos_hat = np.zeros(2)
    vel_hat = np.zeros(2)
    running = True
    counter = 0
    while running:
        counter += 1
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        new_wpt, wpt_pos = guidance.update(pos_hat)
        if new_wpt:
            print("New waypoint issued!", wpt_pos)
            vehicle_controller.setWaypoint(wpt_pos)
        accel_cmd = vehicle_controller.pursueWaypoint(pos_hat, vel_hat)
        vehicle_dynamics.update(accel_cmd)
        force = vehicle_dynamics.force/vehicle_dynamics.mass
        if counter % 1000 == 0 and False:
            pos_meas = np.random.multivariate_normal(vehicle_dynamics.pos, 0.3*np.eye(2))
        else:
            pos_meas = None
        if counter % 10 == 0:
            landmark_meas = landmark_sensor.getMeasurements(
                vehicle_dynamics.pos,
                [l.pos for l in landmark_vizs]
            )
        else:
            landmark_meas = None
        accel_meas = accelerometer.getMeasurements(force)
        predicted_map_and_state = ekf_slam.update(accel_cmd, pos_meas, accel_meas, landmark_meas)
        # predicted_state = kf.update(accel_cmd, pos_meas, accel_meas)
        pos_hat = predicted_map_and_state[0:2]
        vel_hat = predicted_map_and_state[2:4]
        # print(predicted_map_and_state[6:])
        pos_cov = ekf_slam.P[0:2, 0:2]
        # vel_cov = kf.P[2:4, 2:4]
        # acc_cov = kf.P[4:6, 4:6]
        # print(pos_cov)
        screen_pos_hat = screen_converter.convertWorldToScreen(*pos_hat)
        pygame.draw.circle(screen, (255, 0, 0), screen_pos_hat, 6, 1)
        vehicle_viz.updateWorldPosition(vehicle_dynamics.pos)
        pygame.draw.circle(screen, vehicle_viz.color, vehicle_viz.getScreenPosition(), 10, 1)
        for landmark in landmark_vizs:
            pygame.draw.circle(screen, landmark.color, landmark.getScreenPosition(), 4, 1)
        for i in range(len(predicted_map_and_state[6:])//2):
            screen_landmark_pos = screen_converter.convertWorldToScreen(
                *predicted_map_and_state[(6 + 2*i):(6 + 2*i + 2)]
            )
            pygame.draw.circle(
                screen,
                (200, 120, 0),
                screen_landmark_pos,
                3,
                2
            )
        pygame.display.flip()

if __name__ == "__main__":
    main()