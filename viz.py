import numpy as np
import pygame
import sys
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
    vehicle_viz = VehicleViz((0, 0, 255), (0, 0), screen_converter)
    vehicle_dynamics = VehicleDynamics([0.0, 0.0], [0.0, 0.0], 0.5, 1)
    vehicle_controller = VehicleController(1, 0, 0.1, -5, 5)
    vehicle_controller.setWaypoint([30, 10])

    running = True
    while running:
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        accel_cmd = vehicle_controller.pursueWaypoint(vehicle_dynamics.pos, vehicle_dynamics.vel)
        vehicle_dynamics.update(accel_cmd)
        vehicle_viz.updateWorldPosition(vehicle_dynamics.pos)
        pygame.draw.circle(screen, vehicle_viz.color, vehicle_viz.getScreenPosition(), 10, 1)
        pygame.display.flip()

if __name__ == "__main__":
    main()