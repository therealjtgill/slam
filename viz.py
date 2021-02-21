import pygame
from vehicle import Vehicle
from vehicle_dynamics import VehicleDynamics

def main():
    (width, height) = (800, 600)
    screen = pygame.display.set_mode((width, height))
    pygame.display.flip()
    screen.fill((255, 255, 255))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

if __name__ == "__main__":
    main()