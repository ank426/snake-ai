import os
import pygame
from snake_env import Env

env = Env(render_mode="ansi", size=8)
state, info = env.reset()

pygame.init()
screen = pygame.display.set_mode((100, 100))
clock = pygame.time.Clock()
pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])

action = -1
terminated = truncated = False
while not (terminated or truncated):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            terminated = True

    os.system("clear")
    print(env.render())

    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_DOWN] or pressed[pygame.K_s] or pressed[pygame.K_j]:
        action = 0
    elif pressed[pygame.K_RIGHT] or pressed[pygame.K_d] or pressed[pygame.K_l]:
        action = 1
    elif pressed[pygame.K_UP] or pressed[pygame.K_w] or pressed[pygame.K_k]:
        action = 2
    elif pressed[pygame.K_LEFT] or pressed[pygame.K_a] or pressed[pygame.K_h]:
        action = 3

    if action >= 0:
        observation, reward, terminated, truncated, info = env.step(action)

    clock.tick(5)
    # pygame.display.flip()
