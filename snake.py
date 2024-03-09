import pygame
import random

pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

comicsans = pygame.font.SysFont("comicsans", 50)

snake = [(400, 400)]
v = (0, 0)
size = 40
eat = True

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    if eat:
        while True:
            food = (
                random.randrange(800 // size) * size,
                random.randrange(800 // size) * size,
            )
            if food not in snake:
                break
        eat = False

    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_UP] or pressed[pygame.K_w] or pressed[pygame.K_k]:
        v = (0, -size)
    elif pressed[pygame.K_DOWN] or pressed[pygame.K_s] or pressed[pygame.K_j]:
        v = (0, size)
    elif pressed[pygame.K_LEFT] or pressed[pygame.K_a] or pressed[pygame.K_h]:
        v = (-size, 0)
    elif pressed[pygame.K_RIGHT] or pressed[pygame.K_d] or pressed[pygame.K_l]:
        v = (size, 0)

    snake.append((snake[-1][0] + v[0], snake[-1][1] + v[1]))

    if (
        not (0 <= snake[-1][0] <= 800 and 0 <= snake[-1][1] <= 800)
        or snake[-1] in snake[:-2]
    ):
        done = True

    if snake[-1] == food:
        eat = True
    elif v:
        snake.pop(0)

    screen.fill(black)

    pygame.draw.rect(
        screen,
        red,
        pygame.Rect(food[0] + size // 4, food[1] + size // 4, size // 2, size // 2),
    )
    for x, y in snake:
        pygame.draw.rect(screen, white, pygame.Rect(x, y, size, size))

    text = comicsans.render(str(len(snake)), True, white)
    textbox = text.get_rect()
    textbox.center = (400, 100)
    screen.blit(text, textbox)

    clock.tick(10)
    pygame.display.flip()
