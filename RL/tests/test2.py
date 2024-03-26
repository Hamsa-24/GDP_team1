import pygame
import sys

# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre
WIDTH, HEIGHT = 800, 600

# Couleurs
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Création de la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Environnement 3D Pygame")

# Position initiale de la caméra
camera_x, camera_y = 0, 0

# Position initiale du point mobile
point_x, point_y, point_z = WIDTH // 2, HEIGHT // 2, 0

# Boucle principale
running = True
while running:
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Mettre à jour l'environnement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        camera_x -= 5
    if keys[pygame.K_RIGHT]:
        camera_x += 5
    if keys[pygame.K_UP]:
        camera_y -= 5
    if keys[pygame.K_DOWN]:
        camera_y += 5
    if keys[pygame.K_a]:
        point_x -= 5
    if keys[pygame.K_d]:
        point_x += 5
    if keys[pygame.K_w]:
        point_y -= 5
    if keys[pygame.K_s]:
        point_y += 5
    if keys[pygame.K_q]:
        point_z += 5
    if keys[pygame.K_e]:
        point_z -= 5

    # Effacer l'écran
    screen.fill(WHITE)

    # Dessiner les éléments de l'environnement en perspective
    # Simulation de vue 3D en ajustant la position et la taille des objets en fonction de leur distance par rapport à la caméra
    pygame.draw.rect(screen, RED, pygame.Rect(300 - camera_x, 200 - camera_y, 100, 100))  # Pavé droit rouge
    pygame.draw.rect(screen, GREEN, pygame.Rect(200 - camera_x, 300 - camera_y, 150, 150))  # Pavé droit vert
    pygame.draw.circle(screen, BLUE, (point_x, point_y), 10)  # Point mobile bleu

    # Mettre à jour l'affichage
    pygame.display.flip()

    # Délai pour contrôler la vitesse de la boucle
    pygame.time.delay(30)

# Quitter Pygame
pygame.quit()
sys.exit()
