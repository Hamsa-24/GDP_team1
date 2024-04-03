import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class Cube:
    def __init__(self, color=(1,0,0)):
        self.v = []
        self.edges = []
        self.surfaces = []
        #self.colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (1,0.5,0)]
        self.colors = [color for _ in range(6)]
        self.pos_x = 0
        self.pos_y = 0

    def init_from_matrix(self, matrix):
        self.v = [(matrix[0][0], matrix[0][1], matrix[0][2]),   # Coin inférieur gauche
                  (matrix[1][0], matrix[0][1], matrix[0][2]),   # Coin inférieur droit
                  (matrix[1][0], matrix[1][1], matrix[0][2]),   # Coin supérieur droit
                  (matrix[0][0], matrix[1][1], matrix[0][2]),   # Coin supérieur gauche
                  (matrix[0][0], matrix[0][1], matrix[1][2]),   # Coin inférieur gauche arrière
                  (matrix[1][0], matrix[0][1], matrix[1][2]),   # Coin inférieur droit arrière
                  (matrix[1][0], matrix[1][1], matrix[1][2]),   # Coin supérieur droit arrière
                  (matrix[0][0], matrix[1][1], matrix[1][2])]   # Coin supérieur gauche arrière
        self.edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        self.surfaces = [(0,1,2,3), (5,4,7,6), (4,0,3,7),(1,5,6,2), (4,5,1,0), (3,2,6,7)]

    def draw(self):
        glEnable(GL_DEPTH_TEST)

        glLineWidth(5)
        glColor3fv((0, 0, 0))
        glBegin(GL_LINES)
        for e in self.edges:
            glVertex3fv(self.v[e[0]])
            glVertex3fv(self.v[e[1]])
        glEnd()

        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset( 1.0, 1.0 )

        glBegin(GL_QUADS)
        for i, quad in enumerate(self.surfaces):
            glColor3fv(self.colors[i])
            for iv in quad:
                glVertex3fv(self.v[iv])
        glEnd()

        glDisable(GL_POLYGON_OFFSET_FILL)


class Point:
    def __init__(self, pos):
        self.pos = pos

    def draw(self):
        glEnable(GL_DEPTH_TEST)
        glPointSize(8)
        glColor3f(0, 0, 1.0)
        glBegin(GL_POINTS)
        glVertex3fv(self.pos)
        glEnd()

def draw_line(point, orientation):
    line_end = point.pos + 3 * np.array([np.cos(orientation),
                                         np.sin(orientation),
                                         0])
    glLineWidth(2)
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3fv(point.pos)
    glVertex3fv(line_end)
    glEnd()

def drawText(x, y, text):
    font = pygame.font.SysFont('arial', 32)                                                
    textSurface = font.render(text, True, (255, 255, 66, 255), (0, 66, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def set_projection(w, h):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w / h, 0.1, 50.0)
    glTranslatef(-7, -3, -20) 
    glRotatef(45, 1, -1, 0)  # Rotation pour obtenir une vue 3/4
    glRotatef(-90, 1, 0, 0)
    glMatrixMode(GL_MODELVIEW)


def set_projection1(pos, theta):
    glLoadIdentity()
    camera_pos = pos

    glTranslatef(-camera_pos[0], -camera_pos[1], -camera_pos[2])

    # Appliquer une rotation pour orienter la caméra selon l'orientation de l'agent
    angle_degrees = np.degrees(theta)
    glRotatef(90, 0, 1, 0)  # Rotation autour de l'axe z (vertical)

    # Assurez-vous que le mode de matrice soit GL_MODELVIEW
    glMatrixMode(GL_MODELVIEW)