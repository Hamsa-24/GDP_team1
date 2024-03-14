import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd
import os


def plot_environment3d(count, robot_position, robot_orientation, target_zone, forbidden_zone, ax):

    ax.scatter(robot_position[0], robot_position[1], robot_position[2], c='b', label='Robot', s=0.5)

    if count == 1:
        plot_cube3d(target_zone[0], target_zone[1], ax, color='b')
        plot_cube3d(forbidden_zone[0], forbidden_zone[1], ax, color='r')

    lines = plot_line_of_vision3d(robot_position, robot_orientation, ax)
    return lines


def plot_environment2d(count, robot_position, robot_orientation, target_zone, forbidden_zone, ax):

    ax.scatter(robot_position[0], robot_position[1], c='b', s=2)

    if count == 1:
        plot_cube2d(target_zone[0], target_zone[1], ax, color='b')
        plot_cube2d(forbidden_zone[0], forbidden_zone[1], ax, color='r')

    line1, line2 = plot_line_of_vision2d(robot_position, robot_orientation, ax)
    return line1, line2


def plot_cube3d(lower_corner, opposite_corner, ax, color):
    # Coordonnées des coins du cube
    x = [lower_corner[0], opposite_corner[0], opposite_corner[0], lower_corner[0], lower_corner[0], opposite_corner[0], opposite_corner[0], lower_corner[0]]
    y = [lower_corner[1], lower_corner[1], opposite_corner[1], opposite_corner[1], lower_corner[1], lower_corner[1], opposite_corner[1], opposite_corner[1]]
    z = [lower_corner[2], lower_corner[2], lower_corner[2], lower_corner[2], opposite_corner[2], opposite_corner[2], opposite_corner[2], opposite_corner[2]]

    # Arêtes du cube
    edges = [
        [[x[0], x[1]], [y[0], y[1]], [z[0], z[1]]],
        [[x[1], x[2]], [y[1], y[2]], [z[1], z[2]]],
        [[x[2], x[3]], [y[2], y[3]], [z[2], z[3]]],
        [[x[3], x[0]], [y[3], y[0]], [z[3], z[0]]],
        [[x[4], x[5]], [y[4], y[5]], [z[4], z[5]]],
        [[x[5], x[6]], [y[5], y[6]], [z[5], z[6]]],
        [[x[6], x[7]], [y[6], y[7]], [z[6], z[7]]],
        [[x[7], x[4]], [y[7], y[4]], [z[7], z[4]]],
        [[x[0], x[4]], [y[0], y[4]], [z[0], z[4]]],
        [[x[1], x[5]], [y[1], y[5]], [z[1], z[5]]],
        [[x[2], x[6]], [y[2], y[6]], [z[2], z[6]]],
        [[x[3], x[7]], [y[3], y[7]], [z[3], z[7]]],
    ]

    # Tracer les arêtes
    for edge in edges:
        ax.plot(edge[0], edge[1], edge[2], color=color)


def plot_cube2d(lower_corner, opposite_corner, ax, color):

    x = [lower_corner[0], opposite_corner[0], opposite_corner[0], lower_corner[0]]
    y = [lower_corner[1], lower_corner[1], opposite_corner[1], opposite_corner[1]]

    edges = [
        [[x[0], x[1]], [y[0], y[1]]],
        [[x[1], x[2]], [y[1], y[2]]],
        [[x[2], x[3]], [y[2], y[3]]],
        [[x[3], x[0]], [y[3], y[0]]]
        ]
    for edge in edges:
        ax.plot(edge[0], edge[1], color=color)



def plot_line_of_vision3d(position, orientation, ax, length_edge=6, angle_vision=np.pi/12):
    # Calculer les vecteurs des extremités du champ de vision de l'agent
    
    direction_l = np.array([np.cos(orientation[0] + angle_vision/2) * np.cos(orientation[1]),
                            np.sin(orientation[0] + angle_vision/2) * np.cos(orientation[1]),
                            np.sin(orientation[1])])
    
    direction_r = np.array([np.cos(orientation[0] - angle_vision/2) * np.cos(orientation[1]),
                            np.sin(orientation[0] - angle_vision/2) * np.cos(orientation[1]),
                            np.sin(orientation[1])])
    
    direction_upp = np.array([np.cos(orientation[0]) * np.cos(orientation[1] + angle_vision/2),
                              np.sin(orientation[0]) * np.cos(orientation[1] + angle_vision/2),
                              np.sin(orientation[1] - angle_vision/2)])
    
    direction_low = np.array([np.cos(orientation[0]) * np.cos(orientation[1] - angle_vision/2),
                              np.sin(orientation[0]) * np.cos(orientation[1] - angle_vision/2),
                              np.sin(orientation[1] + angle_vision/2)])
    
    directions = [direction_l, direction_r, direction_upp, direction_low]

    edges = []

    for direction in directions:
        # Calculer le vecteur représentant le segment de champ de vision
        edges += [position + length_edge * direction]
    
    lines = []

    for edge in edges:
        # Afficher le segment de champ de vision

        line, =ax.plot([position[0], edge[0]], 
                             [position[1], edge[1]], 
                             [position[2], edge[2]], 
                             color='g', linewidth=1.5)
        lines.append(line)
        
    line1, = ax.plot([edges[0][0], edges[2][0]], 
                               [edges[0][1], edges[2][1]], 
                               [edges[0][2], edges[2][2]], 
                                color='g', linewidth=1.5)
    line2, = ax.plot([edges[1][0], edges[3][0]], 
                               [edges[1][1], edges[3][1]], 
                               [edges[1][2], edges[3][2]], 
                                color='g', linewidth=1.5)
    line3, = ax.plot([edges[3][0], edges[0][0]], 
                               [edges[3][1], edges[0][1]], 
                               [edges[3][2], edges[0][2]], 
                                color='g', linewidth=1.5)
    line4, = ax.plot([edges[1][0], edges[2][0]], 
                               [edges[1][1], edges[2][1]], 
                               [edges[1][2], edges[2][2]], 
                                color='g', linewidth=1.5)
    
    lines.append(line1)
    lines.append(line2)
    lines.append(line3)
    lines.append(line4)
    return lines


def plot_line_of_vision2d(position, orientation, ax, length_edge=6, angle_vision=np.pi/12):
    # Calculer les vecteurs des extremités du champ de vision de l'agent
    
    direction_l = np.array([np.cos(orientation[0] + angle_vision/2),
                            np.sin(orientation[0] + angle_vision/2)])
    
    direction_r = np.array([np.cos(orientation[0] - angle_vision/2),
                            np.sin(orientation[0] - angle_vision/2)])
    
    directions = [direction_l, direction_r]

    edges = []

    for direction in directions:
        # Calculer le vecteur représentant le segment de champ de vision
        edges += [position + length_edge * direction]

    line1, = ax.plot([position[0], edges[0][0]], 
                    [position[1], edges[0][1]],  
                     color='g', linewidth=1.5)
    
    line2, = ax.plot([position[0], edges[1][0]], 
                    [position[1], edges[1][1]],  
                     color='g', linewidth=1.5)
    
    return line1, line2

def save_rewards(rewards, filename):
    data = pd.DataFrame({'Episode': range(1,len(rewards)+1), 'Reward': rewards})
    path = os.path.join('train_results', filename + '.csv')
    data.to_csv(path, index=False)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))