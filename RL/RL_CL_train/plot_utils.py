import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_environment3d(count, agent_position, agent_orientation, target_zone, forbidden_zone, ax):

    ax.scatter(agent_position[0], agent_position[1], agent_position[2], c='b', s=0.5)

    if count == 1:
        plot_cube3d(target_zone[0], target_zone[1], ax, color='b')
        for obstacle in forbidden_zone:
            plot_cube3d(obstacle[0], obstacle[1], ax, color='r')

    lines = plot_line_of_vision3d(agent_position, agent_orientation, ax)
    return lines


def plot_environment2d(count, agent_position, agent_orientation, target_zone, forbidden_zone, ax):
    
    ax.scatter(agent_position[0], agent_position[1], c='b', s=2)

    if count == 1:
        plot_cube2d(target_zone[0], target_zone[1], ax, color='b')
        for obstacle in forbidden_zone:
            plot_cube2d(obstacle[0], obstacle[1], ax, color='r')

    line1, line2 = plot_line_of_vision2d(agent_position, agent_orientation, ax)
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
    
    direction_l = np.array([np.cos(orientation + angle_vision/2),
                            np.sin(orientation + angle_vision/2),
                            np.sin(0)])
    
    direction_r = np.array([np.cos(orientation - angle_vision/2),
                            np.sin(orientation - angle_vision/2),
                            np.sin(0)])
    
    direction_upp = np.array([np.cos(orientation) * np.cos(0 + angle_vision/2),
                              np.sin(orientation) * np.cos(0 + angle_vision/2),
                              np.sin(0 - angle_vision/2)])
    
    direction_low = np.array([np.cos(orientation) * np.cos(0 - angle_vision/2),
                              np.sin(orientation) * np.cos(0 - angle_vision/2),
                              np.sin(0 + angle_vision/2)])
    
    directions = [direction_l, direction_r, direction_upp, direction_low]

    edges = []

    for direction in directions:
        # Calculer le vecteur représentant le segment de champ de vision
        edges += [position + length_edge * direction]
    
    lines = []

    for edge in edges:
        # Afficher le segment de champ de vision

        line, = ax.plot([position[0], edge[0]], 
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
    
    direction_l = np.array([np.cos(orientation + angle_vision/2),
                            np.sin(orientation + angle_vision/2)])
    
    direction_r = np.array([np.cos(orientation - angle_vision/2),
                            np.sin(orientation - angle_vision/2)])
    
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


def initialize_plot(PLOT2d, PLOT3d):
    if PLOT2d:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 12)
        text1, text2, text3, line1, line2 = ax.text(-100,-100,''), ax.text(-100,-100,''), ax.text(-100,-100,''), \
                                            ax.text(-100,-100,''), ax.text(-100,-100,'')
        tmp_objects = [text1, text2, text3, line1, line2]
        return fig, ax, tmp_objects
    elif PLOT3d:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 12)
        ax.set_zlim(-1, 12)
        text1, text2, text3, lines = ax.text(-100,-100,-100,''), ax.text(-100,-100,-100,''), ax.text(-100,-100,-100,''), \
                                    [ax.text(-100,-100,-100,'') for i in range(4)]
        text_instr = [text1, text2, text3]
        tmp_objects = []
        for line in lines:
             tmp_objects.append(line)
        return fig, ax, tmp_objects, text_instr
    return None, None, None
    

def update_plot(env):
        
        for object in env.tmp_objects:
            object.remove()
        for text in env.text_instr:
            text.remove()
        lines = plot_environment3d(env.count, env.agent_position, env.agent_orientation, env.target_zone, env.forbidden_zone, env.ax)

        text_instr = []
        if env.show_instr_d_theta:
            if env.instr_d_theta > 0:
                text1=env.ax.scatter(-100, -100, -100, label=f"Turn left by {env.instr_d_theta*180/np.pi:.0f}°", c='r')
            else:
                text1=env.ax.scatter(-100, -100, -100, label=f"Turn right by {-env.instr_d_theta*180/np.pi:.0f}°", c='r')
            text_instr.append(text1)

        if env.show_instr_vz:
            if env.instr_vz > 0:
                text2=env.ax.scatter(-100, -100, -100, label=f"Go up by {env.instr_vz:.0f}", c='r')
            else: 
                text2=env.ax.scatter(-100, -100, -100, label=f"Go down and land on roof", c='r')
            text_instr.append(text2)

        if env.show_angle_to_xf:
            text3=env.ax.scatter(-100, -100, -100, label=f"Angle to target: {env.angle_to_target()*180/np.pi:.0f}°", c='r')
            text_instr.append(text3)
        if env.show_dist_to_xf:
            text4=env.ax.scatter(-100, -100, -100, label=f"Distance to target: {env.dist_to_target():.1f}", c='r')
            text_instr.append(text4)

        text5=env.ax.scatter(-100, -100, -100,label=f"Orientation: {env.agent_orientation*180/np.pi:.0f}°", c='r')
        text6=env.ax.scatter(-100, -100, -100,label=f"Altitude: {env.agent_position[-1]:.2f}°", c='r')
        text_instr.append(text5)
        text_instr.append(text6)
        plt.legend()
        plt.pause(0.00001)
        env.fig.canvas.draw_idle()
        env.fig.canvas.flush_events()

        tmp_objects = lines
        return tmp_objects, text_instr