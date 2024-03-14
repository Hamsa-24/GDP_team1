import numpy as np
from rl_utils import update_orientation


class ActionSpace():
    def __init__(self, shape=[2], low=-np.pi/12, high=np.pi/12): #doesn't work, see actor_critic.take_action()
        self.shape = shape
        self.high = high
        self.low = low

class Environment2D():
    def __init__(self, state_dim, action_dim):
        #np.random.seed(12)
        self.robot_size = 0.1
        self.state_dim = state_dim
        self.action_space = ActionSpace(shape=[action_dim])
        self.count = 0
        self.death_count = max(1,int(np.random.normal(100, 10)))
        self.robot_orientation = np.random.uniform(-np.pi, np.pi,(1,))
        self.n_obstacles = 1
        x0, target, obstacles = self.initialize_random(self.n_obstacles)
        self.init_dist_to_target = np.linalg.norm(x0-np.mean(target, axis=0))
        self.robot_position = x0
        self.target_zone = target
        self.forbidden_zone = obstacles

    def initialize_random(self, n_obstacles):
        x_f = np.random.uniform(0,9,(2,))
        target = np.vstack((x_f, x_f+1))
        
        objects = [target]
        obstacles = []
        for _ in range(n_obstacles):
            obstacle_sizes = np.random.uniform(0.5, 2, (2,))
            obstacle_pos = np.random.uniform(0,8, (2,))
            obstacle = np.vstack((obstacle_pos, obstacle_pos+obstacle_sizes))
            
            while self.object_in_conflict(obstacle, objects):
                obstacle_sizes = np.random.uniform(0.5, 2, (2,))
                obstacle_pos = np.random.uniform(0, 8, (2,))
                obstacle = np.vstack((obstacle_pos, obstacle_pos+obstacle_sizes))
            obstacles.append(obstacle)
            objects.append(obstacle)
        
        x_0 = np.random.uniform(0, 10-self.robot_size, (2,))
        robot = np.vstack((x_0-self.robot_size/2, x_0+self.robot_size/2))
        while self.object_in_conflict(robot, objects):
            x_0 = np.random.uniform(0, 10-self.robot_size, (2,))
            robot = np.vstack((x_0-self.robot_size/2, x_0+self.robot_size/2))
        
        return x_0, target, obstacles

    def object_in_conflict(self, object, objects):
        for other_object in objects:
            if self.object_is_in(object, other_object):
                return True
        return False

    def object_is_in(self, object1, object2):
        coordA1, coordA2 = object1[0], object1[1]
        coordB1, coordB2 = object2[0], object2[1]
        xA_1, yA_1, xA_2, yA_2 = coordA1[0], coordA1[1], coordA2[0], coordA2[1]
        xB_1, yB_1, xB_2, yB_2 = coordB1[0], coordB1[1], coordB2[0], coordB2[1]

        if (xA_1 < xB_2 and yA_2 > yB_1) and \
           (xA_1 > xB_1 and yA_2 < yB_2):
            return True
        if (xA_2 > xB_1 and yA_2 > yB_1) and \
           (xA_2 < xB_2 and yA_2 < yB_2):
            return True
        if (xA_2 > xB_1 and yA_1 < yB_2) and \
           (xA_2 < xB_2 and yA_1 > yB_1):
            return True
        if (xA_1 < xB_2 and yA_1 < yB_2) and \
           (xA_1 > xB_1 and yA_1 > yB_1):
            return True
        return False


    def reset(self):
        self.count = 0
        self.robot_orientation = np.random.uniform(-np.pi, np.pi, (1,))
        self.death_count = max(1,int(np.random.normal(100, 10)))
        x0, target, obstacles = self.initialize_random(self.n_obstacles)
        self.robot_position = x0
        self.target_zone = target
        self.forbidden_zone = obstacles
        return self.get_state()
    
    def dist_to_target(self):
        return np.linalg.norm(self.robot_position - self.target_zone.mean(axis=0))
    
    def obstacle_in_field_of_vision(self, angle_vision=np.pi/12):
        # Convertir l'orientation en vecteur directionnel
        position = self.robot_position
        orientation = self.robot_orientation[0]
        direction = np.array([np.cos(orientation),
                              np.sin(orientation)])
        
        for obstacle in self.forbidden_zone:
            # Calculer le vecteur entre la position de l'agent et le centre de l'obstacle
            vector_agent_obstacle = obstacle.mean(axis=0) - position

            # Calculer la distance entre l'agent et le centre de l'obstacle
            distance_agent_obstacle = np.linalg.norm(vector_agent_obstacle)

            # Vérifier si l'obstacle est dans le champ de vision de l'agent
            if distance_agent_obstacle > 0:
                angle_obstacle = np.arccos(np.dot(direction, vector_agent_obstacle) / distance_agent_obstacle)
                if angle_obstacle < angle_vision:
                    return 1
        return 0

    def step(self, d_orientation):
        orientation_ = self.robot_orientation + d_orientation
        if self.robot_orientation[0] + d_orientation[0] > np.pi:
            orientation_[0] = self.robot_orientation[0] + d_orientation[0] - 2*np.pi
        if self.robot_orientation[0] + d_orientation[0] < -np.pi:
            orientation_[0] = 2*np.pi + self.robot_orientation[0] + d_orientation[0]

        self.robot_orientation = orientation_
        
        self.robot_position += 1/4 * np.array([np.cos(self.robot_orientation[0]), 
                                               np.sin(self.robot_orientation[0])])
        
        robot = np.vstack((self.robot_position-self.robot_size/2, 
                           self.robot_position+self.robot_size/2))

        if self.object_in_conflict(robot, self.forbidden_zone):
            return self.get_state(), -100, True  
        
        if self.check_target_reached():
            return self.get_state(), 100, True  
        
        if self.is_crashed():
            return self.get_state(), -self.dist_to_target()*3, True
        
        self.count += 1
        return self.get_state(), -self.dist_to_target()/self.init_dist_to_target, False
    

    def check_target_reached(self):
        # Calcule les coordonnées du coin inférieur et du coin supérieur du cube du drone
        drone_lower_corner = self.robot_position - self.robot_size / 2
        drone_upper_corner = self.robot_position + self.robot_size / 2
    
        # Vérifie si tous les coins du cube du drone sont contenus dans la zone d'arrivée
        return (np.all(drone_lower_corner >= self.target_zone[0]) and 
                np.all(drone_upper_corner <= self.target_zone[1]))
    
    def is_crashed(self):
        return self.count == self.death_count
    
    def angle_to_target(self):
        v1 = np.array([np.cos(self.robot_orientation[0]), 
                       np.sin(self.robot_orientation[0])])
        v2 = np.array((self.target_zone[0]+self.target_zone[1])/2 - self.robot_position)
        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        theta = np.arccos(dot_product / (norm_v1 * norm_v2))
        return theta

    def get_state(self):
        return np.concatenate(([self.angle_to_target()],
                               [self.dist_to_target()/self.init_dist_to_target],
                               [self.obstacle_in_field_of_vision()])
                               )


class Environment3D():
    def __init__(self, state_dim, action_dim):
        #np.random.seed(12)
        self.state_dim = state_dim
        self.action_space = ActionSpace(shape=[action_dim])
        self.count = 0
        self.death_count = max(1,int(np.random.normal(100, 10)))
        self.robot_orientation = np.array([np.random.uniform(-np.pi, np.pi),
                                           np.random.uniform(-np.pi/2, np.pi/2)])
        self.robot_position = np.zeros(3)  # position initiale du robot
        self.robot_size = 0.1  # taille du côté du cube représentant le robot
        self.target_zone = np.array([[9, 9, 9], [10, 10, 10]])# position cible
        self.forbidden_zone = np.array([[4.5, 4.5, 4.5], [6.5, 6.5, 6.5]])  # coins de la zone interdite

    # réinitialisation de la position et de la direction du robot
    def reset(self):
        self.count = 0
        self.robot_position = np.zeros(3)  
        self.robot_orientation = np.array([np.random.uniform(-np.pi, np.pi),
                                           np.random.uniform(-np.pi/2, np.pi/2)])
        self.death_count = max(1,int(np.random.normal(100, 10)))
        return self.get_state()
    
    def dist_to_target(self):
        return np.linalg.norm(self.robot_position - self.target_zone.mean(axis=0))
    
    def obstacle_in_field_of_vision(self, angle_vision=np.pi/12):
        # Convertir l'orientation en vecteur directionnel
        position = self.robot_position
        orientation = self.robot_orientation
        position_obstacle = self.forbidden_zone
        direction = np.array([np.cos(orientation[0]) * np.cos(orientation[1]),
                              np.sin(orientation[0]) * np.cos(orientation[1]),
                              np.sin(orientation[1])])

        # Calculer le vecteur entre la position de l'agent et le centre de l'obstacle
        vector_agent_obstacle = (position_obstacle[0] + position_obstacle[1]) / 2 - position

        # Calculer la distance entre l'agent et le centre de l'obstacle
        distance_agent_obstacle = np.linalg.norm(vector_agent_obstacle)

        # Vérifier si l'obstacle est dans le champ de vision de l'agent
        if distance_agent_obstacle > 0:
            angle_obstacle = np.arccos(np.dot(direction, vector_agent_obstacle) / distance_agent_obstacle)
            if angle_obstacle < angle_vision:
                return 1
        return 0

    def step(self, d_orientation):

        self.robot_orientation = update_orientation(self.robot_orientation, d_orientation)
        self.robot_position += 1/4 * np.array([np.cos(self.robot_orientation[0]) * np.cos(self.robot_orientation[1]), 
                                               np.sin(self.robot_orientation[0]) * np.cos(self.robot_orientation[1]), 
                                               np.sin(self.robot_orientation[1])])

        if self.check_collision(self.robot_position):
            return self.get_state(), -100, True  
        
        # Vérifie si le robot a atteint la position cible
        if self.check_target_reached(self.robot_position):
            return self.get_state(), 100, True  
        
        if self.is_crashed():
            return self.get_state(), -3*self.dist_to_target(), True
        
        self.count += 1
        return self.get_state(), -self.dist_to_target()/20, False

    def check_collision(self, position):
        # Vérifie si le robot est en collision avec la zone interdite
        robot_corners = self.get_robot_corners(position)
        for corner in robot_corners:
            if np.all(corner >= self.forbidden_zone[0]) and \
               np.all(corner <= self.forbidden_zone[1]):
                return True
        return False

    def get_robot_corners(self, position):
        # Retourne les coins du cube représentant le robot
        half_size = self.robot_size / 2
        robot_corners = np.array([
            [position[0] - half_size, position[1] - half_size, position[2] - half_size],
            [position[0] - half_size, position[1] - half_size, position[2] + half_size],
            [position[0] - half_size, position[1] + half_size, position[2] - half_size],
            [position[0] - half_size, position[1] + half_size, position[2] + half_size],
            [position[0] + half_size, position[1] - half_size, position[2] - half_size],
            [position[0] + half_size, position[1] - half_size, position[2] + half_size],
            [position[0] + half_size, position[1] + half_size, position[2] - half_size],
            [position[0] + half_size, position[1] + half_size, position[2] + half_size]
        ])
        return robot_corners

    def check_target_reached(self, position):
    
        # Calcule les coordonnées du coin inférieur et du coin supérieur du cube du drone
        drone_lower_corner = position - self.robot_size / 2
        drone_upper_corner = position + self.robot_size / 2
    
        # Vérifie si tous les coins du cube du drone sont contenus dans la zone en forme de pavé droit
        return (np.all(drone_lower_corner >= self.target_zone[0]) and 
                np.all(drone_upper_corner <= self.target_zone[1]))
    
    def is_crashed(self):
        #return np.random.rand() < prob_crash(nstep)
        return self.count == self.death_count
    
    def angle_to_target(self):
        v1 = np.array([np.cos(self.robot_orientation[0]) * np.cos(self.robot_orientation[1]), 
                       np.sin(self.robot_orientation[0]) * np.cos(self.robot_orientation[1]), 
                       np.sin(self.robot_orientation[1])])
        v2 = np.array((self.target_zone[0]+self.target_zone[1])/2 - self.robot_position)
        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        theta = np.arccos(dot_product / (norm_v1 * norm_v2))
        return theta

    def get_state(self):
        # Retourne l'état actuel de l'environnement
        return np.concatenate(([self.angle_to_target()],
                               [self.dist_to_target()/20],
                               [self.obstacle_in_field_of_vision()]))