import numpy as np
import random


class ActionSpace():
    def __init__(self, shape=[2], low=[-np.pi/12], high=[np.pi/12]): 
        self.shape = shape
        self.high = high
        self.low = low

class Environment2D():
    def __init__(self, state_dim, action_dim):
        #np.random.seed(12)
        self.agent_size = 0.1
        self.state_dim = state_dim
        self.action_space = ActionSpace(shape=[action_dim])
        self.count = 0
        self.death_count = max(1,int(np.random.normal(100, 10)))
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.n_obstacles = np.random.randint(0,25)
        x0, target, obstacles = self.initialize_random(self.n_obstacles)
        self.agent_position = x0
        self.target_zone = target
        self.forbidden_zone = obstacles
        self.init_dist_to_target = self.dist_to_target()

    def reset(self):
        self.count = 0
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.death_count = max(1,int(np.random.normal(100, 10)))
        self.n_obstacles = np.random.randint(0,25)
        x0, target, obstacles = self.initialize_random(self.n_obstacles)
        self.agent_position = x0
        self.target_zone = target
        self.forbidden_zone = obstacles
        self.init_dist_to_target = self.dist_to_target()
        return self.get_state()
    

    def initialize_random(self, n_obstacles):
        max_obj_size = 2 * 1/np.sqrt(n_obstacles)
        x_f = np.random.uniform(0,9,(2,))
        target = np.vstack((x_f, x_f+1))
        
        objects = [target]
        obstacles = []
        for _ in range(n_obstacles):
            obstacle_sizes = np.random.uniform(0.2*max_obj_size, max_obj_size, (2,))
            obstacle_pos = np.random.uniform(0, 10-max_obj_size, (2,))
            obstacle = np.vstack((obstacle_pos, obstacle_pos+obstacle_sizes))
            
            while self.object_in_conflict(obstacle, objects):
                obstacle_sizes = np.random.uniform(0.2*max_obj_size, max_obj_size, (2,))
                obstacle_pos = np.random.uniform(0, 8, (2,))
                obstacle = np.vstack((obstacle_pos, obstacle_pos+obstacle_sizes))
            obstacles.append(obstacle)
            objects.append(obstacle)
        
        x_0 = np.random.uniform(0, 10-self.agent_size, (2,))
        agent = np.vstack((x_0-self.agent_size/2, x_0+self.agent_size/2))
        while self.object_in_conflict(agent, objects):
            x_0 = np.random.uniform(0, 10-self.agent_size, (2,))
            agent = np.vstack((x_0-self.agent_size/2, x_0+self.agent_size/2))
        
        return x_0, target, obstacles

    def object_in_conflict(self, object, objects):
        for other_object in objects:
            if self.object_is_in(object, other_object):
                return True
            if self.object_is_in(other_object, object): 
                return True
        return False

    def object_is_in(self, object1, object2):
        coordA1, coordA2 = object1
        coordB1, coordB2 = object2
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
        x0, y0 = np.mean(object1, axis=0)
        if x0 > xB_1 and x0 < xB_2 and y0 > yB_1 and y0 < yB_2:
            return True
        return False

    
    def dist_to_target(self):
        return np.linalg.norm(self.agent_position - self.target_zone.mean(axis=0))
    
    def obstacle_in_field_of_vision(self, angle_vision=np.pi/12, scope=3):
        # Convertir l'orientation en vecteur directionnel
        position = self.agent_position
        orientation = self.agent_orientation
        direction = np.array([np.cos(orientation),
                              np.sin(orientation)])
        
        collision_risk = 0
        obstacle_ahead = 0
        collision_risks = [collision_risk]

        for obstacle in self.forbidden_zone:
            vector_agent_obstacle = obstacle.mean(axis=0) - position
            distance_agent_obstacle = np.linalg.norm(vector_agent_obstacle)
            if distance_agent_obstacle > 0:
                angle_obstacle = np.arccos(np.dot(direction, vector_agent_obstacle) / distance_agent_obstacle)
                if angle_obstacle < angle_vision and distance_agent_obstacle < scope:
                    collision_risk = (scope-distance_agent_obstacle)/scope
                    collision_risks.append(collision_risk)
                    obstacle_ahead = 1
    
        return np.max(collision_risks), obstacle_ahead


    def check_target_reached(self):
        # Calcule les coordonnées du coin inférieur et du coin supérieur du cube du drone
        drone_lower_corner = self.agent_position - self.agent_size / 2
        drone_upper_corner = self.agent_position + self.agent_size / 2
    
        # Vérifie si tous les coins du cube du drone sont contenus dans la zone d'arrivée
        return (np.all(drone_lower_corner >= self.target_zone[0]) and 
                np.all(drone_upper_corner <= self.target_zone[1]))
    
    def is_crashed(self):
        return self.count == self.death_count
    
    def angle_to_target1(self):
        v1 = np.array([np.cos(self.agent_orientation), 
                       np.sin(self.agent_orientation)])
        v2 = np.array((self.target_zone[0]+self.target_zone[1])/2 - self.agent_position)
        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        theta = np.arccos(dot_product / (norm_v1 * norm_v2))
        return theta
    
    def angle_to_target(self):
        v1 = np.array([np.cos(self.agent_orientation), 
                       np.sin(self.agent_orientation)])
        v2 = np.array((self.target_zone[0]+self.target_zone[1])/2 - self.agent_position) 
        dot_product = np.dot(v1, v2)
        cross_product = np.cross(v1, v2)
        theta = np.arctan2(cross_product, dot_product)
        return theta
    
    
    def step(self, action):
        d_orientation = action[0]
        orientation_ = self.agent_orientation + d_orientation

        if self.agent_orientation + d_orientation > np.pi:
            orientation_ = self.agent_orientation + d_orientation - 2*np.pi
        if self.agent_orientation + d_orientation < -np.pi:
            orientation_ = 2*np.pi + self.agent_orientation + d_orientation

        self.agent_orientation = orientation_
        
        self.agent_position += 1/6 * np.array([np.cos(self.agent_orientation), 
                                               np.sin(self.agent_orientation)])
        
        agent = np.vstack((self.agent_position-self.agent_size/2, 
                           self.agent_position+self.agent_size/2))

        if self.object_in_conflict(agent, self.forbidden_zone):
            return self.get_state(), -100, True  
        
        if self.check_target_reached():
            return self.get_state(), 150, True  
        
        if self.is_crashed():
            return self.get_state(), -self.dist_to_target()*3, True
        
        self.count += 1
        return self.get_state(), \
              -self.dist_to_target()/self.init_dist_to_target/2 - np.absolute(d_orientation)/12, False
    

    def get_state(self):
        return np.concatenate(([(self.angle_to_target())/(np.pi)],
                               [self.dist_to_target()/self.init_dist_to_target],
                                self.obstacle_in_field_of_vision())
                               )


class Environment3D():
    def __init__(self, state_dim, action_dim):
        #np.random.seed(12)
        self.agent_size = 0.1
        self.state_dim = state_dim
        self.action_space = ActionSpace(shape=[action_dim], 
                                        low=[-np.pi/12, -1/12],
                                        high=[np.pi/12, 1/12])
        self.count = 0
        self.death_count = max(1,int(np.random.normal(100, 10)))
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.n_obstacles = np.random.randint(5,25)
        x0, target, obstacles = self.initialize_random(self.n_obstacles)
        self.vspeed = 0
        self.agent_position = x0
        self.target_zone = target
        self.forbidden_zone = obstacles
        self.init_dist_to_target = self.dist_to_target()
        self.init_deniv_to_target = self.deniv_to_target()

    # réinitialisation de la position et de la direction du agent
    def reset(self):
        self.count = 0
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.death_count = max(1,int(np.random.normal(100, 10)))
        self.n_obstacles = np.random.randint(5,25)
        x0, target, obstacles = self.initialize_random(self.n_obstacles)
        self.agent_position = x0
        self.vspeed = 0
        self.target_zone = target
        self.forbidden_zone = obstacles
        self.init_dist_to_target = self.dist_to_target()
        self.init_deniv_to_target = self.deniv_to_target()
        return self.get_state()
    
    def initialize_random(self, n_obstacles):
        max_obj_size = 4 * 1/np.sqrt(n_obstacles)
        
        obstacles = []
        for _ in range(n_obstacles):
            obstacle_base = np.random.uniform(0.2*max_obj_size, max_obj_size, (2,))
            obstacle_height = np.random.uniform(1, 6, (1,))
            obstacle_sizes = np.concatenate((obstacle_base, obstacle_height))
            obstacle_pos = np.random.uniform(0, 10-max_obj_size, (2,))
            obstacle_pos = np.append(obstacle_pos, 0)
            obstacle = np.vstack((obstacle_pos, obstacle_pos+obstacle_sizes))
            
            while self.object_in_conflict(obstacle, obstacles):
                obstacle_base = np.random.uniform(0.2*max_obj_size, max_obj_size, (2,))
                obstacle_height = np.random.uniform(1, 9, (1,))
                obstacle_sizes = np.concatenate((obstacle_base, obstacle_height))
                obstacle_pos = np.random.uniform(0, 10-max_obj_size, (2,))
                obstacle_pos = np.append(obstacle_pos, 0)
                obstacle = np.vstack((obstacle_pos, obstacle_pos+obstacle_sizes))
            obstacles.append(obstacle)
        
        x_0 = np.random.uniform(1, 10-self.agent_size, (3,))
        agent = np.vstack((x_0-self.agent_size/2, x_0+self.agent_size/2))
        while self.object_in_conflict(agent, obstacles):
            x_0 = np.random.uniform(0, 10-self.agent_size, (3,))
            agent = np.vstack((x_0-self.agent_size/2, x_0+self.agent_size/2))

        target = random.choice(obstacles)
        target = target + np.array([[0, 0, target[1,2]],
                                    [0, 0, 1]
                                    ])
        return x_0, target, obstacles
    

    def object_in_conflict(self, object, objects, obj_is_agent=False):
        for other_object in objects:
            if self.object_is_in(object, other_object, obj1_is_agent=obj_is_agent):
                return True
            if not obj_is_agent:
                if self.object_is_in(other_object, object): 
                    return True
        return False
    
    def object_is_in(self, object1, object2, obj1_is_agent=False):
        coordA1, coordA2 = object1
        coordB1, coordB2 = object2
        xA_1, yA_1, _ = coordA1
        xA_2, yA_2, _ = coordA2
        xB_1, yB_1, _ = coordB1
        xB_2, yB_2, _ = coordB2

        flag = False
        if (xA_1 < xB_2 and yA_2 > yB_1) and \
           (xA_1 > xB_1 and yA_2 < yB_2):
            flag = True
        if (xA_2 > xB_1 and yA_2 > yB_1) and \
           (xA_2 < xB_2 and yA_2 < yB_2):
            flag = True
        if (xA_2 > xB_1 and yA_1 < yB_2) and \
           (xA_2 < xB_2 and yA_1 > yB_1):
            flag = True
        if (xA_1 < xB_2 and yA_1 < yB_2) and \
           (xA_1 > xB_1 and yA_1 > yB_1):
            flag = True
        x0, y0 = np.mean(object1[:,:-1], axis=0)
        if x0 > xB_1 and x0 < xB_2 and y0 > yB_1 and y0 < yB_2:
            flag = True
        
        if obj1_is_agent and flag:
            if self.agent_position[-1] < object2[-1,-1]: return True
            else: return False
        return flag

    
    def dist_to_target(self):
        return np.linalg.norm(self.agent_position - self.target_zone.mean(axis=0))
    
    def deniv_to_target(self):
        return self.target_zone.mean(axis=0)[-1] - self.agent_position[-1]
    
    def obstacle_in_field_of_vision(self, angle_vision=np.pi/12, scope=3):

        position = self.agent_position
        orientation = self.agent_orientation
        direction = np.array([np.cos(orientation), np.sin(orientation), 0])

        dist_closest_obstacle = 100
        for obstacle in self.forbidden_zone:
            vector_agent_obstacle = obstacle.mean(axis=0) - position

            # Calculer la distance entre l'agent et le centre de l'obstacle
            distance_agent_obstacle = np.linalg.norm(vector_agent_obstacle)
            if distance_agent_obstacle < dist_closest_obstacle:
                dist_closest_obstacle = distance_agent_obstacle

            # Vérifier si l'obstacle est dans le champ de vision de l'agent
            if distance_agent_obstacle > 0:
                angle_obstacle = np.arccos(np.dot(direction, vector_agent_obstacle) / distance_agent_obstacle)
                if (angle_obstacle < angle_vision) and \
                   (distance_agent_obstacle < scope) and \
                   (position[-1] < obstacle[-1,-1]):
                    return distance_agent_obstacle/scope, 1
        return dist_closest_obstacle/scope, 0


    def check_target_reached(self):
        drone_lower_corner = self.agent_position - self.agent_size / 2
        drone_upper_corner = self.agent_position + self.agent_size / 2
        return (np.all(drone_lower_corner >= self.target_zone[0]) and 
                np.all(drone_upper_corner <= self.target_zone[1]))
    
    def is_crashed(self):
        return self.count == self.death_count
    
    def angle_to_target(self):
        v1 = np.array([np.cos(self.agent_orientation), 
                       np.sin(self.agent_orientation)])
        v2 = np.array((self.target_zone[0,:-1]+self.target_zone[1,:-1])/2 - self.agent_position[:-1])
        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        theta = np.arccos(dot_product / (norm_v1 * norm_v2))
        return theta

    def get_state(self):
        return np.concatenate(([self.angle_to_target()],
                               [self.dist_to_target()/self.init_dist_to_target],
                               [self.deniv_to_target()/self.init_deniv_to_target],
                                self.obstacle_in_field_of_vision()))
    

    def step(self, action):
        d_orientation, d_vspeed = action
        orientation_ = self.agent_orientation + d_orientation
        if orientation_ > np.pi:
            orientation_ = orientation_ - 2*np.pi
        if orientation_ < -np.pi:
            orientation_ = 2*np.pi + orientation_
        self.agent_orientation = orientation_

        d_xy = 1/6 * np.array([np.cos(self.agent_orientation),
                               np.sin(self.agent_orientation)])
        
        vspeed_ = min(1/8, max(self.vspeed + d_vspeed,-1/8))
        self.vspeed = vspeed_

        self.agent_position += np.concatenate((d_xy, [self.vspeed]))

        agent = np.vstack((self.agent_position - self.agent_size/2, 
                           self.agent_position + self.agent_size/2))

        if self.object_in_conflict(agent, self.forbidden_zone, obj_is_agent=True):
            return self.get_state(), -100, True  
        
        if self.check_target_reached():
            return self.get_state(), 150, True  
        
        if self.is_crashed():
            return self.get_state(), -3*self.dist_to_target(), True
        
        self.count += 1
        return self.get_state(), \
              -self.dist_to_target()/self.init_dist_to_target/2 \
              -np.absolute(d_orientation)/12, False