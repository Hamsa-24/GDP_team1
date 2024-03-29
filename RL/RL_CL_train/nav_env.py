import numpy as np

class NavEnvironment():
    def __init__(self, state_dim, action_dim, env):
        #np.random.seed(12)
        self.agent_size = np.copy(env.agent_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.count = np.copy(env.count)
        self.death_count = np.copy(env.death_count)
        self.agent_orientation = np.copy(env.agent_orientation)
        self.agent_position = np.copy(env.agent_position)
        self.target_zone = np.copy(env.target_zone)
        self.forbidden_zone = np.copy(env.forbidden_zone)
        self.init_dist_to_target = self.dist_to_target()

    def reset(self, env):
        self.count = np.copy(env.count)
        self.agent_orientation = np.copy(env.agent_orientation)
        self.death_count = np.copy(env.death_count)
        self.agent_position = np.copy(env.agent_position)
        self.target_zone = np.copy(env.target_zone)
        self.forbidden_zone = np.copy(env.forbidden_zone)
        self.init_dist_to_target = self.dist_to_target()
        return self.get_state()
    

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
        return np.linalg.norm(self.agent_position[:-1] - self.target_zone[:,:-1].mean(axis=0))
    
    def deniv_to_target(self):
        return self.target_zone[-1,-1] - self.agent_position[-1]
    
    def obstacle_in_field_of_vision1(self, angle_vision=np.pi/12, scope=3):
        position = self.agent_position
        orientation = self.agent_orientation
        direction = np.array([np.cos(orientation), np.sin(orientation), 0])

        dist_closest_obstacle = 100
        for obstacle in self.forbidden_zone:
            horizontal_vector = (obstacle.mean(axis=0) - position)[:-1]
            horizontal_dist = np.linalg.norm(horizontal_vector)

            if horizontal_dist < dist_closest_obstacle and position[-1] < obstacle[-1,-1]:
                dist_closest_obstacle = horizontal_dist

            if horizontal_dist > 0:
                angle_obstacle = np.arccos(np.dot(direction[:-1], horizontal_vector) / horizontal_dist)
                if (angle_obstacle < angle_vision) and \
                   (horizontal_dist < scope) and \
                   (position[-1] < obstacle[-1,-1]):
                    return horizontal_dist/scope, 1
        return dist_closest_obstacle/scope, 0
    

    def obstacle_in_field_of_vision(self, angle_vision=np.pi/12, scope=3):
        position = self.agent_position[:-1]
        alt = self.agent_position[-1]
        orientation = self.agent_orientation
        direction = np.array([np.cos(orientation),
                              np.sin(orientation)])
        
        collision_risk = 0
        obstacle_ahead = 0
        collision_risks = [collision_risk]

        for obstacle in self.forbidden_zone:
            vector_agent_obstacle = (obstacle[:,:-1].mean(axis=0) - position)
            distance_agent_obstacle = np.linalg.norm(vector_agent_obstacle)
            if distance_agent_obstacle > 0 and alt < obstacle[-1,-1]:
                angle_obstacle = np.arccos(np.dot(direction, vector_agent_obstacle) / distance_agent_obstacle)
                if angle_obstacle < angle_vision and distance_agent_obstacle < scope:
                    collision_risk = (scope-distance_agent_obstacle)/scope
                    collision_risks.append(collision_risk)
                    obstacle_ahead = 1
    
        return np.max(collision_risks), obstacle_ahead


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
        v2 = np.array(self.target_zone[:,:-1].mean(axis=0) - self.agent_position[:-1])
        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        theta = np.arccos(dot_product / (norm_v1 * norm_v2))
        return theta


    def get_state(self):
        return np.concatenate(([self.angle_to_target()/np.pi],
                               [self.dist_to_target()/self.init_dist_to_target],
                                self.obstacle_in_field_of_vision()))
    

    def step(self, action):
        d_orientation = action[0]
        orientation_ = self.agent_orientation + d_orientation
        if orientation_ > np.pi:
            orientation_ = orientation_ - 2*np.pi
        if orientation_ < -np.pi:
            orientation_ = 2*np.pi + orientation_
        self.agent_orientation = orientation_

        d_xy = 1/6 * np.array([np.cos(self.agent_orientation),
                               np.sin(self.agent_orientation)])

        self.agent_position += np.concatenate((d_xy, [0]))
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