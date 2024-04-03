import numpy as np
import random
import pygame
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from geometry import Cube, Point, draw_line, set_projection, drawText


class Environment3D():
    def __init__(self, state_dim, action_dim, safe_altitude=6, 
                 n_sample_paths=50, n_actions_per_step=15, time_per_step=3):
        #np.random.seed(12)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_per_step = time_per_step
        self.safe_altitude = safe_altitude
        self.n_sample_paths = n_sample_paths
        self.n_actions_per_step = n_actions_per_step

        self.time_init = time.time()
        self.death_time = np.random.uniform(60, 8)
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.ep_CL = 0
        self.vz = 0

        self.n_obstacles = np.random.randint(5,25)
        self.info_obstacle = (0, 0)
        self.agent_size = 0.1
        self.agent_position, self.target_zone, self.forbidden_zone = self.initialize_random(self.n_obstacles)
        self.init_dist_to_target = self.dist_to_target()

        self.count = 0

        self.instr_d_theta = 0
        self.instr_vz = 0
        self.show_instr_vz = False
        self.show_instr_d_theta = False
        self.show_angle_to_xf = False
        self.show_dist_to_xf = False
        self.show_nothing = False




        self.points = []

    # réinitialisation de la position et de la direction du agent
    def reset(self, nav_env, nav_agent):
        self.time_since_failure = 0
        self.death_count = max(1,int(np.random.normal(2400, 240)))
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.ep_CL = 0
        self.vz = 0
        self.agent_size = 0.1

        self.time_init = time.time()
        self.death_time = np.random.uniform(60, 8)
        self.n_obstacles = np.random.randint(5,25)
        self.info_obstacle = (0, 0)
        self.agent_position, self.target_zone, self.forbidden_zone = self.initialize_random(self.n_obstacles)
        self.init_dist_to_target = self.dist_to_target()

        self.count = 0

        self.get_instructions(nav_env, nav_agent)
        self.show_instr_vz = False
        self.show_instr_d_theta = False
        self.show_angle_to_xf = False
        self.show_dist_to_xf = False
        self.show_nothing = False
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
        
        xy_0 = np.random.uniform(self.agent_size, 10-self.agent_size, (2,))
        z_0 = np.random.uniform(6, 10-self.agent_size, (1,))
        pos_0 = np.concatenate((xy_0, z_0))
        agent = np.vstack((pos_0-self.agent_size/2, pos_0+self.agent_size/2))
        while self.object_in_conflict(agent, obstacles):
            xy_0 = np.random.uniform(self.agent_size, 10-self.agent_size, (2,))
            z_0 = np.random.uniform(6, 10-self.agent_size, (1,))
            pos_0 = np.concatenate((xy_0, z_0))
            agent = np.vstack((pos_0-self.agent_size/2, pos_0+self.agent_size/2))

        target = random.choice(obstacles)
        target = target + np.array([[0, 0, target[1,2]],
                                    [0, 0, 1]
                                    ])
        return pos_0, target, obstacles
    

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

    def check_target_reached(self):
        drone_lower_corner = self.agent_position - self.agent_size / 2
        drone_upper_corner = self.agent_position + self.agent_size / 2
        return (np.all(drone_lower_corner >= self.target_zone[0]) and 
                np.all(drone_upper_corner <= self.target_zone[1]))
    
    def is_crashed(self):
        return (time.time() >= self.time_init + self.death_time) or (self.agent_position[-1] <= 0)
    
    def angle_to_target(self):
        v1 = np.array([np.cos(self.agent_orientation), 
                       np.sin(self.agent_orientation)])
        v2 = np.array(self.target_zone[:,:-1].mean(axis=0) - self.agent_position[:-1])
        dot_product = np.dot(v1, v2)
        cross_product = np.cross(v1, v2)
        theta = np.arctan2(cross_product, dot_product)
        return theta
    
    def get_CL(self):
        return 0.5

    def get_instructions(self, nav_env, nav_agent):
        agent_pos = self.agent_position[:-1]
        agent_alt = self.agent_position[-1]
        target_zone = self.target_zone[:,:-1]
        agent_above_target = agent_pos[0] >= target_zone[0,0] and agent_pos[0] <= target_zone[1,0] \
                         and agent_pos[1] >= target_zone[0,1] and agent_pos[1] <= target_zone[1,1]
        instr_d_theta = 0
        instr_vz = 0
        path = []
        if not agent_above_target:
            if agent_alt > self.safe_altitude:
                path = []
                reward_nav = -np.inf
                for _ in range(self.n_sample_paths):
                    nav_env.reset(self)
                    path_, reward_nav_ = self.calculate_path(nav_env, nav_agent)
                    if reward_nav_ > reward_nav: 
                        reward_nav = reward_nav_
                        path = path_
                instr_d_theta = np.mean(path)
            else: 
                instr_vz = 1
        else:
            instr_vz = -1
        self.instr_d_theta = instr_d_theta
        self.instr_vz = instr_vz
        return path

    def calculate_path(self, nav_env, nav_agent):
        path = []
        reward_nav = 0
        done = False
        state = nav_env.get_state()
        for _ in range(self.n_actions_per_step):
            action = nav_agent.take_action(state)
            next_state, reward, done = nav_env.step(action)
            path.append(action[0])
            reward_nav += reward
            state = next_state
            if done: break
        return path, reward_nav
    

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
        self.info_obstacle = np.max(collision_risks), obstacle_ahead
        return np.max(collision_risks), obstacle_ahead


    def get_state(self):
        return np.concatenate(([self.instr_d_theta/np.pi],
                               [self.instr_vz],
                               [self.angle_to_target()/np.pi],
                               [self.dist_to_target()/self.init_dist_to_target],
                               [self.get_CL()],
                               [self.time_since_failure/60],
                                self.obstacle_in_field_of_vision(),
                               [self.show_instr_d_theta],
                               [self.show_instr_vz]))
    

    def step(self, action, nav_env, nav_agent, window):
        show_instr_d_theta_, show_instr_vz_, show_angle_to_xf_, \
            show_dist_to_xf_, show_nothing_ = [i == action for i in range(self.action_dim)]
        #show_instr_d_theta_, show_instr_vz_, show_angle_to_xf_, show_dist_to_xf_ = action
        self.show_instr_d_theta = show_instr_d_theta_
        self.show_instr_vz = show_instr_vz_
        self.show_angle_to_xf = show_angle_to_xf_
        self.show_dist_to_xf = show_dist_to_xf_
        self.show_nothing = show_nothing_

        cubes = []
        for obstacle in self.forbidden_zone:
            cube = Cube()
            cube.init_from_matrix(obstacle)
            cubes.append(cube)
        target = Cube(color=(0,0,1))
        target.init_from_matrix(self.target_zone)
        point = Point(self.agent_position)
        pygame.font.init()

        orient = self.agent_orientation
        start_time, run = time.time(), True
        while time.time() < start_time + self.time_per_step and run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.VIDEORESIZE:
                    glViewport(0, 0, event.w, event.h)
                    set_projection(event.w, event.h)
                    #set_projection(self.agent_position, self.agent_orientation)
            d_theta = 0
            vz = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                d_theta = np.pi/180
            if keys[pygame.K_d]:
                d_theta = -np.pi/180
            if keys[pygame.K_SPACE]:
                vz = 1/2
            if keys[pygame.K_a]:
                vz = -1/2

            orientation_ = self.agent_orientation + d_theta
            if orientation_ > np.pi:
                orientation_ = orientation_ - 2*np.pi
            if orientation_ < -np.pi:
                orientation_ = 2*np.pi + orientation_
            self.agent_orientation = orientation_

            d_xyz = 1/300 * np.array([np.cos(self.agent_orientation),
                                      np.sin(self.agent_orientation),
                                      vz])
            self.agent_position += d_xyz
            self.vz = vz
            point.pos = self.agent_position

            glLoadIdentity()
            glClearColor(0.5, 0.5, 0.5, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            for cube in cubes:
                cube.draw()   
            target.draw()
            point.draw()
            #######
            for p in self.points:
                p.draw()
            draw_line(point, self.agent_orientation)
            drawText(40, 20, f"Orientation: {self.agent_orientation*180/np.pi:.0f}°")
            drawText(40, 70, f"Altitude: {self.agent_position[-1]:.2f}°")
            
            if not self.show_nothing:
                if self.show_instr_d_theta:
                    drawText(window.get_width()-400, 50, f"Reach orientation {(orient+self.instr_d_theta)*180/np.pi:.0f}°")
                if self.show_instr_vz:
                    if self.instr_vz >= 0:
                        drawText(window.get_width()-400, 50, f"Reach altitude {self.safe_altitude:.1f}")
                    else:
                        drawText(window.get_width()-400, 50, f"Go down and land on roof")
                if self.show_angle_to_xf:
                    drawText(window.get_width()-400, 50, f"Angle to target: {self.angle_to_target()*180/np.pi:.0f}°")
                if self.show_dist_to_xf:
                    drawText(window.get_width()-400, 50, f"Distance to target: {self.dist_to_target():.1f}")

            agent = np.vstack((self.agent_position - self.agent_size/2, 
                               self.agent_position + self.agent_size/2))
            pygame.display.flip()

            if self.object_in_conflict(agent, self.forbidden_zone, obj_is_agent=True):
                return self.get_state(), -100, True  
            if self.check_target_reached():
                return self.get_state(), 200, True  
            if self.is_crashed():
                return self.get_state(), -self.ep_CL, True
        
        self.ep_CL += self.get_CL()
        path = self.get_instructions(nav_env, nav_agent)
######
        position_ = self.agent_position
        orientation_ = self.agent_orientation
        for instr in path:
            orientation_ = orientation_ + instr
            position_ = position_ + 1/6*np.array([np.cos(orientation_), np.sin(orientation_), 0])
            point = Point(position_)
            self.points.append(point)
#######
        return self.get_state(), -self.get_CL(), False