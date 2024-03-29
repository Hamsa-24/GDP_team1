import numpy as np
import time
import csv
import os
import pyttsx3
from geometry import drawText
import pygame

class Environment3D():
    def __init__(self, state_dim, action_dim, safe_altitude=6, 
                 n_sample_paths=50, n_actions_per_step=15, time_per_step=3):
        #np.random.seed(12)
        self.agent_size = 0.5
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_per_step = time_per_step
        self.safe_altitude = safe_altitude
        self.n_sample_paths = n_sample_paths
        self.n_actions_per_step = n_actions_per_step

        #self.time_init = time.time()
        #self.lifespan = max(1,int(np.random.normal(60, 6)))
        self.CL = 0.5
        self.ep_CL = 0
        self.vz = 0
        self.info_obstacle = (0, 0)  # Collision risk, obstacle ahead (boolean)
        self.agent_position = self.get_pos()
        self.agent_orientation = self.get_orientation()
        self.target_zone = self.get_target()
        self.init_dist_to_target = self.dist_to_target()

        self.instr_d_theta = 0
        self.instr_vz = 0
        self.show_instr_vz = False
        self.show_instr_d_theta = False
        self.show_angle_to_xf = False
        self.show_dist_to_xf = False
        self.show_nothing = False

    def reset(self, nav_env, nav_agent):
        #self.time_init = time.time()
        #self.lifespan = max(1,int(np.random.normal(60, 6)))
        self.CL = 0.5
        self.ep_CL = 0
        self.vz = 0
        self.info_obstacle = (0, 0)
        self.agent_position = self.get_pos()
        self.agent_orientation = self.get_orientation()
        self.target_zone = self.get_target()
        self.init_dist_to_target = self.dist_to_target()

        self.show_instr_vz = False
        self.show_instr_d_theta = False
        self.show_angle_to_xf = False
        self.show_dist_to_xf = False
        self.show_nothing = False

        self.get_instructions(nav_env, nav_agent)
        return self.get_state()
    

    def get_info(self, path='csv.csv', scope=3, min_heartrate=60,  max_heartrate=120): ################################################### CHANGER PATH
        while os.path.getsize(path) == 0:
            time.sleep(1)

        with open(path, 'r', newline='') as csvfile:
            while not last_line:
                csv_reader = csv.reader(csvfile)
                last_line = None
                for line in csv_reader:
                    last_line = line
            time_init, time, target_min, target_max, x, y, z, orientation, dist_obstacle, heartrate, state = tuple(last_line)
            self.time_init = time_init
            self.time = time
            self.target_zone = np.vstack((target_min, target_max))/10
            self.agent_position = np.array([x, y, z])/10
            self.agent_orientation = orientation
            collision_risk, obstacle_ahead = 0, 0
            dist_obstacle = dist_obstacle/10
            if dist_obstacle < scope:
                collision_risk, obstacle_ahead = (scope-dist_obstacle)/scope, 1
            self.collision_risk = collision_risk
            self.obstacle_ahead = obstacle_ahead
            self.CL = (heartrate-min_heartrate)/(max_heartrate-min_heartrate)
        
        with open(path, 'w', newline='') as csvfile:
            pass
        return state
            

    def say_info(self, engine, instr):
        engine.say(instr)
        engine.runAndWait()

    def write_info(self, window, width, height, instr):
        font = pygame.font.Font(None, 36)
        text = font.render(instr, True, (255,255,255))
        window.blit(text, (width, height))
        
    
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

    
    def dist_to_target(self):
        return np.linalg.norm(self.agent_position - self.target_zone.mean(axis=0))
    
    def angle_to_target(self):
        v1 = np.array([np.cos(self.agent_orientation), 
                       np.sin(self.agent_orientation)])
        v2 = np.array((self.target_zone[0,:-1]+self.target_zone[1,:-1])/2 - self.agent_position[:-1])
        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        theta = np.arccos(dot_product / (norm_v1 * norm_v2))
        return theta
    
    def check_target_reached(self):
        drone_lower_corner = self.agent_position - self.agent_size / 2
        drone_upper_corner = self.agent_position + self.agent_size / 2
        return (np.all(drone_lower_corner >= self.target_zone[0]) and 
                np.all(drone_upper_corner <= self.target_zone[1]))

    def get_state(self):
        return np.concatenate(([self.instr_d_theta/np.pi],
                               [self.instr_vz],
                               [self.angle_to_target()/np.pi],
                               [self.dist_to_target()/self.init_dist_to_target],
                               [self.CL],
                               [(time.time()-self.time_init)/60],
                                self.info_obstacle,
                               [self.show_instr_d_theta],
                               [self.show_instr_vz]))
    

    def step(self, action, nav_env, nav_agent, window=None):
        show_instr_d_theta_, show_instr_vz_, show_angle_to_xf_, \
            show_dist_to_xf_, show_nothing_ = [i == action for i in range(self.action_dim)]
        #show_instr_d_theta_, show_instr_vz_, show_angle_to_xf_, show_dist_to_xf_ = action
        self.show_instr_d_theta = show_instr_d_theta_
        self.show_instr_vz = show_instr_vz_
        self.show_angle_to_xf = show_angle_to_xf_
        self.show_dist_to_xf = show_dist_to_xf_
        self.show_nothing = show_nothing_

        engine = pyttsx3.init()
        time_init = time.time()
        orient = self.agent_orientation
        while time.time() < time_init + self.time_per_step:
            state = self.get_info()  # 0 = flying, 1 = landed, 2 = crashed, 3 = exploded

            for event in pygame.event.get():
                pass
            window.fill((0, 0, 0))
            self.write_info(window, 50,70,f"Orientation: {self.agent_orientation*180/np.pi:.0f}°")
            self.write_info(window, 50,150,f"Altitude: {self.agent_position[-1]:.2f}°")
            pygame.display.flip()
            if not self.show_nothing:
                if self.show_instr_d_theta:
                    self.say_info(engine, f"Reach orientation {(orient+self.instr_d_theta)*180/np.pi:.0f}°")
                if self.show_instr_vz:
                    if self.instr_vz >= 0:
                        self.say_info(engine, f"Reach altitude {self.safe_altitude:.1f}")
                    else:
                        self.say_info(engine, f"Go down and land on roof")
                if self.show_angle_to_xf:
                    self.say_info(engine, f"Angle to target: {self.angle_to_target()*180/np.pi:.0f}°")
                if self.show_dist_to_xf:
                    self.say_info(engine, f"Distance to target: {self.dist_to_target():.1f}")

            if state == 2:
                return self.get_state(), -100, True  
            if state == 1:
                return self.get_state(), 200, True  
            if state == 3:
                return self.get_state(), -self.ep_CL, True
            
        self.ep_CL += self.CL
        self.get_instructions(nav_env, nav_agent)

        if state == 0:
            return self.get_state(), -self.CL, False