import matplotlib.pyplot as plt
import numpy as np


robot_position = np.random.uniform(0,10,(2,))
print(robot_position)
robot_size = 0.1
robot = np.vstack((robot_position, 
                       robot_position+robot_size))
print(robot)

x0 = np.array([0, 0])
target = np.array([[-0.5, 5],[0.5, 6]])
init_dist_to_target = np.linalg.norm(x0-np.mean(target, axis=0))
print(init_dist_to_target)