import matplotlib.pyplot as plt
import numpy as np


robot_position = np.random.uniform(0,10,(2,))
print(robot_position)
robot_size = 0.1
robot = np.vstack((robot_position, 
                       robot_position+robot_size))
print(robot)
