#!/usr/bin/env python3

import numpy as np

size_apartment = [20.08,22.19,14.79]
size_fast_food = [24.91, 15.725,11.015]
size_gas_station = [17.52,25.53,7.675]
size_house_1 = [15.50,16.39,7.68]
size_house_2 = [12.48,8.94,7.19]
size_house_3 = [4.57,11.79,10.61]
size_law_office = [6.84,5.43,13.92]
size_osrf_first_office = [26.18,18.30,5.73]
size_radio_tower = [11.72,13.39,44.19]
size_salon = [7.21,5.37,11.38]
size_thrift_shop = [7.21,5.43,11.38]
size_post_office = [10.40,7.30,3.95]


class Environment3D:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros(size, dtype=float)  # Initialize grid with all cells being allowed

    def set_forbidden_positions(self, positions):
        for pos in positions:
            self.grid[pos[0], pos[1], pos[2]] = 1  # Set forbidden positions to 1

    def is_position_allowed(self, position):
        return self.grid[round(position[0]), round(position[1]), round(position[2])] == 0  # Check if the position is allowed (value 0)

    def create_building(self, position, yaw, dimensions):

        if yaw == 1.57:
            new_width = dimensions[0]
            new_length = dimensions[1]
            dimensions[0] = new_length
            dimensions[1] = new_width

        
        position_start = position.copy()
        position_start[0] = position_start[0] - dimensions[0]/2
        position_start[1] = position_start[1] - dimensions[1]/2
        position_end = position.copy() 
        position_end[0] = position_end[0] + dimensions[0]/2
        position_end[1] = position_end[1] + dimensions[1]/2
        position_end[2] = position_end[2] + dimensions[2]
        start = np.floor(position_start).astype(int)
        end = np.ceil(position_end).astype(int)

        forbidden_positions = []
        for x in range (start[0],end[0]):
            for y in range (start[1],end[1]):
                for z in range (start[2],end[2]):
                    forbidden_positions.append([x,y,z])
        self.set_forbidden_positions(forbidden_positions)
        # slices = tuple(slice(max(0, s), min(self.size[i], e)) for i, (s, e) in enumerate(zip(start, end)))
        # self.grid[slices[0], slices[1], slices[2]] = 1  # Set cells within the building's dimensions to forbidden

#         # for x in range(position[0] - dimensions[0]/2, position[0] + dimensions[0]/2):
#         #     for y in range(position[1] - dimensions[1]/2, position[1] + dimensions[1]/2):
#         #         for z in range(position[2], position[2] + dimensions[2]):
#         #             if (0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]):
#         #                 self.grid[x, y, z] = 1  # Set cells within the building's dimensions to forbidden