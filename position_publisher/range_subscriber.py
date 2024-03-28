#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Range
import csv

class RangeListener(Node):
    def __init__(self):
        super().__init__('range_listener')
        self.subscription = self.create_subscription(
            Range,
            '/simple_drone/sonar/out',
            self.tf_callback,
            10)  # Set QoS depth to 10000
        self.timer = self.create_timer(1, self.save_to_csv)  # Timer to save to CSV every 1 second
        self.csv_file = 'simple_drone_positions.csv'
        self.latest_range = None


    def tf_callback(self, msg : Range):
        # # Extract position information from the TransformStamped message
        # position = msg.position
        # print(position)
        # self.positions.append([position.x, position.y, position.z])
        range = msg.range

        print("distance from obstacle in front of robot:")
        print("d:", range)
 

        # # Add positions to list
        # self.positions.append([position_x, position_y, position_z])
        self.latest_range = range
  

    def save_to_csv(self):
        # # Save positions to CSV file
        # with open(self.csv_file, mode='a') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.positions)
        # self.positions = []  # Clear positions list for next cycle
        if self.latest_range is not None:
            
                # Save the latest range to CSV file
                with open(self.csv_file, mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.latest_range])
                # Clear the latest position
                self.latest_range = None

    # def append_from_csv(self, input_csv_file):
    #     # Read values from another CSV file and append them to simple_drone_positions.csv
    #     with open(input_csv_file, mode='r') as file:
    #         reader = csv.reader(file)
    #         next(reader)  # Skip header if present
    #         for row in reader:
    #             # Write each row to simple_drone_positions.csv
    #             with open(self.csv_file, mode='a') as output_file:
    #                 writer = csv.writer(output_file)
    #                 writer.writerow(row)

def main(args=None):
    rclpy.init(args=args)
    range_listener = RangeListener()
    rclpy.spin(range_listener)
    range_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()