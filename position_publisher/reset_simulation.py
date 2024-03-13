#!/usr/bin/env python3

# import rclpy
# from std_msgs.msg import Empty

# def reset_simulation():
#     rclpy.init()
#     node = rclpy.create_node('reset_publisher')

#     # Create a publisher for the Empty message on the /simple_drone/reset topic
#     publisher = node.create_publisher(Empty, '/simple_drone/reset', 10)

#     # Create an instance of the Empty message
#     empty_msg = Empty()

#     # Publish the Empty message to trigger the reset
#     publisher.publish(empty_msg)

#     # Spin briefly to allow the message to be published
#     rclpy.spin_once(node)

#     # Clean up
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     reset_simulation()

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

def reset_simu():
    rclpy.init()
    node = rclpy.create_node('reset_client')

    # Create a client to call the /reset_world service
    client = node.create_client(Trigger, '/reset_world')

    # Wait for the service to be available
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Service /reset_world not available, waiting...')

    # Create a request object for the service
    request = Trigger.Request()

    # Call the service
    future = client.call_async(request)

    # Wait for the service call to complete
    rclpy.spin_until_future_complete(node, future)

    # Check if the service call was successful
    if future.result() is not None:
        node.get_logger().info('Reset world successful')
    else:
        node.get_logger().error('Failed to reset world')

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    reset_simu()