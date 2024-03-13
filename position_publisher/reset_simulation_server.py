#!/usr/bin/env python3

import subprocess

def reset_world():
    # Define the command to call the service
    command = "ros2 service call /reset_world std_srvs/srv/Empty"

    try:
        # Execute the command in the terminal
        subprocess.run(command, shell=True, check=True)
        print("Reset world command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing reset world command: {e}")

if __name__ == "__main__":
    reset_world()