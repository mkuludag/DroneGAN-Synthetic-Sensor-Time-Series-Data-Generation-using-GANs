#!/usr/bin/env python3

# TO help generate multiple files easily: 

import subprocess

# List of arguments to pass to GAN_timeseries_window.py
arguments_list = [
    # "accel_x",
    # "accel_y",
    # "accel_z",
    # "gyro_x",
    # "gyro_y",
    # "gyro_z",
    # "gps_c_x",
    # "gps_c_y",
    # "gps_c_z",
    # "gps_x",
    # "gps_y",
    # "gps_z"
    # "accelerometer_m_s2[0]",
    # "accelerometer_m_s2[1]",
    # "accelerometer_m_s2[2]",
     "gyro_rad[0]",
     "gyro_rad[1]",
     "gyro_rad[2]",


]

# Loop through the list of arguments and run the script
for arguments in arguments_list:
    command = f"python3 GAN_timeseries_window.py {arguments}"
    try:
        # Run the script with subprocess
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the script with arguments: {arguments}")
