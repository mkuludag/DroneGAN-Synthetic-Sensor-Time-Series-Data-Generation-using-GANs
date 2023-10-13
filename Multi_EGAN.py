#!/usr/bin/env python3

import subprocess

# Define the list of sensor configurations
sensor_configurations = [
    # ["accelerometer_m_s2[0]", "accelerometer_m_s2[1]"],   # accelX + accelY
    # ["accelerometer_m_s2[0]", "accelerometer_m_s2[2]"],   # accelX + accelZ
    # ["accelerometer_m_s2[1]", "accelerometer_m_s2[2]"],   # accelY + accelZ
    # ["accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"],  # accelX + accelY + accelZ
    # ["gyro_rad[0]", "gyro_rad[1]"],                       # gyroX + gyroY
    # ["gyro_rad[1]", "gyro_rad[2]"],                       # gyroY + gyroZ
    ["gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]"],        # gyroX + gyroY + gyroZ
    # ["accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]", "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]"],  # All sensors
]

# Path to ensemble_gan.py
ensemble_gan_script = "ensemble_gan.py"

# Loop through sensor configurations and run the script for each configuration
for config in sensor_configurations:
    # Construct the command to run ensemble_gan.py with the current configuration
    command = ["python", ensemble_gan_script] + config
    subprocess.run(command)

print("All configurations have been processed.")

