import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns

real_data = pd.read_csv("test_data/benign_data/sensor_combined.csv")
generated_data1 = pd.read_csv("test_data/gan_data/comined_data.csv")

generated_data_ensemble = pd.read_csv("test_data/ensemble_gan_data/egan_all_sensors.csv")



spoofed_data = pd.read_csv("test_data/gps_spoofing_data/sensor_combined.csv")
jammed_data = pd.read_csv("test_data/gps_jamming_data/sensor_combined.csv")


sensor_names = [
    #"gyro_rad[0]", "gyro_rad[2]",  "xyz[0]"
    # "accelerometer_m_s2[0]",  "accelerometer_m_s2[1]",  "accelerometer_m_s2[2]",
    # 
    # "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z", "gps_x", "gps_y", "gps_z", "mag_x", "mag_y", "mag_z"
    "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]",
    "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"

    ] 

total_number_of_samples = 1000
print("Total number of Actual Samples")
print(total_number_of_samples)


new_real_data = real_data[sensor_names].iloc[:total_number_of_samples]
new_spoofed_data = spoofed_data[sensor_names].iloc[:total_number_of_samples]
new_jammed_data = jammed_data[sensor_names].iloc[:total_number_of_samples]


new_generated_data = generated_data1[sensor_names].iloc[:total_number_of_samples]
new_generated_data_ensemble = generated_data_ensemble[sensor_names].iloc[:total_number_of_samples]

new_real_data["label"] = "benign"
new_jammed_data["label"] = "jammed"
new_spoofed_data["label"] = "spoofed"
new_generated_data["label"] = "gan data"
new_generated_data_ensemble["label"] = "ensemble data"

# data_frames = [new_real_data, new_jammed_data, new_spoofed_data, new_generated_data]
data_frames = [new_real_data, new_generated_data_ensemble]

all_data = pd.concat(data_frames)

all_data.reset_index(inplace=True)
# print(all_data)
sns.pairplot(all_data, hue ='label')
plt.show()
# # features = all_data.drop(['label'],1)
# labels = all_data['label']

# print(labels.head())
# print(features.head())

# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(features)


# df = pd.DataFrame()
# df["y"] = labels
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 2),
#                 data=df).set(title="Iris data T-SNE projection") 

# plt.show()