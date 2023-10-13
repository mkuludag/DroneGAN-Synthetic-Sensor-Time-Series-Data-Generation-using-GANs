import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import StandardScaler


benign_data = pd.read_csv("test_data/data_for_visualization_analysis/Benign.csv")

generated_data_ensemble = pd.read_csv("test_data/data_for_visualization_analysis/ensemble_gan.csv")

generated_data_singular = pd.read_csv("test_data/data_for_visualization_analysis/seperate_gans.csv")

# generated_data1 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[0].csv")
# generated_data2 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[1].csv")
# generated_data3 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[2].csv")

# generated_data4 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[0].csv")
# generated_data5 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[1].csv")
# generated_data6 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[2].csv")

sensor_names = [
    "gyro_rad[0]", 
    "gyro_rad[1]", 
    "gyro_rad[2]",
    "accelerometer_m_s2[0]", 
    "accelerometer_m_s2[1]", 
    "accelerometer_m_s2[2]"
    ] 

total_number_of_samples = 100


# step 1 - scale data
scaled_features = StandardScaler().fit_transform(benign_data.values)
benign_data_scaled = pd.DataFrame(scaled_features, index=benign_data.index, columns=benign_data.columns)

scaled_features_gan = StandardScaler().fit_transform(generated_data_ensemble.values)
gan_data_scaled_ensemble = pd.DataFrame(scaled_features_gan, index=generated_data_ensemble.index, columns=generated_data_ensemble.columns)

scaled_features_gan_singular = StandardScaler().fit_transform(generated_data_singular.values)
gan_data_scaled_singular = pd.DataFrame(scaled_features_gan_singular, index=generated_data_singular.index, columns=generated_data_singular.columns)


# scaled_features_gan1 = StandardScaler().fit_transform(generated_data1.values)
# gan_data_scaled1 = pd.DataFrame(scaled_features_gan1, index=generated_data1.index, columns=generated_data1.columns)

# scaled_features_gan2 = StandardScaler().fit_transform(generated_data2.values)
# gan_data_scaled2 = pd.DataFrame(scaled_features_gan2, index=generated_data2.index, columns=generated_data2.columns)

# # scaled_features_gan3 = StandardScaler().fit_transform(generated_data3.values)
# # gan_data_scaled3 = pd.DataFrame(scaled_features_gan3, index=generated_data3.index, columns=generated_data3.columns)

# scaled_features_gan4 = StandardScaler().fit_transform(generated_data4.values)
# gan_data_scaled4 = pd.DataFrame(scaled_features_gan4, index=generated_data4.index, columns=generated_data4.columns)

# scaled_features_gan5 = StandardScaler().fit_transform(generated_data5.values)
# gan_data_scaled5 = pd.DataFrame(scaled_features_gan5, index=generated_data5.index, columns=generated_data5.columns)

# scaled_features_gan6 = StandardScaler().fit_transform(generated_data6.values)
# gan_data_scaled6 = pd.DataFrame(scaled_features_gan6, index=generated_data6.index, columns=generated_data6.columns)


# step 2 - normalize between 0 and 1
benign_data_normalized = benign_data_scaled.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized_ensemble = gan_data_scaled_ensemble.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized_singular = gan_data_scaled_singular.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# generated_data_normalized1 = gan_data_scaled1.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# generated_data_normalized2 = gan_data_scaled2.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# # generated_data_normalized3 = gan_data_scaled3.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# generated_data_normalized4 = gan_data_scaled4.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# generated_data_normalized5 = gan_data_scaled5.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# generated_data_normalized6 = gan_data_scaled6.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# real data split
benign_data_features = []
benign_data_labels = []
for i in range(total_number_of_samples):
    sample = benign_data_normalized[sensor_names].iloc[i].values
    # np.append(sample, i)
    # label = real_data_normalized[sensor_names].iloc[i+window_size].values
    benign_data_features.append(sample)
    benign_data_labels.append(0)

# ensemble gan data
gan_ensemble_data_features = []
gan_ensemble_data_labels = []

for i in range(total_number_of_samples):
    sample_real_data = generated_data_normalized_ensemble[sensor_names].iloc[i].values
    # np.append(sample_real_data, i)
    # if np.array(sample_real_data) not in X_train:
    # gan_data.append(sample)
    # sample_real_data[0] = generated_data_normalized['gyro_rad[0]'].iloc[i]
    # sample_real_data[1] = generated_data_normalized['gyro_rad[1]'].iloc[i]
    # sample_real_data[2] = generated_data_normalized['gyro_rad[2]'].iloc[i]
    # sample_real_data[3] = generated_data_normalized['accelerometer_m_s2[0]'].iloc[i]
    # sample_real_data[4] = generated_data_normalized['accelerometer_m_s2[1]'].iloc[i]
    # sample_real_data[5] = generated_data_normalized['accelerometer_m_s2[2]'].iloc[i]
    gan_ensemble_data_features.append(sample_real_data)
    gan_ensemble_data_labels.append(1)


# separate gan data
gan_singular_data_features = []
gan_singular_data_label = []

for i in range(total_number_of_samples):
    sample_real_data = generated_data_normalized_singular[sensor_names].iloc[i].values
    # np.append(sample_real_data, i)
    # if np.array(sample_real_data) not in X_train:
    # gan_data.append(sample)
    # sample_real_data[0] = generated_data_normalized['gyro_rad[0]'].iloc[i]
    # sample_real_data[1] = generated_data_normalized['gyro_rad[1]'].iloc[i]
    # sample_real_data[2] = generated_data_normalized['gyro_rad[2]'].iloc[i]
    # sample_real_data[3] = generated_data_normalized['accelerometer_m_s2[0]'].iloc[i]
    # sample_real_data[4] = generated_data_normalized['accelerometer_m_s2[1]'].iloc[i]
    # sample_real_data[5] = generated_data_normalized['accelerometer_m_s2[2]'].iloc[i]
    gan_singular_data_features.append(sample_real_data)
    gan_singular_data_label.append(2)

# for i in range(total_number_of_samples):
#     sample_real_data = benign_data_normalized[sensor_names].iloc[i].values
#     # if np.array(sample_real_data) not in X_train:
#     # gan_data.append(sample)
#     sample_real_data[0] = generated_data_normalized4['gyro_rad[0]'].iloc[i]
#     sample_real_data[1] = generated_data_normalized5['gyro_rad[1]'].iloc[i]
#     sample_real_data[2] = generated_data_normalized6['gyro_rad[2]'].iloc[i]
#     # sample_real_data[3] = generated_data_normalized1['accelerometer_m_s2[0]'].iloc[i]
#     # sample_real_data[4] = generated_data_normalized2['accelerometer_m_s2[1]'].iloc[i]
#     # sample_real_data[5] = generated_data_normalized3['accelerometer_m_s2[2]'].iloc[i]
#     gan_singular_data_features.append(sample_real_data)
#     gan_singular_data_label.append(2)

fig, (ax1, ax2, ax3) = plt.subplots(3)

df2 = benign_data_normalized[sensor_names]

df3 = df2.iloc[100:300]

# Generate example data
# print(len(benign_data_features))
# print(type(benign_data_features[:,0]))
# print(benign_data_features[:,0])

ax1.plot(df3["gyro_rad[0]"], label='gyro x')
ax1.plot(df3["gyro_rad[1]"], label='gyro y')
ax1.plot(df3["gyro_rad[2]"], label='gyro z')
# ax1.plot(df3["accelerometer_m_s2[0]"], label='accel x')
# ax1.plot(df3["accelerometer_m_s2[1]"], label='accel y')
# ax1.plot(df3["accelerometer_m_s2[2]"], label='accel z')
ax1.legend()

df2 = generated_data_normalized_singular[sensor_names]

df3 = df2.iloc[100:300]

# Generate example data
# print(len(benign_data_features))
# print(type(benign_data_features[:,0]))
# print(benign_data_features[:,0])

ax2.plot(df3["gyro_rad[0]"], label='gyro x')
ax2.plot(df3["gyro_rad[1]"], label='gyro y')
ax2.plot(df3["gyro_rad[2]"], label='gyro z')
# ax2.plot(df3["accelerometer_m_s2[0]"], label='accel x')
# ax2.plot(df3["accelerometer_m_s2[1]"], label='accel y')
# ax2.plot(df3["accelerometer_m_s2[2]"], label='accel z')
ax2.legend()

df2 = generated_data_normalized_ensemble[sensor_names]

df3 = df2.iloc[100:300]

# Generate example data
# print(len(benign_data_features))
# print(type(benign_data_features[:,0]))
# print(benign_data_features[:,0])


ax3.plot(df3["gyro_rad[0]"], label='gyro x')
ax3.plot(df3["gyro_rad[1]"], label='gyro y')
ax3.plot(df3["gyro_rad[2]"], label='gyro z')
# ax3.plot(df3["accelerometer_m_s2[0]"], label='accel x')
# ax3.plot(df3["accelerometer_m_s2[1]"], label='accel y')
# ax3.plot(df3["accelerometer_m_s2[2]"], label='accel z')
ax3.legend()

# all_data = np.concatenate((benign_data_features, gan_ensemble_data_features,gan_singular_data_features), axis=0)
# all_labels = np.concatenate((benign_data_labels, gan_ensemble_data_labels, gan_singular_data_label), axis=0)


# plt.legend()
plt.show()
# fig.show()
# plt.savefig("t_SNE_graph.pdf")