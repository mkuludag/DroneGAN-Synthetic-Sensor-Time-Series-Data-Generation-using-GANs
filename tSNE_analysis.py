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

total_number_of_samples = 2000


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
    # label = real_data_normalized[sensor_names].iloc[i+window_size].values
    benign_data_features.append(sample)
    benign_data_labels.append(0)

# ensemble gan data
gan_ensemble_data_features = []
gan_ensemble_data_labels = []

for i in range(total_number_of_samples):
    sample_real_data = generated_data_normalized_ensemble[sensor_names].iloc[i].values
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

# Generate example data

all_data = np.concatenate((benign_data_features, gan_ensemble_data_features,gan_singular_data_features), axis=0)
all_labels = np.concatenate((benign_data_labels, gan_ensemble_data_labels, gan_singular_data_label), axis=0)


# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=3, random_state=0)
data_2D = tsne.fit_transform(all_data)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data_2D[all_labels == 0, 0], data_2D[all_labels == 0, 1], label='Benign', c='b')
plt.scatter(data_2D[all_labels == 1, 0], data_2D[all_labels == 1, 1], label='EDSD-GAN', c='r')
plt.scatter(data_2D[all_labels == 2, 0], data_2D[all_labels == 2, 1], label='DSD-GAN', c='g')

# plt.scatter(data_2D[all_labels == 0, 0], data_2D[all_labels == 0, 1], label='Benign', c='b')
# plt.scatter(data_2D[all_labels == 1, 0], data_2D[all_labels == 1, 1], label='Fake', c='r')
# plt.title('t-SNE Plot for Benign vs Fake Data')
# plt.legend()
#

# Add axis labels
plt.xlabel("x-tSNE ")
plt.ylabel("y-tSNE")

# plt.title('t-SNE Plot with Axes and Labels for Data with Three Categories')
plt.legend()
# plt.show()
plt.savefig("t_SNE_graph.pdf")