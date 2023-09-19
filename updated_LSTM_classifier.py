import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pdb
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr
from itertools import combinations


real_data = pd.read_csv("sensor-csvfiles/Accelerometer_real.csv")
generated_data = pd.read_csv("sensor-csvfiles/Accelerometer_gen.csv")
spoofed_data = pd.read_csv("sensor-csvfiles/Accelerometer_spoof.csv")

sensor_names = [
    #"gyro_rad[0]", "gyro_rad[2]",  "xyz[0]"
    "accelerometer_m_s2[0]",  "accelerometer_m_s2[1]",  "accelerometer_m_s2[2]",
    # 
    ] #"hover_thrust",

window_size = 16
# num_samples = 400
# total_number_of_samples = len(real_data)
total_number_of_samples = 20000
print(total_number_of_samples)

# # Create empty lists to store the windowed data
# X = []
# y = []


# normalizing values between 0 and 1, based on SD and average

# step 1 - scale data
scaled_features = StandardScaler().fit_transform(real_data.values)
real_data_scaled = pd.DataFrame(scaled_features, index=real_data.index, columns=real_data.columns)

# step 2 - normalize between 0 and 1
real_data_normalized = real_data_scaled.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized = generated_data.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

spoofed_data_normalized = spoofed_data.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# print(real_data_normalized.iloc[0])

# Create list of samples and labels based on window size 
X = []
y = []

for i in range(total_number_of_samples):
    if i+window_size < total_number_of_samples:
        sample = real_data_normalized[sensor_names].iloc[i:i+window_size].values
        label = real_data_normalized[sensor_names].iloc[i+window_size].values
        X.append(sample)
        y.append(label)


print(len(X))
print(len(y))

gan_data = []

for i in range(len(generated_data_normalized)):
    if i+window_size < len(generated_data_normalized):
        sample = generated_data_normalized[sensor_names].iloc[i:i+window_size].values
        gan_data.append(sample)

gan_data = np.array(gan_data)

sppofed_data_test = []

for i in range(len(spoofed_data_normalized)):
    if i+window_size < len(spoofed_data_normalized):
        sample = spoofed_data_normalized[sensor_names].iloc[i:i+window_size].values
        sppofed_data_test.append(sample)

sppofed_data_test = np.array(sppofed_data_test)
# create list of labels for each sample
# for i in range(total_number_of_samples)

# # Iterate through the real_data to create windows
# for i in range(num_samples):
#     window = real_data[sensor_names].iloc[i:i+window_size].values
#     target = real_data[sensor_names].iloc[i+window_size].values
#     X.append(window)
#     y.append(target)

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

print("test")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)


# # Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(window_size, len(sensor_names))), #why len sensor_names???, requires window size to match up matrix
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(128, return_sequences=False),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='linear')  # Modified output layer
])

model.summary()

# Compile the model
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001)

# # Train the model
epochs = 1
batch_size = 16

print(X_train.shape)
print(y_train.shape)

# y_train = y_train.reshape(-1, 1, len(sensor_names))
# y_val = y_val.reshape(-1, 1, len(sensor_names))

# print(X_train.shape)
# print(y_train.shape)
print("training is commensing:" )
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])

predictions = model.predict(X_val)
print(predictions.shape)

# # model.save('my_model.keras')



# # # NOTE: evaluation
# # # Evaluate the model and detect anomalies

# predictions = model.predict(X_val)

mean_values = np.mean(X_train, axis=(0, 1))

mean_values = np.mean(X_train, axis=(0, 1))
std_values = np.std(X_train, axis=(0, 1))

anomaly_threshold = 3*std_values
top_border = mean_values + 3*std_values
bottom_border = mean_values - 3*std_values

# print(mean_values)
# print(std_values)
# print(anomaly_threshold)
# print(top_border)
# print(bottom_border)

# # print(top_border)
# # print(bottom_border)
# # # #top_border[2] = 1
# # # #pdb.set_trace()

total_anomalies = 0
t_anomalies_x_axis = 0
t_anomalies_y_axis = 0
t_anomalies_z_axis = 0
total_normal = 0

for p in predictions:
    anomalies = (p > top_border) | (p < bottom_border)
    if np.any(anomalies):
        if (p[0] > top_border[0]) | (p[0] < bottom_border[0]):
            t_anomalies_x_axis += 1
        if (p[1] > top_border[1]) | (p[1] < bottom_border[1]):
            t_anomalies_y_axis += 1
        if (p[2] > top_border[2]) | (p[2] < bottom_border[2]):
            t_anomalies_z_axis += 1
        
        # print("Anomaly detected")
        total_anomalies += 1
    else:
        # print("No anomaly detected")
        total_normal += 1

print(total_anomalies)
print(total_normal)
print("total_anomalies in x")
print(t_anomalies_x_axis)
print("total_anomalies in y")
print(t_anomalies_y_axis)
print("total_anomalies in z")
print(t_anomalies_z_axis)

# Test of Spoofed DAta
print("Testing on Spoofed data")

predictions_spoofed = model.predict(sppofed_data_test)
print(predictions_spoofed.shape)

# # model.save('my_model.keras')



# # # NOTE: evaluation
# # # Evaluate the model and detect anomalies

# predictions = model.predict(X_val)

mean_values = np.mean(X_train, axis=(0, 1))

mean_values = np.mean(X_train, axis=(0, 1))
std_values = np.std(X_train, axis=(0, 1))

anomaly_threshold = 3*std_values
top_border = mean_values + 3*std_values
bottom_border = mean_values - 3*std_values

# print(mean_values)
# print(std_values)
# print(anomaly_threshold)
# print(top_border)
# print(bottom_border)

# # print(top_border)
# # print(bottom_border)
# # # #top_border[2] = 1
# # # #pdb.set_trace()

total_anomalies = 0
t_anomalies_x_axis = 0
t_anomalies_y_axis = 0
t_anomalies_z_axis = 0
total_normal = 0

for p in predictions_spoofed:
    anomalies = (p > top_border) | (p < bottom_border)
    if np.any(anomalies):
        if (p[0] > top_border[0]) | (p[0] < bottom_border[0]):
            t_anomalies_x_axis += 1
        if (p[1] > top_border[1]) | (p[1] < bottom_border[1]):
            t_anomalies_y_axis += 1
        if (p[2] > top_border[2]) | (p[2] < bottom_border[2]):
            t_anomalies_z_axis += 1
        
        # print("Anomaly detected")
        total_anomalies += 1
    else:
        # print("No anomaly detected")
        total_normal += 1

print(total_anomalies)
print(total_normal)
print("total_anomalies in x")
print(t_anomalies_x_axis)
print("total_anomalies in y")
print(t_anomalies_y_axis)
print("total_anomalies in z")
print(t_anomalies_z_axis)


# Test of GAN DAta
print("Testing on GAN data")

predictions_gan = model.predict(gan_data)
print(predictions_gan.shape)

# # model.save('my_model.keras')



# # # NOTE: evaluation
# # # Evaluate the model and detect anomalies

# predictions = model.predict(X_val)

mean_values = np.mean(X_train, axis=(0, 1))

mean_values = np.mean(X_train, axis=(0, 1))
std_values = np.std(X_train, axis=(0, 1))

anomaly_threshold = 3*std_values
top_border = mean_values + 3*std_values
bottom_border = mean_values - 3*std_values

# print(mean_values)
# print(std_values)
# print(anomaly_threshold)
# print(top_border)
# print(bottom_border)

# # print(top_border)
# # print(bottom_border)
# # # #top_border[2] = 1
# # # #pdb.set_trace()

total_anomalies = 0
t_anomalies_x_axis = 0
t_anomalies_y_axis = 0
t_anomalies_z_axis = 0
total_normal = 0

for p in predictions_gan:
    anomalies = (p > top_border) | (p < bottom_border)
    if np.any(anomalies):
        if (p[0] > top_border[0]) | (p[0] < bottom_border[0]):
            t_anomalies_x_axis += 1
        if (p[1] > top_border[1]) | (p[1] < bottom_border[1]):
            t_anomalies_y_axis += 1
        if (p[2] > top_border[2]) | (p[2] < bottom_border[2]):
            t_anomalies_z_axis += 1
        
        # print("Anomaly detected")
        total_anomalies += 1
    else:
        # print("No anomaly detected")
        total_normal += 1

print(total_anomalies)
print(total_normal)
print("total_anomalies in x")
print(t_anomalies_x_axis)
print("total_anomalies in y")
print(t_anomalies_y_axis)
print("total_anomalies in z")
print(t_anomalies_z_axis)


# reshaped_train = X_train.transpose(2,0,1).reshape(3,-1)


# # calculate calculations
# first_axis = y_train[:, 0]
# second_axis = y_train[:, 1]
# third_axis = y_train[:, 2]

# print(y_train[0])
# print(y_train[len(y_train-1)])

# print(first_axis[0])
# print(first_axis[len(first_axis-1)])

# print(second_axis[0])
# print(second_axis[len(second_axis-1)])

# print(third_axis[0])
# print(third_axis[len(third_axis-1)])
# list_of_vars = [y_train[:, 0]]
# correlations_results = [pearsonr(*pair) for pair in combinations(list_of_vars, 2)]

# for anomalous_samples in 

# correlation_threshold = 0.1**2
# # # test Generated: 

# # X_test = []

# # # Iterate through the generated_data to create test windows
# # for i in range(num_samples):
# #     window = generated_data[sensor_names].iloc[i:i+window_size].values
# #     X_test.append(window)

# # # Convert the list to a numpy array
# # X_test = np.array(X_test)

# # # Use the trained model to make predictions
# # predictions = model.predict(X_test)

# # #evaluations