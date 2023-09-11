import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pdb
import pandas as pd

real_data = pd.read_csv("sensor-csvfiles/combined_sensors_real.csv")
generated_data = pd.read_csv("sensor-csvfiles/combined_sensors_generated.csv")

sensor_names = ["gyro_rad[0]", "gyro_rad[2]",  "xyz[0]"] #"hover_thrust",

# Create windowed data 
window_size = 5
num_samples = 1000
num_samples_real = num_samples // 2
real_windowed_data = []
for i in range(num_samples_real):
    window = real_data[sensor_names].iloc[i:i+window_size].values
    real_windowed_data.append(window)
real_windowed_data = np.array(real_windowed_data)

num_samples_generated = num_samples // 2
generated_windowed_data = []
for i in range(num_samples_generated):
    window = generated_data[sensor_names].iloc[i:i+window_size].values
    generated_windowed_data.append(window)
generated_windowed_data = np.array(generated_windowed_data)


labels_real = np.zeros((num_samples_real, 1))
labels_generated = np.ones((num_samples_generated, 1))

# Combine windowed data and labels
combined_data = np.concatenate([real_windowed_data, generated_windowed_data], axis=0)
combined_labels = np.concatenate([labels_real, labels_generated], axis=0)

# Split the data and labels into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(combined_data, combined_labels, test_size=0.2, shuffle=False, random_state=42)


# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(window_size, len(sensor_names))),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Modified output layer
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001)

# Train the model
epochs = 2000
batch_size = 32

# y_train = y_train.reshape(-1, 1, len(sensor_names))
# y_val = y_val.reshape(-1, 1, len(sensor_names))

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])



# NOTE: evaluation
# Evaluate the model and detect anomalies

predictions = model.predict(X_val)

mean_values = np.mean(X_train, axis=(0, 1))
std_values = np.std(X_train, axis=(0, 1))

anomaly_threshold = 3*std_values
top_border = mean_values + 3*std_values
bottom_border = mean_values - 3*std_values

# Detect anomalies based on predictions and borders
for p in predictions:
    anomalies = np.logical_or(p < bottom_border, p > top_border)
    if np.any(anomalies):
        print("Anomaly detected")
    else:
        print("No anomaly detected")


print(predictions.shape)
pdb.set_trace()
