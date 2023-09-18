import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pdb
import pandas as pd

real_data = pd.read_csv("sensor-csvfiles/Accelerometer_real.csv")
generated_data = pd.read_csv("sensor-csvfiles/Accelerometer_gen.csv")

sensor_names = [
    #"gyro_rad[0]", "gyro_rad[2]",  "xyz[0]"
    "accelerometer_m_s2[0]",  "accelerometer_m_s2[1]",  "accelerometer_m_s2[2]",
    # 
    ] #"hover_thrust",

window_size = 3
num_samples = 400

# Create empty lists to store the windowed data
X = []
y = []

# Iterate through the real_data to create windows
for i in range(num_samples):
    window = real_data[sensor_names].iloc[i:i+window_size].values
    target = real_data[sensor_names].iloc[i+window_size].values
    X.append(window)
    y.append(target)

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

print("test")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)


# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(window_size, len(sensor_names))), #why len sensor_names???, requires window size to match up matrix
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
print("training is commensing:" )
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])



# NOTE: evaluation
# Evaluate the model and detect anomalies

predictions = model.predict(X_val)

mean_values = np.mean(X_train, axis=(0, 1))
std_values = np.std(X_train, axis=(0, 1))

anomaly_threshold = 3*std_values
top_border = mean_values + 3*std_values
bottom_border = mean_values - 3*std_values

#top_border[2] = 1
#pdb.set_trace()

for p in predictions:
    anomalies = (p > top_border) | (p < bottom_border)
    if np.any(anomalies):
        print("Anomaly detected")
    else:
        print("No anomaly detected")



# test Generated: 

X_test = []

# Iterate through the generated_data to create test windows
for i in range(num_samples):
    window = generated_data[sensor_names].iloc[i:i+window_size].values
    X_test.append(window)

# Convert the list to a numpy array
X_test = np.array(X_test)

# Use the trained model to make predictions
predictions = model.predict(X_test)

#evaluations