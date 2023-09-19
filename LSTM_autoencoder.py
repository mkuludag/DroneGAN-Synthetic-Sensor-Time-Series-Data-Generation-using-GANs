from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class AutoEncoder(Model):
  """
  Parameters
  ----------
  output_units: int
    Number of output units
  
  code_size: int
    Number of units in bottle neck
  """

  def __init__(self, output_units, code_size=8):
    super().__init__()
    self.encoder = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(code_size, activation='relu')
    ])
    self.decoder = Sequential([
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(output_units, activation='sigmoid')
    ])
  
  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded
  
def find_threshold(model, x_train_scaled):
  reconstructions = model.predict(x_train_scaled)
  # provides losses of individual instances
  reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
  # threshold for anomaly scores
  threshold = np.mean(reconstruction_errors.numpy()) \
      + np.std(reconstruction_errors.numpy())
  return threshold

def get_predictions(model, x_test_scaled, threshold):
  predictions = model.predict(x_test_scaled)
  # provides losses of individual instances
  errors = tf.keras.losses.msle(predictions, x_test_scaled)
  # 0 = anomaly, 1 = normal
  anomaly_mask = pd.Series(errors) > threshold
  preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
  return preds



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
total_number_of_samples = 5000
print(total_number_of_samples)


# step 1 - scale data
scaled_features = StandardScaler().fit_transform(real_data.values)
real_data_scaled = pd.DataFrame(scaled_features, index=real_data.index, columns=real_data.columns)

# step 2 - normalize between 0 and 1
real_data_normalized = real_data_scaled.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized = generated_data.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

spoofed_data_normalized = spoofed_data.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

X = []
y = []

for i in range(total_number_of_samples):
    if i+window_size < total_number_of_samples:
        # sample = real_data_normalized[sensor_names].iloc[i:i+window_size].values
        sample = real_data_normalized[sensor_names].iloc[i].values
        # label = real_data_normalized[sensor_names].iloc[i+window_size].values
        X.append(sample)
        y.append(0)


# print(len(X))
# print(len(y))

# gan_data = []

# for i in range(len(generated_data_normalized)):
#     if i+window_size < len(generated_data_normalized):
#         sample = generated_data_normalized[sensor_names].iloc[i:i+window_size].values
#         gan_data.append(sample)

# gan_data = np.array(gan_data)

spoofed_data_test = []

for i in range(total_number_of_samples):
    if i+window_size < total_number_of_samples:
        sample = spoofed_data_normalized[sensor_names].iloc[i].values
        # sample = spoofed_data_normalized[sensor_names].iloc[i:i+window_size].values
        # sppofed_data_test.append(sample)
        X.append(sample)
        y.append(1)

# sppofed_data_test = np.array(spoofed_data_test)

X = np.array(X)
y = np.array(y)

print("test")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)


print(X_train.shape)

model = AutoEncoder(output_units=X_train.shape[1])
# configurations of model
model.compile(loss='msle', metrics=['mse'], optimizer='adam')

# model.build()
# model.summary()

# use case is novelty detection so use only the normal data
# for training
# train_index = y_train[y_train == 1].index
train_index = np.where[y_train == 0]
train_data = X_train.loc[train_index]

history = model.fit(
    X_train,
    X_train,
    epochs=30,
    batch_size=256,
    validation_data=(X_val, X_val)
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('MSLE Loss')
plt.legend(['loss', 'val_loss'])
plt.show()

threshold = find_threshold(model, X_train)
print(f"Threshold: {threshold}")
# Threshold: 0.01001314025746261
predictions = get_predictions(model, X_val, threshold)
acc = accuracy_score(predictions, y_val)
print(acc)