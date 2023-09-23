from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, roc_auc_score
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
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(16, activation='relu'),
      Dropout(0.1),
    #   Dense(8, activation='relu'),
    #   Dropout(0.1),
      Dense(code_size, activation='relu')
    ])
    self.decoder = Sequential([
    #   Dense(8, activation='relu'),
    #   Dropout(0.1),
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
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
  # 1 = anomaly, 0 = normal
#   preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
  preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0)
  return preds



# real_data = pd.read_csv("sensor-csvfiles/Accelerometer_real.csv")
# generated_data = pd.read_csv("sensor-csvfiles/Accelerometer_gen.csv")
# spoofed_data = pd.read_csv("sensor-csvfiles/Accelerometer_spoof.csv")

real_data = pd.read_csv("test_data/benign_data/sensor_combined.csv")
# generated_data1 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[0].csv")
# generated_data2 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[1].csv")
# generated_data3 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[2].csv")

# generated_data4 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[0].csv")
# generated_data5 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[1].csv")
# generated_data6 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[2].csv")

spoofed_data = pd.read_csv("test_data/gps_spoofing_data/sensor_combined.csv")
jammed_data = pd.read_csv("test_data/gps_jamming_data/sensor_combined.csv")


generated_data = pd.read_csv("test_data/ensemble_gan_data/Egan_['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'].csv")

sensor_names = [
    #"gyro_rad[0]", "gyro_rad[2]",  "xyz[0]"
    # "accelerometer_m_s2[0]",  "accelerometer_m_s2[1]",  "accelerometer_m_s2[2]",
    # 
    # "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z", "gps_x", "gps_y", "gps_z", "mag_x", "mag_y", "mag_z"
    "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]",
    "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"

    ] #"hover_thrust",

gan_sensor_names1 = [
   "accelerometer_m_s2[0]"
   ]

gan_sensor_names2 = [
   "accelerometer_m_s2[1]"
   ]

gan_sensor_names3 = [
   "accelerometer_m_s2[2]"
   ]

gan_sensor_names4 = [
   "gyro_rad[0]"
   ]

gan_sensor_names5 = [
   "gyro_rad[1]"
   ]

gan_sensor_names6 = [
   "gyro_rad[2]"
   ]

window_size = 16
# num_samples = 400
# total_number_of_samples = len(real_data)
total_number_of_samples = 20000
print("Total number of Actual Samples")
print(total_number_of_samples)


# step 1 - scale data
scaled_features = StandardScaler().fit_transform(real_data.values)
real_data_scaled = pd.DataFrame(scaled_features, index=real_data.index, columns=real_data.columns)
scaled_features_spoofed = StandardScaler().fit_transform(spoofed_data.values)
spoofed_data_scaled = pd.DataFrame(scaled_features_spoofed, index=spoofed_data.index, columns=spoofed_data.columns)
scaled_features_jammed = StandardScaler().fit_transform(jammed_data.values)
jammed_data_scaled = pd.DataFrame(scaled_features_jammed, index=jammed_data.index, columns=jammed_data.columns)

# step 2 - normalize between 0 and 1
real_data_normalized = real_data_scaled.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

# generated_data_normalized = generated_data.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

spoofed_data_normalized = spoofed_data_scaled.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

jammed_data_normalized = jammed_data_scaled.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))


X = []
y = []

for i in range(total_number_of_samples):
    sample = real_data_normalized[sensor_names].iloc[i].values
    # label = real_data_normalized[sensor_names].iloc[i+window_size].values
    X.append(sample)
    y.append(0)


for i in range(int(total_number_of_samples/2)):
    sample = spoofed_data_normalized[sensor_names].iloc[i].values
    # sample = spoofed_data_normalized[sensor_names].iloc[i:i+window_size].values
    # sppofed_data_test.append(sample)
    X.append(sample)
    y.append(1)

for i in range(int(total_number_of_samples/2)):
    sample = jammed_data_normalized[sensor_names].iloc[i].values
    # sample = spoofed_data_normalized[sensor_names].iloc[i:i+window_size].values
    # sppofed_data_test.append(sample)
    X.append(sample)
    y.append(1)

X = np.array(X)
y = np.array(y)

print("test")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)


print(X_train.shape)

total_number_of_gan_samples = int(total_number_of_samples*0.3)
if len(generated_data) < int(total_number_of_samples*0.3):
   total_number_of_gan_samples = len(generated_data)
print("Total number of GAN Samples")
print(total_number_of_gan_samples)

# step 1 - scale data
scaled_features_gan = StandardScaler().fit_transform(generated_data.values)
gan_data_scaled = pd.DataFrame(scaled_features_gan, index=generated_data.index, columns=generated_data.columns)

# step 3 - normalize data
generated_data_normalized = gan_data_scaled.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))


gan_data = []
gan_label = []

gan_test_data = []
gan_test_label = []

for i in range(total_number_of_gan_samples):
    sample_real_data = real_data_normalized[sensor_names].iloc[total_number_of_samples+i].values
    # if np.array(sample_real_data) not in X_train:
    # gan_data.append(sample)
    # sample_real_data[0] = generated_data_normalized['gyro_rad[0]'].iloc[i]
    # sample_real_data[1] = generated_data_normalized['gyro_rad[1]'].iloc[i]
    # sample_real_data[2] = generated_data_normalized['gyro_rad[2]'].iloc[i]
    sample_real_data[3] = generated_data_normalized['accelerometer_m_s2[0]'].iloc[i]
    sample_real_data[4] = generated_data_normalized['accelerometer_m_s2[1]'].iloc[i]
    sample_real_data[5] = generated_data_normalized['accelerometer_m_s2[2]'].iloc[i]
    gan_data.append(sample_real_data)
    gan_test_data.append(sample_real_data)
    gan_label.append(1)
    gan_test_label.append(1)



for i in range(total_number_of_gan_samples):
    sample_real_data = real_data_normalized[sensor_names].iloc[total_number_of_samples + total_number_of_gan_samples+i].values
    gan_test_data.append(sample_real_data)
    gan_test_label.append(0)


# gan_data['accelerometer_m_s2[0]'] = generated_data_normalized['accelerometer_m_s2[0]']
# # gan_data = np.array(gan_data)
gan_data = np.array(gan_data)
gan_label = np.array(gan_label)


gan_test_data = np.array(gan_test_data)
gan_test_label = np.array(gan_test_label)

print(len(gan_data))
print(len(gan_test_label))

model = AutoEncoder(output_units=X_train.shape[1])
# configurations of model
model.compile(loss='msle', metrics=['mse'], optimizer='adam')

# model.build()
# model.summary()

# use case is novelty detection so use only the normal data
# for training
# train_index = y_train[y_train == 1].index
train_index = np.where(y_train == 0)
# train_data = X_train.loc[train_index]
train_data = X_train[train_index]

history = model.fit(
    train_data,
    train_data,
    epochs=50,
    batch_size=128,
    validation_data=(X_val, X_val)
)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('Epochs')
# plt.ylabel('MSLE Loss')
# plt.legend(['loss', 'val_loss'])
# plt.show()

print("Regular benchmark test")
threshold = find_threshold(model, train_data)
print(f"Threshold: {threshold}")
# Threshold: 0.01001314025746261
# threshold = 0.02
predictions = get_predictions(model, X_val, threshold)
acc = accuracy_score(predictions, y_val)
print("Accuracy:")
print(acc)

tn, fp, fn, tp = confusion_matrix(y_val, predictions).ravel()
false_positive_rate = fp / (fp + tn)
print("FPR:")
print(false_positive_rate)

true_positive_rate = tp / (tp + fn)
print("TPR:")
print(true_positive_rate)

false_negative_rate = fn / (tp + fn)
print("FNR:")
print(false_negative_rate)

true_negative_rate = tn / (tn + fp)
print("TNR or specificity:")
print(false_negative_rate)

# roc_auc = roc_auc_score(gan_label, predictions, multi_class='ovo')
# print("ROC AUC:")
# print(roc_auc)

print("GAN and benign data test")

threshold = find_threshold(model, train_data)
print(f"Threshold: {threshold}")
# Threshold: 0.01001314025746261
# threshold = 0.02
predictions = get_predictions(model, gan_test_data, threshold)
acc = accuracy_score(predictions, gan_test_label)
print("Accuracy:")
print(acc)

tn, fp, fn, tp = confusion_matrix(gan_test_label, predictions).ravel()
false_positive_rate = fp / (fp + tn)
print("FPR:")
print(false_positive_rate)

true_positive_rate = tp / (tp + fn)
print("TPR:")
print(true_positive_rate)

false_negative_rate = fn / (tp + fn)
print("FNR:")
print(false_negative_rate)

true_negative_rate = tn / (tn + fp)
print("TNR or specificity:")
print(false_negative_rate)

print("GAN only data test")

threshold = find_threshold(model, train_data)
print(f"Threshold: {threshold}")
# Threshold: 0.01001314025746261
# threshold = 0.02
predictions = get_predictions(model, gan_data, threshold)
acc = accuracy_score(predictions, gan_label)
print("Accuracy:")
print(acc)

tn, fp, fn, tp = confusion_matrix(gan_label, predictions).ravel()
false_positive_rate = fp / (fp + tn)
print("FPR:")
print(false_positive_rate)

true_positive_rate = tp / (tp + fn)
print("TPR:")
print(true_positive_rate)

false_negative_rate = fn / (tp + fn)
print("FNR:")
print(false_negative_rate)

true_negative_rate = tn / (tn + fp)
print("TNR or specificity:")
print(false_negative_rate)

# roc_auc = roc_auc_score(gan_label, predictions, multi_class='ovo')
# print("ROC AUC:")
# print(roc_auc)
