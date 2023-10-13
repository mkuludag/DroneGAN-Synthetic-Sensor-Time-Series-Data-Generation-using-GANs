#!/usr/bin/env python3

# Author: Mehmed Kerem Uludag (muludag)
# Email: muludag@umich.edu
# Institution: University of Michigan
# Department: Computer Science & Engineering and Robotics
# Date: July 28, 2023
# Work: This work was started in the ADWISE Lab at Florida International University as part of the Research Expereince for Undergraduates program in the Summer of 2023.
#  
# Description: This file evaluates the GAN generated (G) data by first training a CNN model on benign (B) and spoofed (S) data, and then uses
# different combinations of B, S and G data to test the model acuracies. From Milcom Paper

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, ConvLSTM2D
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import plot_model
from scipy import stats
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

from Class_models import first_model

np.random.seed(123)


shuffleSplit = True # Shuffle data during splitting between Train/Test
shufflefit = True # Shuffle epoches during model run

# p1 = 'sensor-csvfiles/' + 'Accelerometer_real.csv'
# p2 = 'sensor-csvfiles/' + 'Accelerometer_spoof.csv'
# p3 = 'sensor-csvfiles/' + 'Accelerometer_egen.csv'

real_data = pd.read_csv("test_data/benign_data/sensor_combined.csv")
spoofed_data = pd.read_csv("test_data/gps_spoofing_data/sensor_combined.csv")
jammed_data = pd.read_csv("test_data/gps_jamming_data/sensor_combined.csv")

sensor_names = [
    "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]",
    "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"
    ] 


generated_data1 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[0].csv")
generated_data2 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[1].csv")
generated_data3 = pd.read_csv("test_data/latest_gan_data_singular/accelerometer_m_s2[2].csv")

generated_data4 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[0].csv")
generated_data5 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[1].csv")
generated_data6 = pd.read_csv("test_data/latest_gan_data_singular/gyro_rad[2].csv")


window_size = 5
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

stride = window_size
window_data = []
gen_window_data = []

for i in range(0, total_number_of_samples - window_size + 1, stride):
    # window = df[i:i+window_size]
    window = real_data_normalized[sensor_names].iloc[i:i+window_size].values
    X.append(window)
    y.append(0)

# real_window_data = np.array(window_data)

for i in range(0, int(total_number_of_samples/2) - window_size + 1, stride):
    # window = df2[i:i+K]
    window = spoofed_data_normalized[sensor_names].iloc[i:i+window_size].values
    X.append(window)
    y.append(1)  

for i in range(0, int(total_number_of_samples/2) - window_size + 1, stride):
    # window = gen[i:i+K]
    window = jammed_data_normalized[sensor_names].iloc[i:i+window_size].values
    X.append(window)
    y.append(1)

# window_data = np.array(window_data)
# #window_data = window_data[:-1] #matching issues with some datasets, (off by 1)
# labels = np.concatenate((np.zeros(window_data.shape[0] // 2), np.ones(window_data.shape[0] // 2)))
# gen_window_data = np.array(gen_window_data)
# print(window_data.shape)
# print(f.shape)

X = np.array(X)
y = np.array(y)


# step 1 - scale data
scaled_features_gan1 = StandardScaler().fit_transform(generated_data1.values)
gan_data_scaled1 = pd.DataFrame(scaled_features_gan1, index=generated_data1.index, columns=generated_data1.columns)

scaled_features_gan2 = StandardScaler().fit_transform(generated_data2.values)
gan_data_scaled2 = pd.DataFrame(scaled_features_gan2, index=generated_data2.index, columns=generated_data2.columns)

scaled_features_gan3 = StandardScaler().fit_transform(generated_data3.values)
gan_data_scaled3 = pd.DataFrame(scaled_features_gan3, index=generated_data3.index, columns=generated_data3.columns)

scaled_features_gan4 = StandardScaler().fit_transform(generated_data4.values)
gan_data_scaled4 = pd.DataFrame(scaled_features_gan4, index=generated_data4.index, columns=generated_data4.columns)

scaled_features_gan5 = StandardScaler().fit_transform(generated_data5.values)
gan_data_scaled5 = pd.DataFrame(scaled_features_gan5, index=generated_data5.index, columns=generated_data5.columns)

scaled_features_gan6 = StandardScaler().fit_transform(generated_data6.values)
gan_data_scaled6 = pd.DataFrame(scaled_features_gan6, index=generated_data6.index, columns=generated_data6.columns)


generated_data_normalized1 = gan_data_scaled1.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized2 = gan_data_scaled2.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized3 = gan_data_scaled3.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized4 = gan_data_scaled4.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized5 = gan_data_scaled5.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))

generated_data_normalized6 = gan_data_scaled6.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))


gan_data = []
gan_label = []

gan_test_data = []
gan_test_label = []

total_number_of_gan_samples = int(total_number_of_samples*0.3)
if len(generated_data1) < int(total_number_of_samples*0.3):
   total_number_of_gan_samples = len(generated_data1)
print("Total number of GAN Samples")
print(total_number_of_gan_samples)

for i in range(0, total_number_of_gan_samples-window_size + 1, stride):
    sample_real_data = real_data_normalized[sensor_names].iloc[total_number_of_samples+i:total_number_of_samples+i+window_size].values
    # if np.array(sample_real_data) not in X_train:
    # gan_data.append(sample)
    # sample_real_data[0] = generated_data_normalized4['gyro_rad[0]'].iloc[i]
    # sample_real_data[1] = generated_data_normalized5['gyro_rad[1]'].iloc[i]
    # sample_real_data[2] = generated_data_normalized6['gyro_rad[2]'].iloc[i]
    new_sample_data = np.zeros(sample_real_data.shape)
    d = i
    for c, single_item in enumerate(sample_real_data):
        # single_item[0] = generated_data_normalized4['gyro_rad[0]'].iloc[d]
        # single_item[1] = generated_data_normalized5['gyro_rad[1]'].iloc[d]
        # single_item[2] = generated_data_normalized6['gyro_rad[2]'].iloc[d]
        single_item[3] = generated_data_normalized1['accelerometer_m_s2[0]'].iloc[d]
        single_item[4] = generated_data_normalized2['accelerometer_m_s2[1]'].iloc[d]
        # single_item[5] = generated_data_normalized3['accelerometer_m_s2[2]'].iloc[d]
        new_sample_data[c] = single_item
        d += 1


    # sample_real_data[3] = generated_data_normalized1['accelerometer_m_s2[0]'].iloc[i]
    # sample_real_data[4] = generated_data_normalized2['accelerometer_m_s2[1]'].iloc[i]
    # sample_real_data[5] = generated_data_normalized3['accelerometer_m_s2[2]'].iloc[i]
    gan_data.append(new_sample_data)
    gan_test_data.append(sample_real_data)
    gan_label.append(1)
    gan_test_label.append(1)

for i in range(0, total_number_of_gan_samples-window_size + 1, stride):
    sample_real_data = real_data_normalized[sensor_names].iloc[total_number_of_samples + total_number_of_gan_samples+i:total_number_of_samples + total_number_of_gan_samples+i+window_size].values
    gan_test_data.append(sample_real_data)
    gan_test_label.append(0)

gan_data = np.array(gan_data)
gan_label = np.array(gan_label)

gan_test_data = np.array(gan_test_data)
gan_test_label = np.array(gan_test_label)

print(len(gan_data))
print(len(gan_test_label))

print("test")
# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42) # set to 0 the balance out 0 and 1 labels in train/test
#x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=shuffleSplit, random_state=4)
# print(y_test.size)
# print(np.sum(y_test == 1))




# Compile Model
input_shape = (window_size, len(sensor_names))
model = first_model(input_shape)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    x_train, y_train, 
    epochs=20, batch_size=32, 
    validation_data=(x_test, y_test), 
    shuffle=True)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend(['loss', 'val_loss'])
# plt.show()

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.xlabel('Epochs')
# plt.ylabel('MSLE Loss')
# plt.legend(['loss', 'val_loss'])
# plt.show()

print("Regular benchmark test")

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Benchmark accuracy (half benign/half anomaly): {accuracy:.4f}')

predictions = model.predict(x_test)

threshold = 0.2
# anomaly_mask = pd.Series(predictions) > threshold
# predicted_labels = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0)
predicted_labels = (predictions > threshold).astype(int).reshape(y_test.shape)


# print(predicted_labels[:10])
# print(type(predicted_labels))
# print(predicted_labels.shape)

# print(y_test[:10])
# print(type(y_test))
# print(y_test.shape)

acc = accuracy_score(predicted_labels, y_test)
print("Accuracy:")
print(acc)

tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels).ravel()
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

print("GAN and benign data test")


predictions = model.predict(gan_test_data)
predicted_labels = (predictions > threshold).astype(int).reshape(gan_test_label.shape)

acc = accuracy_score(predicted_labels, gan_test_label)
print("Accuracy:")
print(acc)

tn, fp, fn, tp = confusion_matrix(gan_test_label, predicted_labels).ravel()
print(tn)
print(fp)
print(fn)
print(tp)
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


# Threshold: 0.01001314025746261
# threshold = 0.02
predictions = model.predict(gan_data)
predicted_labels = (predictions > threshold).astype(int).reshape(gan_label.shape)

acc = accuracy_score(predicted_labels, gan_label)
print("Accuracy:")
print(acc)

tn, fp, fn, tp = confusion_matrix(gan_label, predicted_labels).ravel()
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

# threshold = 0.5
# predicted_labels = (predictions > threshold).astype(int).reshape(y_test.shape)
# print(predicted_labels)
# cm = confusion_matrix(y_test, predicted_labels)

# TN, FP, FN, TP = cm.ravel()

# TPR = TP / (TP + FN)
# TNR = TN / (TN + FP)
# FPR = FP / (FP + TN)
# FNR = FN / (FN + TP)

# # Calculate ROC AUC score
# roc_auc = roc_auc_score(y_test, predictions)

# # Print the results
# print("True Positive Rate (TPR):", TPR)
# print("True Negative Rate (TNR):", TNR)
# print("False Positive Rate (FPR):", FPR)
# print("False Negative Rate (FNR):", FNR)
# print("ROC AUC:", roc_auc)



# print("number of ones in y_test:")
# print(np.sum(y_test == 1))
# print(y_test.size)
# #pdb.set_trace()

# # Test generated data:

# y_test = np.ones(gen_window_data.shape[0])
# predictions = model.predict(gen_window_data)
# predicted_labels = (predictions > threshold).astype(int).reshape(y_test.shape)
# cm = confusion_matrix(y_test, predicted_labels)
# #pdb.set_trace()
# X = np.min((X, int(gen.shape[0])))
# data = np.concatenate((df[:X, :], gen[:X, :]))

# # test real data for benchmark:
# zeross = np.zeros(real_window_data.shape[0])
# predictions2 = model.predict(real_window_data)
# predicted_labels2 = (predictions2 > threshold).astype(int).reshape(zeross.shape)

# #plotting everything:
# print("Onto plotting the acquired results")

# plt.imshow(cm, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('Actual Labels')
# plt.colorbar()
# plt.xticks([0, 1], ['Real', 'Fake'])
# plt.yticks([0, 1], ['Real', 'Fake'])
# plt.savefig("class_plots/confustionMatrix")
# plt.show()
# plt.clf()
# #pdb.set_trace()
# # Calculate and plot the ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, predictions)
# roc_auc = auc(fpr, tpr)

# plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# #plt.show()
# plt.clf()

# correct_predictions = np.sum(predicted_labels == y_test)
# total_predictions = len(y_test)
# accuracy = correct_predictions / total_predictions
# percentage_correct = accuracy * 100

# correct_predictions2 = np.sum(predicted_labels2 == zeross)
# total_predictions2 = len(zeross)
# accuracy2 = correct_predictions2 / total_predictions2
# percentage_correct2 = accuracy2 * 100


# print(predicted_labels) # if 100% accuracy, then TP => TN for some reason
# print("number of 1s in gen_data", np.sum(predicted_labels == 1))
# print("Percentage of Correct Predictions: {:.2f}%".format(percentage_correct))

# print(predicted_labels2) 
# print("number of 0s in real_data", np.sum(predicted_labels2 == 0))
# print("Percentage of Correct REal Predictions: {:.2f}%".format(percentage_correct2))


# # test both gen_window_data, real_window_data
# print('=====cm for generated and real=====')
# window_data = []
# for i in range(0, len(df2) - K + 1, stride):
#     window = df2[i:i+K]
#     window_data.append(window)
# fake_window_data = np.array(window_data)
# #gen_window_data = fake_window_data
# min_size = min(gen_window_data.shape[0], real_window_data.shape[0])
# gen_window_data2 = gen_window_data[:min_size]
# real_window_data = real_window_data[:min_size]

# gen_labels = np.ones(gen_window_data2.shape[0])
# real_labels = np.zeros(real_window_data.shape[0])

# combined_data = np.concatenate((gen_window_data2, real_window_data), axis=0)
# combined_labels = np.concatenate((gen_labels, real_labels), axis=0)

# combined_data, combined_labels = shuffle(combined_data, combined_labels)

# class_probabilities = model.predict(combined_data)
# class_predictions = (class_probabilities > 0.5).astype(int)
# #pdb.set_trace()
# cm = confusion_matrix(combined_labels, class_predictions)

# false_positive = cm[0, 1]
# total_negative = len(gen_labels)

# total_samples = cm.sum()
# false_positive_percentage = (false_positive / total_negative) * 100

# print(f"False Positive Percentage: {false_positive_percentage:.2f}%")


# correct_predictions = cm[0, 0] + cm[1, 1]

# accuracy = (correct_predictions / total_samples) * 100

# print(f"half B half G or S Accuracy: {accuracy:.2f}%")

# plt.imshow(cm, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('Actual Labels')
# plt.colorbar()
# plt.xticks([0, 1], ['Real', 'Fake'])
# plt.yticks([0, 1], ['Real', 'Fake'])
# plt.savefig("class_plots/confustionMatrix2")
# plt.show()
# plt.clf()


# # all G data
# labels = np.ones(gen_window_data.shape[0])
# predictions = model.predict(gen_window_data)
# class_correct_predictions = (predictions > 0.5).astype(int)
# accuracy = accuracy_score(labels, class_correct_predictions)

# # Print the accuracy
# print("All G Accuracy:", accuracy)
# cm = confusion_matrix(labels, class_correct_predictions)

# TN, FP, FN, TP = cm.ravel()

# # Calculate True Positive Rate (TPR), True Negative Rate (TNR), False Positive Rate (FPR), and False Negative Rate (FNR)
# TPR = TP / (TP + FN)
# TNR = TN / (TN + FP)
# FPR = FP / (FP + TN)
# FNR = FN / (FN + TP)

# # Calculate ROC AUC score
# #roc_auc = roc_auc_score(labels, class_correct_predictions)

# # Print the results
# print("True Positive Rate (TPR):", TPR)
# print("True Negative Rate (TNR):", TNR)
# print("False Positive Rate (FPR):", FPR)
# print("False Negative Rate (FNR):", FNR)
# #print("ROC AUC:", roc_auc)


# # Create a single figure for all graphs
# M = data.shape[1]
# fig, axs = plt.subplots(1, M, figsize=(10, 5)) 

# for i in range(M): # Iterate over each column and create a dot graph
#     # Get the data for the current column
#     column_data = data[:, i]

#     # Divide the data into two halves
#     half = len(column_data) // 2
#     first_half = column_data[:half]
#     second_half = column_data[half:]

#     # 1st half
#     if data_dim != 1:
#         axs[i].plot(range(half), first_half, 'ro', label='Real Data')

#         # 2nd 
#         axs[i].plot(range(half, len(column_data)), second_half, 'bo', label='Fake Data')

#         axs[i].set_title(f'Column {i+1}') 
#         axs[i].legend() 
#     else: # Axes object becomes unsubscriptable if there is only 1 column/graph
#         # 1st half
#         axs.plot(range(half), first_half, 'ro', label='Real Data')

#         # 2nd 
#         axs.plot(range(half, len(column_data)), second_half, 'bo', label='Fake Data')

#         axs.set_title(f'Column {i+1}') 
#         axs.legend() 

# plt.tight_layout()
# plt.savefig("class_plots/all_data_w_dims")
# plt.show()


# plt.clf()

# #pdb.set_trace()


# # Calculate the moving average using a window size 
# window_size = 10
# moving_avg = np.convolve(column_data, np.ones(window_size)/window_size, mode='same')

# # Calculate z-scores 
# z_scores = stats.zscore(column_data)

# # Set threshold to determine outliers
# z_score_threshold = 2.5
# outliers = np.where(np.abs(z_scores) > z_score_threshold)[0]

# # Plot original data and moving average
# plt.plot(column_data, label='Original Data')
# plt.plot(moving_avg, label='Moving Average')

# # Highlight outliers
# plt.scatter(outliers, column_data[outliers], color='red', label='Outliers')

# # Add a red line to represent the separation between real and fake data
# separation_line = len(column_data) // 2
# plt.axvline(x=separation_line, color='red', linestyle='--', label='Real/Spoof Separator')


# plt.xlabel('Data Point')
# plt.ylabel('Value')
# plt.legend()
# plt.savefig("class_plots/trends_in_data")
# plt.show()


# plt.clf()