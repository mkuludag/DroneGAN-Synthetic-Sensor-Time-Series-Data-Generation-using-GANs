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

from Class_models import first_model

np.random.seed(123)


N = 4000
K = 50

shuffleSplit = True # Shuffle data during splitting between Train/Test
shufflefit = True # Shuffle epoches during model run

p1 = 'sensor-csvfiles/' + 'Accelerometer_real.csv'
p2 = 'sensor-csvfiles/' + 'Accelerometer_spoof.csv'
p3 = 'sensor-csvfiles/' + 'Accelerometer_egen.csv'


df = pd.read_csv(p1) # Benign
df2 = pd.read_csv(p2) # Spoofed
gen = pd.read_csv(p3) # Generated

sensor_values = [
    "accelerometer_m_s2[0]",  "accelerometer_m_s2[1]",  "accelerometer_m_s2[2]",
    #"gyro_rad[0]", "hover_thrust",  "xyz[0]",
    ]
data_dim = len(sensor_values)

df = df[sensor_values]
df2 = df2[sensor_values]
gen = gen[sensor_values]

#X = int(N/2 - K/2)
#X2 = int(N/2 + K/2)
X = int(N/2)
df = df.values #np.concatenate((df.values, df.values, df.values), axis=0)
df2 = df2.values #np.concatenate((df2.values, df2.values, df2.values), axis=0)
gen = gen.values
df = df[0:X, :]
df2 = df2[X:N, :]
gen = gen[:X, :] # evaluate all columns
#gen = np.column_stack((gen[:, 0], df[:, -2:])) # just first col
#gen = np.column_stack((df[:, 0], gen[:, 1], df[:, 2])) # just middle col

#pdb.set_trace()
#df2 = np.arange(0, X * 0.06, 0.01).reshape(X, data_dim)
#df2 = np.random.randint(1,50, size=(X,data_dim)) * 0.001

data = np.concatenate((df, df2), axis=0)
#labels = np.concatenate((np.zeros((X)), np.ones((X2))))

stride = 1
window_data = []
gen_window_data = []

for i in range(0, len(df) - K + 1, stride):
    window = df[i:i+K]
    window_data.append(window)
real_window_data = np.array(window_data)
for i in range(0, len(df2) - K + 1, stride):
    window = df2[i:i+K]
    window_data.append(window)    

for i in range(0, len(gen) - K + 1, stride):
    window = gen[i:i+K]
    gen_window_data.append(window)

window_data = np.array(window_data)
#window_data = window_data[:-1] #matching issues with some datasets, (off by 1)
labels = np.concatenate((np.zeros(window_data.shape[0] // 2), np.ones(window_data.shape[0] // 2)))
gen_window_data = np.array(gen_window_data)
print(window_data.shape)
print(labels.shape)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(window_data, labels, test_size=0.2, shuffle=shuffleSplit, random_state=0, stratify=labels) # set to 0 the balance out 0 and 1 labels in train/test
#x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=shuffleSplit, random_state=4)
print(y_test.size)
print(np.sum(y_test == 1))




# Compile Model
input_shape = (K, data_dim)
model = first_model(input_shape)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), shuffle=shufflefit)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Benchmark accuracy (half benign/half spoofed): {accuracy:.4f}')

predictions = model.predict(x_test)
threshold = 0.5
predicted_labels = (predictions > threshold).astype(int).reshape(y_test.shape)
print(predicted_labels)
cm = confusion_matrix(y_test, predicted_labels)

TN, FP, FN, TP = cm.ravel()

TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, predictions)

# Print the results
print("True Positive Rate (TPR):", TPR)
print("True Negative Rate (TNR):", TNR)
print("False Positive Rate (FPR):", FPR)
print("False Negative Rate (FNR):", FNR)
print("ROC AUC:", roc_auc)



print("number of ones in y_test:")
print(np.sum(y_test == 1))
print(y_test.size)
#pdb.set_trace()

# Test generated data:

y_test = np.ones(gen_window_data.shape[0])
predictions = model.predict(gen_window_data)
predicted_labels = (predictions > threshold).astype(int).reshape(y_test.shape)
cm = confusion_matrix(y_test, predicted_labels)
#pdb.set_trace()
X = np.min((X, int(gen.shape[0])))
data = np.concatenate((df[:X, :], gen[:X, :]))

# test real data for benchmark:
zeross = np.zeros(real_window_data.shape[0])
predictions2 = model.predict(real_window_data)
predicted_labels2 = (predictions2 > threshold).astype(int).reshape(zeross.shape)

#plotting everything:
print("Onto plotting the acquired results")

plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.colorbar()
plt.xticks([0, 1], ['Real', 'Fake'])
plt.yticks([0, 1], ['Real', 'Fake'])
plt.savefig("class_plots/confustionMatrix")
plt.show()
plt.clf()
#pdb.set_trace()
# Calculate and plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
#plt.show()
plt.clf()

correct_predictions = np.sum(predicted_labels == y_test)
total_predictions = len(y_test)
accuracy = correct_predictions / total_predictions
percentage_correct = accuracy * 100

correct_predictions2 = np.sum(predicted_labels2 == zeross)
total_predictions2 = len(zeross)
accuracy2 = correct_predictions2 / total_predictions2
percentage_correct2 = accuracy2 * 100


print(predicted_labels) # if 100% accuracy, then TP => TN for some reason
print("number of 1s in gen_data", np.sum(predicted_labels == 1))
print("Percentage of Correct Predictions: {:.2f}%".format(percentage_correct))

print(predicted_labels2) 
print("number of 0s in real_data", np.sum(predicted_labels2 == 0))
print("Percentage of Correct REal Predictions: {:.2f}%".format(percentage_correct2))


# test both gen_window_data, real_window_data
print('=====cm for generated and real=====')
window_data = []
for i in range(0, len(df2) - K + 1, stride):
    window = df2[i:i+K]
    window_data.append(window)
fake_window_data = np.array(window_data)
#gen_window_data = fake_window_data
min_size = min(gen_window_data.shape[0], real_window_data.shape[0])
gen_window_data2 = gen_window_data[:min_size]
real_window_data = real_window_data[:min_size]

gen_labels = np.ones(gen_window_data2.shape[0])
real_labels = np.zeros(real_window_data.shape[0])

combined_data = np.concatenate((gen_window_data2, real_window_data), axis=0)
combined_labels = np.concatenate((gen_labels, real_labels), axis=0)

combined_data, combined_labels = shuffle(combined_data, combined_labels)

class_probabilities = model.predict(combined_data)
class_predictions = (class_probabilities > 0.5).astype(int)
#pdb.set_trace()
cm = confusion_matrix(combined_labels, class_predictions)

false_positive = cm[0, 1]
total_negative = len(gen_labels)

total_samples = cm.sum()
false_positive_percentage = (false_positive / total_negative) * 100

print(f"False Positive Percentage: {false_positive_percentage:.2f}%")


correct_predictions = cm[0, 0] + cm[1, 1]

accuracy = (correct_predictions / total_samples) * 100

print(f"half B half G or S Accuracy: {accuracy:.2f}%")

plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.colorbar()
plt.xticks([0, 1], ['Real', 'Fake'])
plt.yticks([0, 1], ['Real', 'Fake'])
plt.savefig("class_plots/confustionMatrix2")
plt.show()
plt.clf()


# all G data
labels = np.ones(gen_window_data.shape[0])
predictions = model.predict(gen_window_data)
class_correct_predictions = (predictions > 0.5).astype(int)
accuracy = accuracy_score(labels, class_correct_predictions)

# Print the accuracy
print("All G Accuracy:", accuracy)
cm = confusion_matrix(labels, class_correct_predictions)

TN, FP, FN, TP = cm.ravel()

# Calculate True Positive Rate (TPR), True Negative Rate (TNR), False Positive Rate (FPR), and False Negative Rate (FNR)
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

# Calculate ROC AUC score
#roc_auc = roc_auc_score(labels, class_correct_predictions)

# Print the results
print("True Positive Rate (TPR):", TPR)
print("True Negative Rate (TNR):", TNR)
print("False Positive Rate (FPR):", FPR)
print("False Negative Rate (FNR):", FNR)
#print("ROC AUC:", roc_auc)


# Create a single figure for all graphs
M = data.shape[1]
fig, axs = plt.subplots(1, M, figsize=(10, 5)) 

for i in range(M): # Iterate over each column and create a dot graph
    # Get the data for the current column
    column_data = data[:, i]

    # Divide the data into two halves
    half = len(column_data) // 2
    first_half = column_data[:half]
    second_half = column_data[half:]

    # 1st half
    if data_dim != 1:
        axs[i].plot(range(half), first_half, 'ro', label='Real Data')

        # 2nd 
        axs[i].plot(range(half, len(column_data)), second_half, 'bo', label='Fake Data')

        axs[i].set_title(f'Column {i+1}') 
        axs[i].legend() 
    else: # Axes object becomes unsubscriptable if there is only 1 column/graph
        # 1st half
        axs.plot(range(half), first_half, 'ro', label='Real Data')

        # 2nd 
        axs.plot(range(half, len(column_data)), second_half, 'bo', label='Fake Data')

        axs.set_title(f'Column {i+1}') 
        axs.legend() 

plt.tight_layout()
plt.savefig("class_plots/all_data_w_dims")
plt.show()


plt.clf()

#pdb.set_trace()


# Calculate the moving average using a window size 
window_size = 10
moving_avg = np.convolve(column_data, np.ones(window_size)/window_size, mode='same')

# Calculate z-scores 
z_scores = stats.zscore(column_data)

# Set threshold to determine outliers
z_score_threshold = 2.5
outliers = np.where(np.abs(z_scores) > z_score_threshold)[0]

# Plot original data and moving average
plt.plot(column_data, label='Original Data')
plt.plot(moving_avg, label='Moving Average')

# Highlight outliers
plt.scatter(outliers, column_data[outliers], color='red', label='Outliers')

# Add a red line to represent the separation between real and fake data
separation_line = len(column_data) // 2
plt.axvline(x=separation_line, color='red', linestyle='--', label='Real/Spoof Separator')


plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.savefig("class_plots/trends_in_data")
plt.show()


plt.clf()