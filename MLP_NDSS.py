import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import pdb
# We consider a feed-forward neural net (i.e.,
# MLP) with 2 hidden layers with Leaky-Relu (LRelu) activation
# to make the model non-linear to enable learning the non-linear
# dynamics. The model attempts to predict the next position of
# the UAV and, hence, learn the sensor dynamics of the mission.
# Therefore, given the an example xi at time step i, the label yi
# is the position of xi+1.
# We will need to implement this part for the dataset training, basically give 3 consequtive samples of sensors: for input 1 the output (label) has to be 2, for 2 - output is 3, etc. 

# NOTE: there is no specifics about the structure of hidden layers itself


#convert to cartesian if neccessary: 
p1 = 'sensor-csvfiles/' + 'NDSS_data_real' + '.csv'
p2 = 'sensor-csvfiles/' + 'NDSS_data_gen' + '.csv'
p3 = 'sensor-csvfiles/' + 'NDSS_data_spoof' + '.csv'
df = pd.read_csv(p1)
df_g = pd.read_csv(p2)
df_s = pd.read_csv(p3)

# # Convert GPS coordinates to Cartesian coordinates
# R = 6371  # Earth's radius in kilometers (you can adjust this if needed)
# df['gps_c_x'] = R * np.cos(np.radians(df['gps_x'])) * np.cos(np.radians(df['gps_y']))
# df['gps_c_y'] = R * np.cos(np.radians(df['gps_x'])) * np.sin(np.radians(df['gps_y']))
# df['gps_c_z'] = R * np.sin(np.radians(df['gps_x']))

# # Rename the columns
# df.rename(columns={'gps_x': 'gps_c_x', 'gps_y': 'gps_c_y', 'gps_z': 'gps_c_z'}, inplace=True)

# # Save the modified DataFrame back to the CSV file
# df.to_csv(p1, index=False)


sensors_1 =  [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    
    "gps_x",
    "gps_y",
    "gps_z", 
    
    "mag_x",
    "mag_y",
    "mag_z",
    
    # "gps_c_x",
    # "gps_c_y",
    # "gps_c_z"
    ]
input_dim = len(sensors_1)
output_dim = input_dim  

num_samples = 135 # 135, 225
X = df[sensors_1][:num_samples]
extracted_df3 = df_s[sensors_1][:num_samples]

y = X[1:]
X = X[:-1]

# #X = np.concatenate((extracted_df, extracted_df3), axis=0)
# y = np.zeros_like(extracted_df)
# # y2 = np.ones_like(extracted_df3)
# # y = np.concatenate((y, y2), axis=0)


# Normalize data using MinMaxScaler (Scaling data)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# NOTE: plcaeholder to complete the code. Split data into training and testing sets. You can adjust it as needed. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Define the architecture of the MLP
def build_mlp(input_dim, hidden_units, output_dim):
    model = Sequential()
    
    # Input layer
    model.add(Dense(hidden_units[0], input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))  # Leaky ReLU activation
    
    # Hidden layers
    for units in hidden_units[1:]:
        model.add(Dense(units))
        model.add(LeakyReLU(alpha=0.2))  # Leaky ReLU activation
    
    # Output layer
    model.add(Dense(output_dim))
    
    return model

# Define parameters
hidden_units = [64, 32]  # Number of units in each hidden layer

# Build the MLP model
mlp_model = build_mlp(input_dim, hidden_units, output_dim)

# Compile the model
mlp_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

# Train the model
mlp_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = mlp_model.evaluate(X_test, y_test) 
# threshold = 0.5
# y_pred_prob = mlp_model.predict(X_test)  
# y_pred = (y_pred_prob >= threshold).astype(int)  
# accuracy = np.mean(y_pred == y_test)

print(f'Test Loss: {loss}')
#print(f'Test Accuracy: {accuracy}')

print("Benign testing for each column for benchmark:")


predictions = mlp_model.predict(X_test)
prediction_errors = np.abs(predictions - y_test)

column_means = np.mean(prediction_errors, axis=0)
for column_name, mean_value in column_means.items():
    mean_value = round(mean_value, 3)
    print(f" {mean_value}")
    #print(f"'{column_name}': {mean_value}")

column_medians = prediction_errors.median()
# for column_name, median_value in column_medians.items():
#     print(f"Median of Column '{column_name}': {median_value}")

# Test Generated: 
sensors_2 =  [
    #"accel_x",
    # "accel_y",
     "accel_z",
    #"gyro_x",
    # "gyro_y",
    # "gyro_z",
     
    # "gps_x",
    #  "gps_y",
    # "gps_z",
    
     # "gps_c_x",
    # "gps_c_y",
    # "gps_c_z",  
    ]

updated_data = df_s[sensors_2][:num_samples]

test = pd.DataFrame(X, columns=sensors_2)
c_preds = pd.DataFrame(y, columns=sensors_2)

test[sensors_2] = updated_data.iloc[:-1].values
c_preds[sensors_2] = updated_data.iloc[1:].values

remaining_cols = [col for col in X.columns if col not in sensors_2]
test[remaining_cols] = X[remaining_cols]
c_preds[remaining_cols] = y[remaining_cols]

test = test.reindex(X.columns, axis=1)
c_preds = c_preds.reindex(y.columns, axis=1)
#pdb.set_trace()

# updated_data = df_s[sensors_2][:num_samples]
# test = X.copy() 
# test[sensors_2] = updated_data[sensors_2][:-1]

# c_preds = y.copy()
# column_mask = np.isin(test.columns, sensors_2)
# c_preds[:, column_mask] = updated_data[sensors_2][1:]


# loss, accuracy2 = mlp_model.evaluate(test, test_labels) 
# threshold = 0.5
# pred_prob = mlp_model.predict(test)  
# pred = (pred_prob >= threshold).astype(int)  
# accuracy = np.mean(pred == test_labels)

# print(f'Test Loss: {loss}')
# print(f'Test Accuracy: {accuracy}')

#pdb.set_trace()


# Evaluate the model and detect anomalies
predictions = mlp_model.predict(test)
# Calculate prediction errors
prediction_errors = np.abs(predictions - c_preds)

column_means = np.mean(prediction_errors, axis=0)
for column_name, mean_value in column_means.items():
    mean_value = round(mean_value, 3)
    print(f"{mean_value}")
    #print(f"'{column_name}': {mean_value}")

column_medians = prediction_errors.median()
# for column_name, median_value in column_medians.items():
#     print(f"Median of Column '{column_name}': {median_value}")

# Flatten prediction errors for histogram
prediction_errors_flat = np.array(prediction_errors).flatten()

# Plot the prediction error distribution
plt.hist(prediction_errors_flat, bins=50, alpha=0.7, color='blue')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.grid(True)
#plt.show()


# NOTE: training process
# Achived average error rate of 0.322m on test set with accuracy 0.957 --> 4800
# standard deviation for the error 0.283 with median 0.281
