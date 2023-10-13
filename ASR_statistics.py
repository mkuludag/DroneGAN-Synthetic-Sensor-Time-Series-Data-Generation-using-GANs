import matplotlib.pyplot as plt

# Sample data: 2 values for 6 items
items = ['accel X', 'accel Y', 'accel Z', 'gyro X', 'gyro Y', 'gyro Z']
value1 = [96.40, 97.85, 5.2, 96.8, 98.65, 84.5]
value2 = [83.75, 92.5, 63.75, 100, 100, 100]

# Define the width of the bars
bar_width = 0.4

# Define the x-axis positions for the bars
x = range(len(items))

# Create the first set of bars
bar1 = plt.bar(x, value1, width=bar_width, label='Autoencoder', align='center')

# Create the second set of bars shifted to the right
bar2 = plt.bar([i + bar_width for i in x], value2, width=bar_width, label='CNN Classifier', align='center')

# Set labels for the x-axis and y-axis
plt.xlabel('Sensor', fontsize=16)
plt.ylabel('Attack Success Rate (ASR)', fontsize=16)

# Set the x-axis ticks and labels
plt.xticks([i + bar_width / 2 for i in x], items,  fontsize=14)

# Add values at the top of each bar
for bar in bar1 + bar2:
    height = bar.get_height()
    plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',  fontsize=16)

plt.ylim(0, max(max(value1), max(value2)) + 10)

# Move the legend outside the chart
# plt.legend(loc='lower center')
# Add a legend
# plt.legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)
plt.legend(bbox_to_anchor =(0.47,-0.35), loc='lower center', ncol=2,  fontsize=16)
plt.tight_layout()
# Show the plot
plt.show()

