# Basic import
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow import keras
import numpy as np

# Create the model
model = Sequential(Dense(units = 1, input_shape = [1]))

# Compile the model
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# Provide data
xs = np.array([-1, 0, 1, 2, 3, 4], dtype = float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype = float)

# Train the model
model.fit(xs, ys, epochs = 500)

# Predict the model
print(model.predict([10.0]))
