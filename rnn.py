import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


X_train = np.random.rand(100, 10, 1)  
y_train = np.random.rand(100, 1)   

model = Sequential()
model.add(SimpleRNN(64, input_shape=(10, 1), return_sequences=False))  
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=10, batch_size=16)

output = model.predict(X_train[:5])  
print("Model Output:", output)
