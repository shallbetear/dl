from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
import numpy as np

(x,_),_=mnist.load_data()
x=x.astype("float32")/255.0
x=x.reshape(-1,28,28,1)

i=Input((28,28,1))
e=Dense(32,activation='relu')(Flatten()(i))
d=Reshape((28,28,1))(Dense(784,activation='sigmoid')(Dense(64,activation='relu')(e)))
m=Model(i,d)
m.compile(optimizer='adam',loss='mse')
m.fit(x,x,epochs=10,batch_size=128,verbose=0)

r=m.predict(x[:10])
mse=np.mean((x[:10]-r)**2,axis=(1,2,3))
print("Anomalies:",mse>0.02)
