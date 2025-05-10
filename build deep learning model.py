from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x,y=make_classification(n_samples=1000,n_features=20,n_informative=15,n_redundant=5)
xtr,xt,ytr,yt=train_test_split(x,y,test_size=0.2)
m=Sequential([
  Dense(16,activation='relu',input_shape=(20,)),
  Dense(8,activation='relu'),
  Dense(1,activation='sigmoid')
])
m.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
m.fit(xtr,ytr,epochs=20,batch_size=32,verbose=0)
yp=(m.predict(xt)>0.5).astype(int).flatten()
print("Accuracy:",accuracy_score(yt,yp))
