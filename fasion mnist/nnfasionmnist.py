import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.regularizers import l2
# data frame loading for training set
df=pd.read_csv(r'C:\Users\OKKKK\Desktop\stocktesting2\fasion mnist\fashion-mnist_train.csv')
X_train=df.drop('label',axis=1).astype('float32').values/255.0
Y_train=df['label'].astype('int32').values
# data frame loading for test set
df1=pd.read_csv(r'C:\Users\OKKKK\Desktop\stocktesting2\fasion mnist\fashion-mnist_test.csv')
X_test=df1.drop('label',axis=1).astype('float32').values/255.0
Y_test=df1['label'].astype('int32').values

model=Sequential([
    Dense(64,input_shape=(X_train.shape[1],),activation='relu',kernel_regularizer=l2(0.0001)),
    Dense(32,'relu',kernel_regularizer=l2(0.0001)),
    Dense(10,'linear')
])
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),optimizer=Adam(0.0001),metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=30,validation_split=0.1)

logits=model.predict(X_test)
prediction=np.argmax(logits,axis=1)
print("accuracy: ",accuracy_score(Y_test,prediction))
model.save("fasion1.h5")
