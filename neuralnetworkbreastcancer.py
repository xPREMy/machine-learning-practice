from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
data =load_breast_cancer()
X=data.data
y=data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
split=int(0.7*len(df))
dftr=df[:split]
dfte=df[split:]

X_train=dftr.drop('target',axis=1).astype('float32').values
Y_train=dftr['target'].astype('int32').values

X_test=dfte.drop('target',axis=1).astype('float32').values
Y_test=dfte['target'].astype('int32').values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model=Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(units=16,activation='relu'),
    Dense(units=12,activation='relu'),
    Dense(units=4,activation='relu'),
    Dense(units=1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=200,validation_split=0.2)
pred=model.predict(X_test)
pr=(pred>=0.5).astype('int32')
print("acurracy: ", accuracy_score(Y_test,pr))
model.save("breast_cancer_model.h5")
