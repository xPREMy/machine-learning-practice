from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from neuralnetwork import Model , layer
from neuralnetwork import*
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model=Model([
    layer(32,'relu'),
    layer(16,'relu'),
    layer(12,'relu'),
    layer(units=4,activation='relu'),
    layer(units=1,activation='sigmoid')
])
model.initialize(X_train,loss='BCE',learningrate=0.2)
model.fit(X_train,Y_train,epochs=100)
pred=model.predict(X_test)
pr=(pred>=0.5).astype('int32')
print("acurracy: ", accuracy_score(Y_test,pr))

plt.figure(figsize=(8, 5))
plt.plot(model.loss_history, label='Training Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
