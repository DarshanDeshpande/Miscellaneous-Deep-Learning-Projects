import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


df = pd.read_csv("D:\\Download\\aadr.us.txt")
print(df.head(5))
df.drop(columns=['High','Low','OpenInt'],inplace=True)
print(df.head(5))

X = np.array(df[['Open']])
Y = np.array(df['Close'])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,shuffle=False,train_size=0.9,test_size=0.1)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(256,input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences= True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(256,return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation='linear')))

adam = tf.keras.optimizers.Adam(lr=0.001)

reducr_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

model.compile(optimizer=adam,loss='mae',metrics=['mse'],callbacks=[reducr_lr])
model.fit(X_train,Y_train,epochs=300,batch_size=32)
model.save("StockPredictionModel.h5")

# UNCOMMENT THIS AND COMMENT THE ABOVE THREE LINES TO CONTINUE TRAINING
# model = tf.keras.models.load_model("StockPredictionModel.h5")
# model.fit(X_train,Y_train,epochs=150,batch_size=32)
# model.save("StockPredictionModel1.h5")


model = tf.keras.models.load_model("StockPredictionModel.h5")
pred = model.predict(X_test)
plt.plot(Y_test,label='actual',color='blue')
plt.plot(pred.reshape(157,1),label='predictions',color='black')
plt.legend()
plt.show()
