import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
#import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras import Input
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
import pickle



# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 14, 8

# RANDOM_SEED = 42

# np.random.seed(RANDOM_SEED)

csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/BTC-USD.csv"

df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values('Date')
print(df.head())
print (df.shape)

ax = df.plot(x='Date', y='Close');
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")
# plt.show()

scaler = MinMaxScaler()

close_price = df.Close.values.reshape(-1, 1)

scaled_close = scaler.fit_transform(close_price)
print(scaled_close.shape)

print(np.isnan(scaled_close).any())
scaled_close = scaled_close[~np.isnan(scaled_close)]
scaled_close = scaled_close.reshape(-1, 1)
print(np.isnan(scaled_close).any())


#preprocessing

SEQ_LEN = 100

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.95)
print(X_train.shape)
print(y_train.shape)


#MODEL
DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

model = keras.Sequential()

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True), input_shape=(WINDOW_SIZE, X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

model.add(Dense(units=1))

model.add(Activation('linear'))

#Training
model.compile(
    loss='mean_squared_error', 
    optimizer='adam'
)
BATCH_SIZE = 64

history = model.fit(
    X_train, 
    y_train, 
    epochs=25, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    validation_split=0.1
)

model_json = model.to_json()
print(model_json)
with open("C:/Users/BHAVYA NANGIA/Desktop/FIL Training bitbucket/CryptOS/home/model1.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
model.save_weights("C:/Users/BHAVYA NANGIA/Desktop/FIL Training bitbucket/CryptOS/home/model1.h5")
print("Saved model to disk")

# json_file = open('E:\FIL-Project\CryptOS\home\model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights("E:\FIL-Project\CryptOS\home\model.h5")
# # print("Loaded model from disk")

# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
print(model.evaluate(X_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)
 
plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')
 
plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')
 
plt.show();
#model.save('saved_model/trained_data')