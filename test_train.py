from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
from keras.layers import LSTM
import numpy as np


feature_hash = 4023239360457093510
experiment_hash = 8841606304273805265
tempLength = 7
def run_experiment(X_train, y_train, X_val, y_val):
    print 'Constructing model'
    model = Sequential()
    model.add(Bidirectional(LSTM(7, return_sequences=False), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, W_regularizer=l2(1.0), activation='sigmoid'))
    model.summary()
    np.random.seed(0)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
    print 'Training model'
    model.fit(X_train, y_train, nb_epoch=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(patience=3)])
    model.save('model.h5')

    print 'Saving predictions'

    X = np.load('X_%d.npy' % feature_hash)

    model.save('model_%d.h5' % experiment_hash)
    X = X.reshape((X.shape[0], tempLength, X.shape[1]//tempLength))
    y_prob = model.predict_proba(X)
    y_hat = model.predict_classes(X)

    np.save('y_prob_%d.npy' % experiment_hash, y_prob)
    np.save('y_hat_%d.npy' % experiment_hash, y_hat)

    print("Done Training...")