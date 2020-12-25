import os
import numpy as np
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Flatten, Dense, Dropout, AveragePooling2D

DATASET_PATH = './specset'


def load_data(path):
    r = os.path.join(DATASET_PATH, path)
    data = []
    for i in os.listdir(r):
        p = os.path.join(r, i)
        tmp = np.load(p)
        data.append(tmp)

    return np.array(data)


def rand_sample(data, samples=10):
    ind = np.arange(len(data))
    np.random.shuffle(ind)
    return data[ind[:samples]]


equal = load_data('equal')
pure = load_data('pure')
pytha = load_data('pytha')

# model parameters
DROP_RATE = 0.8
L1 = 1e-2

model = Sequential()
model.add(Dropout(DROP_RATE, input_shape=(1025, 469, 1)))
model.add(AveragePooling2D((2, 50)))
model.add(MaxPooling2D((1, 9)))
model.add(Flatten())
model.add(Dense(3, activation='softmax', kernel_regularizer=regularizers.l1(L1)))

Adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['acc'])
model.summary()
model.save_weights('initial.hdf5')

# `SAMPEL` songs for each temperament
# `TRAIN_NUM` of which for training, while others for validating
SAMPLE = 10
TRAIN_NUM = 7
VAL_NUM = SAMPLE - TRAIN_NUM

acc_list = np.zeros((10))
for rep in range(10):
    # prepare data
    data = np.concatenate((rand_sample(equal), rand_sample(pure), rand_sample(pytha)), axis=0)

    train_x = np.zeros((TRAIN_NUM * 3, np.shape(data)[1], np.shape(data)[2], 1))
    train_y = np.zeros((TRAIN_NUM * 3, 3))
    val_x = np.zeros((VAL_NUM * 3, np.shape(data)[1], np.shape(data)[2], 1))
    val_y = np.zeros((VAL_NUM * 3, 3))

    for i in range(3):
        train_x[(TRAIN_NUM * i):((TRAIN_NUM * (i + 1))), :, :, 0] = data[(SAMPLE * i):(SAMPLE * i + TRAIN_NUM)]
        val_x[(VAL_NUM * i):((VAL_NUM * (i + 1))), :, :, 0] = data[(SAMPLE * i + TRAIN_NUM):(SAMPLE * (i + 1))]
        train_y[(TRAIN_NUM * i):((TRAIN_NUM * (i + 1))), i] = 1
        val_y[(VAL_NUM * i):((VAL_NUM * (i + 1))), i] = 1

    # train model
    filepath = "model.hdf5"

    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    save_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    callbacks = [early_stop, save_best]
    model.load_weights('initial.hdf5')
    model.fit(train_x, train_y, epochs=20000, batch_size=7, validation_data=(val_x, val_y), callbacks=callbacks)
    model.load_weights(filepath)
    loss, acc = model.evaluate(val_x, val_y)
    acc_list[rep] = acc
    print(acc)

print(np.mean(acc_list))
print(np.std(acc_list) / (10 - 1)**0.5)

np.save('T7.npy', acc_list)
