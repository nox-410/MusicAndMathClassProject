import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization, MaxPooling2D, Activation, ActivityRegularization, GaussianNoise
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt 

DATASET_PATH = './specset'


def Load_Data(path):
    r = os.path.join(DATASET_PATH, path)
    data = []
    for i in os.listdir(r):
        p = os.path.join(r,i)
        tmp = np.load(p)
        data.append(tmp)  
    
    return np.array(data)

def Rand_Sample(data, samples = 10):
    ind = np.arange(len(data))
    np.random.shuffle(ind)
    return data[ind[:samples]]
    

#%% load data
equal = Load_Data('equal')
pure = Load_Data('pure')
pytha = Load_Data('pytha')

#%% define model
Drop_R = 0.8
AP_x = 2
AP_y = 50
MP_x = 1
MP_y = 9
L1 = 1e-2

model = Sequential()
model.add(Dropout(Drop_R,input_shape=(1025,469,1)))
model.add(AveragePooling2D((AP_x,AP_y)))
model.add(MaxPooling2D((MP_x,MP_y)))
model.add(Flatten())
model.add(Dense(3,activation='softmax', kernel_regularizer=regularizers.l1(L1)))


Adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['acc'])
model.summary()
model.save_weights('initial.hdf5')



#%% 
Train_num = 7
val_num = 3
Sample = 10

acc_list = np.zeros((10))
for rep in range(10):
    # prepare train & val
    data = np.concatenate((Rand_Sample(equal),Rand_Sample(pure),Rand_Sample(pytha)),axis =0 )
    
    train_x = np.zeros((Train_num*3,np.shape(data)[1],np.shape(data)[2],1))
    train_y = np.zeros((Train_num*3,3))
    val_x = np.zeros((val_num*3,np.shape(data)[1],np.shape(data)[2],1))
    val_y = np.zeros((val_num*3,3))
    
    for i in range(3):
        train_x[(Train_num*i):((Train_num*(i+1))),:,:,0] = data[(Sample*i):(Sample*i+Train_num)]
        val_x[(val_num*i):((val_num*(i+1))),:,:,0] = data[(Sample*i+Train_num):(Sample*(i+1))]
        train_y[(Train_num*i):((Train_num*(i+1))),i] = 1
        val_y[(val_num*i):((val_num*(i+1))),i] = 1
    
    # train model
    filepath="model.hdf5"
    
    earlyStopping=EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    
    callbacks_list = [earlyStopping,saveBestModel]
    model.load_weights('initial.hdf5')
    model.fit(train_x, train_y, epochs=20000, batch_size=7, validation_data=(val_x, val_y), callbacks=callbacks_list)
    model.load_weights(filepath)
    loss,acc = model.evaluate(val_x,val_y)
    acc_list[rep] = acc
    print(acc)

#%%
print(np.mean(acc_list))
print(np.std(acc_list)/(10-1)**0.5)

np.save('T7.npy',acc_list)






