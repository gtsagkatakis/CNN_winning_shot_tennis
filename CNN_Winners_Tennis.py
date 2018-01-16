"""
Created on Wed Jan  3 12:16:07 2018

@author: Grigorios Tsagkatakis
"""
import numpy as np
from matplotlib import pyplot as plt
import keras
import os
from keras.layers import Dense, Flatten, Dropout, Merge
from keras.models import Sequential
from keras.optimizers import  SGD
from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.callbacks import CSVLogger
from IPython.display import clear_output
from PIL import Image

dim_x=140
dim_y=78
dim_t=30
num_classes=2
num_ex_per_class=100;

winner_data=np.load('winner_data.npy')
nowinner_data=np.load('nowinner_data.npy')

winner_feat=np.load('winner_data_OF.npy')
nowinner_feat=np.load('nowinner_data_OF.npy')

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_acc'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="Training accuracy")
        plt.plot(self.x, self.val_losses, label="Testing accuracy")
        plt.legend()
        plt.show();
  


      
plot_losses = PlotLosses()


np.random.seed(1111)
train_sel=np.random.choice(100, 100)

Train_data=np.empty([2*num_ex_per_class,dim_y,dim_x,dim_t])
Train_data[0:num_ex_per_class,:,:,:]=  winner_data[train_sel[0:num_ex_per_class],:,:,:]
Train_data[num_ex_per_class:2*num_ex_per_class,:,:,:]=  nowinner_data[train_sel[0:num_ex_per_class],:,:,:]
Train_data = Train_data.reshape(Train_data.shape[0], dim_y, dim_x, dim_t, 1)
    
Train_feat=np.empty([2*num_ex_per_class,dim_y,dim_x,dim_t])
Train_feat[0:num_ex_per_class,:,:,:]=  winner_feat[train_sel[0:num_ex_per_class],:,:,:]
Train_feat[num_ex_per_class:2*num_ex_per_class,:,:,:]=  nowinner_feat[train_sel[0:num_ex_per_class],:,:,:]
Train_feat = Train_feat.reshape(Train_feat.shape[0], dim_y, dim_x, dim_t, 1)
    
y_train=np.zeros([2*num_ex_per_class,1])
y_train[0:num_ex_per_class]=0
y_train[num_ex_per_class:2*num_ex_per_class]=1
Train_data_label = keras.utils.to_categorical(y_train, num_classes)
    

model0 = Sequential();
model0.add(Conv3D(32, (3,3,3), activation='relu', input_shape=[dim_y,dim_x,dim_t,1]))
model0.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1)))
model0.add(Conv3D(64, (3,3,3), activation='relu'))
model0.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model0.add(Conv3D(128, (3,3,3), activation='relu'))
model0.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model0.add(Flatten())
    
model1 = Sequential();
model1.add(Conv3D(32, (3,3,3), activation='relu', input_shape=[dim_y,dim_x,dim_t,1]))
model1.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1)))
model1.add(Conv3D(64, (3,3,3), activation='relu'))
model1.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model1.add(Conv3D(128, (3,3,3), activation='relu'))
model1.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model1.add(Flatten())
    
model =  Sequential();
model.add(Merge([model0, model1], mode = 'concat'))
    
   
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
    
sgd = SGD(lr=0.001, decay=1e-1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
fname_log="Testlog_%d.log" % (num_ex_per_class)
csv_logger = CSVLogger(fname_log)
history=model.fit([Train_data,Train_feat], Train_data_label,epochs=100,batch_size=10,callbacks=[csv_logger,plot_losses],validation_split=0.2,shuffle=True)
