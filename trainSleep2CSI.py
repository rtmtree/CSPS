"""
The Codes in this file are used to classify Human Activity using Channel State Information. 
The deep learning architecture used here is Bidirectional LSTM stacked with One Attention Layer.
Author: https://github.com/ludlows
2019-12
"""
import numpy as np 
import random
import tensorflow as tf
import glob
import os
import csv
import json
import pylab
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM,Conv1D,Conv2D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Input, Embedding, Concatenate
from modules.draw import Plotter3d, draw_poses
from functions.csi_util import rawCSItoAmp,filterNullSC,csiIndices_sec,poseIndices_sec,samplingCSISleep,featureEngineer,featureEngineerNorm
from functions.pose_util import poseToPAM,PAMtoPose,rotate_poses,getPCK
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import stdev
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from math import sqrt, atan2 ,isnan



class AttenLayer(tf.keras.layers.Layer):
    """
    Attention Layers used to Compute Weighted Features along Time axis
    Args:
        num_state :  number of hidden Attention state
    
    2019-12, https://github.com/ludlows
    """
    def __init__(self, num_state, **kw):
        super(AttenLayer, self).__init__(**kw)
        self.num_state = num_state
    
    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[input_shape[-1], self.num_state])
        self.bias = self.add_weight('bias', shape=[self.num_state])
        self.prob_kernel = self.add_weight('prob_kernel', shape=[self.num_state])

    def call(self, input_tensor):
        atten_state = tf.tanh(tf.tensordot(input_tensor, self.kernel, axes=1) + self.bias)
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
        prob = tf.nn.softmax(logits)
        weighted_feature = tf.reduce_sum(tf.multiply(input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature
    
    # for saving the model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state,})
        return config


class CSIModelConfig:
    """
    class for Human Activity Recognition ("bed", "fall", "pickup", "run", "sitdown", "standup", "walk")
    Using CSI (Channel State Information)
    Specifically, the author here wants to classify Human Activity using Channel State Information. 
    The deep learning architecture used here is Bidirectional LSTM stacked with One Attention Layer.
       2019-12, https://github.com/ludlows
    Args:
        win_len   :  integer (1000 default) window length for batching sequence
        step      :  integer (200  default) sliding window by this step
        thrshd    :  float   (0.6  default) used to check if the activity is intensive inside a window
        downsample:  integer >=1 (2 default) downsample along the time axis
    """
    def __init__(self, win_len=1000, step=200, thrshd=0.6, downsample=2):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ("bed", "fall", "pickup", "run", "sitdown", "standup", "walk")
        self._downsample = downsample

    def build_model2(self, n_unit_lstm=200, n_unit_atten=400,label_n=2):
        """
        Returns the Tensorflow Model which uses AttenLayer
        """
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
            x_in = tf.keras.Input(shape=(length, 52))
        else:
            x_in = tf.keras.Input(shape=(self._win_len, 52))
        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True))(x_in)
        x_tensor = AttenLayer(n_unit_atten)(x_tensor)
        pred = tf.keras.layers.Dense(label_n, activation='softmax')(x_tensor)
        model = tf.keras.Model(inputs=x_in, outputs=pred)
        return model

    @staticmethod
    def load_model(hdf5path):
        """
        Returns the Tensorflow Model for AttenLayer
        Args:
            hdf5path: str, the model file path
        """
        model = tf.keras.models.load_model(hdf5path, custom_objects={'AttenLayer':AttenLayer})
        return model
    
if __name__ == "__main__":
 
    path = 'drive/MyDrive/Project/'
    # path = ''
    realData = True

    labels=['sleep29-11-2021end1020','sleep30-11-2021end1020']
    labelsAlt=[]
    batch_size = 64
    runTrain=True
    sleepWinSize=1
    samplingedCSIWinSize = 200
    epoch=300

    if realData:
        label_n = 4
        minCSIthreshold= samplingedCSIWinSize
        modelFileName='test_models/csi2sleepM_e'+str(epoch)+'.hdf5'
    else:
        label_n = 2
        modelFileName="test.hdf5"

    # preprocessing
    csiFilePaths=[]
    sleepFilePaths=[]
    csiFilePathsAlt=[]
    sleepFilePathsAlt=[]
    for label in labels:
        csiFilePaths.append('data/CSI'+label+'.csv')
        sleepFilePaths.append('data/SS'+label+'.csv')
    for label in labelsAlt:
        csiFilePathsAlt.append('data/CSI'+label+'.csv')
        sleepFilePathsAlt.append('data/SS'+label+'.csv')
    X = []
    Y = []
    Xalt = []
    Yalt = []
    if realData:#load file
        for fileIndx in range(len(csiFilePaths)+len(csiFilePathsAlt)):
            if(fileIndx<len(csiFilePaths)):
                csiFileName=csiFilePaths[fileIndx]
                sleepFileName=sleepFilePaths[fileIndx]
                isAlt=False
            else:
                csiFileName=csiFilePathsAlt[fileIndx-len(csiFilePaths)]
                sleepFileName=sleepFilePathsAlt[fileIndx-len(csiFilePaths)]
                isAlt=True
            csiList=[]
            sleepList=[]
            csiList.extend(pd.read_csv(path+csiFileName,delimiter=',',header=None).values)
            sleepList.extend(pd.read_csv(path+sleepFileName,delimiter=',',header=None).values)

            dataLooper=len(sleepList)
            dataStep=sleepWinSize
            for i in range(0,dataLooper,dataStep):

                # get pose in the time period
                sleepIndices,startTime,endTime=poseIndices_sec(i,sleepList,sec=30)
                print("index",i,"/",dataLooper)
                print("sleepIndices",sleepIndices)
                print("startTime",startTime)
                print("endTime",endTime)

                csiIndices=csiIndices_sec(startTime,endTime,csiList)
                print("len csiIndices",len(csiIndices))
                if (len(csiIndices)>0):
                  print("csiIndices",csiIndices[0],"-",csiIndices[-1])

                  # check if there is csi more than minCSIthreshold
                  if(len(csiIndices)<minCSIthreshold):
                      print("too low csi number",len(csiIndices),minCSIthreshold)
                      continue
                  
                  print(len(csiIndices),"csis to ",len(sleepIndices),"SS")

                  curCSIs,_=samplingCSISleep(csiList,csiIndices,sleepList,sleepIndices,samplingedCSIWinSize)
                  stage = [0,0,0,0]
                  if(sleepList[sleepIndices[0]][1] == 1):
                      stage = [1,0,0,0]
                  elif(sleepList[sleepIndices[0]][1] == 2):
                      stage = [0,1,0,0]
                  elif(sleepList[sleepIndices[0]][1] == 3):
                      stage = [0,0,1,0]
                  elif(sleepList[sleepIndices[0]][1] == 4):
                      stage = [0,0,0,1]
                  curSleeps = stage
                  
                  if(isAlt==False):
                      X.append(curCSIs)
                      Y.append(curSleeps)
                  else:
                      Xalt.append(curCSIs)
                      Yalt.append(curSleeps)

        X=np.array(X)
        Y=np.array(Y)
        print('bf shape X',(X.shape))
        print('bf shape Y',(Y.shape))

        Xalt=np.array(Xalt)
        Yalt=np.array(Yalt)

        print('shape Xalt',(Xalt.shape))
        print('shape Yalt',(Yalt.shape))


    else:#load fake data
        lenFake = 100
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=400)
        print('shape X_train',(X_train.shape))
        print(len(X_train[0][0:53]))
        X = [[[X_train[i*j][k] if len(X_train[i*j])>52 else 0 for k in range(52)] for j in range(samplingedCSIWinSize)] for i in range(lenFake)]
        Y = [ [1,0] if y_train[i]==0 else [0,1] for i in range(lenFake)]
        Xalt = [[[X_test[i*j][k] if len(X_test[i*j])>52 else 0 for k in range(52)] for j in range(samplingedCSIWinSize)] for i in range(lenFake)]
        Yalt = [ [1,0] if y_test[i]==0 else [0,1] for i in range(lenFake)]

        print('len X',len(X))
        print('len Y',len(Y))
        X=np.array(X)
        Y=np.array(Y)
        Xalt=np.array(Xalt)
        Yalt=np.array(Yalt)
        print('shape X',(X.shape))
        print('shape Y',(Y.shape))
        
    if len(Xalt)==0:
      x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size=0.2, random_state=3)
    else:
      x_train = X
      y_train = Y
      x_test = Xalt
      y_test = Yalt
      # way 2
      # np.random.seed(42) 
      # np.random.shuffle(X) 
      # np.random.seed(42) 
      # np.random.shuffle(Y) 

    print('shape x_train',(x_train.shape))
    print('shape x_test',(x_test.shape))
    print('shape y_train',(y_train.shape))
    print('shape y_test',(y_test.shape))
    

    if runTrain:

        # parameters for Deep Learning Model
        cfg = CSIModelConfig(win_len=400, step=200, thrshd=0.6, downsample=2)
        model = cfg.build_model2(n_unit_lstm=200, n_unit_atten=400,label_n=label_n)
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
        print(model.summary())
        model.fit(
        x_train,
        y_train,
        batch_size=batch_size, epochs=epoch,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(path+modelFileName,
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                save_weights_only=False)
            ])

        # model.save(path+modelFileName)

    if True:
        # load the best model
        model = cfg.load_model(path+modelFileName)
        y_pred = model.predict(x_test)

        print("evaluate")
        matchCounter = 0
        for i in range(0,len(y_test)):
          maximum_test = np.max(y_test[i])
          maximum_pred = np.max(y_pred[i])
          index_of_maximum_test = np.where(y_test[i] == maximum_test)
          index_of_maximum_pred = np.where(y_pred[i] == maximum_pred)
          print(i,"test",index_of_maximum_test[0])
          print(i,"pred",index_of_maximum_pred[0])
          if(index_of_maximum_test[0]==index_of_maximum_pred[0]):
            print("match")
            matchCounter = matchCounter+1
          else:
            print("unmatch")
        print("score",matchCounter,"/",len(y_test))
