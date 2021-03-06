"""
The Codes in this file are used to classify Human Activity using Channel State Information. 
The deep learning architecture used here is Bidirectional LSTM stacked with One Attention Layer.
Author: https://github.com/ludlows
2019-12
"""
import numpy as np 
import random
from numpy.core.numeric import True_
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
from functions.csi_util import rawCSItoAmp,filterNullSC,csiIndices_sec,poseIndices_sec,samplingCSISleep,sleepIdx2csiIndices_timestamp
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

def build_model(downsample=1,win_len=1000, n_unit_lstm=200, n_unit_atten=400,label_n=2):
    """
    Returns the Tensorflow Model which uses AttenLayer
    """
    if downsample > 1:
        length = len(np.ones((win_len,))[::downsample])
        x_in = tf.keras.Input(shape=(length, 52))
    else:
        x_in = tf.keras.Input(shape=(win_len, 52))
    x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True))(x_in)
    x_tensor = AttenLayer(n_unit_atten)(x_tensor)
    pred = tf.keras.layers.Dense(label_n, activation='softmax')(x_tensor)
    model = tf.keras.Model(inputs=x_in, outputs=pred)
    return model

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
    timeLen=30
    runTrain=True
    runEval=True

    #========== adjustable
    labels=[
      # 'sleep24-12-2021end1212'
      # 'sleep24-12-2021end0954'
      'sleep21-12-2021end1200',
      'sleep20-12-2021end1123',
      'sleep18-12-2021end1106'
      , 
      'sleep17-12-2021end1155',
      # 'sleep29-11-2021end1020',
      'sleep30-11-2021end1020',
      'sleep08-12-2021end1000','sleep11-12-2021end0930',
      'sleep12-12-2021end1000','sleep13-12-2021end1010',
      'sleep14-12-2021end1200','sleep15-12-2021end1220',
      'sleep16-12-2021end1210'
      ]
    labelsAlt=[
      # 'sleep24-12-2021end1212',
      # 'sleep24-12-2021end0954'
      'sleep29-11-2021end1020'
      # 'sleep16-12-2021end1210'
    ]
    wakeIncluded = False
    batch_size = 128
    sleepWinSize=1 #sigma
    samplingedCSIWinSize = 600 #delta
    epoch=300
    n_unit_lstm=200
    n_unit_atten=400
    downsample=2

    #========== adjustable end
    minCSIthreshold= samplingedCSIWinSize
    modelFileName='test_models/csi2sleepM_e'+str(epoch)+'spCSI_'+str(samplingedCSIWinSize)+('wakeIncluded' if wakeIncluded else 'wakeExcluded')+'.hdf5'

    if wakeIncluded:
        label_n = 4
    else:
        label_n = 3

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
    if True:#load file
        for fileIndx in range(len(csiFilePaths)+len(csiFilePathsAlt)):
            if(fileIndx<len(csiFilePaths)):
                csiFileName=csiFilePaths[fileIndx]
                sleepFileName=sleepFilePaths[fileIndx]
                isAlt=False
            else:
                csiFileName=csiFilePathsAlt[fileIndx-len(csiFilePaths)]
                sleepFileName=sleepFilePathsAlt[fileIndx-len(csiFilePaths)]
                isAlt=True
            print("fileIndx",fileIndx,csiFileName)
            csiList=[]
            sleepList=[]
            csiList.extend(pd.read_csv(path+csiFileName,delimiter=',',header=None).values)
            sleepList.extend(pd.read_csv(path+sleepFileName,delimiter=',',header=None).values)

            dataLooper=len(sleepList)
            dataStep=sleepWinSize
            csiStartIdx=0
            for i in range(0,dataLooper,dataStep):

                # get sleep stage in the time period
                # sleepIndices,startTime,endTime=poseIndices_sec(i,sleepList,sec=30)
                sleepIdx=i
                print("index",i,"/",dataLooper)
                print("sleepIdx",sleepIdx,'ts',sleepList[sleepIdx][0])

                # sleep stage matrix formation
                stage = False
                if wakeIncluded :
                    if(sleepList[sleepIdx][1] == 1):
                        stage = [1,0,0,0]
                    elif(sleepList[sleepIdx][1] == 2):
                        stage = [0,1,0,0]
                    elif(sleepList[sleepIdx][1] == 3):
                        stage = [0,0,1,0]
                    elif(sleepList[sleepIdx][1] == 4):
                        stage = [0,0,0,1]
                    if(stage==False):
                        continue
                    curSleeps = stage
                else :
                    if(sleepList[sleepIdx][1] == 1):
                        continue
                    elif(sleepList[sleepIdx][1] == 2):
                        stage = [1,0,0]
                    elif(sleepList[sleepIdx][1] == 3):
                        stage = [0,1,0]
                    elif(sleepList[sleepIdx][1] == 4):
                        stage = [0,0,1]
                    if(stage==False):
                        continue
                    curSleeps = stage

                # get CSI indices in the time period
                # way 1
                # startTime=sleepList[sleepIdx][0]-timeLen
                # endTime=sleepList[sleepIdx][0]
                # print("startTime",startTime)
                # print("endTime",endTime)
                # csiIndices=csiIndices_sec(startTime,endTime,csiList)
                # way 2
                csiIndices = sleepIdx2csiIndices_timestamp(sleepIdx, sleepList, csiStartIdx, csiList, timeLen=timeLen)
                print("len csiIndices",len(csiIndices))
                if (len(csiIndices)==0):
                  continue

                print(len(csiIndices),"csis to 1 SS")
                csiStartIdx = csiIndices[-1]
                print("csiIndices",csiIndices[0],"-",csiIndices[-1])

                # check if there is csi more than minCSIthreshold
                if(len(csiIndices)<minCSIthreshold):
                  print("too low csi number",len(csiIndices),minCSIthreshold)
                  continue

                # CSI matrix formation
                curCSIs,_=samplingCSISleep(csiList, csiIndices, sleepList, sleepIdx, samplingedCSIWinSize,timeLen=timeLen)

                if(isAlt==False):
                    X.append(curCSIs)
                    Y.append(curSleeps)
                else:
                    Xalt.append(curCSIs)
                    Yalt.append(curSleeps)

        X=np.array(X)
        Y=np.array(Y)
        print('shape X',(X.shape))
        print('shape Y',(Y.shape))

        Xalt=np.array(Xalt)
        Yalt=np.array(Yalt)

        print('shape Xalt',(Xalt.shape))
        print('shape Yalt',(Yalt.shape))

    if len(Xalt)==0:
      x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size=0.2, random_state=18)
    else:
      np.random.seed(42) 
      np.random.shuffle(X) 
      np.random.seed(42) 
      np.random.shuffle(Y) 
      x_train = X
      y_train = Y

      x_test = Xalt
      y_test = Yalt

    print('shape x_train',(x_train.shape))
    print('shape x_test',(x_test.shape))
    print('shape y_train',(y_train.shape))
    print('shape y_test',(y_test.shape))
    
    if runTrain:

        model = build_model(downsample=downsample,win_len=samplingedCSIWinSize*2,n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten,label_n=label_n)
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

    if runEval:
        # load the best model
        model = load_model(path+modelFileName)
        y_pred = model.predict(x_test)

        print("evaluate")
        matchCounter = 0
        stageTestCounter = [0,0,0,0]
        stagePredCounter = [0,0,0,0]

        wakePredCounter = [0,0,0,0]
        remPredCounter = [0,0,0,0]
        lightPredCounter = [0,0,0,0]
        deepPredCounter = [0,0,0,0]

        for i in range(0,len(y_test)):
          maximum_test = np.max(y_test[i])
          maximum_pred = np.max(y_pred[i])
          index_of_maximum_test = np.where(y_test[i] == maximum_test)
          index_of_maximum_pred = np.where(y_pred[i] == maximum_pred)
          curTest=index_of_maximum_test[0][0]
          curPred=index_of_maximum_pred[0][0]
          # print("curTest",curTest)
          # print("curPred",curPred)
          if(curTest==curPred):
            matchCounter = matchCounter+1

          stagePredCounter[curPred] = stagePredCounter[curPred] + 1
          stageTestCounter[curTest] = stageTestCounter[curTest] + 1 

          if wakeIncluded:
            if(curTest == 0):
                wakePredCounter[curPred] = wakePredCounter[curPred] + 1
            if(curTest == 1):
                remPredCounter[curPred] = remPredCounter[curPred] + 1
            elif(curTest == 2):
                lightPredCounter[curPred] = lightPredCounter[curPred] + 1
            elif(curTest == 3):
                deepPredCounter[curPred] = deepPredCounter[curPred] + 1
          else :
            if(curTest == 0):
                remPredCounter[curPred] = remPredCounter[curPred] + 1
            elif(curTest == 1):
                lightPredCounter[curPred] = lightPredCounter[curPred] + 1
            elif(curTest == 2):
                deepPredCounter[curPred] = deepPredCounter[curPred] + 1
          
        print("stagePredCounter",stagePredCounter)
        print("stageTestCounter",stageTestCounter)
        print("score",matchCounter,"/",len(y_test))
        print("score percent",matchCounter/len(y_test)*100,"/100")
        
        wakePredCounter = np.array(wakePredCounter) / stageTestCounter[0]
        remPredCounter = np.array(remPredCounter) / stageTestCounter[1]
        lightPredCounter = np.array(lightPredCounter) / stageTestCounter[2]
        deepPredCounter = np.array(deepPredCounter) / stageTestCounter[3]

        #toFixed 2
        wakePredCounter = np.around(wakePredCounter,decimals=2)
        remPredCounter = np.around(remPredCounter,decimals=2)
        lightPredCounter = np.around(lightPredCounter,decimals=2) 
        deepPredCounter = np.around(deepPredCounter,decimals=2)

        print("        wake rem light deep")
        print("wake  ",wakePredCounter)
        print("rem   ",remPredCounter)
        print("light ",lightPredCounter)
        print("deep  ",deepPredCounter)
