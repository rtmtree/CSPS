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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM,Conv1D,Conv2D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Input, Embedding, Concatenate
from modules.draw import Plotter3d, draw_poses
from functions.csi_util import rawCSItoAmp,filterNullSC,csiIndices_sec,poseIndices_sec,samplingCSI,featureEngineer,featureEngineerNorm
from functions.pose_util import poseToPAM,PAMtoPose,rotate_poses,getPCK
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import stdev
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from math import sqrt, atan2 ,isnan

def merge_csi_label(csifile, labelfile, win_len=1000, thrshd=0.6, step=200):
    """
    Merge CSV files into a Numpy Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 90)
    Args:
        csifile  :  str, csv file containing CSI data
        labelfile:  str, csv fiel with activity label 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    activity = []
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            label  = line[0]
            if label == 'NoActivity':
                activity.append(0)
            else:
                activity.append(1)
    activity = np.array(activity)
    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            line_array = np.array([float(v) for v in line])
            # extract the amplitude only
            line_array = line_array[1:91]
            csi.append(line_array[np.newaxis,...])
    csi = np.concatenate(csi, axis=0)
    assert(csi.shape[0] == activity.shape[0])
    # screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index+win_len]
        if np.sum(cur_activity)  <  thrshd * win_len:
            index += step
            continue
        cur_feature = np.zeros((1, win_len, 90))
        cur_feature[0] = csi[index:index+win_len, :]
        feature.append(cur_feature)
        index += step
    return np.concatenate(feature, axis=0)


def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=1000, thrshd=0.6, step=200):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_foler: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError("The label {} should be among 'bed','fall','pickup','run','sitdown','standup','walk'".format(labels))
    
    data_path_pattern = os.path.join(raw_folder, 'input_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    annot_csv_files = [os.path.basename(fname).replace('input_', 'annotation_') for fname in input_csv_files]
    annot_csv_files = [os.path.join(raw_folder, fname) for fname in annot_csv_files]
    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100,label))
    
    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}_win_{}_thrshd_{}percent_step_{}.npz".format(
            label, win_len, int(thrshd*100), step), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def train_valid_split(numpy_tuple, train_portion=0.9, seed=379):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        split_len = int(train_portion * x_arr.shape[0])
        x_train.append(x_arr[index[:split_len], ...])
        tmpy = np.zeros((split_len,7))
        tmpy[:, i] = 1
        y_train.append(tmpy)
        x_valid.append(x_arr[index[split_len:],...])
        tmpy = np.zeros((x_arr.shape[0]-split_len,7))
        tmpy[:, i] = 1
        y_valid.append(tmpy)
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index = np.random.permutation([i for i in range(x_train.shape[0])])
    x_train = x_train[index, ...]
    y_train = y_train[index, ...]
    return x_train, y_train, x_valid, y_valid
    
    

def extract_csi(raw_folder, labels, save=False, win_len=1000, thrshd=0.6, step=200):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array 
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)


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

    def preprocessing(self, raw_folder, save=False):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            raw_folder: the folder containing raw CSI 
            save      : choose if save the numpy array
        """
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        if self._downsample > 1:
            return tuple([v[:, ::self._downsample,...] if i%2 ==0 else v for i, v in enumerate(numpy_tuple)])
        return numpy_tuple
    
    def load_csi_data_from_files(self, np_files):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            np_files: ('x_bed.npz', 'x_fall.npz', 'x_pickup.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz')
        """
        if len(np_files) != 7:
            raise ValueError('There should be 7 numpy files for bed, fall, pickup, run, sitdown, standup, walk.')
        x = [np.load(f)['arr_0'] for f in np_files]
        if self._downsample > 1:
            x = [arr[:,::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:,i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)


    
    def build_model(self, n_unit_lstm=200, n_unit_atten=400):
        """
        Returns the Tensorflow Model which uses AttenLayer
        """
        # input_len=90
        input_len=52
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
            x_in = tf.keras.Input(shape=(length, input_len))
        else:
            x_in = tf.keras.Input(shape=(self._win_len, input_len))
        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True))(x_in)
        x_tensor = AttenLayer(n_unit_atten)(x_tensor)
        # pred = tf.keras.layers.Dense(len(self._labels), activation='softmax')(x_tensor)
        pred = tf.keras.layers.Dense(200 * 3 * 19 * 19, activation='softmax')(x_tensor)
        model = tf.keras.Model(inputs=x_in, outputs=pred)
        return model
    
    
    def build_model_rath(self, n_unit_lstm=200, n_unit_atten=400,seqLen=30):

        input_len=52   
        model = tf.keras.Sequential()

        #csi feature extractor

        # model.add(Conv2D(320, 2, activation='relu',input_shape=(seqLen,input_len)))
        # model.add(Conv2D(150, 2, activation='relu',input_shape=(seqLen,input_len)))
        # model.add(Conv2D(300, 2, activation='relu',input_shape=(seqLen,input_len)))
        # model.add(Conv2D(300, 2, activation='relu',input_shape=(seqLen,input_len)))


        # encoder layer
        model.add(Bidirectional(LSTM(100, activation='relu', input_shape=(seqLen,input_len ))))

        # repeat vector
        # model.add(RepeatVector(3*3*19*19))
        model.add(RepeatVector(seqLen))
        # model.add(RepeatVector(19*19))

        # decoder layer
        model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))

        model.add(TimeDistributed(Dense(1083)))

        return model
    def build_model_rath0(self, n_unit_lstm=200, n_unit_atten=400,seqLen=30):
        from tensorflow.keras import backend as K 
        K.clear_session()
        max_text_len=80
        latent_dim = 200
        embedding_dim=110

        # Encoder
        encoder_inputs = Input(shape=(max_text_len,))

        #embedding layer
        enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

        #encoder lstm 1
        encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        #encoder lstm 2
        encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        #encoder lstm 3
        encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))

        #embedding layer
        dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
        decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

        # Attention layer
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention input and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        #dense layer
        decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Define the model 
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.summary()
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
 

    # labels=['02','03','04','05','07','09','10','11']
    labels=['23','25','27','22']
    # labels=['20','21']
    labelsAlt=['20']
    # labels=['02','03','04']
    # labelsAlt=['02']
    # isActSDthreshold=150
    isActSDthreshold=50
    # isActSDthreshold=20
    sliceData=True
    seqLen=15
    epoch=1000


    # path = 'drive/MyDrive/Project/'
    path = ''
    minCSIthreshold= int((seqLen/30) * 80)
    runTrain = True

    realData = True
    batch_size = None
    runFeatureEngineer=True

    if realData:
        
        runPCK = True
        runPlot = True
        checkIndex = 0
        modelFileName='test_models/model_01_e' \
        +str(epoch)+'_Actthes_'+str(isActSDthreshold) \
        +'_seqLen_'+str(seqLen) \
        +'_'+('FE' if runFeatureEngineer else 'NoFE') \
        +'_'+('SD' if sliceData else 'NoSD') \
        +'_'+'.hdf5'
    else:
        runPCK = False
        runPlot = False
        modelFileName="test.hdf5"

    # preprocessing
    csiFilePaths=[]
    poseFilePaths=[]
    csiFilePathsAlt=[]
    poseFilePathsAlt=[]
    for label in labels:
        csiFilePaths.append('data/parsedCSI'+label+'.csv')
        poseFilePaths.append('data/parsedPose3D'+label+'.csv')
    for label in labelsAlt:
        csiFilePathsAlt.append('data/parsedCSI'+label+'.csv')
        poseFilePathsAlt.append('data/parsedPose3D'+label+'.csv')
    X = []
    Y = []
    Xalt = []
    Yalt = []
    if realData:#load file
        for label in (labels)+(labelsAlt):
            if(label in labels):
                isAlt=False
            else:
                isAlt=True
            csiFileName=('data/parsedCSI'+label+'.csv')
            poseFileName=('data/parsedPose3D'+label+'.csv')
            csiList=[]
            poseList=[]
            print(isAlt)
            print(csiFileName)
            print(poseFileName)
            csiList.extend(pd.read_csv(path+csiFileName,delimiter=',',header=None).values)
            poseList.extend(pd.read_csv(path+poseFileName,delimiter=',',header=None).values)

            # guassian filter
            if False:
                tempCsiList=[]
                for i in range(0,64):
                    # if (6<=i<32 or 33<=i<59):
                    subClist=[]
                    for j in range(0,len(csiList)):
                        subClist.append(csiList[j][i])
                    tempCsiList.append(gaussian_filter(subClist,sigma=2))
                
                for i in range(0,len(csiList)):
                    for j in range(0,64):
                        csiList[i][j]=tempCsiList[j][i]

                print(len(csiList))
            # guassian filter End
            if(sliceData):
                dataLooper=len(poseList)-seqLen+1
                dataStep=1
            else:
                dataLooper=int(len(poseList)/seqLen)
                dataStep=seqLen
            for i in range(0,dataLooper,dataStep):

                # get pose in the time period
                poseIndices,startTime,endTime=poseIndices_sec(i,poseList,seqLen=seqLen)
                # print("poseIndices",poseIndices)

                # check if there is pose less than seqLen
                if(len(poseIndices)<seqLen):
                    # print("too low pose number")
                    continue

                # check if there is some null in pose
                isNullPose=False
                for j in poseIndices:
                    # print(label,poseList[j])
                    if(poseList[j].all()==0):
                        print(label,"sq",i,"null pose at index",j)
                        isNullPose=True
                        break
                if(isNullPose):
                    continue

                csiIndices=csiIndices_sec(startTime,endTime,csiList)
                # print("csiIndices",csiIndices)

                # check if there is csi more than minCSIthreshold
                if(len(csiIndices)<minCSIthreshold):
                    print(label,"too low csi number",len(csiIndices))
                    continue


                # check if there is some null in csi
                # isNullCSI=False
                # for j in csiIndices:
                #     if(csiList[j].all()==0):
                #         print("sq",i,"null csi at index",j)
                #         isNullCSI=True
                #         break
                # if(isNullCSI):
                #     continue
                
                print(label,'sq',i,'pass null filter')
                print(label,len(csiIndices),"csis to ",len(poseIndices),"poses")

                # check if this seqence has an activity
                isAct=True
                # check with raw amplitude
                # normalAmp=[filterNullSC( rawCSItoAmp(   csiList[j][1:]  )  )    for j in csiIndices]
                # check with samplinged amplitude
                normalAmp,_=samplingCSI(csiList,csiIndices,poseList,poseIndices,paddingTo=seqLen)

                sdSum=0
                for j in range(0,52):
                    subClist=[]
                    for k in range(len(normalAmp)):
                        subClist.append( normalAmp[k][j] )
                    sdAmp=stdev(subClist)
                    sdSum+=sdAmp
                print(label,"sum_SD sq",i,'is',sdSum)
                if(isActSDthreshold > sdSum):
                    print(label,"no activity")
                    isAct=False
                    continue
                else:
                    print(label,"activity!!!")
                    isAct=True
                
                # filtered only valid annotation
                if(isAct):
                    # curCSIs,_=samplingCSI(csiList,csiIndices,poseList,poseIndices,paddingTo=seqLen)
                    curCSIs = normalAmp

                    # reshape all pose to vector and to PAM and to vector again
                    curposes = [ poseToPAM(  np.array( [ poseList[j][1:].reshape(19,3) ] )  ) for j in poseIndices ]
                    # curposes = [  np.array(poseList[j])   for j in poseIndices ]
                    
                    if len(curposes)!= seqLen:
                        print(label,"invalid len at ",label,poseIndices)
                        break
                    isValid = True
                    for j in range(len(curposes)):
                        # print((curposes[j].shape))
                        if (curposes[j].shape)!= (3,19,19):
                            print(label,"invalid len at ",label,poseIndices,j)
                            isValid=False
                            break
                    if(isValid==False):
                        break

                    print(label,"Before Add")
                    print('isAlt',isAlt,'curCSIs',len(curCSIs),'curposes',len(curposes))
                    if(isAlt==False):
                        X.append(curCSIs)
                        Y.append(curposes)
                    else:
                        Xalt.append(curCSIs)
                        Yalt.append(curposes)

        X=np.array(X)
        Y=np.array(Y)
        print('shape X',(X.shape))
        print('shape Y',(Y.shape))
        X =(X).reshape(X.shape[0], seqLen, 52)
        Y =(Y).reshape(Y.shape[0], seqLen, 3 * 19 * 19)
        print('shape X',(X.shape))
        print('shape Y',(Y.shape))
        Xalt=np.array(Xalt)
        Yalt=np.array(Yalt)
        Xalt =(Xalt).reshape(Xalt.shape[0], seqLen, 52)
        Yalt =(Yalt).reshape(Yalt.shape[0], seqLen, 3 * 19 * 19)
        print('shape Xalt',(Xalt.shape))
        print('shape Yalt',(Yalt.shape))

        # print("X")
        # print(X)
        # print("Y")
        # print(Y)

    else:#load fake data
        X=[]
        Y=[]
        for i in range(1,1001):
            xSeq=[]
            ySeq=[]
            randd=np.random.randint(100)
            for j in range(1,seqLen+1):
                randd2=np.random.randint(100)
                if(j==1):
                    xSeq.append(np.ones((52))*(i*(j))+randd)
                    # ySeq.append(np.ones((1083))*(i*(j))+(randd-5)+(randd2*2))
                    # ySeq.append(np.ones((1083))*(i*(j))+(randd2*2))
                    ySeq.append(np.ones((1083))*(i*(j))+77)
                else:
                    xSeq.append(xSeq[-1]+randd2)
                    ySeq.append(ySeq[-1]+(randd2*2))
            xSeq=np.array(xSeq)
            ySeq=np.array(ySeq)
            X.append(xSeq) 
            Y.append(ySeq) 
        print('len X',len(X))
        print('len Y',len(Y))

        for i in range(len(X)):
            print(i,len(X[i]),"csi to ",len(Y[i]),"pose")
        X=np.array(X)
        Y=np.array(Y)
        print('shape X',(X.shape))
        print('shape Y',(Y.shape))

        print("X")
        print(X)
        print("Y")
        print(Y)

    if runFeatureEngineer:
        X=featureEngineer(X)
        Xalt=featureEngineer(Xalt)

        # print("normalized X")
        # print(X)
        # print("normalized Y")
        # print(Y)
    if True:
        random_state=3
        if len(labelsAlt)==0:
            x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size=0.1, random_state=random_state)
        else:
            np.random.seed(random_state) 
            np.random.shuffle(X) 
            np.random.seed(random_state) 
            np.random.shuffle(Y) 
            np.random.seed(random_state) 
            np.random.shuffle(Xalt) 
            np.random.seed(random_state) 
            np.random.shuffle(Yalt) 
            x_train = X
            y_train = Y
            x_test = Xalt
            y_test = Yalt

        # X=np.concatenate((X, Xalt))
        # Y=np.concatenate((Y, Yalt))

        # x_train,_,y_train,_=train_test_split(X, Y, test_size=0.01, random_state=3)
        # x_test,_,y_test,_=train_test_split(Xalt, Yalt, test_size=0.01, random_state=3)        

        
        cfg = CSIModelConfig(win_len=1000, step=2000, thrshd=0.6, downsample=2)

        print('shape x_train',(x_train.shape))
        print('shape x_test',(x_test.shape))
    if runTrain:


        # parameters for Deep Learning Model
        # model = cfg.build_model(n_unit_lstm=200, n_unit_atten=400)
        model = cfg.build_model_rath(n_unit_lstm=200, n_unit_atten=400,seqLen=seqLen)

        # model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
        # history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

        # train
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        #     loss='categorical_crossentropy', 
        #     metrics=['accuracy'])
        
        # model.fit(
        #     x_train,
        #     y_train,
        #     batch_size=128, epochs=60,
        #     validation_data=(x_valid, y_valid),
        #     callbacks=[
        #         tf.keras.callbacks.ModelCheckpoint('best_atten.hdf5',
        #                                             monitor='val_accuracy',
        #                                             save_best_only=True,
        #                                             save_weights_only=False)
        #         ])
        model.compile(optimizer='adam', loss='mse')
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
        model.fit(x_train, y_train, epochs=epoch, validation_split=0.2, verbose=1, batch_size=batch_size
        )
        print(model.summary())

        model.save(path+modelFileName)

    if True:
        # load the best model
        model = cfg.load_model(path+modelFileName)

        print(x_test.shape)
        # print("x_test[0]")
        # print(x_test[0])
        # print("x_test[1]")
        # print(x_test[1])
        # y_pred = model.predict(x_test, verbose=0)
        y_pred = model.predict(x_test)
        # print("y_test[0]")
        # print(y_test[0])
        # print("y_test[1]")
        # print(y_test[1])
        # print("y_pred[0]")
        # print(y_pred[0])
        # print("y_pred[1]")
        # print(y_pred[1])

    if runPCK: # calculate PCK
        print(y_pred.shape)
        PCKthres=[5,10,20,30,40,50]
        PCKAll=[]
        # for predIdx in range(y_pred.shape[0]):
        for predIdx in range(y_pred.shape[0]):
            if(y_test[predIdx].all!=0):
                PCKresult=[]
                # print('PCK idx',str(predIdx))
                for threshold in PCKthres:
                    PCKkpsum=0
                    # print('thres',str(threshold))
                    for keypointIdx in range(19):
                        PCK=getPCK(y_pred[predIdx],y_test[predIdx],keypointIdx,frame=seqLen,threshold=threshold)
                        PCKkpsum+=PCK
                        # print(PCK)
                    PCKkpsum=PCKkpsum/19
                    PCKresult.append(PCKkpsum)
                print('PCK avg idx',str(predIdx))
                print(PCKresult)
                PCKAll.append(PCKresult)
        for thresholdIndex in range(len(PCKthres)):
            print('avg '+str(PCKthres[thresholdIndex]))
            sum=0
            for each in PCKAll:
                sum+=each[thresholdIndex]
            print(sum/len(PCKAll))
        print(modelFileName)
        print(labels)
        print(labelsAlt)

    if runPlot: # plot pose3D

        fig = plt.figure()
        gs = gridspec.GridSpec(2,2)
        ax0=fig.add_subplot(gs[0,0])
        ax1=fig.add_subplot(gs[0,1])
        ax2=fig.add_subplot(gs[1,:])
        x_values=[[] for i in range(64)]
        y_values=[[] for i in range(64)]

        def updatefig(i):

            global y_pred
            global y_test
            global checkIndex
            # print("updatefig",i)
            i=i%seqLen
            if i==seqLen-1:
                checkIndex=checkIndex+1
            if checkIndex==len(y_pred):
                checkIndex=0
            ax0.cla()
            ax1.cla()
            ax2.cla()

            poseIdx=i

            # if (poseIdx == len(csiList)-1):
            #     print('close')
            #     plt.close(fig)

            csiIdx=i

            
            # Plot 3D Pose
            poses_3dFromImage=PAMtoPose(y_pred[checkIndex][poseIdx].reshape(3,19,19))

            edgesFromImage = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3dFromImage.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            canvas_3d = np.zeros((450, 450, 3), dtype=np.uint8)
            plotter = Plotter3d(canvas_3d.shape[:2])
            plotter.plot(canvas_3d, poses_3dFromImage, edgesFromImage)
            ax0.imshow(canvas_3d)

            poses_3dFromImageGT=PAMtoPose(y_test[checkIndex][poseIdx].reshape(3,19,19))
            # poses_3dFromImageGT[0][9]=np.array([ 0,0,0]) # right shoulder
            # poses_3dFromImageGT[0][6]=np.array([ 0,0,0]) # left hip

            edgesFromImageGT = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3dFromImageGT.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            canvas_3dGT = np.zeros((450, 450, 3), dtype=np.uint8)
            plotterGT = Plotter3d(canvas_3dGT.shape[:2])
            plotterGT.plot(canvas_3dGT, poses_3dFromImageGT, edgesFromImageGT)
            ax1.imshow(canvas_3dGT)

            # Plot CSI
            for j in range(0,52):
                textX=[]
                textY=[]
                for k in range(len(x_test[checkIndex])):
                    curCsi=x_test[checkIndex][k]
                    textX.append(k)
                    textY.append(curCsi[j])
                ax2.plot(textX,gaussian_filter(textY,sigma=0), label='CSI subcarrier')

            return ax0,ax1,ax2 


        ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
        plt.show()