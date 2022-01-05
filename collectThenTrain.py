import numpy as np
import json
from modules.draw import Plotter3d, draw_poses
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import re
from scipy.ndimage import gaussian_filter
from math import sqrt, atan2, isnan, floor
import matplotlib.gridspec as gridspec
import glob
from datetime import datetime,timedelta
import json
import imageio
import cv2
from modules.inference_engine_pytorch import InferenceEnginePyTorch

# train stuffs
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM, Conv1D, Conv2D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Input, Embedding, Concatenate
from modules.draw import Plotter3d, draw_poses
from functions.csi_util import rawCSItoAmp, filterNullSC, csiIndices_sec, poseIndices_sec, samplingCSISleep, sleepIdx2csiIndices_timestamp
from functions.pose_util import poseToPAM, PAMtoPose, rotate_poses, getPCK
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import stdev
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from math import sqrt, atan2, isnan


def rawCSItoAmp(data, len=128):
    if(data == False):
        return False
    amplitudes = []
    for j in range(0, 128, 2):
        amplitudes.append(sqrt(data[j] ** 2 + data[j+1] ** 2))
    return amplitudes


def parseCSI(csi):
    try:
        csi_string = re.findall(r"\[(.*)\]", csi)[0]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
        return csi_raw
    except:
        return False


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def reshape_poses(rotatedPose):
    poses_3d_copy = rotatedPose.copy()
    x = poses_3d_copy[:, 0::4]
    y = poses_3d_copy[:, 1::4]
    z = poses_3d_copy[:, 2::4]
    rotatedPose[:, 0::4], rotatedPose[:,
                                      1::4], rotatedPose[:, 2::4] = -z, x, -y
    rotatedPose = rotatedPose.reshape(rotatedPose.shape[0], 19, -1)[:, :, 0:3]

    return rotatedPose


def imageIdx2csiIdx(durationSec, imageIdx, tsList, fps):
    offsetTime = (tsList[len(tsList)-1]/(10**6)) - durationSec

    # print("offsetTime",offsetTime)
    timeInVid = (imageIdx/vidLength) * durationSec
    # print("time in vid",timeInVid)

    timeInCSI = (timeInVid-offsetTime)
    # print("timeInCSI",timeInCSI)

    csiIndex = min(range(len(tsList)), key=lambda i: abs(
        tsList[i]-(timeInCSI*(10**6))))
    # print("csiIndex",csiIndex)
    # print("csiIndex exp",timeInCSI)
    # print("csiIndex real",tsList[csiIndex]/(10**6))
    # if(np.abs(tsList[csiIndex]-timeInCSI)<100000):
    if(True):
        return csiIndex
    else:
        return False


def imageIdx2csiIndicesPrecise(durationSec, imageIdx, tsList, vidLength, lastsec):
    durationMicroSec = durationSec*(10**6)
    offsetTime = lastsec - durationMicroSec
    print("last CSI ts", (lastsec))
    print("offsetTime", offsetTime)
    timeInVid = ((imageIdx+1)/vidLength) * durationMicroSec
    prevTimeInVid = ((imageIdx)/vidLength) * durationMicroSec

    prevParsedTimeInVid = prevTimeInVid + offsetTime
    parsedTimeInVid = timeInVid + offsetTime

    print("prevTimeInVid", (prevTimeInVid))
    print("timeInVid", (timeInVid))
    print("prevParsedTimeInVid", prevParsedTimeInVid)
    print("parsedTimeInVid", (parsedTimeInVid))
    csiIndices = []
    for i in range(len(tsList)):
        if(prevParsedTimeInVid < tsList[i] and tsList[i] <= parsedTimeInVid):
            csiIndices.append(i)

    return csiIndices, parsedTimeInVid


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
        self.kernel = self.add_weight(
            'kernel', shape=[input_shape[-1], self.num_state])
        self.bias = self.add_weight('bias', shape=[self.num_state])
        self.prob_kernel = self.add_weight(
            'prob_kernel', shape=[self.num_state])

    def call(self, input_tensor):
        atten_state = tf.tanh(tf.tensordot(
            input_tensor, self.kernel, axes=1) + self.bias)
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
        prob = tf.nn.softmax(logits)
        weighted_feature = tf.reduce_sum(tf.multiply(
            input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature

    # for saving the model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state, })
        return config


def build_model(downsample=1, win_len=1000, n_unit_lstm=200, n_unit_atten=400, label_n=2):
    """
    Returns the Tensorflow Model which uses AttenLayer
    """
    if downsample > 1:
        length = len(np.ones((win_len,))[::downsample])
        x_in = tf.keras.Input(shape=(length, 52))
    else:
        x_in = tf.keras.Input(shape=(win_len, 52))
    x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=n_unit_lstm, return_sequences=True))(x_in)
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
    model = tf.keras.models.load_model(hdf5path, custom_objects={
                                       'AttenLayer': AttenLayer})
    return model


path = 'drive/MyDrive/Project/'
# path=''
labels = [
     'sleep30-11-2021end1020',
      'sleep08-12-2021end1000','sleep11-12-2021end0930',
      'sleep12-12-2021end1000','sleep13-12-2021end1010',
      'sleep14-12-2021end1200','sleep15-12-2021end1220',
      'sleep16-12-2021end1210',
    'sleep2022-01-04start0336',
    # 'sleep2022-01-03start0233',
    'sleep2022-01-02start0236',
    'sleep2022-01-01start0247',
    'sleep2021-12-30start0315',
    'sleep2021-12-29start0515',
    'sleep2021-12-29start0002',
    'sleep2021-12-27start0334'
]
labelsAlt = [
  'sleep2022-01-03start0233'
  #  'sleep2022-01-01start0247'
    # 'sleep2021-12-30start0315'
    # 'sleep2021-12-29start0515'
    # 'sleep2021-12-27start0334'
    # 'sleep2021-12-29start0002'
]

wakeIncluded = False
batch_size = 128
sleepWinSize = 1  # sigma
samplingedCSIWinSize = 900  # delta
epoch = 300
n_unit_lstm = 200
n_unit_atten = 400
downsample = 2

justCollect = False
runTrain = True
runEval = True

# ========== adjustable end
Sleeplabels = []
isTestLabels = []
sleepIndxs = []
validLenCounter = []
for label in labels:
    Sleeplabels.append(label)
    isTestLabels.append(False)
    sleepIndxs.append(0)
for label in labelsAlt:
    Sleeplabels.append(label)
    isTestLabels.append(True)
    sleepIndxs.append(0)
CSIlabels = Sleeplabels
minCSIthreshold = samplingedCSIWinSize
modelFileName = 'test_models/csi2sleepM_e'+str(epoch)+'spCSI_'+str(
    samplingedCSIWinSize)+('wakeIncluded' if wakeIncluded else 'wakeExcluded')+'.hdf5'

if wakeIncluded:
    label_n = 4
else:
    label_n = 3


# timestampColName = 'local_timestamp'
timestampColName = 'real_timestamp'
timeperoid = 30
decimalShiftTs = 0

collectedCSI = []
collectedSS = []
X = []
Y = []
Xalt = []
Yalt = []
for fileIdx in range(len(CSIlabels)):
    # Organize sleep log file
    thisLabel = CSIlabels[fileIdx]
    thisSync = thisLabel[15:17]
    if thisSync == 'st':
        filePath = path+'raw_data/'+CSIlabels[fileIdx]
    else:
        filePath = path+'raw_data/old/'+CSIlabels[fileIdx]
    print("read sleep file", filePath)
    f = open(filePath+'.json')
    data = json.load(f)
    sleepSData = data['sleep'][sleepIndxs[fileIdx]]['levels']['data']

    # find sleep length

    # get first ts of sleep stage log
    firstsecSleep = sleepSData[0]['dateTime']
    print('first date/time in Sleep is', firstsecSleep)
    futc_time = datetime.strptime(firstsecSleep, "%Y-%m-%dT%H:%M:%S.%f")
    sleepFirstTs = (futc_time - datetime(1970, 1, 1)).total_seconds()
    print('first timestamp in Sleep is', sleepFirstTs)

    # get last ts of real world ts CSI log
    lastsecSleep = sleepSData[-1]['dateTime']
    print('last date/time in Sleep is', lastsecSleep)
    lutc_time = datetime.strptime(lastsecSleep, "%Y-%m-%dT%H:%M:%S.%f")
    sleepLastTs = (lutc_time - datetime(1970, 1, 1)
                   ).total_seconds() + sleepSData[-1]['seconds']
    print('last timestamp in Sleep is', sleepLastTs)

    vidLength = int((sleepLastTs-sleepFirstTs)/30)
    print('Sleep time Length', vidLength)

    # Organize CSI log file
    print("read CSI file", filePath)
    colsName = ["type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
                "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "real_time_set", "real_timestamp", "len", "CSI_DATA"]
    curFile = pd.read_csv(filePath+'.csv', names=colsName)

    # filter out the rows unvalid timestamp. CLEAN Data
    curFile[timestampColName] = curFile[timestampColName].astype('str')
    print("len file bf", len(curFile.index))
    curFile = curFile[curFile[timestampColName].str.match(
        r"[+-]?([0-9]*[.])?[0-9]+") == True]
    curFile = curFile[curFile[timestampColName].str.contains(
        '[A-Za-z :]') == False]
    curFile = curFile[curFile[timestampColName].str.contains("[.]") == True]
    # curFile = curFile[curFile[timestampColName].str.contains(":")==False]

    print("len file af", len(curFile.index))
    curFile[timestampColName] = curFile[timestampColName].astype('float')
    curFile = curFile.sort_values(timestampColName)

    # print(curFile)

    # find diffEpoch to sync time
    tsList = (list(x/(10**decimalShiftTs) for x in curFile[timestampColName]))
    
    
    if thisSync == 'st':
        CSIsyncTime = thisLabel[5:15]+'T'+thisLabel[-4:-2]+':'+thisLabel[-2:]
        # get first ts of real world ts CSI log
        csiRealFirstTs = CSIsyncTime+':01.000'
        print('first real world date/time in CSI is', csiRealFirstTs)
        utc_time = datetime.strptime(csiRealFirstTs, "%Y-%m-%dT%H:%M:%S.%f")
        realFirstTs = (utc_time - datetime(1970, 1, 1)).total_seconds()
        print('last real world timestamp in CSI is', realFirstTs)
        diffEpoch = realFirstTs
    else:
        CSIsyncTime = thisLabel[11:15]+'-'+thisLabel[8:10]+'-'+thisLabel[5:7]+'T'+thisLabel[-4:-2]+':'+thisLabel[-2:]
        # get last ts of real world ts CSI log
        csiRealLastTs = CSIsyncTime+':00.000'
        print('last real world date/time in CSI is', csiRealLastTs)
        utc_time = datetime.strptime(csiRealLastTs, "%Y-%m-%dT%H:%M:%S.%f")
        utc_time += timedelta(days=1)
        realLastTs = (utc_time - datetime(1970, 1, 1)).total_seconds()
        print('last real world timestamp in CSI is', realLastTs)
        # get last ts of CSI log
        csiLastTs = tsList[-1]
        print('last self timestamp in CSI is', csiLastTs)
        diffEpoch = realLastTs - csiLastTs
    print("diffEpoch is", diffEpoch)

    # filter for usable CSI data
    # my_filter_address="7C:9E:BD:D2:D8:9C"
    my_filter_address = "98:F4:AB:7D:DD:1D"
    curFile = curFile[(curFile['mac'] == my_filter_address)]
    curFile = curFile[(curFile['len'] == 384)]
    curFile = curFile[(curFile['stbc'] == 0)]
    curFile = curFile[(curFile['rx_state'] == 0)]
    curFile = curFile[(curFile['sig_mode'] == 1)]
    curFile = curFile[(curFile['bandwidth'] == 1)]
    curFile = curFile[(curFile['secondary_channel'] == 1)]
    print("filtering CSI done")
    print("len CSI", len(curFile.index))
    # curCSI = curFile['CSI_DATA']
    # csiList = list(rawCSItoAmp(parseCSI(x),128) for x in curCSI)
    # csiList = list(x for x in curCSI)
    # curRSSI = curFile['rssi']
    # rssiList=list(x for x in curRSSI)

    ss_value = []
    csi_value = []

    def updatefig(i):
        # global realLastTs
        global sleepSData
        global sleepFirstTs
        global curFile
        global diffEpoch

        sleepIdx = i+startFrom

        sleepTs = int((sleepIdx*30)+sleepFirstTs)
        print("sleepTs", sleepTs)
        # if(sleepTs>realLastTs):
        #   print("Exceed realLastTs. DONE!")
        #   return False

        stage = 0
        for j in range(len(sleepSData)):
            # print(sleepSData[j])
            tutc_time = datetime.strptime(
                sleepSData[j]['dateTime'], "%Y-%m-%dT%H:%M:%S.%f")
            tepoch_time = int(
                (tutc_time - datetime(1970, 1, 1)).total_seconds())
            # print(int(sleepTs))
            # print(int(tepoch_time))
            # print(int(sleepTs)>=int(tepoch_time))
            if((sleepTs) >= (tepoch_time)):
                if(sleepSData[j]['level'] == "wake"):
                    stage = 1
                elif(sleepSData[j]['level'] == "rem"):
                    stage = 2
                elif(sleepSData[j]['level'] == "light"):
                    stage = 3
                elif(sleepSData[j]['level'] == "deep"):
                    stage = 4
            else:
                # print("stage",stage)
                break
        if(stage == 0):
            print("catch no sleep stage detected")
            return False
        ss_value.append([sleepTs]+[stage])

        if(True):
            # Setting the values for all axes.
            # csiIndices,parsedTimeInVid=imageIdx2csiIndicesPrecise(duration_in_sec,imageIdx,tsList,vidLength,lastsec)
            csiTsEnd = (sleepTs - diffEpoch)
            csiTs = csiTsEnd-timeperoid
            # parse to microsecond
            csiTsEnd = (10**decimalShiftTs) * csiTsEnd
            csiTs = (10**decimalShiftTs) * csiTs

            print("csiTs", csiTs)
            print("csiTsEnd", csiTsEnd)

            # dataInPeriod = curFile[(curFile[timestampColName]>=csiTs)]
            # dataInPeriod = dataInPeriod[(dataInPeriod[timestampColName]<csiTsEnd)]

            dataInPeriod = curFile[(curFile[timestampColName] >= csiTs) & (
                curFile[timestampColName] < csiTsEnd)]

            print("len csidataInPeriod", len(dataInPeriod.index))
            tsInPeriod = list(
                (x/(10**decimalShiftTs))+diffEpoch for x in dataInPeriod[timestampColName])
            csiInPeriod = list(parseCSI(x)
                               for x in dataInPeriod['CSI_DATA'])
            if(len(csiInPeriod) > 0):
                for k in range(len(csiInPeriod)):
                    curParseCSI = (csiInPeriod[k])
                    curParseTs = tsInPeriod[k]
                    if(k > 0 and curParseCSI == csiInPeriod[k-1] and curParseTs == tsInPeriod[k-1]):
                        # print("duplicate CSI row found. SKIP")
                        continue
                    # print("adding ",curParseCSI)
                    if(curParseCSI != False):
                        # print("len check")
                        # print(k,len(curParseCSI),curParseTs)
                        if(len(curParseCSI) != 384):
                            print("len not 384")
                            continue
                        # print("isFloat check")
                        isInt = True
                        for l in range(384):
                            if(isinstance(curParseCSI[l], int) == False):
                                # print(curParseCSI[l]," is not int")
                                isInt = False
                                break
                        if isInt == False:
                            continue
                        csi_value.append([curParseTs]+curParseCSI)
                        # print("added ",len(curParseCSI))
                    else:
                        continue
                        # csi_value.append([curParseTs]+[0 for l in range(384)])
                        # print("added ",k,'as 0s')
            # print("====",i,"====")
            return False

    startFrom = 0
    collectLength = vidLength-startFrom
    print('startFrom', startFrom)
    print('length', collectLength)
    for i in range(collectLength):
        updatefig(i)
        print("collecting ====", i, "/", collectLength,
              "====", fileIdx, "/", len(CSIlabels))

    print('collection finishing', CSIlabels[fileIdx])

    csi_value = np.array(csi_value)
    ss_value = np.array(ss_value)
    print(csi_value.shape)
    print(ss_value.shape)
    if(False):
        savePath = 'sample_data/'
        # savePath='drive/MyDrive/Project/data'
        pathSavedFileCSI = savePath+'CSI'+CSIlabels[fileIdx]+'.csv'
        pathSavedFileSleep = savePath+'SS'+CSIlabels[fileIdx]+'.csv'
        # pathSavedFileCSI = 'CSI'+labels[fileIdx]+'.csv'
        # pathSavedFileSleep = 'SS'+labels[fileIdx]+'.csv'
        fmt = '%1.6f,'+('%d,'*383)+'%d'
        np.savetxt(pathSavedFileCSI, csi_value, delimiter=',', fmt=fmt)
        print('saved', pathSavedFileCSI)
        np.savetxt(pathSavedFileSleep, ss_value, delimiter=',', fmt='%d')
        print('saved', pathSavedFileSleep)
    # else:
    #     collectedCSI.append(csi_value)
    #     collectedSS.append(ss_value)
    
    if (justCollect == False):
        # X = []
        # Y = []
        # Xalt = []
        # Yalt = []
        # for colIndx in range(len(collectedCSI)):
        # csiList = collectedCSI[colIndx]
        # sleepList = collectedSS[colIndx]

        dataLooper = len(ss_value)
        dataStep = sleepWinSize
        csiStartIdx = 0
        validLenCounter.append(0)
        for i in range(0, dataLooper, dataStep):

            # get sleep stage in the time period
            # sleepIndices,startTime,endTime=poseIndices_sec(i,collectedSS[colIndx],sec=30)
            sleepIdx = i
            print("training ====", i, "/", dataLooper,
                  "====", fileIdx, "/", len(CSIlabels))
            print("sleepIdx", sleepIdx, 'ts', ss_value[sleepIdx][0])

            # sleep stage matrix formation
            curSleeps = False
            if wakeIncluded == True:
                if(ss_value[sleepIdx][1] == 1):
                    curSleeps = [1, 0, 0, 0]
                elif(ss_value[sleepIdx][1] == 2):
                    curSleeps = [0, 1, 0, 0]
                elif(ss_value[sleepIdx][1] == 3):
                    curSleeps = [0, 0, 1, 0]
                elif(ss_value[sleepIdx][1] == 4):
                    curSleeps = [0, 0, 0, 1]
            else:
                if(ss_value[sleepIdx][1] == 1):
                    continue
                elif(ss_value[sleepIdx][1] == 2):
                    curSleeps = [1, 0, 0]
                elif(ss_value[sleepIdx][1] == 3):
                    curSleeps = [0, 1, 0]
                elif(ss_value[sleepIdx][1] == 4):
                    curSleeps = [0, 0, 1]

            if(curSleeps == False):
                print("invalid sleep stage", curSleeps)
                continue

            # get CSI indices in the time period
            # way 1
            # startTime=ss_value[sleepIdx][0]-timeperoid
            # endTime=ss_value[sleepIdx][0]
            # print("startTime",startTime)
            # print("endTime",endTime)
            # csiIndices=csiIndices_sec(startTime,endTime,collectedCSI[colIndx])
            # way 2
            csiIndices = sleepIdx2csiIndices_timestamp(
                sleepIdx, ss_value, csiStartIdx, csi_value, timeLen=timeperoid)
            print("len csiIndices", len(csiIndices))
            if (len(csiIndices) == 0):
                continue
            # check if there is csi more than minCSIthreshold
            if(len(csiIndices) < minCSIthreshold):
                print("too low csi number", len(
                    csiIndices), minCSIthreshold)
                continue

            print(len(csiIndices), "csis to 1 SS")
            csiStartIdx = csiIndices[-1]
            print("csiIndices", csiIndices[0], "-", csiIndices[-1])

            # CSI matrix formation
            curCSIs, _ = samplingCSISleep(
                csi_value, csiIndices, ss_value, sleepIdx, samplingedCSIWinSize, timeLen=timeperoid)
            validLenCounter[-1] = validLenCounter[-1]+1

            if(isTestLabels[fileIdx] == False):
                X.append(curCSIs)
                Y.append(curSleeps)
            else:
                Xalt.append(curCSIs)
                Yalt.append(curSleeps)


if (justCollect == False):
    X = np.array(X)
    Y = np.array(Y)
    Xalt = np.array(Xalt)
    Yalt = np.array(Yalt)

    if len(Xalt) == 0:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=18)
    else:
        # np.random.seed(42)
        # np.random.shuffle(X)
        # np.random.seed(42)
        # np.random.shuffle(Y)
        x_train = X
        y_train = Y
        x_test = Xalt
        y_test = Yalt

    print('shape x_train', (x_train.shape))
    print('shape x_test', (x_test.shape))
    print('shape y_train', (y_train.shape))
    print('shape y_test', (y_test.shape))

    if runTrain:

        model = build_model(downsample=downsample, win_len=samplingedCSIWinSize*2,
                            n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=label_n)
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
        stageTestCounter = [0, 0, 0, 0]
        stagePredCounter = [0, 0, 0, 0]

        wakePredCounter = [0, 0, 0, 0]
        remPredCounter = [0, 0, 0, 0]
        lightPredCounter = [0, 0, 0, 0]
        deepPredCounter = [0, 0, 0, 0]

        for i in range(0, len(y_test)):
            maximum_test = np.max(y_test[i])
            maximum_pred = np.max(y_pred[i])
            index_of_maximum_test = np.where(y_test[i] == maximum_test)
            index_of_maximum_pred = np.where(y_pred[i] == maximum_pred)
            curTest = index_of_maximum_test[0][0]
            curPred = index_of_maximum_pred[0][0]
            # print("curTest",curTest)
            # print("curPred",curPred)
            if(curTest == curPred):
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
            else:
                if(curTest == 0):
                    remPredCounter[curPred] = remPredCounter[curPred] + 1
                elif(curTest == 1):
                    lightPredCounter[curPred] = lightPredCounter[curPred] + 1
                elif(curTest == 2):
                    deepPredCounter[curPred] = deepPredCounter[curPred] + 1

        print('len X', len(X))
        print('len Y', len(Y))
        print('len Xalt', len(Xalt))
        print('len Yalt', len(Yalt))
        for labelIdx in range(len(Sleeplabels)):
            print(Sleeplabels[labelIdx], 'len is', validLenCounter[labelIdx],
            ("TEST" if isTestLabels[labelIdx] else "TRAIN"))
        print("stagePredCounter", stagePredCounter)
        print("stageTestCounter", stageTestCounter)
        print("score", matchCounter, "/", len(y_test))
        print("score percent", matchCounter/len(y_test)*100, "/100")
        if wakeIncluded:
            wakePredCounter = np.array(
                wakePredCounter) / stageTestCounter[0]
            remPredCounter = np.array(remPredCounter) / stageTestCounter[1]
            lightPredCounter = np.array(
                lightPredCounter) / stageTestCounter[2]
            deepPredCounter = np.array(
                deepPredCounter) / stageTestCounter[3]

            # toFixed 2
            wakePredCounter = np.around(wakePredCounter, decimals=2)
            remPredCounter = np.around(remPredCounter, decimals=2)
            lightPredCounter = np.around(lightPredCounter, decimals=2)
            deepPredCounter = np.around(deepPredCounter, decimals=2)

            print("        wake rem light deep")
            print("wake  ", wakePredCounter)
            print("rem   ", remPredCounter)
            print("light ", lightPredCounter)
            print("deep  ", deepPredCounter)
        else:
            remPredCounter = np.array(remPredCounter) / stageTestCounter[0]
            lightPredCounter = np.array(
                lightPredCounter) / stageTestCounter[1]
            deepPredCounter = np.array(
                deepPredCounter) / stageTestCounter[2]

            # toFixed 2
            remPredCounter = np.around(remPredCounter, decimals=2)
            lightPredCounter = np.around(lightPredCounter, decimals=2)
            deepPredCounter = np.around(deepPredCounter, decimals=2)
            print("        rem light deep")
            print("rem   ", remPredCounter)
            print("light ", lightPredCounter)
            print("deep  ", deepPredCounter)
