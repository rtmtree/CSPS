from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from functions.csi_util import filterNullSC, rawCSItoAmp, parseCSI, samplingCSISleep, sleepIdx2csiIndices_timestamp, samplingRSSISleep,featureEngineerDataSeqSc
from functions.sleepstage_util import getSleepStageByTime, oneSS2Mat, twoSS2Mat, mat2OneSS, mat2TwoSS
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import json
import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
plotAsTimeline = True
if(dir_path=="/Users/ratthamontreeburimas/Documents/works/CSPS"):
    print("running in LOCAL")
    path = ''
    loadDataFromFile = True
    runTrain = True
    runTrainUnsup = True
    runEval = False
    runPlot = True
else:
    print("running in COLAB")
    path = 'drive/MyDrive/Project/'
    loadDataFromFile = False
    runTrain = True
    runEval = True
    runPlot = False

# from functions.pose_util import poseToPAM, PAMtoPose, rotate_poses, getPCK
useTestData = False
savePredictedData = False
if runPlot:
    import imageio
    import cv2

if runTrain or runEval:
    if runTrainUnsup:
        from sklearn.datasets import make_blobs
        from sklearn.cluster import KMeans
    else:
        # train stuffs
        from modules.inference_engine_pytorch import InferenceEnginePyTorch
        import tensorflow as tf
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import pandas as pd
        import matplotlib.gridspec as gridspec
        from scipy.ndimage import gaussian_filter

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

        def build_model(downsample=1, win_len=1000, n_unit_lstm=200, n_unit_atten=400, label_n=2, data_len=52):
            """
            Returns the Tensorflow Model which uses AttenLayer
            """
            # print("data_len",data_len)
            if downsample > 1:
                length = len(np.ones((win_len,))[::downsample])
                x_in = tf.keras.Input(shape=(length, data_len))
            else:
                x_in = tf.keras.Input(shape=(win_len, data_len))
            x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                units=n_unit_lstm, return_sequences=True))(x_in)
            x_tensor = AttenLayer(n_unit_atten)(x_tensor)
            pred = tf.keras.layers.Dense(label_n, activation='softmax')(x_tensor)
            model = tf.keras.Model(inputs=x_in, outputs=pred)
            return model

        def build_modelMD(downsample=1, win_len=10000, n_unit_lstm=200, n_unit_atten=400, label_n=4, data_len=52):
            model = tf.keras.Sequential()

            model.add(tf.keras.layers.LSTM(10, input_shape=(
                win_len, data_len), return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.LSTM(100, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.LSTM(
                label_n, return_sequences=True, activation='linear'))
            return model

        def build_modelMD2(downsample=1, win_len=1000, n_unit_lstm=200, n_unit_atten=400, label_n=4, data_len=52):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(50, batch_input_shape=(
                None, 10000, data_len), kernel_initializer="he_normal", activation="relu"))
            model.add(tf.keras.layers.Dense(
                20, kernel_initializer="he_normal", activation="relu"))
            model.add(tf.keras.layers.Dense(
                5, kernel_initializer="he_normal", activation="relu"))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(4))
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


# ========== adjustable
labels = [
    '2022-01-23s0200',
    '2022-01-28s0356',
    '2022-01-29s0638',
    '2022-01-31s0409',
    '2022-02-01s0232',
    '2022-02-02s0344',
    '2022-02-03s0200',
    '2022-02-07s0519',
    '2022-02-08s0358',
    '2022-02-11s0525',
    '2022-02-13s0422',
    '2022-02-12s0318',
    '2022-01-25s0153',
    '2022-01-26s0250',
    '2022-01-27s0242',
]
# labelsAlt = []
labelsAlt = [
    '2022-01-27s0242',
    '2022-02-07s0519',
    '2022-02-08s0358',
]
FE = True
gSigma = 0
samplingedCSIWinSize = 30  # delta
limitCollect = 100000
limitEach = 100000
epoch = 200
useCSI = True
SSWeightForm = True
focusStage = [0, 1, 2, 3]
# focusStage = [0,3]
batch_size = 128
sleepWinSize = 1 # sigma
sleepWinSizeCropTo = 1
timeperoid = sleepWinSize*30
maxSSWinSize = sleepWinSize
# minCSIthreshold = samplingedCSIWinSize
minCSIthreshold = samplingedCSIWinSize
n_unit_lstm = 200
n_unit_atten = 400
downsample = 2


# ========== adjustable end
savePath = path+'data/'
Wcounter = 0
Rcounter = 0
Lcounter = 0
Dcounter = 0
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

modelFileName = 'test_models/csi2sleepM_e'+str(epoch)+'spCSI_'+str(
    samplingedCSIWinSize)+('focusStage_'+str(len(focusStage)))+'.hdf5'
# timestampColName = 'local_timestamp'
timestampColName = 'real_timestamp'
decimalShiftTs = 0


X = []
Y = []
Xalt = []
Yalt = []
if useTestData:
    for fileIdx in range(len(CSIlabels)):
        validLenCounter.append(0)
        X.append([np.array([0 for j in range(52)])])
        if(SSWeightForm):
            Y.append([[0 for j in range(len(focusStage))]])
        else:
            Y.append([0])

        Xalt.append([np.array([0 for j in range(52)])])
        if(SSWeightForm):
            Yalt.append([[0 for j in range(len(focusStage))]])
        else:
            Yalt.append([0])
        validLenCounter[-1] += 1
else:
    for fileIdx in range(len(CSIlabels)):
        pathSavedFileCSI = savePath+'parsed' + \
            CSIlabels[fileIdx]+'minCSI'+str(minCSIthreshold)+'sws'+str(samplingedCSIWinSize)+'ss'+str(sleepWinSize)+'-'+str(sleepWinSizeCropTo)+'CSI.csv'
        pathSavedFileSleep = savePath+'parsed' + \
            CSIlabels[fileIdx]+'minCSI'+str(minCSIthreshold)+'sws'+str(samplingedCSIWinSize)+'ss'+str(sleepWinSize)+'-'+str(sleepWinSizeCropTo)+'SS.csv'
        Xfile = []
        Yfile = []
        if loadDataFromFile:
            curXfile = np.loadtxt(pathSavedFileCSI, delimiter=',')
            curXfile = curXfile.reshape(
                (curXfile.shape[0], samplingedCSIWinSize, 52))
            # Xfile= [Xfile]
            Xfile.extend(curXfile)
            print('loaded', pathSavedFileCSI)
            curYfile = np.loadtxt(pathSavedFileSleep, delimiter=',')
            # Yfile= [Yfile]
            Yfile.extend(curYfile)
            print('loaded', pathSavedFileSleep)
            validLenCounter.append(len(Xfile))
            print(len(Xfile))
        else:
            validLenCounter.append(0)
            # Organize sleep log file
            thisLabel = CSIlabels[fileIdx]
            thisSync = thisLabel[10:11]
            if thisSync == 's':
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
            futc_time = datetime.strptime(
                firstsecSleep, "%Y-%m-%dT%H:%M:%S.%f")
            sleepFirstTs = (futc_time - datetime(1970, 1, 1)).total_seconds()
            print('first timestamp in Sleep is', sleepFirstTs)

            # get last ts of real world ts CSI log
            lastsecSleep = sleepSData[-1]['dateTime']
            print('last date/time in Sleep is', lastsecSleep)
            lutc_time = datetime.strptime(lastsecSleep, "%Y-%m-%dT%H:%M:%S.%f")
            sleepLastTs = (lutc_time - datetime(1970, 1, 1)
                           ).total_seconds() + sleepSData[-1]['seconds']
            print('last timestamp in Sleep is', sleepLastTs)

            vidLength = int((sleepLastTs-sleepFirstTs)/timeperoid)
            print('Sleep time Length', vidLength)

            # Organize CSI log file
            print("read CSI file", filePath)
            colsName = ["type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
                        "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "real_time_set", "real_timestamp", "len", "CSI_DATA"]
            try:
                curFile = pd.read_csv(filePath+'.csv', names=colsName)
            except:
                continue
            # filter out the rows unvalid timestamp. CLEAN Data
            curFile[timestampColName] = curFile[timestampColName].astype('str')
            print("len file bf", len(curFile.index))
            curFile = curFile[curFile[timestampColName].str.match(
                r"[+-]?([0-9]*[.])?[0-9]+") == True]
            curFile = curFile[curFile[timestampColName].str.contains(
                '[A-Za-z :]') == False]
            curFile = curFile[curFile[timestampColName].str.contains(
                "[.]") == True]
            # curFile = curFile[curFile[timestampColName].str.contains(":")==False]

            print("len file af", len(curFile.index))
            curFile[timestampColName] = curFile[timestampColName].astype(
                'float')
            curFile = curFile.sort_values(timestampColName)

            # print(curFile)

            # find diffEpoch to sync time
            tsList = (list(x/(10**decimalShiftTs)
                           for x in curFile[timestampColName]))

            if thisSync == 's':
                CSIsyncTime = thisLabel[0:10]+'T' + \
                    thisLabel[-4:-2]+':'+thisLabel[-2:]
                # get first ts of real world ts CSI log
                csiRealFirstTs = CSIsyncTime+':01.000'
                print('first real world date/time in CSI is', csiRealFirstTs)
                utc_time = datetime.strptime(
                    csiRealFirstTs, "%Y-%m-%dT%H:%M:%S.%f")
                realFirstTs = (utc_time - datetime(1970, 1, 1)).total_seconds()
                print('last real world timestamp in CSI is', realFirstTs)
                diffEpoch = realFirstTs
            else:
                CSIsyncTime = thisLabel[11:15]+'-'+thisLabel[8:10]+'-' + \
                    thisLabel[5:7]+'T'+thisLabel[-4:-2]+':'+thisLabel[-2:]
                # get last ts of real world ts CSI log
                csiRealLastTs = CSIsyncTime+':00.000'
                print('last real world date/time in CSI is', csiRealLastTs)
                utc_time = datetime.strptime(
                    csiRealLastTs, "%Y-%m-%dT%H:%M:%S.%f")
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
            my_filter_address = "7C:9E:BD:D2:D8:9D"
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
            lastSleepTs = 0

            def updatedata(i):
                # global realLastTs
                global sleepSData
                global sleepFirstTs
                global curFile
                global diffEpoch
                global lastSleepTs

                sleepIdx = i+startFrom

                sleepTs = int((sleepIdx*30)+sleepFirstTs)
                # print("sleepTs", sleepTs,datetime.fromtimestamp(sleepTs).strftime('%c'))

                stage = getSleepStageByTime(sleepTs, lastSleepTs, sleepSData)
                # print("SS",stage)
                if(stage == -1):
                    print("catch no sleep stage detected")
                    return False
                ss_value.append([sleepTs]+[stage])

                csiTsTo = (sleepTs - diffEpoch)
                csiTsFrom = csiTsTo-timeperoid
                # parse to second
                csiTsTo = (10**decimalShiftTs) * csiTsTo
                csiTsFrom = (10**decimalShiftTs) * csiTsFrom
                print("csiTsFrom", csiTsFrom, "csiTsTo", csiTsTo)

                # dataInPeriod = curFile[(curFile[timestampColName]>=csiTsFrom)]
                # dataInPeriod = dataInPeriod[(dataInPeriod[timestampColName]<csiTsTo)]

                dataInPeriod = curFile[(curFile[timestampColName] >= csiTsFrom) & (
                    curFile[timestampColName] < csiTsTo)]

                print("len csidataInPeriod", len(dataInPeriod.index))
                if(useCSI):
                    csiInPeriod = list(parseCSI(x)
                                       for x in dataInPeriod['CSI_DATA'])
                    # rssiInPeriod = list(float(x) for x in dataInPeriod['rssi'])
                else:
                    csiInPeriod = list(float(x) for x in dataInPeriod['rssi'])
                parseTsInPeriod = list(
                    (x/(10**decimalShiftTs))+diffEpoch for x in dataInPeriod[timestampColName])

                if(len(csiInPeriod) > 0 and len(csiInPeriod) == len(parseTsInPeriod)):
                    for k in range(len(csiInPeriod)):
                        curParseTs = parseTsInPeriod[k]
                        if(useCSI):
                            curParseCSI = (csiInPeriod[k])
                            if(curParseCSI != False):
                                # check if all elements are int
                                isInt = all(isinstance(l, int)
                                            for l in curParseCSI)
                                if isInt == False:
                                    continue
                                csi_value.append([curParseTs]+curParseCSI)
                            else:
                                continue
                        else:  # RSSI
                            csi_value.append([curParseTs]+[csiInPeriod[k]])

            startFrom = 0
            collectLength = vidLength-startFrom
            print('startFrom', startFrom)
            print('length', collectLength)
            for i in range(collectLength):
                updatedata(i)
                print("collecting ====", i+1, "/", collectLength,
                      "====", fileIdx+1, "/", len(CSIlabels))
                if i > limitCollect:
                    break

            print('collecting finishing', CSIlabels[fileIdx])

            csi_value = np.array(csi_value)
            ss_value = np.array(ss_value)
            print(csi_value.shape)
            print(ss_value.shape)

            # Clean data

            dataLooper = len(ss_value)
            # dataStep = sleepWinSize
            dataStep = 1
            dataCacheIdx = 0

            for sleepIdx in range(dataCacheIdx, dataLooper, dataStep):
                if(sleepWinSize==2 and sleepIdx+1==(dataLooper)):
                    break
                lastSSTs = ss_value[sleepIdx][0]
                # if(Wcounter > limitEach and Rcounter > limitEach and Lcounter > limitEach and Dcounter > limitEach):
                #     print("exceed limitEach")
                #     break

                # get sleep stage in the time period
                print("training ====", sleepIdx+1, "/", dataLooper,
                      "====", fileIdx+1, "/", len(CSIlabels))
                print("sleepIdx", sleepIdx, 'ts', ss_value[sleepIdx][0])

                # get CSI indices in the time period
                csiIndices = sleepIdx2csiIndices_timestamp(
                    lastSSTs, dataCacheIdx, csi_value, timeLen=timeperoid)
                # check if there is csi more than minCSIthreshold
                if(len(csiIndices) < minCSIthreshold):
                    print("too low csi number", len(
                        csiIndices), minCSIthreshold)
                    continue
                if(len(csiIndices)>0):
                    print("csiIndices", csiIndices[0], "-", csiIndices[-1])
                else:
                    csiIndices = [dataCacheIdx]
                print("len csiIndices", len(csiIndices))
                # print(csiIndices)
                # update dataCacheIdx for reducing time wasting
                if(useCSI):
                    # CSI matrix formation
                    curCSIs, _ = samplingCSISleep(
                        csi_value, csiIndices, lastSSTs, samplingedCSIWinSize, timeLen=timeperoid)
                else:
                    # use RSSI
                    curCSIs, _ = samplingRSSISleep(
                        csi_value, csiIndices, ss_value, sleepIdx, samplingedCSIWinSize, timeLen=timeperoid)

                # sleep stage matrix formation
                curParsedSleeps = False
                curSleeps = ss_value[sleepIdx][1]
                if(sleepWinSize == 2):
                    curSleeps2 = ss_value[sleepIdx+1][1]
                if SSWeightForm:
                    if(sleepWinSize == 1):
                        curParsedSleeps = oneSS2Mat(focusStage, curSleeps)
                    elif(sleepWinSize == 2):
                        # print(curSleeps,"=>",curSleeps2)
                        if(sleepWinSize!=sleepWinSizeCropTo):
                            if(sleepWinSizeCropTo==1):
                                curParsedSleeps = oneSS2Mat(focusStage, curSleeps2)
                        else:
                            curParsedSleeps = twoSS2Mat(
                                focusStage, curSleeps, curSleeps2)
                        # print(curParsedSleeps)
                        # counter
                else:
                    curParsedSleeps = ss_value[sleepIdx][1]
                if curParsedSleeps != False:
                    if(sleepWinSize >= 1):
                        if(sleepWinSizeCropTo==sleepWinSize):
                            if(curSleeps == 0):
                                Wcounter += 1
                            elif(curSleeps == 1):
                                Rcounter += 1
                            elif(curSleeps == 2):
                                Lcounter += 1
                            elif(curSleeps == 3):
                                Dcounter += 1
                    if(sleepWinSize >= 2):
                        if(curSleeps2 == 0):
                            Wcounter += 1
                        elif(curSleeps2 == 1):
                            Rcounter += 1
                        elif(curSleeps2 == 2):
                            Lcounter += 1
                        elif(curSleeps2 == 3):
                            Dcounter += 1
                    dataCacheIdx = csiIndices[0]
                    validLenCounter[-1] += 1
                    Xfile.append(curCSIs)
                    Yfile.append(curParsedSleeps)
                # else:
                #     if(curSleeps != False):
                #         Xfile.extend(curCSIs)
                #         Yfile.extend([curParsedSleeps for j in range(samplingedCSIWinSize)])
            XfileP = np.array(Xfile)
            np.savetxt(pathSavedFileCSI, XfileP.reshape(
                (XfileP.shape[0], XfileP.shape[1]*XfileP.shape[2])), delimiter=',', fmt='%1.6f')
            print('saved', pathSavedFileCSI, XfileP.shape)
            np.savetxt(pathSavedFileSleep, Yfile, delimiter=',', fmt='%d')
            print('saved', pathSavedFileSleep)
        
        [Xfile] = featureEngineerDataSeqSc([Xfile],FE,gSigma)
        if(isTestLabels[fileIdx] == False):
            X.append(Xfile)
            Y.append(Yfile)
        else:
            Xalt.append(Xfile)
            Yalt.append(Yfile)

# Sum data
if False:
    print('len X', len(X))
    for i in range(len(X)):
        print(i, np.array(X[i]).shape)
    for i in range(len(Y)):
        print(i, np.array(Y[i]).shape)
print('len Y', len(Y))
print('len Xalt', len(Xalt))
print('len Yalt', len(Yalt))
# print('len X0', len(X[0]))
# print('len Y0', len(Y[0]))
# print('len Xalt0', len(Xalt[0]))
# print('len Yalt0', len(Yalt[0]))

# if (FE):
#    X=featureEngineerDataSeqSc(X)
#    Xalt=featureEngineerDataSeqSc(Xalt)


if False:
    for i in range(len(X)):
        while(len(X[i]) < maxSSWinSize):
            X[i].append(np.array([0 for j in range(52)]))
            if(SSWeightForm):
                Y[i].append([0 for j in range(len(focusStage))])
            else:
                Y[i].append(0)
    for i in range(len(Xalt)):
        while(len(Xalt[i]) < maxSSWinSize):
            Xalt[i].append(np.array([0 for j in range(52)]))
            if(SSWeightForm):
                Yalt[i].append([0 for j in range(len(focusStage))])
            else:
                Yalt[i].append(0)
Xmerged = []
Ymerged = []
for x in X:
    Xmerged.extend(x)
for y in Y:
    Ymerged.extend(y)
Xmerged = np.array(Xmerged)
Ymerged = np.array(Ymerged)
# Xalt = np.array(Xalt)
# Yalt = np.array(Yalt)
# print('shape X', (X.shape))
# print('shape Y', (Y.shape))
# print('shape Xall', (Xall.shape))
# print('shape Yall', (Yall.shape))
# print('shape Xalt', (Xalt.shape))
# print('shape Yalt', (Yalt.shape))


if len(Xalt) == 0:
    # x_train, x_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=18)
    x_train = Xmerged
    y_train = Xmerged
    x_test = Xmerged
    y_test = Xmerged
else:
    # np.random.seed(42)
    # np.random.shuffle(X)
    # np.random.seed(42)
    # np.random.shuffle(Y)
    # x_train, _, y_train, _ = train_test_split(
    #     X, Y, test_size=0.01, random_state=18)
    x_train = Xmerged
    y_train = Xmerged
    x_testSet = Xalt
    y_testSet = Yalt

print('shape x_train', (x_train.shape))
print('shape y_train', (y_train.shape))
print('len x_testSet', len(x_testSet))
print('len y_testSet', len(y_testSet))
# print('shape x_test', (x_test.shape))
# print('shape y_test', (y_test.shape))

cluster_n = len(focusStage)*len(focusStage)
y_predSet = []
if runTrain:
    if runTrainUnsup:
        # Create the dataset - Returns X, y or data[0], data[1].
        # Dataset made by "make_blobs". Parameters are set.
        # data = make_blobs(n_samples=1000, centers=len(focusStage), cluster_std=1.2, random_state=10)
        # data2 = make_blobs(n_samples=1000, centers=len(focusStage), cluster_std=1.2, random_state=3)

        # Setting scatter plot title.
        # plt.title('Data Points')

        # # Show the scatter plot of the data points.
        # plt.scatter(data[0][:,0], data[0][:,1], edgecolors='black', linewidths=.5);

        # plt.show()

        # Creating algorithm and setting it's parameters.
        K_Means = KMeans(n_clusters=cluster_n)

        newX_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
        # Training - "Fitting to the data".
        K_Means.fit(newX_train)
        for testIdx in range(len(x_testSet)):
            x_testCur = np.array(x_testSet[testIdx])
            newX_test = x_testCur.reshape((x_testCur.shape[0],x_testCur.shape[1]*x_testCur.shape[2]))
            # Make predictions.
            y_pred = K_Means.predict(newX_test)
            y_predSet.append(y_pred)

        # Setting scatter plot title.
        # # plt.title('Data Points in Clusters')
        # # # Show scatter plot - The color (c) is determined by the clusters.
        # # plt.scatter(data2[0][:,0], data2[0][:,1], c=clusters, edgecolors='black', linewidths=.5)

        # plt.show()
    else:
        if(useCSI):
            if(sleepWinSize==2):
                if(sleepWinSizeCropTo==1):
                    model = build_model(downsample=downsample, win_len=samplingedCSIWinSize*2,
                                        n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=len(focusStage), data_len=52)
                else:
                    model = build_model(downsample=downsample, win_len=samplingedCSIWinSize*2,
                                        n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=len(focusStage)*len(focusStage), data_len=52)
            else:
                model = build_model(downsample=downsample, win_len=samplingedCSIWinSize*2,
                n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=len(focusStage) if SSWeightForm else 10000, data_len=52)
            # n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=(10000,len(focusStage)), data_len=52)
            # model = build_modelMD(downsample=downsample, win_len=maxSSWinSize,
            #                       n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=len(focusStage) if SSWeightForm else maxSSWinSize, data_len=52)
        else:
            model = build_model(downsample=downsample, win_len=samplingedCSIWinSize*2*1000,
                                n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=len(focusStage) if SSWeightForm else maxSSWinSize, data_len=1)
        if True:
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
        else:
            model.compile(optimizer='adam', loss='mse')
            print(model.summary())
            model.fit(x_train, y_train, epochs=epoch, verbose=2,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(path+modelFileName,
                                                            monitor='val_accuracy',
                                                            save_best_only=True,
                                                            save_weights_only=False)
                    ])

if runEval:
    if sleepWinSize==1 or sleepWinSizeCropTo==1:
        dataLooper = len(focusStage)
    else:
        dataLooper = len(focusStage)*len(focusStage)
    for labelIdx in range(len(Sleeplabels)):
        print(Sleeplabels[labelIdx], 'len is', validLenCounter[labelIdx],
              ("TEST" if isTestLabels[labelIdx] else "TRAIN"))

    # load the best model
    model = load_model(path+modelFileName)
    y_pred = model.predict(x_test)

    matchCounter = 0
    stageTestCounter = [0 for i in range(dataLooper)]
    stagePredCounter = [0 for i in range(dataLooper)]
    PredCounter = [[0.0 for i in range(dataLooper)] for j in range(dataLooper)]

    for i in range(len(y_test)):
        if savePredictedData and len(labelsAlt) > 0:
            pathSavedFileTestSleep = savePath+'tested' + \
                labelsAlt[i]+'sws'+str(samplingedCSIWinSize)+'SS.csv'
            pathSavedFilePredSleep = savePath+'predicted' + \
                labelsAlt[i]+'sws'+str(samplingedCSIWinSize)+'SS.csv'
            np.savetxt(pathSavedFileTestSleep,
                       y_test[i], delimiter=',', fmt='%d')
            np.savetxt(pathSavedFilePredSleep,
                       y_pred[i], delimiter=',', fmt='%d')
        if sleepWinSize==1 or sleepWinSizeCropTo==1:
            curTest = mat2OneSS(y_test[i],focusStage)
            curPred = mat2OneSS (y_pred[i],focusStage)
        else:
            print(y_test[i])
            print(y_pred[i])
            curTest,curTest2 = mat2TwoSS(y_test[i],focusStage)
            curPred,curPred2 = mat2TwoSS (y_pred[i],focusStage)
        if sleepWinSize==1 or sleepWinSizeCropTo==1:
            if(curTest == curPred):
                matchCounter += 1
            stagePredCounter[curPred] += 1
            stageTestCounter[curTest] += 1
            PredCounter[curTest][curPred] += 1
        elif sleepWinSize==2:
            if(curTest == curPred and curTest2 == curPred2):
                matchCounter += 1
            stagePredCounter[(curPred*len(focusStage))+curPred2] += 1
            stageTestCounter[(curTest*len(focusStage))+curTest2] += 1
            PredCounter[(curTest*len(focusStage))+curTest2][(curPred*len(focusStage))+curPred2] += 1

    print("   0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15")
    for i in range(dataLooper):
        # for j in range(len(focusStage)*len(focusStage)):
        index = i
        PredCounter[i] = np.array(PredCounter[i]) if stageTestCounter[i]==0 else np.array(PredCounter[i]) / stageTestCounter[i]
        # PredCounter[i] = np.array(PredCounter[i])
        PredCounter[i] = np.around(PredCounter[i], decimals=2)
        print(("" if index>9 else "0") + str(index)," ", PredCounter[i])

    print("stageTestCounter", stageTestCounter)
    print("stagePredCounter", stagePredCounter)
    print("score", matchCounter, "/", (len(y_test)))
    print("score percent", matchCounter /
          (len(y_test))*100, "/100")

if runPlot:
    # fig = plt.figure()
    # gs = gridspec.GridSpec(7, 2)
    # axSS = fig.add_subplot(gs[0:3, :])
    # axCSI = fig.add_subplot(gs[4:6, :])
    fig, [axSS, axUSL,axCSI] = plt.subplots(3, 1)
    def init():
        axSS.invert_yaxis()
        axUSL.invert_yaxis()
        plt.setp(axSS, xlabel="Frame (1/"+str(timeperoid)+"s)", ylabel="Sleep Stage")
        plt.setp(axUSL, xlabel="Frame (1/"+str(timeperoid)+"s)", ylabel="Unsupervised Label")
        plt.setp(axCSI, xlabel="Frame ("+str(timeperoid)+"s)", ylabel="Amplitude(dB)")
        axSS.set_ylim([4,-1])
        axSS.yaxis.set_major_locator(ticker.FixedLocator([3, 2, 1, 0]))
        axSS.yaxis.set_major_formatter(ticker.FixedFormatter(['deep', 'light', 'rem', 'wake']))
        if useCSI:
            axCSI.set_ylim([-10, +40])
        else:
            axCSI.set_ylim([-100, 0])

        x_plotLen = 1000
        if plotAsTimeline:
            axSS.set_xlim([0, (x_plotLen)])
        else:    
            axSS.set_xlim([1, 2])

        if plotAsTimeline:
            axCSI.set_xlim([0, x_plotLen * samplingedCSIWinSize])
        else:
            axCSI.set_xlim([0, x_plotLen])

        if plotAsTimeline:
            axUSL.set_xlim([0, x_plotLen])
        else:
            axUSL.set_xlim([1, 2])
        axUSL.set_ylim([cluster_n+1,-1])

        return [axSS, axUSL,axCSI]

    x_plot = x_testSet
    y_plot = y_testSet
    y_plot2 = y_predSet
    def updatefig(i):
        print("bf plotCur index",i)
        # i=0
        axSS.cla()
        axCSI.cla()
        axUSL.cla()
        init()
        if(i >= len(x_plot)):
            i = i % len(x_plot)
        print("real plotCur index",i)
        x_plotCur = np.array(x_plot[i])
        y_plotCur = np.array(y_plot[i])
        y_plot2Cur = np.array(y_plot2[i])

        # SS part
        if SSWeightForm:
            plotXSS = []
            plotYSS = []
            if plotAsTimeline:
                for ii in range(len(y_plotCur)):
                    curPred = mat2OneSS(y_plotCur[ii], focusStage)
                    plotXSS.append(ii)
                    plotYSS.append(curPred)
            else:
                plotXSS = []
                plotYSS = []
                if maxSSWinSize == 2:
                    print(y_plotCur)
                    curPred, curPred2 = mat2TwoSS(y_plotCur, focusStage)
                    plotXSS.extend([1,2])
                    plotYSS.extend([curPred,curPred2])
                elif maxSSWinSize == 1:
                    curPred = mat2OneSS(y_plotCur, focusStage)
                    plotXSS.extend([1,2])
                    plotYSS.extend([curPred,curPred])
            axSS.plot(plotXSS, plotYSS,
                    label='Sleep stage')
        else:
            plotXSS = [j for j in range(len(y_plotCur))]
            plotYSS = y_plot[i]
            axSS.plot(plotXSS, plotYSS,
                     label='Sleep stage')

        # CSI part
        if useCSI:
            if plotAsTimeline:
                for j in range(len(x_plotCur[0][0])): #52
                    plotXCSI = []
                    plotYCSI = []
                    for ii in range(len((x_plotCur))):
                        for k in range(len(x_plotCur[ii])):
                            curCsi = x_plotCur[ii][k][j]
                            curCsiIdx = ( ii*len(x_plotCur[ii]) + k )
                            plotXCSI.append(curCsiIdx)
                            plotYCSI.append(curCsi)
                    axCSI.plot(plotXCSI, plotYCSI,
                            label='CSI samplinged subcarrier')
            else:
                for j in range(len((x_plotCur[i][0]))):
                    plotXCSI = []
                    plotYCSI = []
                    for k in range(len(x_plotCur[i])):
                        curCsi = ((x_plotCur[i][k]))
                        plotXCSI.append(k)
                        plotYCSI.append(curCsi[j]*magnify)
                    axCSI.plot(plotXCSI, plotYCSI,
                            label='CSI samplinged subcarrier')
        else:
            plotXCSI = []
            plotYCSI = []
            for k in range(len(x_plotCur[i])):
                curRssi = x_plotCur[i][k]
                plotXCSI.append(k)
                plotYCSI.append(curRssi)
            axCSI.plot(plotXCSI, plotYCSI,
                     label='RSSI samplinged subcarrier')

        # unsupervised labels part
        if runTrainUnsup:
            axUSL.plot(np.arange(len(y_plot2Cur)), y_plot2Cur,
                     label='Sleep stage')

        return [axSS, axUSL,axCSI]

    ani = animation.FuncAnimation(fig, updatefig, interval=60000,
                                  #  frames=len(x_plot),
                                    init_func=init,
                                  blit=True)
    plt.show()
