from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from functions.csi_util import filterNullSC,rawCSItoAmp, parseCSI, samplingCSISleep, sleepIdx2csiIndices_timestamp, samplingRSSISleep
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import json
import numpy as np
runTrain = False
runPlot = True
runEval = runTrain

# from functions.pose_util import poseToPAM, PAMtoPose, rotate_poses, getPCK

if runPlot:
    import imageio
    import cv2

if runTrain:
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
# path = 'drive/MyDrive/Project/'
path = ''
labels = [
    #   '2022-01-22s0417',
      '2022-01-23s0200',
      '2022-01-25s0153',
      '2022-01-26s0250',
      '2022-01-27s0242',
      '2022-01-28s0356',
    #   '2022-01-29s0638',
    #   '2022-01-31s0409',
    #   '2022-02-01s0232',
    #   '2022-02-02s0344',
    #   '2022-02-03s0200',
    #   '2022-02-05s0333',
    # '2022-02-07s0519',
    # '2022-02-08s0358',
    #   '2022-02-11s0525',
    #   '2022-02-12s0318'
    #   '2022-02-13s0422'
]
labelsAlt = [
    # '2022-02-08s0358',
    #   '2022-02-11s0525',
    #   '2022-02-12s0318',
    #   '2022-02-13s0422'
]
limitCollect = 100000
limitEach = 100000
epoch = 200
useCSI = True
focusStage = [0,1,2,3]
# focusStage = [0,3]
batch_size = 128
sleepWinSize = 1  # sigma
samplingedCSIWinSize = 90  # delta
n_unit_lstm = 200
n_unit_atten = 400
downsample = 2


# ========== adjustable end

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
minCSIthreshold = samplingedCSIWinSize
modelFileName = 'test_models/csi2sleepM_e'+str(epoch)+'spCSI_'+str(
    samplingedCSIWinSize)+('focusStage_'+str(len(focusStage)))+'.hdf5'


# timestampColName = 'local_timestamp'
timestampColName = 'real_timestamp'
decimalShiftTs = 0
timeperoid = 30

X = []
Y = []
Xalt = []
Yalt = []
for fileIdx in range(len(CSIlabels)):
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
    curFile = curFile[curFile[timestampColName].str.contains("[.]") == True]
    # curFile = curFile[curFile[timestampColName].str.contains(":")==False]

    print("len file af", len(curFile.index))
    curFile[timestampColName] = curFile[timestampColName].astype('float')
    curFile = curFile.sort_values(timestampColName)

    # print(curFile)

    # find diffEpoch to sync time
    tsList = (list(x/(10**decimalShiftTs) for x in curFile[timestampColName]))

    if thisSync == 's':
        CSIsyncTime = thisLabel[0:10]+'T'+thisLabel[-4:-2]+':'+thisLabel[-2:]
        # get first ts of real world ts CSI log
        csiRealFirstTs = CSIsyncTime+':01.000'
        print('first real world date/time in CSI is', csiRealFirstTs)
        utc_time = datetime.strptime(csiRealFirstTs, "%Y-%m-%dT%H:%M:%S.%f")
        realFirstTs = (utc_time - datetime(1970, 1, 1)).total_seconds()
        print('last real world timestamp in CSI is', realFirstTs)
        diffEpoch = realFirstTs
    else:
        CSIsyncTime = thisLabel[11:15]+'-'+thisLabel[8:10]+'-' + \
            thisLabel[5:7]+'T'+thisLabel[-4:-2]+':'+thisLabel[-2:]
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
        print("sleepTs", sleepTs)

        stage = -1
        for j in range(lastSleepTs, len(sleepSData)):
            # print(sleepSData[j])
            tutc_time = datetime.strptime(
                sleepSData[j]['dateTime'], "%Y-%m-%dT%H:%M:%S.%f")
            tepoch_time = int(
                (tutc_time - datetime(1970, 1, 1)).total_seconds())
            if((sleepTs) >= (tepoch_time)):
                if(sleepSData[j]['level'] == "wake"):
                    stage = 0
                elif(sleepSData[j]['level'] == "rem"):
                    stage = 1
                elif(sleepSData[j]['level'] == "light"):
                    stage = 2
                elif(sleepSData[j]['level'] == "deep"):
                    stage = 3
            else:
                lastSleepTs = 0 if j == 0 else j-1
                break
        if(stage == -1):
            print("catch no sleep stage detected")
            return False
        ss_value.append([sleepTs]+[stage])

        csiTsEnd = (sleepTs - diffEpoch)
        csiTs = csiTsEnd-timeperoid
        # parse to second
        csiTsEnd = (10**decimalShiftTs) * csiTsEnd
        csiTs = (10**decimalShiftTs) * csiTs
        print("csiTs", csiTs, "csiTsEnd", csiTsEnd)

        # dataInPeriod = curFile[(curFile[timestampColName]>=csiTs)]
        # dataInPeriod = dataInPeriod[(dataInPeriod[timestampColName]<csiTsEnd)]

        dataInPeriod = curFile[(curFile[timestampColName] >= csiTs) & (
            curFile[timestampColName] < csiTsEnd)]
        tsInPeriod = list(
            (x/(10**decimalShiftTs))+diffEpoch for x in dataInPeriod[timestampColName])

        print("len csidataInPeriod", len(dataInPeriod.index))
        if(useCSI):
            csiInPeriod = list(parseCSI(x) for x in dataInPeriod['CSI_DATA'])
            # rssiInPeriod = list(float(x) for x in dataInPeriod['rssi'])
        else:
            csiInPeriod = list(float(x) for x in dataInPeriod['rssi'])

        if(len(csiInPeriod) > 0):
            for k in range(len(csiInPeriod)):
                curParseTs = tsInPeriod[k]
                if(useCSI):
                    curParseCSI = (csiInPeriod[k])
                    if(curParseCSI != False):
                        isInt = True
                        for l in range(len(curParseCSI)):
                            if(isinstance(curParseCSI[l], int) == False):
                                isInt = False
                                break
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

    print('collection finishing', CSIlabels[fileIdx])

    csi_value = np.array(csi_value)
    ss_value = np.array(ss_value)
    print(csi_value.shape)
    print(ss_value.shape)

# Clean data

    dataLooper = len(ss_value)
    dataStep = sleepWinSize
    csiStartIdx = 0

    for sleepIdx in range(0, dataLooper, dataStep):

        if(Wcounter > limitEach and Rcounter > limitEach and Lcounter > limitEach and Dcounter > limitEach):
            print("exceed limitEach")
            break

        # get sleep stage in the time period
        print("training ====", sleepIdx+1, "/", dataLooper,
              "====", fileIdx+1, "/", len(CSIlabels))
        print("sleepIdx", sleepIdx, 'ts', ss_value[sleepIdx][0])

        # sleep stage matrix formation
        curSleeps = False
        if len(focusStage) == 4:
            if(ss_value[sleepIdx][1] == 0):
                curSleeps = [1, 0, 0, 0]
            elif(ss_value[sleepIdx][1] == 1):
                curSleeps = [0, 1, 0, 0]
            elif(ss_value[sleepIdx][1] == 2):
                curSleeps = [0, 0, 1, 0]
            elif(ss_value[sleepIdx][1] == 3):
                curSleeps = [0, 0, 0, 1]
        elif len(focusStage) == 3:
            if(ss_value[sleepIdx][1] == focusStage[0]):
                curSleeps = [1, 0, 0]
            elif(ss_value[sleepIdx][1] == focusStage[1]):
                curSleeps = [0, 1, 0]
            elif(ss_value[sleepIdx][1] == focusStage[2]):
                curSleeps = [0, 0, 1]
        elif len(focusStage) == 2:
            if(ss_value[sleepIdx][1] == focusStage[0]):
                curSleeps = [1, 0]
            elif(ss_value[sleepIdx][1] == focusStage[1]):
                curSleeps = [0, 1]

        if(curSleeps == False):
            print("invalid sleep stage", curSleeps)
            continue

        # get CSI indices in the time period
        csiIndices = sleepIdx2csiIndices_timestamp(
            sleepIdx, ss_value, csiStartIdx, csi_value, timeLen=timeperoid)
        print("len csiIndices", len(csiIndices))
        # check if there is csi more than minCSIthreshold
        if(len(csiIndices) < minCSIthreshold):
            print("too low csi number", len(
                csiIndices), minCSIthreshold)
            continue

        print("csiIndices", csiIndices[0], "-", csiIndices[-1])
        csiStartIdx = csiIndices[-1]
        if(useCSI):
            # CSI matrix formation
            curCSIs, _ = samplingCSISleep(
                csi_value, csiIndices, ss_value, sleepIdx, samplingedCSIWinSize, timeLen=timeperoid)
        else:
            # use RSSI
            curCSIs, _ = samplingRSSISleep(
                csi_value, csiIndices, ss_value, sleepIdx, samplingedCSIWinSize, timeLen=timeperoid)

        if(isTestLabels[fileIdx] == False):
            X.append(curCSIs)
            Y.append(curSleeps)
        else:
            Xalt.append(curCSIs)
            Yalt.append(curSleeps)
        # counter
        validLenCounter[-1] += 1
        if(ss_value[sleepIdx][1] == 0):
            Wcounter += 1
        elif(ss_value[sleepIdx][1] == 1):
            Rcounter += 1
        elif(ss_value[sleepIdx][1] == 2):
            Lcounter += 1
        elif(ss_value[sleepIdx][1] == 3):
            Dcounter += 1


# Sum data

X = np.array(X)
Y = np.array(Y)
Xalt = np.array(Xalt)
Yalt = np.array(Yalt)
print('shape X', (X.shape))
print('shape Y', (Y.shape))
print('shape Xalt', (Xalt.shape))
print('shape Yalt', (Yalt.shape))

if len(Xalt) == 0:
    # x_train, x_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=18)
    x_train = X
    y_train = Y
else:
    # np.random.seed(42)
    # np.random.shuffle(X)
    # np.random.seed(42)
    # np.random.shuffle(Y)
    # x_train = X
    # y_train = Y
    x_train, _, y_train, _ = train_test_split(
        X, Y, test_size=0.01, random_state=18)
    x_test = Xalt
    y_test = Yalt

print('shape x_train', (x_train.shape))
print('shape y_train', (y_train.shape))
# print('shape x_test', (x_test.shape))
# print('shape y_test', (y_test.shape))

if runPlot:
    # fig = plt.figure()
    # gs = gridspec.GridSpec(7, 2)
    # ax0 = fig.add_subplot(gs[0:3, :])
    # ax3 = fig.add_subplot(gs[4:6, :])
    fig, [ax0, ax3] = plt.subplots(2, 1)

    def init():
        ax0.set_ylim([-1, 4])
        ax0.set_yticklabels(['', 'wake', 'rem', 'light', 'deep', ''])
        plt.setp(ax0, xlabel="Frame (1/30s)", ylabel="Sleep Stage")
        # ax0.set_xlim([startFrom, startFrom+SSWindowSize])
        ax0.set_xlim([0, 1])
        plt.setp(ax3, xlabel="Frame (30s)", ylabel="Amplitude(dB)")
        ax3.set_ylim([-10, +40])
        ax3.set_xlim([0, len(x_train[0])])
        ln1, = plt.plot([], [], 'ro')
        ln2, = plt.plot([], [], 'ro')
        ln = [ln1, ln2]
        return ln

    def updatefig(i):
        ax0.cla()
        ax3.cla()
        ax0.set_ylim([-1, 4])
        ax0.set_yticklabels(['', 'wake', 'rem', 'light', 'deep', ''])
        plt.setp(ax0, xlabel="Frame (1/30s)", ylabel="Sleep Stage")
        ax0.set_xlim([0, 1])
        plt.setp(ax3, xlabel="Frame (30s)", ylabel="Amplitude(dB)")
        magnify = 2
        if useCSI:
            ax3.set_ylim([-10, +40])
        else:
            ax3.set_ylim([-100, 0])
        ax3.set_xlim([0, len(x_train[0])-1])
        if useCSI:
            for j in range(len((x_train[i][0]))):
                textX = []
                textY = []
                for k in range(len(x_train[i])):
                    curCsi = ((x_train[i][k]))
                    textX.append(k)
                    textY.append(curCsi[j]*magnify)
                ax3.plot(textX, textY,
                        label='CSI samplinged subcarrier')
        else:
            textX = []
            textY = []
            for k in range(len(x_train[i])):
                    curRssi = x_train[i][k]
                    textX.append(k)
                    textY.append(curRssi)
            ax3.plot(textX, textY,
                        label='RSSI samplinged subcarrier')
        maximum_test = np.max(y_train[i])
        curPred = (np.where(y_train[i] == maximum_test))[0][0]
        print('Sleep stage ' +str(focusStage[curPred]))
        ax0.plot([0, 1], [focusStage[curPred], focusStage[curPred]],
                 label='Sleep stage')
        return [ax0, ax3]

    ani = animation.FuncAnimation(fig, updatefig, frames=len(x_train), interval=5000,
                                  blit=True)
    plt.show()

if runTrain:
    if(useCSI):
        model = build_model(downsample=downsample, win_len=samplingedCSIWinSize*2,
                            n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=len(focusStage), data_len=52)
    else:
        model = build_model(downsample=downsample, win_len=samplingedCSIWinSize*2,
                            n_unit_lstm=n_unit_lstm, n_unit_atten=n_unit_atten, label_n=len(focusStage), data_len=1)
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

    matchCounter = 0
    stageTestCounter = [0 for i in range(len(focusStage))]
    stagePredCounter = [0 for i in range(len(focusStage))]
    PredCounter0 = [0 for i in range(len(focusStage))]
    PredCounter1 = [0 for i in range(len(focusStage))]
    PredCounter2 = [0 for i in range(len(focusStage))]
    PredCounter3 = [0 for i in range(len(focusStage))]

    for i in range(len(y_test)):
        maximum_test = np.max(y_test[i])
        maximum_pred = np.max(y_pred[i])
        curTest = (np.where(y_test[i] == maximum_test))[0][0]
        curPred = (np.where(y_pred[i] == maximum_pred))[0][0]
        if(curTest == curPred):
            matchCounter += 1

        stagePredCounter[curPred] += 1
        stageTestCounter[curTest] += 1

        if len(focusStage) == 4:
            if(curTest == 0):
                PredCounter0[curPred] += 1
            if(curTest == 1):
                PredCounter1[curPred] += 1
            elif(curTest == 2):
                PredCounter2[curPred] += 1
            elif(curTest == 3):
                PredCounter3[curPred] += 1
        elif len(focusStage) == 3:
            if(curTest == 0):
                PredCounter0[curPred] += 1
            elif(curTest == 1):
                PredCounter1[curPred] += 1
            elif(curTest == 2):
                PredCounter2[curPred] += 1
        elif len(focusStage) == 2:
            if(curTest == 0):
                PredCounter0[curPred] += 1
            elif(curTest == 1):
                PredCounter1[curPred] += 1

    print('len X', len(X))
    print('len Y', len(Y))
    print('len Xalt', len(Xalt))
    print('len Yalt', len(Yalt))
    for labelIdx in range(len(Sleeplabels)):
        print(Sleeplabels[labelIdx], 'len is', validLenCounter[labelIdx],
              ("TEST" if isTestLabels[labelIdx] else "TRAIN"))

    PredCounter0 = np.array(PredCounter0) / stageTestCounter[0]
    PredCounter1 = np.array(PredCounter1) / stageTestCounter[1]
    if len(focusStage) >= 3:
        PredCounter2 = np.array(PredCounter2) / stageTestCounter[2]
        if len(focusStage) >= 4:
            PredCounter3 = np.array(PredCounter3) / stageTestCounter[3]

    # toFixed 2
    PredCounter0 = np.around(PredCounter0, decimals=2)
    PredCounter1 = np.around(PredCounter1, decimals=2)
    PredCounter2 = np.around(PredCounter2, decimals=2)
    PredCounter3 = np.around(PredCounter3, decimals=2)

    print("   0 1 2 3")
    print("0 ", PredCounter0)
    print("1 ", PredCounter1)
    print("2 ", PredCounter2)
    print("3 ", PredCounter3)

    print("stageTestCounter", stageTestCounter)
    print("stagePredCounter", stagePredCounter)
    print("score", matchCounter, "/", len(y_test))
    print("score percent", matchCounter/len(y_test)*100, "/100")
