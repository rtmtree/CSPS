import numpy as np
from math import sqrt, atan2, isnan
import re
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.signal import butter, lfilter,filtfilt,find_peaks
from hampel import hampel

def CSIfilterer(filePath,my_filter_address,my_filter_length,Channel,timestampColName,tsParser,tail=False):
    print("read file csi", filePath)
    curFileSTA = pd.read_csv(filePath)
    curFileSTA = curFileSTA[(curFileSTA['mac'] == my_filter_address)]
    curFileSTA = curFileSTA[(curFileSTA['len'] == my_filter_length)]
    curFileSTA = curFileSTA[(curFileSTA['stbc'] == 0)]
    curFileSTA = curFileSTA[(curFileSTA['rx_state'] == 0)]
    curFileSTA = curFileSTA[(curFileSTA['sig_mode'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['bandwidth'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['secondary_channel'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['channel'] == Channel)]
    print("read file csi done")
    print("len csi file", len(curFileSTA.index))

    if(tail==False):
        tail = len(curFileSTA.index)

    curCSISTA = curFileSTA['CSI_DATA'].tail(tail)
    curRSSISTA = curFileSTA['rssi'].tail(tail)
    curTSSTA = curFileSTA[timestampColName].tail(tail)
    # for check rssi
    # curRSSI = curFileSTA['rssi'].tail(2)
    # RTrssiList = list((x) for x in curRSSI)
    # print("curRSSI is ",RTrssiList[0])
            
    csiList = list(x for x in curCSISTA)
    rssiList = list(x for x in curRSSISTA)
    tsList = list(x*tsParser for x in curTSSTA)
    return csiList,rssiList,tsList

# def parseCSI(csi):
#     try:
#         csi_string = re.findall(r"\[(.*)\]", csi)[0]
#         csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
#         return csi_raw
#     except:
#         return False


def rawCSItoAmp(data, length=128):
    amplitudes = []
    for j in range(0, length, 2):
        amplitudes.append(sqrt(data[j] ** 2 + data[j+1] ** 2))
    return amplitudes


def filterNullSC(data, length=64):
    amplitudes = []
    for j in range(0, length, 1):
        if (6 <= j < 32 or 33 <= j < 59):
            amplitudes.append(data[j])
    return amplitudes


def featureEngineer(X):
    FE_X = X
    for i in range(len(FE_X)):
        for j in range(len(FE_X[i])-1, 0, -1):
            FE_X[i][j] = FE_X[i][j]-FE_X[i][j-1]
        FE_X[i][0] = FE_X[i][0]-FE_X[i][0]
    return FE_X

def featureEngineerDataSeqSc(X,FE,gSigma):
    FE_X = X
    for iDataIdx in range(len(FE_X)):
        for iWinIdx in range(len(FE_X[iDataIdx])-1,-1,-1):
            if FE:
                array1 = np.array(FE_X[iDataIdx][iWinIdx])
                # array2 = np.array(FE_X[iDataIdx][iWinIdx-1])
                # array2 = np.array(FE_X[iDataIdx][0])
                array2 = np.array(FE_X[0][0])
                subtracted_array = np.subtract(array1, array2)
                subtracted = list(subtracted_array)
                FE_X[iDataIdx][iWinIdx] = subtracted
            FE_X[iDataIdx][iWinIdx] = gaussian_filter(FE_X[iDataIdx][iWinIdx],sigma=gSigma)
    print("featureEngineered",FE,gSigma)
    return FE_X


def featureEngineerNorm(X):
    FE_X = X
    for i in range(len(FE_X)):
        for j in range(len(FE_X[i])):
            FE_X[i][j] = FE_X[i][j]-FE_X[i][0]
    return FE_X


def csiIndices_sec(startTime, endTime, csiList):
    csiIndices = []
    # print("startTime",startTime)
    # print("endTime",endTime)
    for i in range(0, len(csiList)):
        if(csiList[i][0] >= startTime and csiList[i][0] < endTime):
            # print("addeed",csiList[i][0])
            csiIndices.append(i)
        elif(csiList[i][0] >= endTime):
            break
    return csiIndices


def poseIndices_sec(index, poseList, sec=1):
    endTime = poseList[index][0]
    startTime = endTime-sec
    poseIndices = [index]
    # print("startTime",startTime)
    # print("endTime",endTime)
    # for i in range(index,len(poseList)):
    #     if(poseList[i][0]<endTime):
    #         # print("addeed",poseList[i][0])
    #         poseIndices.append(i)
    #     else:
    #         break
    return poseIndices, startTime, endTime


def samplingCSI(csiList, csiIndices, poseList, poseIndices, paddingTo=30):
    simplingedCSIs = []
    expectedTSs = []
    for j in range(paddingTo):
        # print(poseIndices)
        # print(j)
        expectedTS = poseList[poseIndices[j]][0]
        # print("index",j,"expect",expectedTS)

        expectedTSs.append(expectedTS)

        csiIndicesExtended = []
        if(csiIndices[0] != 0 or j != 0):
            csiIndicesExtended += [csiIndices[0]-1]
        else:
            simplingedCSIs.append(np.array(filterNullSC(
                rawCSItoAmp(csiList[csiIndices[0]][1:]))))
            startIndex = 0
            continue

        csiIndicesExtended += csiIndices

        if(csiIndices[-1] != len(csiList)-1):
            csiIndicesExtended += [csiIndices[-1]+1]
        # else:
        #     None

        if j == 0:
            startIndex = csiIndicesExtended[0]
        for k in range(startIndex, csiIndicesExtended[-1]+1):
            if(k >= len(csiList) or csiList[k][0] > expectedTS):
                break
            else:
                startIndex = k
        # if startIndex TS matched expected TS

        if(csiList[startIndex][0] == expectedTS):
            simplingedCSIs.append(np.array(filterNullSC(
                rawCSItoAmp(csiList[startIndex][1:]))))
            continue

        if(startIndex == len(csiList)-1):
            simplingedCSIs.append(np.array(filterNullSC(
                rawCSItoAmp(csiList[startIndex][1:]))))
            continue

        endIndex = startIndex+1

        startCSI = filterNullSC(rawCSItoAmp(csiList[startIndex][1:]))
        endCSI = filterNullSC(rawCSItoAmp(csiList[endIndex][1:]))
        middleCSI = []
        offsetX = csiList[endIndex][0]-csiList[startIndex][0]
        offsetXo = expectedTS - csiList[startIndex][0]
        for k in range(52):
            offsetY = endCSI[k]-startCSI[k]
            offsetYo = (offsetXo*offsetY) / offsetX

            middleCSI.append((startCSI[k]+offsetYo))

        simplingedCSIs.append(np.array(middleCSI))
    return simplingedCSIs, expectedTSs


def samplingCSISleep(csiList, csiIndices, lastTS, newCSILen, timeLen=30):
    simplingedCSIs = []
    expectedTSs = []
    startTS = lastTS - timeLen
    for j in range(newCSILen):
        if(j < newCSILen-1):
            expectedTS = startTS + (j * timeLen/newCSILen)
        else:
            expectedTS = lastTS

        expectedTSs.append(expectedTS)

        csiIndicesExtended = []
        if(csiIndices[0] != 0 or j != 0):
            csiIndicesExtended += [csiIndices[0]-1]
        else:
            simplingedCSIs.append(np.array(filterNullSC(
                rawCSItoAmp(csiList[csiIndices[0]][1:]))))
            startIndex = 0
            continue

        csiIndicesExtended += csiIndices

        if(csiIndices[-1] != len(csiList)-1):
            csiIndicesExtended += [csiIndices[-1]+1]
        # else:
        #     None

        if j == 0:
            startIndex = csiIndicesExtended[0]
        for k in range(startIndex, csiIndicesExtended[-1]+1):
            if(k >= len(csiList) or csiList[k][0] > expectedTS):
                break
            else:
                startIndex = k

        # if startIndex TS matched expected TS
        if(csiList[startIndex][0] == expectedTS):
            simplingedCSIs.append(np.array(filterNullSC(
                rawCSItoAmp(csiList[startIndex][1:]))))
            continue

         # if startIndex is at the end of CSI data
        if(startIndex == len(csiList)-1):
            simplingedCSIs.append(np.array(filterNullSC(
                rawCSItoAmp(csiList[startIndex][1:]))))
            continue

        endIndex = startIndex+1

        # startCSI=filterNullSC( rawCSItoAmp(   csiList[startIndex][1:]  )  )
        # endCSI=filterNullSC( rawCSItoAmp(   csiList[endIndex][1:]  )  )
        startCSI = csiList[startIndex]
        endCSI = csiList[endIndex]
        offsetX = endCSI[0]-startCSI[0]
        offsetXo = expectedTS - startCSI[0]
        middleCSI = []
        for k in range(len(startCSI[1:])):
            offsetY = endCSI[1:][k]-startCSI[1:][k]
            offsetYo = (offsetXo*offsetY) / offsetX
            middleCSI.append((startCSI[1:][k]+offsetYo))
        simplingedCSIs.append(np.array(filterNullSC(rawCSItoAmp(middleCSI))))
    return simplingedCSIs, expectedTSs

def singleLinearInterpolation(csiList, csiIndices, timestampList, digitTs, timeLen=60):
    simplingedCSIs = []
    expectedTSs = []
    # startTS = timestampList[-1] - timeLen
    if(len(timestampList)==0):
        return simplingedCSIs,expectedTSs
    startTS = timestampList[0] 
    lastTS = timestampList[-1] 
    # print("startTS",startTS)
    # print("lastTS",lastTS)
    newCSILen = int( (lastTS - startTS)/(10**digitTs) * timeLen )
    for j in range(newCSILen):
        if(j < newCSILen-1):
            # expectedTS = startTS + (j * timeLen/newCSILen)
            expectedTS = startTS + (j * (lastTS-startTS)/newCSILen)
        else:
            expectedTS = lastTS
        
        expectedTSs.append(expectedTS)
        # print(expectedTS)
        csiIndicesExtended = []
        if(csiIndices[0] != 0 or j != 0):
            csiIndicesExtended = csiIndices
        else:
            
            simplingedCSIs.append(csiList[csiIndices[0]])
            startIndex = 0
            continue
        if j == 0:
            startIndex = csiIndicesExtended[0]
        for k in range(startIndex, csiIndicesExtended[-1]+1):
            if(k >= len(csiList) or timestampList[k] > expectedTS):
                break
            else:
                startIndex = k

        # if startIndex TS matched expected TS
        if(timestampList[startIndex] == expectedTS):
            simplingedCSIs.append(csiList[startIndex])
            continue

         # if startIndex is at the end of CSI data
        if(startIndex == len(csiList)-1):
            simplingedCSIs.append(csiList[startIndex])
            continue

        endIndex = startIndex+1

        startCSI = csiList[startIndex]
        endCSI = csiList[endIndex]
        offsetX = timestampList[endIndex]-timestampList[startIndex]
        offsetXo = expectedTS - timestampList[startIndex]
        offsetY = endCSI-startCSI
        offsetYo = (offsetXo*offsetY) / offsetX
        middleCSI = (startCSI+offsetYo)
        middleCSI = (startCSI)
        simplingedCSIs.append(middleCSI)
    return simplingedCSIs, expectedTSs

def moving_average(x, w):
    return np.convolve(np.array(x), np.ones(w), 'valid') / w

def samplingRSSISleep(csiList, csiIndices, poseList, sleepIndex, newCSILen, timeLen=30):
    simplingedCSIs = []
    expectedTSs = []
    startTS = poseList[sleepIndex][0] - timeLen
    for j in range(newCSILen):
        if(j < newCSILen-1):
            expectedTS = startTS + \
                (j * (poseList[sleepIndex][0] - startTS)/newCSILen)
        else:
            expectedTS = poseList[sleepIndex][0]

        expectedTSs.append(expectedTS)

        csiIndicesExtended = []
        if(csiIndices[0] != 0 or j != 0):
            csiIndicesExtended += [csiIndices[0]-1]
        else:
            simplingedCSIs.append(np.array(csiList[csiIndices[0]][1:]))
            startIndex = 0
            continue

        csiIndicesExtended += csiIndices

        if(csiIndices[-1] != len(csiList)-1):
            csiIndicesExtended += [csiIndices[-1]+1]
        # else:
        #     None

        if j == 0:
            startIndex = csiIndicesExtended[0]
        for k in range(startIndex, csiIndicesExtended[-1]+1):
            if(k >= len(csiList) or csiList[k][0] > expectedTS):
                break
            else:
                startIndex = k
        # if startIndex TS matched expected TS

        if(csiList[startIndex][0] == expectedTS):
            simplingedCSIs.append(np.array(csiList[startIndex][1:]))
            continue

        if(startIndex == len(csiList)-1):
            simplingedCSIs.append(np.array(csiList[startIndex][1:]))
            continue

        endIndex = startIndex+1

        startCSI = csiList[startIndex][1:]
        endCSI = csiList[endIndex][1:]
        middleCSI = []
        offsetX = csiList[endIndex][0]-csiList[startIndex][0]
        offsetXo = expectedTS - csiList[startIndex][0]
        for k in range((1)):
            offsetY = endCSI[k]-startCSI[k]
            offsetYo = (offsetXo*offsetY) / offsetX

            middleCSI.append((startCSI[k]+offsetYo))

        simplingedCSIs.append(np.array(middleCSI))
    return simplingedCSIs, expectedTSs


def imageIdx2csiIndices_timestamp(poseIdx, poseList, csiList, skipframe=1):
    timeInPose = poseList[poseIdx][0]
    if (poseIdx > 0):
        prevTimeInPose = poseList[poseIdx-skipframe][0]
    else:
        prevTimeInPose = 0

    csiIndices = []
    for i in range(len(csiList)):
        if(prevTimeInPose < csiList[i][0] <= timeInPose):
            csiIndices.append(i)

    return csiIndices


def sleepIdx2csiIndices_timestamp( lastTs, csiStartIdx, csiList, timeLen=30):
    timeInPose = lastTs
    prevTimeInPose = lastTs-timeLen
    csiIndices = []
    if(isinstance(prevTimeInPose, np.int64) and isinstance(timeInPose, np.int64)):
        for i in range(csiStartIdx, len(csiList)):
            # print("csiList",type(csiList[i][0]))
            # print("timeInPose",type(timeInPose))
            if(csiList[i][0] >= timeInPose):
                break
            if(isinstance(csiList[i][0], np.float64)):
                if(prevTimeInPose <= csiList[i][0] and csiList[i][0] < timeInPose):
                    csiIndices.append(i)
    return csiIndices

def hampel_filter(raw_signal,win_size=3,sigma=3) :
    copy_signal = np.copy(np. asarray(raw_signal))
    n = len(raw_signal)
    for i in range((win_size), (n-win_size)):
        dataslice = copy_signal[i- win_size:i+ win_size]
        median_abs_dev = med_abs_dev(dataslice)
        median = np.median(dataslice)
        if copy_signal[i] > median + (sigma * median_abs_dev):
            # print("filter",copy_signal[i],"=>",median)
            copy_signal[i] = median
    return copy_signal
def med_abs_dev(datapoints):
    med = np.median(datapoints)
    return np.median(np.abs (datapoints- med))

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def hampel_filter_forloop(input_series, window_size, n_sigmas=3):
    
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    
    # possibly use np.nanmedian 
    for i in range((window_size),(n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices
def hampel_filter_v(data, half_win_length, threshold):
    padded_data = np.concatenate(
        [[np.nan]*half_win_length, data, [np.nan]*half_win_length])
    windows = np.ma.array(
        np.lib.stride_tricks.sliding_window_view(padded_data, 2*50+1))
    windows[np.isnan(windows)] = np.ma.masked
    median = np.ma.median(windows, axis=1)
    mad = np.ma.median(np.abs(windows-np.atleast_2d(median).T), axis=1)
    bad = np.abs(data-median) > (mad*threshold)
    return np.where(bad)[0]

def butter_lowpass_filter(data, cutoff, fs, order,nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
def butter_lowpass_filter_fft(data, cutoff, fs, order):
        nyq = 0.5 * fs # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a,data)
        return y
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def getCentralFrequency(channel):
    cenFreqList = [2412, 2417, 2422, 2427, 2432, 2437,
                   2442, 2447, 24452, 2457, 2462, 2467, 2472]
    return cenFreqList[channel-1]


def getFrequencyRange(cenFreq, bandwidth, subcarrierLength):
    startFreq = cenFreq - (bandwidth/2)
    Xaxis = []
    for i in range(subcarrierLength):
        Xaxis.append(startFreq+(i*(bandwidth/subcarrierLength)))
    return Xaxis


def parseCSI(csi):
    csi_string = re.findall(r"\[(.*)\]", csi)[0]
    csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
    return csi_raw


def findAVGCSI(fileAVG):
    print("read STABLE file", fileAVG)
    curFileSTA = pd.read_csv(fileAVG)
    # curFileSTA = curFileSTA[(curFileSTA['mac'] == my_filter_address)]
    # curFileSTA = curFileSTA[(curFileSTA['len'] == my_filter_length)]
    curFileSTA = curFileSTA[(curFileSTA['stbc'] == 0)]
    curFileSTA = curFileSTA[(curFileSTA['rx_state'] == 0)]
    curFileSTA = curFileSTA[(curFileSTA['sig_mode'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['bandwidth'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['secondary_channel'] == 1)]
    print("read STABLE file done")
    print("len file", len(curFileSTA.index))
    tail = int((int(len(curFileSTA.index)/2))+(PacketLength/2))
    head = PacketLength
    print(tail)
    print(head)
    curCSISTA = curFileSTA['CSI_DATA'].tail(tail).head(head)
    csiSTAList = list(x for x in curCSISTA)

    parseCSIList = [parseCSI(csiSTAList[i]) for i in range(len(csiSTAList)-1)]
    AVGCSI = []
    for i in range(0, subcarrierLengthX2, 2):
        sumCSICol = 0
        for j in range(len(parseCSIList)):
            # raw
            # sumCSICol=sumCSICol+parseCSIList[j][i]
            # amplitude
            sumCSICol = sumCSICol + \
                (sqrt(parseCSIList[j][i] * 2 + parseCSIList[j][i+1] * 2))
        AVGCSI.append(sumCSICol/len(parseCSIList))
    return AVGCSI
    # return []
