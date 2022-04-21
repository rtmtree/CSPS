import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import count
import sys
from math import sqrt, atan2, isnan
import numpy as np
from matplotlib import style
from matplotlib import collections as matcoll
import time
from datetime import datetime
import random
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from scipy.ndimage import gaussian_filter
import re
from scipy import pi
from statistics import stdev
from functions.csi_util import filterNullSC, rawCSItoAmp, singleSamplingCSISleep,moving_average
from hampel import hampel
from scipy.signal import butter, lfilter

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
    curFileSTA = curFileSTA[(curFileSTA['mac'] == my_filter_address)]
    curFileSTA = curFileSTA[(curFileSTA['len'] == 384)]
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


def checkWithThreshold(rssiList, threshold):
    if(rssiList[0] > threshold
       and rssiList[1] > threshold
       and rssiList[2] > threshold
       and rssiList[3] > threshold
       and rssiList[4] > threshold):
        return True
    else:
        return False


def basicDetection(rssiList):
    len(rssiList)
    if(checkWithThreshold(rssiList, -55)):
        print("!!!======== There can be human in 0.5m ==========!!!")
    elif(checkWithThreshold(rssiList, -60)):
        print("!!!======== There can be human in 1m ==========!!!")
    else:
        print("!!!======== There can be nothing ==========!!!")


def calculateI_FFT(n, amplitude_spect, phase_spect):

    data = list()

    for mag, phase in zip(amplitude_spect, phase_spect):
        data.append((mag*n/2)*(np.cos(phase)+1j * np.sin(phase)))
    full_data = list(data)
    i_data = np.fft.irfft(data)
    return i_data

 # timestampColName = 'local_timestamp'
timestampColName = 'real_timestamp'
# style.use('dark_background')
# my_filter_address = "98:F4:AB:7D:DD:1D"
my_filter_address = "7C:9E:BD:D2:D8:9D"
# =================================================
isRealtime = False
startFrom = 0
frameLength = 2000
# 1536 #xx2 == 00:19
# 3000 #xx3 == 00:35
filePaths = ['get_csi/esp32-csi-tool/active_sta/xx5.csv']
# filePaths = ['active_sta/sleep18-11-2021.csv']
# =================================================
interval = 1000 if isRealtime else 1000
my_filter_length = 128
Channels = [6]
channelColors = ['red', 'green', 'blue', 'yellow', 'pink']
plotOffset = False
filePathOffset = 'active_ap/csit_06_stable.csv'
colorCSIs = ['red', 'green', 'blue']
colorPDP = ['red', 'green', 'blue']
nullSubcarrier = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
shows = [True, True, True]

stableCSI = 0.1

subcarrierLength = 64
subcarrierLengthX2 = int(subcarrierLength*2)
PacketLength = frameLength
csiSTAList = []
rssiSTAList = []
tsSTAList = []

csiAPList = []
rssiAPList = []
tsAPList = []

csiList = []
rssiList = []
tsList = []
# channelList = []
if(isRealtime == False):
    print("read file", filePaths[0])
    curFileSTA = pd.read_csv(filePaths[0])
    curFileSTA = curFileSTA[(curFileSTA['mac'] == my_filter_address)]
    curFileSTA = curFileSTA[(curFileSTA['len'] == 384)]
    curFileSTA = curFileSTA[(curFileSTA['stbc'] == 0)]
    curFileSTA = curFileSTA[(curFileSTA['rx_state'] == 0)]
    curFileSTA = curFileSTA[(curFileSTA['sig_mode'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['bandwidth'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['secondary_channel'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['channel'] == Channels[0])]
    print("read file done")
    print("len file", len(curFileSTA.index))
    # head = len(curFileSTA.index)
    # tail = int((int(len(curFileSTA.index)/2))+(PacketLength/2))
    tail = len(curFileSTA.index)
    print(tail)
    # print(head)
    curCSISTA = curFileSTA['CSI_DATA'].tail(tail)
    csiSTAList = list(x for x in curCSISTA)
    curRSSISTA = curFileSTA['rssi'].tail(tail)
    rssiSTAList = list(x for x in curRSSISTA)

    curTSSTA = curFileSTA[timestampColName].tail(tail)
    tsSTAList = list(x*(10**6) for x in curTSSTA)
    print("tsSTAList[0]")
    print(tsSTAList[0])
    # tsSTAList = list(x for x in curTSSTA)
    curChannelSTA = curFileSTA['channel'].tail(tail)
    channelSTAList = list(x for x in curChannelSTA)
    csiList = (csiSTAList)
    rssiList = (rssiSTAList)
    tsList = (tsSTAList)
    # channelList.append(channelSTAList)

if(plotOffset):
    AVGCSI = findAVGCSI(filePathOffset)

    print("AVGCSI")
    print(AVGCSI)
    print(len(AVGCSI))


index = count()
# fig, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=(6, 9))
fig, [ax2, ax1, ax0] = plt.subplots(3, 1, figsize=(6, 9))


def animate(line):

    ax0.cla()
    ax1.cla()
    ax2.cla()
    # print("animate",line)
    valueSTA = 0.1
    valueAP = 0.1

    plt.cla()
    fileIdx = 0
    print("============filePath1", filePaths[fileIdx])
    # print("============color CSI", colorCSIs[fileIdx])
    valueSTA = 0.1
    valueAP = 0.1
    if(isRealtime == True):
        try:
            curFileSTA = pd.read_csv(filePaths[fileIdx])
            curFileSTA = curFileSTA[(
                curFileSTA['mac'] == my_filter_address)]
            curFileSTA = curFileSTA[(curFileSTA['len'] == 384)]
            curFileSTA = curFileSTA[(curFileSTA['stbc'] == 0)]
            curFileSTA = curFileSTA[(curFileSTA['rx_state'] == 0)]
            curFileSTA = curFileSTA[(curFileSTA['sig_mode'] == 1)]
            curFileSTA = curFileSTA[(curFileSTA['bandwidth'] == 1)]
            curFileSTA = curFileSTA[(curFileSTA['channel'] == Channels[0])]
            curFileSTA = curFileSTA[(curFileSTA['secondary_channel'] == 1)]
            tail = frameLength + 1
            curCSI = curFileSTA['CSI_DATA'].tail(tail)
            RTcsiList = list((x) for x in curCSI)
            curRSSI = curFileSTA['rssi'].tail(tail)
            RTrssiList = list(x for x in curRSSI)
            curTS = curFileSTA[timestampColName].tail(tail)
            tsSTAList = list(x for x in curTS)
            # curChannel = curFileSTA['channel'].tail(tail)
            # channelSTAList = list(x for x in curChannel)
            print("#Index", curCSI.index[0])
            print("frameLength", frameLength)
            print("last ts", tsSTAList[-2])
            print("last rssi", RTrssiList[-2])
            valueSTA = []
            tsSTA = []
            rssiSTA = []
            for i in range(0, frameLength):  # cut last element
                valueSTA.append(parseCSI(RTcsiList[i]))
                tsSTA.append(tsSTAList[i]*1000)
                rssiSTA.append(RTrssiList[i])
            print(len(valueSTA))
        except:
            print("catch animate RT")
            return
    else:
        line = line+startFrom
        global csiList
        global rssiList
        global tsList

        print("batch Index", line)
        valueSTA = []
        tsSTA = []
        rssiSTA = []

        if False:
            # crop with frame len
            print("last ts", tsList[line+frameLength-1])
            # print("frameLength", frameLength)
            for i in range(line, line+frameLength):
                valueSTA.append(parseCSI(csiList[i]))
                tsSTA.append(tsList[i])
                rssiSTA.append(rssiList[i])
        else:
            maxTs = line*interval*1000
            print("batch maxTs", maxTs)
            # crop with timestamp
            lineIndex = 0
            while(tsList[lineIndex] < maxTs):
                lineIndex += 1
            curLength = (lineIndex-frameLength) if (lineIndex -
                                                    frameLength) > 0 else 0
            for i in range(lineIndex, curLength, -1):
                valueSTA.append(parseCSI(csiList[i]))
                tsSTA.append(tsList[i])
                rssiSTA.append(rssiList[i])
            valueSTA.reverse()
            tsSTA.reverse()
            rssiSTA.reverse()

    if(isinstance(valueSTA, float) == False and len(valueSTA) > 0):

        amplitudesAll = [filterNullSC(rawCSItoAmp(curAmp))
                         for curAmp in valueSTA]
        tsAll = tsSTA
        # Ploting Start
        # new plot subcarrier:
        sumY = [0 for j in range(len(amplitudesAll))]
        for j in range(len(amplitudesAll[0])):  # 52/64
            textX = []
            textY = []
            for k in range(len(amplitudesAll)):
                tsAll[k] = int(tsAll[k])
                textX.append(tsAll[k])
                textY.append(amplitudesAll[k][j])
                sumY[k] += amplitudesAll[k][j]
            ax0.plot(textX, textY, label='CSI subcarrier')
            # ax0.plot(textX, gaussian_filter(
            #     textY, sigma=5), label='CSI subcarrier')
            # ax1.plot(textX, gaussian_filter(
            #     textY, sigma=10), label='CSI subcarrier')
            if(j==40):
                # ax2.plot(textX, gaussian_filter(textY,sigma=15), label='CSI subcarrier')
                seriesCSI = pd.Series(textY)
                hameFil = hampel(seriesCSI, window_size=5, n=3, imputation=True)
                ax1.plot(textX,hameFil , label='CSI subcarrier Hampel Filter')
                # linear interpolation to fit 60 Hz sampling frequency
                linIntCSI,linIntTS = singleSamplingCSISleep(textY,[k for k in range(len(hameFil))],textX,6,60)
                # print(linIntCSI)
                # print((linIntTS))
                # print(len(linIntCSI))
                # ax2.plot(linIntTS, linIntCSI , label='CSI subcarrier Linear Interpolation')  
                # linIntCSI = [ int(k) for k in linIntCSI]
                # print(linIntCSI) 
                mAFilter =  moving_average(linIntCSI,2)
                # ax2.plot([k for k in range(len(mAFilter))], mAFilter , label='CSI subcarrier Moving Average Filter')  
                fs = 5000.0
                lowcut = 500.0
                highcut = 1250.0
                BBFilter = butter_bandpass_filter(mAFilter, lowcut, highcut, fs, order=5)
                ax2.plot([k for k in range(len(mAFilter))], BBFilter , label='CSI subcarrier Butterworth Filter')  
        if False:
            for j in range(len(sumY)):
                sumY[j] = sumY[j]/len(amplitudesAll[0])
            SDY = [0 for j in range(len(amplitudesAll))]
            for k in range(len(amplitudesAll)):
                for j in range(len(amplitudesAll[0])):  # 52/64
                    SDY[k] += (amplitudesAll[k][j] - sumY[j])**2
                SDY[k] = SDY[k]/(len(amplitudesAll[0])-1)
            ax2.plot(textX, gaussian_filter(
                SDY, sigma=15), label='CSI subcarrier')
        # ax2.plot(textX, gaussian_filter(textY, sigma=10), label='CSI subcarrier')
    plt.xlabel("Frame")
    plt.ylabel("Amplitude(dB)")
    ax0.set_ylim([-10, +40])
    ax1.set_ylim([-10, +40])
    ax2.set_ylim([-10, +40])


ani = animation.FuncAnimation(
    fig, animate, interval=interval)
plt.show()
