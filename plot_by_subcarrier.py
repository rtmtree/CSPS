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
from functions.csi_util import filterNullSC,rawCSItoAmp

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


# style.use('dark_background')
# my_filter_address = "98:F4:AB:7D:DD:1D"
my_filter_address="7C:9E:BD:D2:D8:9D"
# =================================================
isRealtime = True
<<<<<<< HEAD
# filePaths = ['get_csi/esp32-csi-tool/active_sta/0358.csv']
=======
>>>>>>> 6b17d88 (change path of log csv)
filePaths = ['get_csi/esp32-csi-tool/active_sta/xx.csv']
# filePaths = ['active_sta/sleep18-11-2021.csv']
# =================================================
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
frameLength = 200
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
    curTSSTA = curFileSTA['local_timestamp'].tail(tail)
    tsSTAList = list(x*(10**6) for x in curTSSTA)
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

x_values = [[] for i in range(200)]
y_values = [[] for i in range(200)]

index = count()
fig, ax = plt.subplots(1, 1, figsize=(6, 6))


def animate(line):

    ax.cla()
    # print("animate",line)
    valueSTA = 0.1
    valueAP = 0.1

    plt.cla()
    fileIdx = 0
    print("============filePath1", filePaths[fileIdx])
    print("============color CSI", colorCSIs[fileIdx])
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
            curTS = curFileSTA['local_timestamp'].tail(tail)
            tsSTAList = list(x for x in curTS)
            # curChannel = curFileSTA['channel'].tail(tail)
            # channelSTAList = list(x for x in curChannel)
            print("startIndex", curCSI.index[0])
            print("frameLength", frameLength)
            print("last ts", tsSTAList[-2])
            print("last csi", parseCSI(RTcsiList[-2])[40])
            print("last rssi", RTrssiList[-2])
            valueSTA = []
            tsSTA = []
            rssiSTA = []
            for i in range(0, frameLength):  # cut last element
                valueSTA.append(parseCSI(RTcsiList[i]))
                tsSTA.append(tsSTAList[i])
                rssiSTA.append(RTrssiList[i])
        except:
            print("catch animate RT")
            return
    else:
        global csiList
        global rssiList
        global tsList
        print("batch Index", line)
        print("frameLength", frameLength)
        print("last ts", tsList[line+frameLength-1])
        print("last csi", parseCSI(csiList[line+frameLength-1])[40])
        print("last rssi", rssiList[line+frameLength-1])
        valueSTA = []
        tsSTA = []
        rssiSTA = []
        for i in range(line, line+frameLength):
            valueSTA.append(parseCSI(csiList[i]))
            tsSTA.append(tsList[i])
            rssiSTA.append(rssiList[i])
    if(isinstance(valueSTA, float) == False and len(valueSTA) > 0):

        amplitudesAll = [ filterNullSC( rawCSItoAmp( curAmp )) for curAmp in valueSTA]
        tsAll = tsSTA
        # Ploting Start
        # new plot subcarrier:
        for j in range(len(amplitudesAll[0])):
            textX = []
            textY = []
            for k in range(len(amplitudesAll)):
                tsAll[k] = int(tsAll[k])
                textX.append(tsAll[k])
                textY.append(amplitudesAll[k][j])
            ax.plot(textX, textY, label='CSI subcarrier')
    plt.xlabel("Frame")
    plt.ylabel("Amplitude(dB)")
    ax.set_ylim([-10, +40])


ani = animation.FuncAnimation(
    fig, animate, interval=1000 if isRealtime else 500)
plt.show()