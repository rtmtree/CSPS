from curses import raw
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
from matplotlib import collections as matcoll
import time
from datetime import datetime
import random
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from scipy.ndimage import gaussian_filter
import re
from scipy import pi
from statistics import stdev, mean
from functions.csi_util import filterNullSC, rawCSItoAmp, singleLinearInterpolation, moving_average, parseCSI, hampel_filter, butter_lowpass_filter_fft, CSIfilterer


def TimeDomain2BPMFTHM(dataList, tsList, fs, secondToCountBreath):
    peaksTS = []
    peaksPS = []
    meanPS = mean(dataList)
    for i in range(1, len(dataList)):
        if(dataList[i-1] < meanPS and dataList[i] > meanPS):
            peaksTS.append(tsList[i])
            peaksPS.append(dataList[i])
    BPM = len(peaksPS)*(fs/secondToCountBreath)
    return peaksTS, peaksPS, meanPS, BPM


def TimeDomain2BPMIOM(dataList, tsList, fs, secondToCountBreath, threshold=1):
    peaksTS = []
    peaksPS = []
    meanPS = mean(dataList)
    upperBound = mean(dataList)+threshold
    lowerBound = mean(dataList)-threshold
    for i in range(1, len(dataList)):
        if((dataList[i] > upperBound and len(peaksPS) == 0) or (len(peaksPS) > 0 and peaksPS[-1] < meanPS and dataList[i] > upperBound)):
            peaksTS.append(tsList[i])
            peaksPS.append(dataList[i])
        elif((dataList[i] < lowerBound and len(peaksPS) == 0) or (len(peaksPS) > 0 and peaksPS[-1] > meanPS and dataList[i] < lowerBound)):
            peaksTS.append(tsList[i])
            peaksPS.append(dataList[i])
    BPM = int(len(peaksPS)*(fs/secondToCountBreath) / 2)
    return peaksTS, peaksPS, meanPS, BPM


def TimeDomain2BPMFPH(dataList, tsList, fs, secondToCountBreath, threshold=1):
    peakX = 0
    maxDistance = 0
    meanPS = mean(dataList)
    for i in range(1, len(dataList)):
        if(abs(dataList[i]-meanPS) > maxDistance):
            # peakX = tsList[i]
            peakX = i
            maxDistance = abs(dataList[i] - meanPS)
    peaksTSF = [tsList[peakX]]
    peaksPSF = [dataList[peakX]]
    for i in range(peakX, len(dataList)):
        if(abs(dataList[i]-peaksPSF[-1]) > threshold and ((dataList[i] > meanPS and peaksPSF[-1] < meanPS) or (dataList[i] < meanPS and peaksPSF[-1] > meanPS))):
            peaksTSF.append(tsList[i])
            peaksPSF.append(dataList[i])
    peaksTSB = [tsList[peakX]]
    peaksPSB = [dataList[peakX]]
    for i in range(peakX, 0, -1):
        if(abs(dataList[i]-peaksPSB[-1]) > threshold and ((dataList[i] > meanPS and peaksPSB[-1] < meanPS) or (dataList[i] < meanPS and peaksPSB[-1] > meanPS))):
            peaksTSB.append(tsList[i])
            peaksPSB.append(dataList[i])
    peaksTS = [tsList[peakX]]
    peaksPS = [dataList[peakX]]
    # extend all except the first peak
    peaksTS.extend(peaksTSF[1:])
    peaksTS.extend(peaksTSB[1:])
    peaksPS.extend(peaksPSF[1:])
    peaksPS.extend(peaksPSB[1:])
    BPM = int(len(peaksPS)*(fs/secondToCountBreath) / 2)
    return peaksTS, peaksPS, meanPS, BPM


def TimeDomain2BPMFPHLP(dataList, tsList, fs, secondToCountBreath, threshold=1):
    peakX = 0
    maxDistance = 0
    meanPS = mean(dataList)
    for i in range(1, len(dataList)):
        if(abs(dataList[i]-meanPS) > maxDistance):
            peakX = i
            maxDistance = abs(dataList[i]-meanPS)
    print("first peak is", dataList[peakX], "at", tsList[peakX])
    # forward
    peaksTSF = [tsList[peakX]]
    peaksPSF = [dataList[peakX]]
    for i in range(peakX, len(dataList)):
        if(((dataList[i] > meanPS and peaksPSF[-1] < meanPS) or (dataList[i] < meanPS and peaksPSF[-1] > meanPS))):
            localPeakX = 0
            localMaxDistance = 0
            for j in range(i, len(dataList)):
                curLocalDistance = abs(dataList[j]-peaksPSF[-1])
                if(curLocalDistance > localMaxDistance):
                    localPeakX = j
                    localMaxDistance = curLocalDistance
                elif((dataList[j] > meanPS and dataList[i] > meanPS) or (dataList[j] < meanPS and dataList[i] < meanPS)):
                    break
            if(localMaxDistance >= threshold):
                peaksTSF.append(tsList[localPeakX])
                peaksPSF.append(dataList[localPeakX])
    # backward
    peaksTSB = [tsList[peakX]]
    peaksPSB = [dataList[peakX]]
    for i in range(peakX, 0, -1):
        if(((dataList[i] > meanPS and peaksPSB[-1] < meanPS) or (dataList[i] < meanPS and peaksPSB[-1] > meanPS))):
            localPeakX = 0
            localMaxDistance = 0
            for j in range(i, 0, -1):
                curLocalDistance = abs(dataList[j]-peaksPSB[-1])
                if(curLocalDistance > localMaxDistance):
                    localPeakX = j
                    localMaxDistance = curLocalDistance
                elif((dataList[j] > meanPS and dataList[i] > meanPS) or (dataList[j] < meanPS and dataList[i] < meanPS)):
                    break
            if(localMaxDistance >= threshold):
                peaksTSB.append(tsList[localPeakX])
                peaksPSB.append(dataList[localPeakX])
    peaksTS = [tsList[peakX]]
    peaksPS = [dataList[peakX]]
    # extend all except the first peak
    peaksTS.extend(peaksTSF[1:])
    peaksTS.extend(peaksTSB[1:])
    peaksPS.extend(peaksPSF[1:])
    peaksPS.extend(peaksPSB[1:])
    BPM = int(len(peaksPS)*(fs/secondToCountBreath) / 2)
    return peaksTS, peaksPS, meanPS, BPM


# timestampColName = 'local_timestamp'
timestampColName = 'real_timestamp'
power2timestamp = 6 if (timestampColName == 'real_timestamp') else 1
tsParser = (10**power2timestamp)
tsParserGT = (10**3)

# =================================================

# xx0  hold breath 104-133
# xx0  breath slowly 155-207
# xx0  breath normally 210-321 //bpm 18
# xx0  breath quickly 325

# xx3[8, 25]
# xx3 breath normally 300-400
# xx3 breath quickly 400-450 ***
# xx3 breath slowly 450-550
# xx3 hold breath 550-610

# xx4 [20,40]
# xx4 345,640
# xx4 breath normally 300-400 345 310,345 345,374
# xx4 breath quickly 400-500 410,440
# xx4 hold breath 500-600 510,540
# xx4 breath slowly 600-700 610,640

fileCode = 'xx1'
# sec
startFrom, endAt = 0, 999
isFreeze = False
showBPMOnPlot = True
# CSIBPMPlotPeriod = [20, 40]
# 0 = raw
# 1 = hampel
# 2 = gaussian
# 3 = linear inter
# 4 = butter_lowpass
# plotOrder=[2,3,4]
plotOrder = [0, 1, 4]
frameLength = 5000 # for RealTime only
timeLength = 70 * tsParser # for batch only


isRealtime = False
isWithGroundTruth = False
isCountBPM = False

# CSI filtering parameter part======================
# hampel_filter
HF_sigma = -50
HF_winsize = 1
# gaussian_filter
GF_sigma = 5
# linear interpolation
LI_digi = 0
LI_timelen = 60
# butterworth low pass filter
BLPF_order = 2       # sin wave can be approx represented as quadratic
BLPF_cutoffFreq = 30
# Respiratory Extraction
# FPHLP
# RRE_threshold = 1 # good for normal,slow,hold / bad for quick
RRE_threshold = 0.7
# RRE_threshold = 0.3 # good for normal,slow,quick / bad for hold
# PRESSURE filtering parameter part======================
isFilterGT = True
GT_GF_sigma = 1
GT_HF_sigma = -100
GT_HF_winsize = 1
# FPHLP
GT_RRE_threshold = 0.4
# GT_RRE_threshold = 0.2
# =================================================

# sync part
fs = LI_timelen  # sample rate, Hz
secondToCountBreath = 30  # second
secPerGap = 30
interval = 1000 if isRealtime else 1000
PDMADList = []
GTMADList = []

# GroundTruth PART=======
gtTSList = []
gtPSList = []
syncTime = 0

# CSI PART============
subcarrierIndex = [1]

my_filter_address = "7C:9E:BD:D2:D8:9D"
# my_filter_addressForAP = "7C:9E:BD:D2:D8:9D"
my_filter_addressForAP = "98:F4:AB:7D:DD:1C"
my_filter_length = 384
Channel = 6
subcarrierLength = 64
subcarrierLengthX2 = int(subcarrierLength*2)
PacketLength = frameLength
# init
csiSTAList = []
rssiSTAList = []
tsSTAList = []
csiList = []
rssiList = []
tsList = []
tsAPList = []
# end CSI PART============

filePaths = ['get_csi/esp32-csi-tool/'+fileCode+'sta.csv']
# filePathsAP = ['get_csi/esp32-csi-tool/active_sta/'+fileCode+'.csv']
filePathsAP = ['get_csi/esp32-csi-tool/'+fileCode+'ap.csv']
filePathsGT = ['get_groundtruth/'+fileCode+'.csv']
if(isRealtime == False):

    csiList, _, tsList = CSIfilterer(
        filePaths[0], my_filter_address, my_filter_length, Channel, timestampColName, tsParser, tail=False)
    csiAPList, _, tsAPList = CSIfilterer(
        filePathsAP[0], my_filter_addressForAP, my_filter_length, Channel, timestampColName, tsParser, tail=False)
    syncTimeSTAsubAP = tsList[-1] - tsAPList[-1]
    print("syncTimeSTAsubAP",syncTimeSTAsubAP)
    tsAPList = [(tsAPList[i])+syncTimeSTAsubAP for i in range(0, len(tsAPList))]
    
    if(isWithGroundTruth):
        print("read file groundtruth", filePathsGT[0])
        curFileGT = pd.read_csv(filePathsGT[0])
        print("read file groundtruth done")
        print("len groundtruth file", len(curFileGT.index))
        tailGT = len(curFileGT.index)
        curPSGT = curFileGT['pressure'].tail(tailGT)
        curTSGT = curFileGT[timestampColName].tail(tailGT)

        gtPSList = list(x for x in curPSGT)
        gtTSList = list(x*tsParserGT for x in curTSGT)
        print("last Timestamp of groundtruth", tsList[len(
            tsList)-1], "last Timestamp of GT", gtTSList[len(gtTSList)-1])
        syncTime = tsList[len(tsList)-1]-gtTSList[len(gtTSList)-1]
        print("syncTime is", syncTime)
        gtTSList = [x+syncTime for x in gtTSList]

# fig, [ax0, ax1, ax2,ax3] = plt.subplots(4, 1, figsize=(6, 9))
fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, figsize=(6, 9))
# fig, [ax2, ax1, ax0] = plt.subplots(3, 1, figsize=(6, 9))
# fig, [ax2, ax1, ax0] = plt.subplots(3, 1, figsize=(6, 9))


def animate(line):
    global gtTSList
    global gtPSList
    global tsList
    # global CSIBPMPlotPeriod

    # ax0.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    # print("animate",line)
    valueSTA = 0.1

    plt.cla()
    fileIdx = 0
    print("============filePath1", filePaths[fileIdx])
    # print("============color CSI", colorCSIs[fileIdx])
    valueSTA = 0.1
    valueAP = 0.1
    if(isRealtime == True):
        # try:
        tail = frameLength + 1
        RTcsiSTAList, _, RTtsSTAList = CSIfilterer(
            filePaths[0], my_filter_address, my_filter_length, Channel, timestampColName, tsParser, tail=tail)
        RTcsiAPList, _, RTtsAPList = CSIfilterer(
            filePathsAP[0], my_filter_addressForAP, my_filter_length, Channel, timestampColName, tsParser, tail=tail)

        # crop out one probably invalid row / bc realtime make uncontrollable snapshot
        RTcsiSTAList = RTcsiSTAList[0:len(RTcsiSTAList)-1]
        RTtsSTAList = RTtsSTAList[0:len(RTtsSTAList)-1]
        
        RTcsiAPList = RTcsiAPList[0:len(RTcsiAPList)-1]
        RTtsAPList = RTtsAPList[0:len(RTtsAPList)-1]
        syncTimeSTAsubAP = RTtsSTAList[-1] - RTtsAPList[-1]
        print("syncTimeSTAsubAP",syncTimeSTAsubAP)

        print("len RTcsiSTAList",len(RTcsiSTAList))
        print("len RTcsiAPList",len(RTcsiAPList))
        # parseCSI and fix size to be ==>frameLength
        valueSTA = [parseCSI(RTcsiSTAList[i]) for i in range(0, frameLength)]
        tsSTA = [(RTtsSTAList[i]) for i in range(0, frameLength)]
        valueAP = [parseCSI(RTcsiAPList[i]) for i in range(0, frameLength)]
        # sync time of AP to STA
        tsAP = [(RTtsAPList[i] + syncTimeSTAsubAP) for i in range(0, frameLength)]

        firstTsCSI = tsSTA[0]
        lastTsCSI = tsSTA[len(tsSTA)-1]
        lastTsCSIforBPM = (firstTsCSI) + (secondToCountBreath*tsParser)

        if(isWithGroundTruth):
            curFileGT = pd.read_csv(filePathsGT[0])
            tailGT = len(curFileGT.index)
            curPSGT = curFileGT['pressure'].tail(tailGT)
            curTSGT = curFileGT[timestampColName].tail(tailGT)

            gtPSList = list(x for x in curPSGT)
            gtTSList = list(x*tsParserGT for x in curTSGT)
            # crop out one probably invalid row
            gtPSList = gtPSList[0:len(gtPSList)-1]
            gtTSList = gtTSList[0:len(gtTSList)-1]
            syncTime = tsSTA[len(tsSTA)-1]-gtTSList[len(gtTSList)-1]
            gtTSList = [x+syncTime for x in gtTSList]
        # except:
        #     print("catch animate RT")
        #     return
    else:
        if(isFreeze):
            line = startFrom
        else:
            line = line+startFrom
        global fs
        global csiList
        global csiAPList
        global tsAPList
        # global rssiList

        print("batch Index", line)
        if(endAt != 0 and line == endAt):
            plt.close(fig)
            # quit()
        
        valueSTA = []
        tsSTA = []
        valueAP = []
        tsAP = []
        rssiSTA = []

        minTs = (line*tsParser)
        maxTs = minTs + timeLength
        print("batch minTs", minTs, "maxTs", maxTs)
        # crop with timestamp

        # CSI part

        # parse STA
        valueSTA = []
        tsSTA = []
        for i in range(0, len(csiList)):
            if(tsList[i] > minTs and tsList[i] <= maxTs):
                valueSTA.append(parseCSI(csiList[i]))
                tsSTA.append(tsList[i])
            elif(tsList[i] > maxTs):
                break

        print("CSI valueSTA len", len(valueSTA))
        print("CSI last ts", tsSTA[-1])

        # parse AP
        valueAP = []
        tsAP = []
        for i in range(0, len(csiAPList)):
            if(tsAPList[i] > minTs and tsAPList[i] <= maxTs):
                valueAP.append(parseCSI(csiAPList[i]))
                tsAP.append(tsAPList[i])
            elif(tsAPList[i] > maxTs):
                break

        print("CSI valueAP len", len(valueAP))
        print("CSI last ts", tsAP[-1])


        firstTsCSI = tsSTA[0]
        lastTsCSI = tsSTA[len(tsSTA)-1]
        lastTsCSIforBPM = (firstTsCSI) + (secondToCountBreath*tsParser)

    if(isWithGroundTruth):
        valueGTPlot = []
        tsGTPlot = []
        psGTBPM = []
        tsGTBPM = []
        print("firstTsCSI", firstTsCSI, "lastTsCSI",
              lastTsCSI, "lastTsCSIforBPM", lastTsCSIforBPM)

        # create GroundTruth within the period same as CSI for ploting
        for i in range(0, len(gtTSList)):
            if(gtTSList[i] > firstTsCSI and gtTSList[i] <= lastTsCSI):
                valueGTPlot.append(gtPSList[i])
                # parse back
                tsGTPlot.append(float(gtTSList[i]/tsParser))
            elif(gtTSList[i] > lastTsCSI):
                break

        if(isFilterGT):
            filGtPSList = hampel_filter(
                valueGTPlot, GT_HF_winsize, GT_HF_sigma)
            filGtPSList = gaussian_filter(filGtPSList, sigma=GT_GF_sigma)
        else:
            filGtPSList = valueGTPlot
        filGtPSList, filGtTSList = singleLinearInterpolation(
            filGtPSList, np.arange(len(filGtPSList)), tsGTPlot, 0, 4)

        ax4.plot(filGtTSList, filGtPSList, label='Ground Truth')
        ax4.set_ylim([95, 105])
        ax4.set_ylabel("Pressure(kPa)")

        for i in range(0, len(filGtTSList)):
            if(filGtTSList[i] <= (lastTsCSIforBPM/tsParser)):
                psGTBPM.append(filGtPSList[i])
                tsGTBPM.append(filGtTSList[i])
            elif(filGtTSList[i] > (lastTsCSIforBPM/tsParser)):
                break
        if isCountBPM:
            if(len(psGTBPM) == 0):
                print("BELT!!! can't calculate BPM")
            else:
                # peaksGT, _ = find_peaks(psGTBPM,threshold=1)
                # peaksGT, _ = find_peaks(psGTBPM)
                # peaksGTTS = [tsGTBPM[x] for x in peaksGT]
                # peaksGTPS = [psGTBPM[x] for x in peaksGT]
                # peaksGTTS,peaksGTPS,meanGTPS,BPM = TimeDomain2BPMFTHM(psGTBPM,tsGTBPM,fs,secondToCountBreath)
                peaksGTTS, peaksGTPS, meanGTPS, BPM = TimeDomain2BPMFPHLP(
                    psGTBPM, tsGTBPM, fs, secondToCountBreath, GT_RRE_threshold)
                print("BELT!!! use data in", int(
                    tsGTBPM[0]), " -", int(tsGTBPM[len(tsGTBPM)-1]), "th sec")
                print("BELT!!! BPM =", BPM)
                GTMADList.append(BPM)

                ax4.axhline(y=meanGTPS, color='r', linestyle='-')
                ax4.plot(peaksGTTS, peaksGTPS, 'x')
                if showBPMOnPlot:
                    ax4.title.set_text('BPM=====>'+str(BPM))

    # CSI PART
    if((isinstance(valueSTA, float) == False and len(valueSTA) > 0) and isinstance(valueAP, float) == False and len(valueAP) > 0):

        amplitudesAll = [filterNullSC(rawCSItoAmp(curCSI))
                         for curCSI in valueSTA]
        amplitudesAllAP = [filterNullSC(rawCSItoAmp(curCSI))
                         for curCSI in valueAP]
        tsAllSTA = [float(curTs/tsParser) for curTs in tsSTA]
        tsAllAP = [float(curTs/tsParser) for curTs in tsAP]
        print("CSI tsAllSTA ts", tsAllSTA[-1])
        print("CSI tsAllAP ts", tsAllAP[-1])
        for j in range(len(amplitudesAll[0])):  # 52/64
            curSCTSSTA = tsAllSTA
            curSCTSAP = tsAllAP
            curSCCSISTA = [amplitudesAll[k][j] for k in range(len(amplitudesAll))]
            curSCCSIAP = [amplitudesAllAP[k][j] for k in range(len(amplitudesAllAP))]
            if(j in subcarrierIndex):
                graphList = []
                graphListAP = []
                # ax1.plot(curSCTS,textY , label='CSI subcarrier')
                graphList.append([curSCTSSTA, curSCCSISTA])
                graphListAP.append([curSCTSAP, curSCCSIAP])

                # Hampel Filter
                hameFil = hampel_filter(curSCCSISTA, HF_winsize, HF_sigma)
                hameFilAP = hampel_filter(curSCCSIAP, HF_winsize, HF_sigma)
                graphList.append([curSCTSSTA, hameFil])
                graphListAP.append([curSCTSAP, hameFilAP])
                # ax1.plot(curSCTS,hameFil , label='CSI subcarrier Hampel Filter')

                # Gaussian Filter
                gaussFil = gaussian_filter(hameFil, sigma=GF_sigma)
                gaussFilAP = gaussian_filter(hameFilAP, sigma=GF_sigma)
                graphList.append([curSCTSSTA, gaussFil])
                graphListAP.append([curSCTSAP, gaussFilAP])
                # ax1.plot(curSCTS,gaussFil , label='CSI subcarrier after Gaussian Filtering')

                # Linear Interpolation
                linIntCSISTA, linIntTSSTA = singleLinearInterpolation(
                    gaussFil, np.arange(len(gaussFil)), curSCTSSTA, LI_digi, LI_timelen)
                linIntCSIAP, linIntTSAP = singleLinearInterpolation(
                    gaussFilAP, np.arange(len(gaussFilAP)), curSCTSAP, LI_digi, LI_timelen)
                graphList.append([linIntTSSTA, linIntCSISTA])
                graphListAP.append([linIntTSAP, linIntCSIAP])
                # ax2.plot(linIntTS, linIntCSI , label='CSI subcarrier Linear Interpolation')

                # Butterworth Filter' low pass
                BLPF_T = len(linIntCSISTA)
                BLPF_sec = BLPF_T/LI_timelen         # Sample Period
                BLPF_cutoff = BLPF_cutoffFreq / BLPF_sec
                BBFilterSTA = butter_lowpass_filter_fft(
                    linIntCSISTA, BLPF_cutoff, LI_timelen, BLPF_order)
                BBFilterAP = butter_lowpass_filter_fft(
                    linIntCSIAP, BLPF_cutoff, LI_timelen, BLPF_order)

                graphList.append([linIntTSSTA, BBFilterSTA])
                graphListAP.append([linIntTSAP, BBFilterAP])
                # ax3.plot(linIntTS, BBFilter , label='CSI subcarrier Moving Average Filter')

                ax1.plot(graphList[plotOrder[0]][0],
                         graphList[plotOrder[0]][1])
                ax2.plot(graphList[plotOrder[1]][0],
                         graphList[plotOrder[1]][1])
                ax3.plot(graphList[plotOrder[2]][0],
                         graphList[plotOrder[2]][1])
                ax1.plot(graphListAP[plotOrder[0]][0],
                         graphListAP[plotOrder[0]][1])
                ax2.plot(graphListAP[plotOrder[1]][0],
                         graphListAP[plotOrder[1]][1])
                ax3.plot(graphListAP[plotOrder[2]][0],
                         graphListAP[plotOrder[2]][1])
                # ax4.plot(graphList[plotOrder[3]][0],graphList[plotOrder[3]][1])

                # find peak_frequency
                if False:
                    # print("sec",sec)
                    time = np.linspace(0, sec, T, endpoint=True)

                    from scipy import fftpack
                    sig_noise_fft = fftpack.fft(linIntCSI)
                    sig_noise_amp = 2 / time.size * np.abs(sig_noise_fft)
                    sig_noise_freq = np.abs(fftpack.fftfreq(time.size, sec/T))
                    # print(sig_noise_amp)
                    signal_amplitude = pd.Series(sig_noise_amp).nlargest(
                        2).round(0).astype(int).tolist()
                    # print("signal_amplitude",signal_amplitude)
                    # Calculate Frequency Magnitude
                    magnitudes = abs(
                        sig_noise_fft[np.where(sig_noise_freq >= 0)])
                    # Get index of top 2 frequencies
                    peak_frequency = np.sort(
                        (np.argpartition(magnitudes, -2)[-2:])/sec)
                    # print("peak_frequency",peak_frequency)
                    # cutoff = peak_frequency[1]     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz

                # Moving Average Filter'
                if False:
                    # linIntCSI = [ int(k) for k in linIntCSI]
                    # mAFilter =  butter_lowpass_filter_fft(linIntCSI,cutoff,fs,order)
                    mAFilter = movingaverage(linIntCSI, 1.5)

                # Butterworth Filter'
                if False:
                    # fs = 60.0
                    # lowcut = 0.2
                    # highcut = 0.4
                    fs = 5000.0
                    lowcut = 500.0
                    highcut = 1250.0
                    BBFilter = butter_bandpass_filter(
                        mAFilter, lowcut, highcut, fs, order=4)
                    ax3.plot(linIntTS, BBFilter,
                             label='CSI subcarrier Butterworth Filter')

                # Find BPM
                if(isCountBPM):
                    # print("len final data",len(BBFilter))
                    lastIndexCSIDataInTime = int(secondToCountBreath*fs)
                    psPDBPM = BBFilterSTA[0:lastIndexCSIDataInTime]
                    tsPDBPM = linIntTSSTA[0:lastIndexCSIDataInTime]
                    # print("len  dataInTime",len(dataInTime))
                    # amplitudeDataInTime = max(tsPDBPM)-min(tsPDBPM)
                    # print("!!! amplitudeDataInTime=",amplitudeDataInTime)
                    # if(amplitudeDataInTime<1):
                    if(False):
                        print("ESP32!!! No breath Detected!!!")
                    else:
                        # peaks, _ = find_peaks(psPDBPM)
                        # peaksPDPS = [psPDBPM[x] for x in peaks]
                        # peaksPDTS = [tsPDBPM[x] for x in peaks]
                        # peaksPDTS,peaksPDPS,meanPDPS,BPM = TimeDomain2BPMFTHM(psPDBPM,tsPDBPM,fs,secondToCountBreath)
                        peaksPDTS, peaksPDPS, meanPDPS, BPM = TimeDomain2BPMFPHLP(
                            psPDBPM, tsPDBPM, fs, secondToCountBreath, RRE_threshold)
                        if(len(linIntTSSTA) > 0 and lastIndexCSIDataInTime < len(linIntTSSTA)):
                            print("ESP32!!! use data in ", int(
                                linIntTSSTA[0]), "-", int(linIntTSSTA[lastIndexCSIDataInTime]), "th sec")
                            print("ESP32!!! BPM =", BPM)
                            PDMADList.append(BPM)

                        ax3.axhline(y=meanPDPS, color='r', linestyle='-')
                        ax3.plot(peaksPDTS, peaksPDPS, 'x')
                        if showBPMOnPlot:
                            ax3.title.set_text('BPM=====>'+str(BPM))
        ax1.xaxis.set_ticks(np.arange(int(min(tsAllSTA)/1)*1,
                            int(max(tsAllSTA)/1)*1, 1*secPerGap))
        ax2.xaxis.set_ticks(np.arange(int(min(tsAllSTA)/1)*1,
                            int(max(tsAllSTA)/1)*1, 1*secPerGap))
        ax3.xaxis.set_ticks(np.arange(int(min(tsAllSTA)/1)*1,
                            int(max(tsAllSTA)/1)*1, 1*secPerGap))
        ax4.xaxis.set_ticks(np.arange(int(min(tsAllSTA)/1)*1,
                            int(max(tsAllSTA)/1)*1, 1*secPerGap))
    ax1.set_ylabel("Amplitude(dB)")
    ax2.set_ylabel("Amplitude(dB)")
    ax3.set_ylabel("Amplitude(dB)")
    ax4.set_xlabel("Frame(sec)")

    ax1.set_ylim([-10, +40])
    ax2.set_ylim([-10, +40])
    # ax3.set_ylim(CSIBPMPlotPeriod)



ani = animation.FuncAnimation(
    fig, animate, interval=interval, repeat=False)
plt.show()

# trim duplication at first index
PDMADList = PDMADList[1:]
GTMADList = GTMADList[1:]

print("PDMADList", len(PDMADList))
print("GTMADList", len(GTMADList))
print(PDMADList)
print(GTMADList)

MADSum = 0

for curMADidx in range(len(GTMADList)):
    MADSum += abs(GTMADList[curMADidx]-PDMADList[curMADidx])

MAD = MADSum/len(GTMADList)

print(str(startFrom), "-", str(endAt), "MAD", MAD)
