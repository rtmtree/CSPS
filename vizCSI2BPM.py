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
from statistics import stdev,mean
from functions.csi_util import filterNullSC, rawCSItoAmp, singleLinearInterpolation,moving_average,parseCSI,hampel_filter,butter_lowpass_filter_fft


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

# xx3 breath normally 300-400
# xx3 breath quickly 400-450 ***
# xx3 breath slowly 450-550
# xx3 hold breath 550-610
# [8, 25]
# xx4 breath normally 300-400
# xx4 breath quickly 400-500 ***
# xx4 hold breath 500-600
# xx4 breath slowly 600-700

fileCode = 'xx4'
CSIBPMPlotPeriod = [20,40]
# 0 = raw
# 1 = hampel
# 2 = gaussian
# 3 = linear inter
# 4 = butter_lowpass
plotOrder=[2,3,3,4]
frameLength = 5000
#sec
startFrom = 345
isFreeze = True
endAt= 1000

isRealtime = False
isWithGroundTruth = False
isCountBPM = False

# CSI filtering parameter part======================
# gaussian_filter
GF_sigma = 5
# hampel_filter
HF_sigma = -100
HF_winsize = 1
# linear interpolation
LI_digi = 0
LI_timelen = 60
# butterworth low pass filter
BLPF_order = 2       # sin wave can be approx represented as quadratic
BLPF_cutoffFreq = 30
# PRESSURE filtering parameter part======================
GT_GF_sigma = 1

# =================================================

# sync part
fs = LI_timelen # sample rate, Hz
secondToCountBreath = 60 #second
secPerGap = 30
interval = 1000 if isRealtime else 1000

# GroundTruth PART=======
gtTSList =[]
gtPSList =[]
syncTime = 0

# CSI PART============
subcarrierIndex = [1]

# my_filter_address = "98:F4:AB:7D:DD:1D"
my_filter_address = "7C:9E:BD:D2:D8:9D"
my_filter_length = 384
Channels = [6]
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
# end CSI PART============

filePaths = ['get_csi/esp32-csi-tool/active_sta/'+fileCode+'.csv']
filePathsGT = ['get_groundtruth/'+fileCode+'.csv']
if(isRealtime == False):
    
    print("read file csi", filePaths[0])
    curFileSTA = pd.read_csv(filePaths[0])
    curFileSTA = curFileSTA[(curFileSTA['mac'] == my_filter_address)]
    curFileSTA = curFileSTA[(curFileSTA['len'] == my_filter_length)]
    curFileSTA = curFileSTA[(curFileSTA['stbc'] == 0)]
    curFileSTA = curFileSTA[(curFileSTA['rx_state'] == 0)]
    curFileSTA = curFileSTA[(curFileSTA['sig_mode'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['bandwidth'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['secondary_channel'] == 1)]
    curFileSTA = curFileSTA[(curFileSTA['channel'] == Channels[0])]
    print("read file csi done")
    print("len csi file", len(curFileSTA.index))

    tail = len(curFileSTA.index)

    curCSISTA = curFileSTA['CSI_DATA'].tail(tail)
    curRSSISTA = curFileSTA['rssi'].tail(tail)

    curTSSTA = curFileSTA[timestampColName].tail(tail)

    csiList = list(x for x in curCSISTA)
    rssiList = list(x for x in curRSSISTA)
    tsList = list(x*tsParser for x in curTSSTA)

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
        print("last Timestamp of groundtruth" , tsList[len(tsList)-1],"last Timestamp of GT",gtTSList[len(gtTSList)-1])
        syncTime = tsList[len(tsList)-1]-gtTSList[len(gtTSList)-1]
        print("syncTime is" , syncTime)
        gtTSList = [x+syncTime for x in gtTSList]

index = count()
# fig, [ax0, ax1, ax2,ax3] = plt.subplots(4, 1, figsize=(6, 9))
fig, [ax1, ax2, ax3,ax4] = plt.subplots(4, 1, figsize=(6, 9))
# fig, [ax2, ax1, ax0] = plt.subplots(3, 1, figsize=(6, 9))
# fig, [ax2, ax1, ax0] = plt.subplots(3, 1, figsize=(6, 9))


def animate(line):
    global gtTSList
    global gtPSList
    global tsList
    global CSIBPMPlotPeriod

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
            curFileSTA = pd.read_csv(filePaths[fileIdx])
            curFileSTA = curFileSTA[(
                curFileSTA['mac'] == my_filter_address)]
            curFileSTA = curFileSTA[(curFileSTA['len'] == my_filter_length)]
            curFileSTA = curFileSTA[(curFileSTA['stbc'] == 0)]
            curFileSTA = curFileSTA[(curFileSTA['rx_state'] == 0)]
            curFileSTA = curFileSTA[(curFileSTA['sig_mode'] == 1)]
            curFileSTA = curFileSTA[(curFileSTA['bandwidth'] == 1)]
            curFileSTA = curFileSTA[(curFileSTA['channel'] == Channels[0])]
            curFileSTA = curFileSTA[(curFileSTA['secondary_channel'] == 1)]
            tail = frameLength + 1
            curCSI = curFileSTA['CSI_DATA'].tail(tail)
            curTS = curFileSTA[timestampColName].tail(tail)
            RTcsiList = list((x) for x in curCSI)
            tsSTAList = list(x*tsParser for x in curTS)
            # crop out one probably invalid row
            RTcsiList = RTcsiList[0:len(RTcsiList)-1]
            tsSTAList = tsSTAList[0:len(tsSTAList)-1]

            print("last ts", tsSTAList[-2])
            # print("last rssi", RTrssiList[-2])
            valueSTA = []
            tsSTA = []
            # rssiSTA = []
            for i in range(0, frameLength):  # cut last element
                if(i>=len(RTcsiList)):
                    break
                valueSTA.append(parseCSI(RTcsiList[i]))
                tsSTA.append(tsSTAList[i])
                # rssiSTA.append(RTrssiList[i])
            print(len(valueSTA))
            firstTsCSI = tsSTA[0]
            lastTsCSI = tsSTA[len(tsSTA)-1]
            lastTsCSIforBPM = (firstTsCSI) + (secondToCountBreath*tsParser)


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
                # crop out one probably invalid row
                gtPSList = gtPSList[0:len(gtPSList)-1]
                gtTSList = gtTSList[0:len(gtTSList)-1]

                print("last Timestamp of groundtruth" , tsSTA[len(tsSTA)-1],"last Timestamp of GT",gtTSList[len(gtTSList)-1])
                syncTime = tsSTA[len(tsSTA)-1]-gtTSList[len(gtTSList)-1]
                print("syncTime is" , syncTime)
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
        global rssiList
        
        

        print("batch Index", line)
        if(endAt!=0 and line==endAt):
            quit() 
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
            minTs = (line*tsParser)
            print("batch minTs", minTs,"frameLength",frameLength)
            # crop with timestamp 

            

            # CSI part
            tsIndex = 0
            while(tsList[tsIndex] < minTs):
                tsIndex += 1
            print("CSI tsIndex",tsIndex,"ts is",tsList[tsIndex])
            curLength = frameLength if (len(tsList) - tsIndex) > frameLength else (len(tsList) - tsIndex)
            print("CSI curLength",curLength)
            for i in range(tsIndex, tsIndex+curLength):
                valueSTA.append(parseCSI(csiList[i]))
                tsSTA.append(tsList[i])
                # rssiSTA.append(rssiList[i])
            firstTsCSI = tsSTA[0]
            lastTsCSI = tsSTA[len(tsSTA)-1]
            lastTsCSIforBPM = (firstTsCSI) + (secondToCountBreath*tsParser)

    if(isWithGroundTruth):
        valueGTPlot = []
        tsGTPlot = []
        psGTBPM = []
        tsGTBPM = []
        print("firstTsCSI",firstTsCSI,"lastTsCSI",lastTsCSI,"lastTsCSIforBPM",lastTsCSIforBPM)
        # print("30",(secondToCountBreath*tsParser))
        # print("firstTsCSI + 30",(firstTsCSI + (secondToCountBreath*tsParser)))
        # create GroundTruth within the period same as CSI for ploting
        print("bf len gtPSList",len(gtPSList))


        filGtPSList,filGtTSList = singleLinearInterpolation(gtPSList,np.arange(len(gtPSList)),gtTSList,6,4)
        
        # print("af len gtPSList",len(gtPSList))
        # print("expTS 0",gtTSList[0]-syncTime)
        # print("expTS last",gtTSList[len(gtTSList)-1]-syncTime)

        
        for i in range(0, len(filGtTSList)):
            if(filGtTSList[i]>=firstTsCSI and filGtTSList[i]<=lastTsCSIforBPM):
                psGTBPM.append(filGtPSList[i])
                # parse back
                tsGTBPM.append(float(filGtTSList[i]/tsParser))
            if(filGtTSList[i]>=firstTsCSI and filGtTSList[i]<=lastTsCSI):
                valueGTPlot.append(filGtPSList[i])
                # parse back
                tsGTPlot.append(float(filGtTSList[i]/tsParser))
            elif(filGtTSList[i]>lastTsCSI and filGtTSList[i]>lastTsCSIforBPM):
                break

        # ax3.set_ylim([95, 105])
        # ax3.set_ylabel("Pressure(kPa)")
        ax4.set_ylim([95, 105])
        ax4.set_ylabel("Pressure(kPa)")

        if(len(tsGTPlot)>0):
            # ax3.plot(tsGTPlot, valueGTPlot, label='Ground Truth')
            valueGTPlot = gaussian_filter(valueGTPlot,sigma=GT_GF_sigma)
            ax4.plot(tsGTPlot, valueGTPlot, label='Ground Truth')
        
        if isCountBPM:
            if(len(psGTBPM)==0):
                print("BELT!!! can't calculate BPM")
            else:
                # peaksGT, _ = find_peaks(psGTBPM,threshold=1)
                # peaksGT, _ = find_peaks(psGTBPM)
                # peaksGTTS = [tsGTBPM[x] for x in peaksGT]
                # peaksGTPS = [psGTBPM[x] for x in peaksGT]
                peaksGTTS = []
                peaksGTPS = []
                meanGTPS = mean(psGTBPM)
                for i in range(1,len(psGTBPM)):
                    if(psGTBPM[i-1]<meanGTPS and psGTBPM[i]>meanGTPS):
                        peaksGTTS.append(tsGTBPM[i])
                        peaksGTPS.append(psGTBPM[i])
                ax4.axhline(y=meanGTPS, color='r', linestyle='-')
                ax4.plot(peaksGTTS, peaksGTPS,'x')
                BPM = len(peaksGTTS)*(fs/secondToCountBreath)
                print("BELT!!! use data in",int(tsGTBPM[0])," -",int(tsGTBPM[len(tsGTBPM)-1]),"th sec")
                print("BELT!!! B =",len(peaksGTTS)," per ",secondToCountBreath,"second =",BPM,"BPM")
                # ax4.title.set_text('BPM________'+str(BPM))


    # CSI PART
    if(isinstance(valueSTA, float) == False and len(valueSTA) > 0):

        amplitudesAll = [filterNullSC(rawCSItoAmp(curCSI))
                         for curCSI in valueSTA]
        tsAll = [float(curTs/tsParser) for curTs in tsSTA]

        #if sum needed
        # sumY = [0 for j in range(len(amplitudesAll))]
        for j in range(len(amplitudesAll[0])):  # 52/64
            textX = tsAll
            textY = [amplitudesAll[k][j] for k in range(len(amplitudesAll))]
            if False: #if sum needed
                for k in range(len(amplitudesAll)):
                    textX.append(tsAll[k])
                    textY.append(amplitudesAll[k][j])
                    sumY[k] += amplitudesAll[k][j]
            
            # if True:
            if(j in subcarrierIndex):
                graphList = []
                # ax1.plot(textX,textY , label='CSI subcarrier')
                graphList.append([textX,textY])


                #Hampel Filter
                hameFil  = hampel_filter(textY, HF_winsize,HF_sigma)
                graphList.append([textX,hameFil])
                # ax1.plot(textX,hameFil , label='CSI subcarrier Hampel Filter')
                
                #Gaussian Filter
                gaussFil = gaussian_filter(hameFil, sigma=GF_sigma)
                graphList.append([textX,gaussFil])
                # ax1.plot(textX,gaussFil , label='CSI subcarrier after Gaussian Filtering')


                # Linear Interpolation
                linIntCSI,linIntTS = singleLinearInterpolation(gaussFil,np.arange(len(hameFil)),textX,LI_digi,LI_timelen)
                graphList.append([linIntTS,linIntCSI])
                # ax1.plot(linIntTS, linIntCSI , label='CSI subcarrier Linear Interpolation')  

                #Butterworth Filter' low pass
                BLPF_T = len(linIntCSI)
                BLPF_sec = BLPF_T/LI_timelen         # Sample Period
                BLPF_cutoff = BLPF_cutoffFreq / BLPF_sec  
                BBFilter =  butter_lowpass_filter_fft(linIntCSI,BLPF_cutoff,LI_timelen,BLPF_order)
                graphList.append([linIntTS,BBFilter])
                # ax3.plot(linIntTS, BBFilter , label='CSI subcarrier Moving Average Filter')  

                ax1.plot(graphList[plotOrder[0]][0],graphList[plotOrder[0]][1])
                ax2.plot(graphList[plotOrder[1]][0],graphList[plotOrder[1]][1])
                ax3.plot(graphList[plotOrder[2]][0],graphList[plotOrder[2]][1])
                ax4.plot(graphList[plotOrder[3]][0],graphList[plotOrder[3]][1])

                # find peak_frequency
                if False:
                    # print("sec",sec)
                    time = np.linspace(0, sec, T, endpoint=True)

                    from scipy import fftpack
                    sig_noise_fft = fftpack.fft(linIntCSI)
                    sig_noise_amp = 2 / time.size * np.abs(sig_noise_fft)
                    sig_noise_freq = np.abs(fftpack.fftfreq(time.size, sec/T))
                    # print(sig_noise_amp)
                    signal_amplitude = pd.Series(sig_noise_amp).nlargest(2).round(0).astype(int).tolist()
                    # print("signal_amplitude",signal_amplitude)
                    #Calculate Frequency Magnitude
                    magnitudes = abs(sig_noise_fft[np.where(sig_noise_freq >= 0)])
                    #Get index of top 2 frequencies
                    peak_frequency = np.sort((np.argpartition(magnitudes, -2)[-2:])/sec)
                    # print("peak_frequency",peak_frequency)
                    # cutoff = peak_frequency[1]     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz


                #Moving Average Filter'
                if False:
                    # linIntCSI = [ int(k) for k in linIntCSI]
                    # mAFilter =  butter_lowpass_filter_fft(linIntCSI,cutoff,fs,order)
                    mAFilter =  movingaverage(linIntCSI,1.5)
                    
                #Butterworth Filter'
                if False:
                    # fs = 60.0
                    # lowcut = 0.2
                    # highcut = 0.4
                    fs = 5000.0
                    lowcut = 500.0
                    highcut = 1250.0
                    BBFilter = butter_bandpass_filter(mAFilter, lowcut, highcut, fs, order=4)
                    ax3.plot(linIntTS, BBFilter , label='CSI subcarrier Butterworth Filter')
                
                # Find BPM 
                if(isCountBPM):
                    # print("len final data",len(BBFilter))
                    lastIndexCSIDataInTime = int(secondToCountBreath*fs)
                    psPDBPM = BBFilter[0:lastIndexCSIDataInTime]
                    tsPDBPM = linIntTS[0:lastIndexCSIDataInTime]
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
                        peaksPDPS = []
                        peaksPDTS = []
                        meanPDPS = mean(psPDBPM)
                        for i in range(1,len(psPDBPM)):
                            if(psPDBPM[i-1]<meanPDPS and psPDBPM[i]>meanPDPS):
                                peaksPDPS.append(psPDBPM[i])
                                peaksPDTS.append(tsPDBPM[i])
                        ax3.axhline(y=meanPDPS, color='r', linestyle='-')
                        ax3.plot(peaksPDTS, peaksPDPS,'x')
                        BPM = len(peaksPDTS)*(fs/secondToCountBreath)
                        print("ESP32!!! use data in ",int(linIntTS[0]),"-",int(linIntTS[lastIndexCSIDataInTime]),"th sec")
                        print("ESP32!!! B =",len(peaksPDTS)," per ",secondToCountBreath,"second","=",BPM,"BPM")
                        # ax3.title.set_text('BPM________'+str(BPM))
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
    # ax0.set_ylabel("Amplitude(dB)")
    # ax1.set_ylabel("Amplitude(dB) 1 SC"+"("+str(subcarrierIndex)+"th)")
    # ax2.set_ylabel("Amp 1 SC HampelF")
    ax1.set_ylabel("Amplitude(dB)")
    ax2.set_ylabel("Amplitude(dB)")
    ax3.set_ylabel("Amplitude(dB)")
    ax4.set_ylabel("Amplitude(dB)")
    ax4.set_xlabel("Frame(sec)")

    # ax0.set_ylim([-10, +40])
    ax1.set_ylim([-10, +40])
    ax2.set_ylim([-10, +40])
    ax3.set_ylim([-10, +40])
    ax4.set_ylim([-10, +40])
    # ax3.set_ylim(CSIBPMPlotPeriod)
    
    
    # ax0.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )
    ax1.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )
    ax2.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )
    ax3.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )
    ax4.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )

    


ani = animation.FuncAnimation(
    fig, animate, interval=interval)
plt.show()
