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
from statistics import stdev
from functions.csi_util import filterNullSC, rawCSItoAmp, singleLinearInterpolation,moving_average
from hampel import hampel
from scipy.signal import butter, lfilter,filtfilt,find_peaks

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
    curFileSTA = curFileSTA[(curFileSTA['mac'] == my_filter_address)]
    curFileSTA = curFileSTA[(curFileSTA['len'] == my_filter_length)]
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


# timestampColName = 'local_timestamp'
timestampColName = 'real_timestamp'
power2timestamp = 6 if (timestampColName == 'real_timestamp') else 1
tsParser = (10**power2timestamp)
tsParserGT = (10**3)

# =================================================
isRealtime = False
isWithGroundTruth = True
#sec
startFrom = 600
endAt= 1000

frameLength = 5000
# xx0  hold breath 104-133 
# xx0  breath slowly 155-207
# xx0  breath normally 210-321 //bpm 18
# xx0  breath quickly 325
fileCode = 'xx2'
filePaths = ['get_csi/esp32-csi-tool/active_sta/'+fileCode+'.csv']
filePathsGT = ['get_groundtruth/'+fileCode+'.csv']
# =================================================

# sync part
fs = 60.0 # sample rate, Hz
secondToCountBreath = 30 #second
secPerGap = 30
interval = 1000 if isRealtime else 1000

# GroundTruth PART=======
gtTSList =[]
gtPSList =[]
syncTime = 0

# CSI PART============
subcarrierIndex = [44]
gsSigma = 5
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
# CSI PART============
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

    if(isWithGroundTruth):
        valueGT = []
        tsGT = []
        # print("firstTsCSI",firstTsCSI,"lastTsCSI",lastTsCSI)
        # print("30",(secondToCountBreath*tsParser))
        # print("firstTsCSI + 30",(firstTsCSI + (secondToCountBreath*tsParser)))
        lastIndexGTDataInTime = 0
        for i in range(0, len(gtTSList)):
            # for finding last index within time period (for BPM cal)
            if( gtTSList[i] < (firstTsCSI + (secondToCountBreath*tsParser)) ):
                lastIndexGTDataInTime+=1

            if(gtTSList[i]>=firstTsCSI and gtTSList[i]<=lastTsCSI):
                valueGT.append(gtPSList[i])
                # parse back
                tsGT.append(float(gtTSList[i]/tsParser))
            elif(gtTSList[i]>lastTsCSI):
                break

            
        # ax4.plot(tsGT, valueGT, label='Ground Truth')
        ax4.set_ylim([95, 105])
        ax4.set_xlabel("Frame(sec)")
        if(len(tsGT)>0):
            ax4.plot(tsGT, gaussian_filter(valueGT, sigma=1), label='Ground Truth')
            ax4.xaxis.set_ticks(np.arange(int(min(tsGT)/1)*1, int(max(tsGT)/1)*1, 1*secPerGap ) )
            print(len(gtTSList))
            print(len(valueGT))
            print(lastIndexGTDataInTime)
            if(lastIndexGTDataInTime==0):
                print("GT!!! can't calculate BPM")
            else:
                dataInTimeGT = gtTSList[0:lastIndexGTDataInTime]
                peaksGT, _ = find_peaks(dataInTimeGT)
                BPM = len(peaksGT)*(fs/secondToCountBreath)
                print("GT!!! use data until",gtTSList[lastIndexGTDataInTime],"th sec")
                print("GT!!! B =",len(peaksGT)," per ",secondToCountBreath,"second =",BPM,"BPM")



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
            
            # ax0.plot(textX, gaussian_filter(
            #     textY, sigma=gsSigma), label='CSI subcarrier')
            if(j in subcarrierIndex):
            # if True:
                initCSI = gaussian_filter(textY, sigma=gsSigma)
                ax1.plot(textX,initCSI , label='CSI subcarrier')

                #Hampel Filter'
                # seriesCSI = pd.Series(initCSI)
                # hameFil = hampel_filter_v(initCSI, 0.5, 1.7)
                # hameFil  = hampel_filter(initCSI, 1,1.7)
                hameFil  = hampel_filter(initCSI, 1,-100)
                ax2.plot(textX,hameFil , label='CSI subcarrier Hampel Filter')

                

                # Linear interpolation to fit 60 Hz sampling frequency
                # print("len hameFil",len(hameFil))
                
                linIntCSI,linIntTS = singleLinearInterpolation(hameFil,np.arange(len(hameFil)),textX,0,fs)
                # print("len linIntCSI",len(linIntCSI))
                # ax3.plot(linIntTS, linIntCSI , label='CSI subcarrier Linear Interpolation')  

                #Moving Average Filter'
                if True:
                    T = len(linIntCSI)
                    sec = T/fs         # Sample Period
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


                    # Filter requirements.
                    cutoff = 20 / sec     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
                    # cutoff = peak_frequency[1]     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
                    # Signal Freq = 6 signal / 5 sec = 1.2 Hz
                    order = 2       # sin wave can be approx represented as quadratic
                    # linIntCSI = [ int(k) for k in linIntCSI]
                    # mAFilter =  butter_lowpass_filter_fft(linIntCSI,cutoff,fs,order)
                    mAFilter =  movingaverage(linIntCSI,1.5)
                    
                if True:    #Butterworth Filter' low pass
                    BBFilter =  butter_lowpass_filter_fft(mAFilter,cutoff,fs,order)
                    ax3.plot(linIntTS, BBFilter , label='CSI subcarrier Moving Average Filter')  

                    

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
                
                
                # print("len final data",len(BBFilter))
                lastIndexCSIDataInTime = int(secondToCountBreath*fs)
                dataInTime = BBFilter[0:lastIndexCSIDataInTime]
                # print("len  dataInTime",len(dataInTime))
                amplitudeDataInTime = max(dataInTime)-min(dataInTime)
                # print("!!! amplitudeDataInTime=",amplitudeDataInTime)
                if(True):
                    if(amplitudeDataInTime<1):
                        print("CSI!!! No breath Detected!!!")
                    else:
                        peaks, _ = find_peaks(dataInTime)
                        BPM = len(peaks)*(fs/secondToCountBreath)
                        print("CSI!!! use data until",textX[lastIndexCSIDataInTime],"th sec")
                        print("CSI!!! B =",len(peaks)," per ",secondToCountBreath,"second","BPM ",BPM)
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
    ax1.set_ylabel("Amplitude(dB) 1 SC"+"("+str(subcarrierIndex)+"th)")
    ax2.set_ylabel("Amp 1 SC HampelF")
    ax3.set_ylabel("Amp 1 SC HF LiInter")
    ax3.set_xlabel("Frame(sec)")

    # ax0.set_ylim([-10, +40])
    ax1.set_ylim([-10, +40])
    ax2.set_ylim([-10, +40])
    # ax3.set_ylim([-10, +40])
    # ax3.set_ylim([15, 25])
    ax3.set_ylim([5, 15])
    
    
    # ax0.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )
    ax1.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )
    ax2.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )
    ax3.xaxis.set_ticks(np.arange(int(min(tsAll)/1)*1, int(max(tsAll)/1)*1, 1*secPerGap ) )

    


ani = animation.FuncAnimation(
    fig, animate, interval=interval)
plt.show()
