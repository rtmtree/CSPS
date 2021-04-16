import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import count
import sys
from math import sqrt, atan2 ,isnan
import numpy as np
from matplotlib import style
from matplotlib import collections as matcoll
import time
from datetime import datetime
import random
import pandas as pd
from scipy.fft import fft, fftfreq,ifft
from scipy.ndimage import gaussian_filter
import re
from scipy import pi 
from statistics import stdev
from trainPose3d import CSIModelConfig
from functions.csi_util import rawCSItoAmp,filterNullSC,csiIndices_sec,poseIndices_sec,samplingCSI,featureEngineer,featureEngineerNorm
import matplotlib.gridspec as gridspec
from modules.draw import Plotter3d, draw_poses
from functions.pose_util import poseToPAM,PAMtoPose,rotate_poses,getPCK,stand3dmatrix


def getCentralFrequency(channel):
	cenFreqList = [2412,2417,2422,2427,2432,2437,2442,2447,24452,2457,2462,2467,2472]
	return cenFreqList[channel-1]
def getFrequencyRange(cenFreq,bandwidth,subcarrierLength):
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
	print("read STABLE file",fileAVG)
	curFileSTA = pd.read_csv(fileAVG)
	curFileSTA = curFileSTA[(curFileSTA['mac']==my_filter_address )]
	curFileSTA = curFileSTA[(curFileSTA['len']==384 )]
	curFileSTA = curFileSTA[(curFileSTA['stbc']==0 )]
	curFileSTA = curFileSTA[(curFileSTA['rx_state']==0 )]
	curFileSTA = curFileSTA[(curFileSTA['sig_mode']==1 )]
	curFileSTA = curFileSTA[(curFileSTA['bandwidth']==1 )]
	curFileSTA = curFileSTA[(curFileSTA['secondary_channel']==1 )]
	print("read STABLE file done")
	print("len file",len(curFileSTA.index))
	tail=int((int(len(curFileSTA.index)/2))+(PacketLength/2))
	head=PacketLength
	print(tail)
	print(head)
	curCSISTA = curFileSTA['CSI_DATA'].tail(tail).head(head)
	csiSTAList = list(x for x in curCSISTA)

	parseCSIList = [parseCSI(csiSTAList[i]) for i in range(len(csiSTAList)-1)]
	AVGCSI = []
	for i in range(0,subcarrierLengthX2,2):
		sumCSICol = 0
		for j in range(len(parseCSIList)):
			# raw
			# sumCSICol=sumCSICol+parseCSIList[j][i]
			# amplitude
			sumCSICol=sumCSICol+(sqrt(parseCSIList[j][i] ** 2 + parseCSIList[j][i+1] ** 2))
		AVGCSI.append(sumCSICol/len(parseCSIList))
	return AVGCSI
	# return []
def checkWithThreshold(rssiList,threshold):
	if(rssiList[0]>threshold
	and rssiList[1]>threshold
	and rssiList[2]>threshold
	and rssiList[3]>threshold
	and rssiList[4]>threshold):
		return True
	else:
		return False
def basicDetection(rssiList):
	len(rssiList)
	if(checkWithThreshold(rssiList,-55)):
		print("!!!======== There can be human in 0.5m ==========!!!")
	elif(checkWithThreshold(rssiList,-60)):
		print("!!!======== There can be human in 1m ==========!!!")
	else:
		print("!!!======== There can be nothing ==========!!!")
				
def calculateI_FFT (n,amplitude_spect,phase_spect):
	
	data=list()
	
	for mag,phase in zip(amplitude_spect,phase_spect):
		data.append((mag*n/2)*(np.cos(phase)+1j* np.sin(phase)))
	full_data=list(data)
	i_data=np.fft.irfft(data)
	return i_data

runFeatureEngineer=True
epoch=1000
seqLen=15
batch_size=None
isActSDthreshold=80
minCSIthreshold= int((seqLen/30) * 80)
modelFileName='test_models/model_01_e'+str(epoch)+'_Actthes_'+str(isActSDthreshold)+'_seqLen_'+str(seqLen)+'_'+('FE' if runFeatureEngineer else 'NoFE')+'_'+'.hdf5'


# style.use('dark_background')
# my_filter_address="98:F4:AB:7D:DD:1D"
my_filter_address="7C:9E:BD:D2:D8:9C"
# my_filter_address="3C:71:BF:6D:2A:78"
my_filter_length=128
show1 =  True
isRealtime = True
plotBySubcarrier = True
filePaths = ['get_csi/active_ap/realtime.csv']
plotMultiChannel = True
Channels = [6]
channelColors = ['red','green','blue','yellow','pink']
colorCSIs = ['red','green','blue']
colorPDP = ['red','green','blue']
nullSubcarrier = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
shows=[True,True,True]

stableCSI=0.1
subcarrierLength = 64
subcarrierLengthX2 = int(subcarrierLength*2)
PacketLength = 5000
csiSTAList=[]
rssiSTAList=[]
tsSTAList=[]

csiAPList=[]
rssiAPList=[]
tsAPList=[]

csiList=[]
rssiList=[]
tsList=[]
channelList=[]


fig = plt.figure()

gs = gridspec.GridSpec(8,2)
# gs = gridspec.GridSpec(3,2)
ax=fig.add_subplot(gs[0,:])
ax0=fig.add_subplot(gs[2,:])
ax1=fig.add_subplot(gs[4:8,:])

cfg = CSIModelConfig(win_len=1000, step=2000, thrshd=0.6, downsample=2)
model = cfg.load_model(modelFileName)
ln1, = plt.plot([], [], 'ro')
ln2, = plt.plot([], [], 'ro')
ln = [ln1, ln2]
def init():
	plt.setp(ax,xlabel="Frame",ylabel="Amplitude(dB)")
	plt.setp(ax0,xlabel="Frame",ylabel="Amplitude(dB)")
	ax.set_ylim([-10, +40])
	ax0.set_ylim([-10, +40])
	ax.set_xlim([ 0,15000000 ])
	ax0.set_xlim([ 0,15000000 ])
	return ln

y_pred = False
def updatefig(line):
	sLine = line%seqLen
	print("animate",sLine)

	if(sLine==seqLen-1):
		fileIdx=0
		print("============filePath1",filePaths[fileIdx])
		print("============color CSI",colorCSIs[fileIdx])
		valueSTA=0.1
		valueAP=0.1
		try:
			curFileSTA = pd.read_csv(filePaths[fileIdx])
			if(len(Channels)==1):
				curFileSTA = curFileSTA[(curFileSTA['mac']==my_filter_address )]
				curFileSTA = curFileSTA[(curFileSTA['len']==384 )]
				curFileSTA = curFileSTA[(curFileSTA['stbc']==0 )]
				curFileSTA = curFileSTA[(curFileSTA['rx_state']==0 )]
				curFileSTA = curFileSTA[(curFileSTA['sig_mode']==1 )]
				curFileSTA = curFileSTA[(curFileSTA['bandwidth']==1 )]
				curFileSTA = curFileSTA[(curFileSTA['secondary_channel']==1 )]

			tail=len(Channels)*2
			tail=500+1
			curCSI = curFileSTA['CSI_DATA'].tail(tail)
			RTcsiList = list(x for x in curCSI)
			curRSSI = curFileSTA['rssi'].tail(tail)
			RTrssiList=list(x for x in curRSSI)
			curTS = curFileSTA['local_timestamp'].tail(tail)
			tsSTAList=list(x for x in curTS)
			curChannel = curFileSTA['channel'].tail(tail)
			channelSTAList=list(x for x in curChannel)
			# print("timestamp",datetime.fromtimestamp(tsSTAList[0]+1611841591.5))
			if(plotBySubcarrier):
				valueSTA=[]
				rssiSTA=[]
				tsSTA=[]
				startIndex=curRSSI.index[0]
				for i in range(len(RTcsiList)-1):
					valueSTA.append(parseCSI(RTcsiList[i]))
					rssiSTA.append(RTrssiList[i])
					tsSTA.append(tsSTAList[i])
		except:
			print("catch animate RT")
			return [ax,ax0,ax1]
			# return [ax,ax0]
			
		if(shows[fileIdx] and isinstance(valueSTA, float)==False and len(valueSTA)>0):
			csiAll=valueSTA
			tsAll = tsSTA
			rssiAll = rssiSTA
			print("last rssiAll",rssiAll[-1])
			lastTS = tsAll[-1]/(10**6)
			lastTSbf = (tsAll[-1]/(10**6)) - (seqLen/30)
			# print("last TS",lastTS) 
			# print("last TSbf",lastTSbf) 

			csiInRange = []
			for i in range(len(csiAll)):
				if(tsAll[i]/(10**6) > lastTSbf ):
				# if(True):
					csiInRange.append([tsAll[i]/(10**6)] + csiAll[i])

			amplitudesAll=[filterNullSC( rawCSItoAmp(   csiInRange[j][1:]  )  )    for j in range(len(csiInRange))]
			csiIndices=[j for j in range(len(csiInRange))]
			poseIndices=[j for j in range((seqLen))]
			poseList=[]
			stepTS=(lastTS-lastTSbf)/seqLen
			for j in range(0,seqLen):
				poseList.append( [  lastTSbf + ((j+1) * (stepTS))   ]) 
			samplingedAmp,expectedTS=samplingCSI(csiInRange,csiIndices,poseList,poseIndices,paddingTo=seqLen)

			# Ploting Start
			for j in range(0,52):
				textX=[]
				textY=[]
				for k in range(len(amplitudesAll)):
					curCsi=amplitudesAll[k][j]
					textX.append(csiInRange[k][0])
					textY.append(curCsi)
				ax.plot(textX,gaussian_filter(textY,sigma=0), label='CSI subcarrier')
				textXSP=[]
				textYSP=[]
				for k in range(len(samplingedAmp)):
					curCsi=samplingedAmp[k][j]
					textXSP.append(expectedTS[k])
					textYSP.append(curCsi)
				ax0.plot(textXSP,gaussian_filter(textYSP,sigma=0), label='CSI subcarrier')
			ax.set_xlim([ textX[0] , textX[-1] ])
			ax0.set_xlim([ textXSP[0] , textXSP[-1] ])
			
			sdSum=0
			for j in range(0,52):
				subClist=[]
				for k in range(len(samplingedAmp)):
					subClist.append( samplingedAmp[k][j] )
				sdAmp=stdev(subClist)
				sdSum+=sdAmp
			print("sum_diff",sdSum)

			if(sdSum> isActSDthreshold ):
			# if(False):
				X=np.array([samplingedAmp])
				X=featureEngineer(X)
				print(X.shape)
				global y_pred
				y_pred = model.predict(X)
			else:
				y_pred=False
				ax1.cla()
		
	
	if(isinstance(y_pred, bool)==False):
		print("have pose")
		if True:
			poses_3dFromImage=PAMtoPose(y_pred[0][sLine].reshape(3,19,19))
		else:
			poses_3dFromImage=stand3dmatrix
		edgesFromImage = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3dFromImage.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
		canvas_3d = np.zeros((450, 450, 3), dtype=np.uint8)
		plotter = Plotter3d(canvas_3d.shape[:2])
		plotter.plot(canvas_3d, poses_3dFromImage, edgesFromImage)
		ax1.imshow(canvas_3d)

	# return [ax,ax0]
	return [ax,ax0,ax1]


ani = animation.FuncAnimation(fig, updatefig, interval=200,
                    init_func=init, blit=True)

plt.show()
		



