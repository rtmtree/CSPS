import numpy as np 
import json
from modules.draw import Plotter3d, draw_poses
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import re
from scipy.ndimage import gaussian_filter
from math import sqrt, atan2 ,isnan
import matplotlib.gridspec as gridspec
import glob
import imageio
import cv2
import math
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from modules.parse_poses import parse_poses
from statistics import stdev

def imageIdx2csiIdx(durationSec,imageIdx,tsList,fps):
    offsetTime=durationSec - (tsList[len(tsList)-1]/(10**6))

    timeInVid= (imageIdx/vidLength) * durationSec

    timeInCSI=(timeInVid-offsetTime) 

    csiIndex = min(range(len(tsList)), key=lambda i: abs(tsList[i]-(timeInCSI*(10**6))))
    if(True):
        return csiIndex
    else:
        return False
def imageIdx2csiIndices(durationSec,imageIdx,tsList,vidLength):
    offsetTime=durationSec - (tsList[len(tsList)-1]/(10**6))
    timeInVid= (imageIdx/vidLength) * durationSec

    parsedTimeInVid= timeInVid - offsetTime

    print("timeInVid",int(timeInVid))
    print("parsedTimeInVid",int(parsedTimeInVid))
    csiIndices=[]
    for i in range(len(tsList)):
        # print("timecsi",int(tsList[i]/(10**6)))
        # print("parsedTimeInVid",int(parsedTimeInVid))

        if(math.floor(tsList[i]/(10**6))==math.floor(parsedTimeInVid)):
            csiIndices.append(i)
    
    return csiIndices
def rawCSItoAmp(data,length=128):
    if(data==False):
        return False
    amplitudes = []
    for j in range(0,length,2):
        amplitudes.append(sqrt(data[j] ** 2 + data[j+1] ** 2))
    return amplitudes
    
def parseCSI(csi):
    try:
        csi_string = re.findall(r"\[(.*)\]", csi)[0]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
        return csi_raw
    except:
        return False


label='10'
withPose = True

csiFilePaths = ['raw_data/CSI'+label+'.csv']
vidFilePaths = ['raw_data/CSI'+label+'.mov']
if withPose:
    poseFilePaths = ['data/parsedPose3D'+label+'.csv']
    poseList = pd.read_csv(poseFilePaths[0],delimiter=',',header=None).values

# csiFilePaths = ['test4.csv']
# vidFilePaths = ['test4_crop.mov']


curFileSTA = pd.read_csv(csiFilePaths[0])
my_filter_address="7C:9E:BD:D2:D8:9C"
curFileSTA = curFileSTA[(curFileSTA['mac']==my_filter_address )]
curFileSTA = curFileSTA[(curFileSTA['len']==384 )]
curFileSTA = curFileSTA[(curFileSTA['stbc']==0 )]
curFileSTA = curFileSTA[(curFileSTA['rx_state']==0 )]
curFileSTA = curFileSTA[(curFileSTA['sig_mode']==1 )]
curFileSTA = curFileSTA[(curFileSTA['bandwidth']==1 )]
curFileSTA = curFileSTA[(curFileSTA['secondary_channel']==1 )]
print("read file done")
print("len file",len(curFileSTA.index))

curCSI = curFileSTA['CSI_DATA']
csiList = list(rawCSItoAmp(parseCSI(x),128) for x in curCSI)
curTS = curFileSTA['local_timestamp']
tsList=list(x for x in curTS)
x_values=[[] for i in range(64)]
y_values=[[] for i in range(64)]

vid = imageio.get_reader(vidFilePaths[0],  'ffmpeg')
meta_data=vid.get_meta_data()
duration_in_sec = meta_data['duration']
cap = cv2.VideoCapture(vidFilePaths[0])
vidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

startFrom=0

if True: # plot pose3D
    fig = plt.figure()
    gs = gridspec.GridSpec(2,2)
    ax0=fig.add_subplot(gs[0,0])
    ax1=fig.add_subplot(gs[0,1])
    ax2=fig.add_subplot(gs[1,:])

    x_values=[[] for i in range(64)]
    y_values=[[] for i in range(64)]

    def updatefig(i):
        print("updatefig",i)
        ax0.cla()
        ax1.cla()
        ax2.cla()

        imageIdx=(i+startFrom)*14

        if (imageIdx >= (vidLength)-1):
            print('close')
            plt.close(fig)
        
        print("imageIdx",imageIdx)

        if (imageIdx == vidLength):
            print(f'imageIdx {imageIdx} == vidLength {vidLength}; closing!')
            plt.close(fig)

        frame = vid.get_data(imageIdx)
        ax0.imshow(frame)    

        if withPose:
            poseIdx = imageIdx
            poses_3dFromImage=np.array([poseList[poseIdx][1:].reshape(19,3)])
            # poses_3dFromImage=stand3dmatrix

            edgesFromImage = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3dFromImage.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            canvas_3d = np.zeros((450, 450, 3), dtype=np.uint8)
            plotter = Plotter3d(canvas_3d.shape[:2])
            plotter.plot(canvas_3d, poses_3dFromImage, edgesFromImage)
            ax0.imshow(canvas_3d)



        #imageIdx to csiIdx
        csiIndices=imageIdx2csiIndices(duration_in_sec,imageIdx,tsList,vidLength)
        if(len(csiIndices)>0):
            startCSIIdx=csiIndices[0]
            endCSIIdx=csiIndices[len(csiIndices)-1]
            print(startCSIIdx,'-',endCSIIdx)
            print(endCSIIdx-startCSIIdx+1)
            
            for j in range(0,64):
                if (6<=j<32 or 33<=j<59):
                    textX=[]
                    textY=[]
                    for k in csiIndices:
                        textX.append(tsList[k]/(10**6))
                        textY.append(csiList[k][j])
                    ax2.plot(textX,gaussian_filter(textY,sigma=1), label='CSI subcarrier')
            print("added")
            print('lastTS',tsList[endCSIIdx])
        ax2.set_ylim([-10, +40])
        # ax2.xlabel("Frame")
        # ax2.ylabel("Amplitude(dB)")
        return ax0,ax1,ax2 

    ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True,frames=len(csiList),repeat=False)
    custom_ylim = (-10, +40)
    custom_xlim = (-10, +40)
    # plt.setp(ax2,xlim=custom_xlim, ylim=custom_ylim,xlabel="Frame",ylabel="Amplitude(dB)")
    plt.show()
