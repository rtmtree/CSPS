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
# import imageio
import cv2
from modules.inference_engine_pytorch import InferenceEnginePyTorch
# from modules.parse_poses import parse_poses
from functions.csi_util import rawCSItoAmp,imageIdx2csiIndices_timestamp,samplingCSI,filterNullSC,featureEngineer



csiFilePaths = ['data/parsedCSI04.csv']
poseFilePaths = ['data/parsedPose3D04.csv']

csiList = pd.read_csv(csiFilePaths[0],delimiter=',',header=None).values
poseList = pd.read_csv(poseFilePaths[0],delimiter=',',header=None).values
startFrom=0

if True: # plot pose3D
    fig = plt.figure()
    # gs = gridspec.GridSpec(3,2)
    gs = gridspec.GridSpec(3,2)
    # ax0=fig.add_subplot(gs[0,0])
    # ax1=fig.add_subplot(gs[0,1])
    ax2=fig.add_subplot(gs[0,:])
    ax3=fig.add_subplot(gs[2,:])

    x_values=[[] for i in range(64)]
    y_values=[[] for i in range(64)]
    skipframe=15
    ln1, = plt.plot([], [], 'ro')
    ln2, = plt.plot([], [], 'ro')
    ln = [ln1, ln2]
    def init():
        plt.setp(ax2,xlabel="Frame",ylabel="Amplitude(dB)")
        plt.setp(ax3,xlabel="Frame",ylabel="Amplitude(dB)")
        ax2.set_ylim([-10, +40])
        ax3.set_ylim([-10, +40])
        ax2.set_xlim([ 0,15000000 ])
        ax3.set_xlim([ 0,15000000 ])
        return ln
    def updatefig(i):
        print("updatefig",i)
        poseIdx=(i+startFrom)*skipframe
        if (poseIdx >= len(poseList)-1):
            print('close')
            plt.close(fig)

        if False:
            if True:
                poses_3dFromImage=np.array([poseList[poseIdx][1:].reshape(19,3)])
            else:
                poses_3dFromImage=stand3dmatrix


            edgesFromImage = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3dFromImage.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            canvas_3d = np.zeros((450, 450, 3), dtype=np.uint8)
            plotter = Plotter3d(canvas_3d.shape[:2])
            plotter.plot(canvas_3d, poses_3dFromImage, edgesFromImage)
            ax0.imshow(canvas_3d)
        
        csiIndices=imageIdx2csiIndices_timestamp(poseIdx,poseList,csiList,skipframe=skipframe)
        if(len(csiIndices)>0):
            startCSIIdx=csiIndices[0]
            endCSIIdx=csiIndices[len(csiIndices)-1]
            print(startCSIIdx,'-',endCSIIdx)
            print(endCSIIdx-startCSIIdx+1)
            # custom_xlim = (csiList[startCSIIdx][0], csiList[endCSIIdx][0])
            # print(custom_xlim)
            # plt.setp(ax2, xlim=custom_xlim)
            print([ csiList[startCSIIdx][0] , csiList[endCSIIdx][0] ])
            # ax2.set_xlim([ csiList[startCSIIdx][0] , csiList[endCSIIdx][0] ])
            
            # plot normal AMPLITUDE
            normalAmp=[filterNullSC( rawCSItoAmp(   csiList[j][1:]  )  )    for j in csiIndices]
            for j in range(0,52):
                textX=[]
                textY=[]
                for k in range(len(normalAmp)):
                    curCsi=normalAmp[k]
                    textX.append(csiList[csiIndices[0]+k][0])
                    textY.append(curCsi[j])
                ax2.plot(textX,gaussian_filter(textY,sigma=0), label='CSI subcarrier')

            # plot resamplinged AMPLITUDE
            poseIndices=[j for j in range(poseIdx-skipframe,poseIdx)]
            samplingedAmp,expectedTSs=samplingCSI(csiList,csiIndices,poseList,poseIndices,paddingTo=skipframe)
            for j in range(0,52):
                textX=[]
                textY=[]
                for k in range(len(samplingedAmp)):
                    curCsi=samplingedAmp[k]
                    textX.append(expectedTSs[k])
                    textY.append(curCsi[j])
                ax3.plot(textX,gaussian_filter(textY,sigma=0), label='CSI samplinged subcarrier')

            # plot fe AMPLITUDE
            FE_X = featureEngineer(samplingedAmp)
            # print(FE_X)
            for j in range(0,52):
                textX=[]
                textY=[]
                for k in range(len(FE_X)):
                    curCsi=FE_X[k]
                    textX.append(expectedTSs[k])
                    textY.append(curCsi[j])
                # ax3.plot(textX,gaussian_filter(textY,sigma=0), label='CSI samplinged subcarrier')


            ax2.set_xlim([ csiList[startCSIIdx][0] , csiList[endCSIIdx][0] ])
            ax3.set_xlim([ expectedTSs[0] , expectedTSs[-1] ])

        # ax2.set_ylim([-10, +40])
        # ax3.set_ylim([-10, +40])
        
        return [ax2,ax3]
    

    # ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True,frames=len(csiList),repeat=False,init_func=init)
    # custom_ylim = (-10, +40)
    ani = animation.FuncAnimation(fig, updatefig, frames=len(csiList),
                    init_func=init, blit=True)
    plt.show()
