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
from modules.inference_engine_pytorch import InferenceEnginePyTorch
# from modules.parse_poses import parse_poses
from functions.csi_util import rawCSItoAmp,imageIdx2csiIndices_timestamp,samplingCSI,filterNullSC,featureEngineer
stand3dmatrix = np.array(
            [[[  17.793797   ,-147.97356   ,  147.04082   ],
            [  10.467414   ,-161.14389   ,  160.92406   ],
            [  15.775149   ,-143.2878    ,   99.19944   ],
            [  26.34641    ,-150.5476    ,  146.2604    ],
            [  31.166206   ,-155.43958   ,  119.391556  ],
            [  27.480389   ,-158.90024   ,  102.42989   ],
            [  21.38687    ,-150.89978   ,   94.70287   ],
            [  21.46943    ,-153.1436    ,   57.474857  ],
            [  19.946053   ,-151.5097    ,   21.78808   ],
            [   8.3965435  ,-150.06699   ,  147.06216   ],
            [   0.6513193  ,-152.40784   ,  121.6008    ],
            [  -1.6908724  ,-159.19742   ,  104.371     ],
            [   3.377078   ,-146.47162   ,  101.29445   ],
            [   1.7780559  ,-149.1533    ,   65.287445  ],
            [  -0.74428445 ,-142.35242   ,   32.657227  ],
            [  13.069774   ,-160.25378   ,  160.9427    ],
            [  19.783405   ,-152.43784   ,  157.54123   ],
            [  10.242576   ,-164.34209   ,  161.9919    ],
            [  10.958148   ,-156.61496   ,  162.87784   ]]]
        )

label='05'
withVid = True


csiFilePaths = ['data/CSI'+label+'.csv']
poseFilePaths = ['data/Pose3D'+label+'.csv']

csiList = pd.read_csv(csiFilePaths[0],delimiter=',',header=None).values
poseList = pd.read_csv(poseFilePaths[0],delimiter=',',header=None).values
startFrom=0

if withVid:
    vidFilePaths = ['raw_data/'+label+'.mov']
    vid = imageio.get_reader(vidFilePaths[0],  'ffmpeg')
    meta_data=vid.get_meta_data()
    duration_in_sec = meta_data['duration']
    cap = cv2.VideoCapture(vidFilePaths[0])
    vidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if True: # plot pose3D
    fig = plt.figure()
    # gs = gridspec.GridSpec(3,2)
    gs = gridspec.GridSpec(3,2)
    ax0=fig.add_subplot(gs[0,0])
    ax1=fig.add_subplot(gs[0,1])
    ax2=fig.add_subplot(gs[1,:])
    ax3=fig.add_subplot(gs[2,:])

    x_values=[[] for i in range(64)]
    y_values=[[] for i in range(64)]
    skipframe=30
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
        print("poseIdx",poseIdx)
        if (poseIdx >= len(poseList)-1):
            print('close')
            plt.close(fig)

        if True:
            if withVid:
                print("get vid")
                imageIdx=poseIdx
                frame = vid.get_data(imageIdx)
                ax1.imshow(frame)
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
        
        return [ax0,ax1,ax2,ax3]
    

    # ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True,frames=len(csiList),repeat=False,init_func=init)
    # custom_ylim = (-10, +40)
    ani = animation.FuncAnimation(fig, updatefig, frames=len(csiList),interval=1000,
                    init_func=init, blit=True)
    plt.show()
