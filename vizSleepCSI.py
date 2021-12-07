import numpy as np
import json
from modules.draw import Plotter3d, draw_poses
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import re
from scipy.ndimage import gaussian_filter
from math import sqrt, atan2, isnan
import matplotlib.gridspec as gridspec
import glob
import imageio
import cv2
from modules.inference_engine_pytorch import InferenceEnginePyTorch
# from modules.parse_poses import parse_poses
from functions.csi_util import rawCSItoAmp, samplingCSISleep, sleepIdx2csiIndices_timestamp, samplingCSISleep, filterNullSC, featureEngineer
stand3dmatrix = np.array(
    [[[17.793797, -147.97356,  147.04082],
      [10.467414, -161.14389,  160.92406],
      [15.775149, -143.2878,   99.19944],
      [26.34641, -150.5476,  146.2604],
      [31.166206, -155.43958,  119.391556],
      [27.480389, -158.90024,  102.42989],
      [21.38687, -150.89978,   94.70287],
      [21.46943, -153.1436,   57.474857],
      [19.946053, -151.5097,   21.78808],
      [8.3965435, -150.06699,  147.06216],
      [0.6513193, -152.40784,  121.6008],
      [-1.6908724, -159.19742,  104.371],
      [3.377078, -146.47162,  101.29445],
      [1.7780559, -149.1533,   65.287445],
      [-0.74428445, -142.35242,   32.657227],
      [13.069774, -160.25378,  160.9427],
      [19.783405, -152.43784,  157.54123],
      [10.242576, -164.34209,  161.9919],
      [10.958148, -156.61496,  162.87784]]]
)

label = 'sleep30-11-2021end1020'
withVid = False


csiFilePaths = ['data/CSI'+label+'.csv']
poseFilePaths = ['data/SS'+label+'.csv']

csiList = pd.read_csv(csiFilePaths[0], delimiter=',', header=None).values
sleepList = pd.read_csv(poseFilePaths[0], delimiter=',', header=None).values
startFrom = 0
print("read files done")
SSWindowSize = 10


if True:  # plot pose3D
    fig = plt.figure()
    # gs = gridspec.GridSpec(3,2)
    gs = gridspec.GridSpec(3, 2)
    ax0 = fig.add_subplot(gs[0, :])
    # ax1=fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])

    x_values = [[] for i in range(64)]
    y_values = [[] for i in range(64)]
    skipframe = 1
    ln1, = plt.plot([], [], 'ro')
    ln2, = plt.plot([], [], 'ro')
    ln = [ln1, ln2]

    def init():
        ax0.set_yticklabels(['wake', 'rem', 'light', 'deep'])
        plt.setp(ax0, xlabel="Frame (1/30s)", ylabel="Sleep Stage")
        ax0.set_xlim([0, SSWindowSize])
        ax0.set_ylim([1, 4])
        plt.setp(ax2, xlabel="Frame (30s)", ylabel="Amplitude(dB)")
        plt.setp(ax3, xlabel="Frame (30s)", ylabel="Amplitude(dB)")
        ax2.set_ylim([-10, +40])
        ax3.set_ylim([-10, +40])
        ax2.set_xlim([0, 15000000])
        ax3.set_xlim([0, 15000000])
        return ln
    SSX = []
    SSY = []

    def updatefig(i):
        print("updatefig", i)
        sleepIdx = (i+startFrom)*skipframe
        print("sleepIdx", sleepIdx)
        if (sleepIdx >= len(sleepList)-1):
            print('close')
            plt.close(fig)
        stage = ''
        if(sleepList[sleepIdx][1] == 1):
            stage = "wake"
        elif(sleepList[sleepIdx][1] == 2):
            stage = "rem"
        elif(sleepList[sleepIdx][1] == 3):
            stage = "light"
        elif(sleepList[sleepIdx][1] == 4):
            stage = "deep"
        print("!!!!!==========stage", sleepList[sleepIdx][1], stage)
        SSX.append(sleepIdx)
        SSY.append(sleepList[sleepIdx][1])
        if(len(SSX) > SSWindowSize):
            SSX.pop()
            SSY.pop()
            print("set", SSX[0], SSX[len(SSX)-1])
            # ax0.set_xlim([ sleepIdx,sleepIdx+SSWindowSize-1 ])
            newXLim = [min(SSX), max(SSX)]
            print("newXLim", newXLim)
            ax0.set_xlim(newXLim)
        ax0.plot(SSX, SSY, label='Sleep stage')

        csiIndices = sleepIdx2csiIndices_timestamp(
            sleepIdx, sleepList, csiList, skipframe=skipframe)
        if(len(csiIndices) > 0):
            startCSIIdx = csiIndices[0]
            endCSIIdx = csiIndices[len(csiIndices)-1]
            print("CSI range", startCSIIdx, '-', endCSIIdx)
            print("len(csiIndices)", len(csiIndices))
            # custom_xlim = (csiList[startCSIIdx][0], csiList[endCSIIdx][0])
            # print(custom_xlim)
            # plt.setp(ax2, xlim=custom_xlim)
            print([csiList[startCSIIdx][0], csiList[endCSIIdx][0]])
            # ax2.set_xlim([ csiList[startCSIIdx][0] , csiList[endCSIIdx][0] ])

            # plot normal AMPLITUDE
            normalAmp = [filterNullSC(rawCSItoAmp(csiList[j][1:]))
                         for j in csiIndices]
            for j in range(0, 52):
                textX = []
                textY = []
                for k in range(len(normalAmp)):
                    curCsi = normalAmp[k]
                    textX.append(csiList[csiIndices[0]+k][0])
                    textY.append(curCsi[j])
                ax2.plot(textX, gaussian_filter(
                    textY, sigma=0), label='CSI subcarrier')

            # plot resamplinged AMPLITUDE
            # sleepIndices=[j for j in range(sleepIdx-skipframe,sleepIdx)]
            # print(sleepIndices)
            samplingedAmp, expectedTSs = samplingCSISleep(
                csiList, csiIndices, sleepList, [sleepIdx, sleepIdx+1], 200)
            for j in range(0, 52):
                textX = []
                textY = []
                for k in range(len(samplingedAmp)):
                    curCsi = samplingedAmp[k]
                    textX.append(expectedTSs[k])
                    textY.append(curCsi[j])
                ax3.plot(textX, gaussian_filter(textY, sigma=0),
                         label='CSI samplinged subcarrier')
            ax2.set_xlim([csiList[startCSIIdx][0], csiList[endCSIIdx][0]])
            ax3.set_xlim([csiList[startCSIIdx][0], csiList[endCSIIdx][0]])

            if False:
                # plot fe AMPLITUDE
                FE_X = featureEngineer(samplingedAmp)
                # print(FE_X)
                for j in range(0, 52):
                    textX = []
                    textY = []
                    for k in range(len(FE_X)):
                        curCsi = FE_X[k]
                        textX.append(expectedTSs[k])
                        textY.append(curCsi[j])
                    # ax3.plot(textX,gaussian_filter(textY,sigma=0), label='CSI samplinged subcarrier')

                ax3.set_xlim([expectedTSs[0], expectedTSs[-1]])
        # ax2.set_ylim([-10, +40])
        # ax3.set_ylim([-10, +40])

        return [ax0, ax2, ax3]
    # ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True,frames=len(csiList),repeat=False,init_func=init)
    # custom_ylim = (-10, +40)
    ani = animation.FuncAnimation(fig, updatefig, frames=len(csiList), interval=5000,
                                  init_func=init, blit=True)
    plt.show()
