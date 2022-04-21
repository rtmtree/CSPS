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

label = 'xx'
# label = 'sleep30-11-2021end1020'
withVid = False


csiFilePaths = ['data/CSI'+label+'.csv']
poseFilePaths = ['data/SS'+label+'.csv']

csiList = pd.read_csv(csiFilePaths[0], delimiter=',', header=None).values
sleepList = pd.read_csv(poseFilePaths[0], delimiter=',', header=None).values
startFrom = 239
print("read files done")
SSWindowSize = 10


if True:  # plot pose3D
    fig = plt.figure()
    # gs = gridspec.GridSpec(3,2)
    gs = gridspec.GridSpec(7, 2)
    ax0 = fig.add_subplot(gs[0:3, :])
    # ax1=fig.add_subplot(gs[0,1])
    # ax2 = fig.add_subplot(gs[3, :])
    ax3 = fig.add_subplot(gs[4:6, :])

    x_values = [[] for i in range(64)]
    y_values = [[] for i in range(64)]
    skipframe = 1
    ln1, = plt.plot([], [], 'ro')
    ln2, = plt.plot([], [], 'ro')
    ln = [ln1, ln2]

    def init():
        ax0.set_yticklabels(['','wake', 'rem', 'light', 'deep',''])
        plt.setp(ax0, xlabel="Frame (1/30s)", ylabel="Sleep Stage")
        # ax0.set_xlim([0, SSWindowSize])
        # ax0.set_xlim([0, len(sleepList)])
        ax0.set_xlim([startFrom, startFrom+SSWindowSize])
        ax0.set_ylim([0, 5])
        # plt.setp(ax2, xlabel="Frame (30s)", ylabel="Amplitude(dB)")
        # ax2.set_ylim([-10, +40])
        # ax2.set_xlim([0, 15000000])
        plt.setp(ax3, xlabel="Frame (30s)", ylabel="Amplitude(dB)")
        ax3.set_ylim([-10, +40])
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
        print(sleepList[sleepIdx][0],"!!!!!==========stage", sleepList[sleepIdx][1], stage)
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
            sleepIdx, sleepList, csiList, timeLen=30)
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
            # normalAmp = [filterNullSC(rawCSItoAmp(csiList[j][1:]))
            #              for j in csiIndices]
            # for j in range(0, 52):
            #     textX = []
            #     textY = []
            #     for k in range(len(normalAmp)):
            #         curCsi = normalAmp[k]
            #         textX.append(csiList[csiIndices[0]+k][0])
            #         textY.append(curCsi[j])
            #     ax2.plot(textX, gaussian_filter(
            #         textY, sigma=0), label='CSI subcarrier')
            # ax2.set_xlim([csiList[startCSIIdx][0], csiList[endCSIIdx][0]])


            # plot resamplinged AMPLITUDE
            samplingedAmp, expectedTSs = samplingCSISleep(
                csiList, csiIndices, sleepList, sleepIdx, 50,30)
            for j in range(0, 52):
                textX = []
                textY = []
                for k in range(len(samplingedAmp)):
                    curCsi = samplingedAmp[k]
                    textX.append(expectedTSs[k])
                    textY.append(curCsi[j])
                ax3.plot(textX, gaussian_filter(textY, sigma=0),
                         label='CSI samplinged subcarrier')
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

        return [ax0, ax3]
    # ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True,frames=len(csiList),repeat=False,init_func=init)
    # custom_ylim = (-10, +40)
    ani = animation.FuncAnimation(fig, updatefig, frames=len(csiList), interval=5000,
                                  init_func=init, blit=True)
    plt.show()
