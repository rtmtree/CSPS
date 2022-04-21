import numpy as np 
import json
from modules.draw import Plotter3d, draw_poses
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import re
from scipy.ndimage import gaussian_filter
from math import sqrt, atan2 ,isnan,floor
import matplotlib.gridspec as gridspec
import glob
import imageio
import cv2
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from modules.parse_poses import parse_poses
# path='drive/MyDrive/Project/'
path=''
net = InferenceEnginePyTorch(path+'human-pose-estimation-3d.pth', "GPU", use_tensorrt=False)
def rawCSItoAmp(data,len=128):
    if(data==False):
        return False
    amplitudes = []
    for j in range(0,128,2):
        amplitudes.append(sqrt(data[j] ** 2 + data[j+1] ** 2))
    return amplitudes
    
def parseCSI(csi):
    try:
        csi_string = re.findall(r"\[(.*)\]", csi)[0]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
        return csi_raw
    except:
        return False

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d
def reshape_poses(rotatedPose):
    poses_3d_copy = rotatedPose.copy()
    x = poses_3d_copy[:, 0::4]
    y = poses_3d_copy[:, 1::4]
    z = poses_3d_copy[:, 2::4]
    rotatedPose[:, 0::4], rotatedPose[:, 1::4], rotatedPose[:, 2::4] = -z, x, -y
    rotatedPose = rotatedPose.reshape(rotatedPose.shape[0], 19, -1)[:, :, 0:3]

    return rotatedPose
def imageIdx2csiIdx(durationSec,imageIdx,tsList,fps):
    offsetTime=  (tsList[len(tsList)-1]/(10**6)) - durationSec

    # print("offsetTime",offsetTime)
    timeInVid= (imageIdx/vidLength) * durationSec
    # print("time in vid",timeInVid)

    timeInCSI=(timeInVid-offsetTime) 
    # print("timeInCSI",timeInCSI)

    csiIndex = min(range(len(tsList)), key=lambda i: abs(tsList[i]-(timeInCSI*(10**6))))
    # print("csiIndex",csiIndex)
    # print("csiIndex exp",timeInCSI)
    # print("csiIndex real",tsList[csiIndex]/(10**6))
    # if(np.abs(tsList[csiIndex]-timeInCSI)<100000):
    if(True):
        return csiIndex
    else:
        return False
def imageIdx2csiIndicesPrecise(durationSec,imageIdx,tsList,vidLength,lastsec):
    durationMicroSec=durationSec*(10**6)
    offsetTime= lastsec - durationMicroSec
    print("last CSI ts",(lastsec))
    print("offsetTime",offsetTime)
    timeInVid= ((imageIdx+1)/vidLength) * durationMicroSec
    prevTimeInVid= ((imageIdx)/vidLength) * durationMicroSec

    prevParsedTimeInVid= prevTimeInVid + offsetTime
    parsedTimeInVid= timeInVid + offsetTime

    print("prevTimeInVid",(prevTimeInVid))
    print("timeInVid",(timeInVid))
    print("prevParsedTimeInVid",prevParsedTimeInVid)
    print("parsedTimeInVid",(parsedTimeInVid))
    csiIndices=[]
    for i in range(len(tsList)):
        if(prevParsedTimeInVid < tsList[i] and tsList[i] <= parsedTimeInVid):
            csiIndices.append(i)
    
    return csiIndices,parsedTimeInVid

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

nullPose3D = np.array([[  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0.,
  -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,
   0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0.,
  -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,
   0.,  0., -1., -1.]])


with open(path+'extrinsics.json','r') as f:
    extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)
    

# labels=['02','03','04','05','07','08','09','10','11']
labels=['05']

for label in labels:
    filePath = path+'raw_data/'+label+'.csv'

    
    my_filter_address="7C:9E:BD:D2:D8:9C"
    PacketLength=100000

    print("read file",filePath)
    curFile = pd.read_csv(filePath)
    curFile = curFile[(curFile['mac']==my_filter_address )]
    curFile = curFile[(curFile['len']==384 )]
    curFile = curFile[(curFile['stbc']==0 )]
    curFile = curFile[(curFile['rx_state']==0 )]
    curFile = curFile[(curFile['sig_mode']==1 )]
    curFile = curFile[(curFile['bandwidth']==1 )]
    curFile = curFile[(curFile['secondary_channel']==1 )]
    print("read file done")
    print("len file",len(curFile.index))
    curCSI = curFile['CSI_DATA']
    # csiList = list(rawCSItoAmp(parseCSI(x),128) for x in curCSI)
    csiList = list(x for x in curCSI)
    curRSSI = curFile['rssi']
    rssiList=list(x for x in curRSSI)
    curTS = curFile['real_timestamp']
    tsList=(list(x*(10**6) for x in curTS))
    lastsec=tsList[-1]
    print('last local_timestamp is',lastsec)

    if False: # plot pose3D
        fig = plt.figure()
        gs = gridspec.GridSpec(2,2)
        ax0=fig.add_subplot(gs[0,0])
        ax1=fig.add_subplot(gs[0,1])
        ax2=fig.add_subplot(gs[1,:])

        x_values=[[] for i in range(64)]
        y_values=[[] for i in range(64)]

    if True: 
        pose3D_value=[]
        csi_value=[]

        vidFile=path+'raw_data/'+label+'.mov'
        is_video = True
        stride = 8
        base_height=256
        vid = imageio.get_reader(vidFile,  'ffmpeg')
        meta_data=vid.get_meta_data()
        duration_in_sec = meta_data['duration']
        print("duration_in_sec",duration_in_sec)
        cap = cv2.VideoCapture(vidFile)
        vidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("vidlength",vidLength)
        FPS=int(vidLength/duration_in_sec)
        print("FPS",FPS)
        
        # startFrom=29*280
        startFrom=0


        def updatefig(i,noPlot=False):
            print("updatefig",i,label)
            # ax0.cla()
            # ax1.cla()
            # ax2.cla()

            imageIdx=i+startFrom
            print("imageIdx",imageIdx)

            if (noPlot==False and imageIdx == vidLength):
                print(f'imageIdx {imageIdx} == vidLength {vidLength}; closing!')
                plt.close(fig)
            try:
                frame = vid.get_data(imageIdx)
                input_scale = base_height / frame.shape[0]
                fx = np.float32(0.8 * frame.shape[1])
                scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
                scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
                
                inference_result = net.infer(scaled_img)
                poses_3dFromImage, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
            except:
                poses_3dFromImage=[]
                poses_2d=[]

            if len(poses_3dFromImage)==0 or len(poses_2d)==0 or poses_3dFromImage.all()==nullPose3D.all():
                print("No pose  detected ")
                # return False if noPlot==True else ax0,ax1,ax2
                # return False 
                poses_3dFromImage=np.array([np.zeros((19,3))])
            else:
                if True:
                    poses_3dFromImage = rotate_poses(poses_3dFromImage, R, t)
                    poses_3dFromImage = reshape_poses(poses_3dFromImage)
                else:
                    poses_3dFromImage=stand3dmatrix


            if(noPlot==False):
                edgesFromImage = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3dFromImage.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
                canvas_3d = np.zeros((450, 450, 3), dtype=np.uint8)
                plotter = Plotter3d(canvas_3d.shape[:2])
                plotter.plot(canvas_3d, poses_3dFromImage, edgesFromImage)
                ax0.imshow(canvas_3d)
                draw_poses(frame, poses_2d)
                ax1.imshow(frame)        

            # Setting the values for all axes.
            csiIndices,parsedTimeInVid=imageIdx2csiIndicesPrecise(duration_in_sec,imageIdx,tsList,vidLength,lastsec)
            
            if(noPlot==True):
                print("parsedTimeInVid",parsedTimeInVid)
                parsedPoses_3dFromImage=np.array(poses_3dFromImage[0]).reshape(3*19)
                parsedTimeInVid_array=np.array([parsedTimeInVid])
                pose3D_value.append(    np.concatenate((parsedTimeInVid_array,parsedPoses_3dFromImage))     )
                if(len(csiIndices)>0):
                    startCSIIdx=csiIndices[0]
                    endCSIIdx=csiIndices[len(csiIndices)-1]
                    print(startCSIIdx,'-',endCSIIdx)
                    print(endCSIIdx-startCSIIdx+1)
                    for k in csiIndices:
                        curParseCSI=parseCSI(csiList[k])
                        print("adding ",curParseCSI)
                        if(curParseCSI!=False):
                            print("len check")
                            print(k,len(curParseCSI),tsList[k])
                            if(len(curParseCSI)!=384):
                                print("len not 384")
                                continue
                            print("isFloat check")
                            isInt = True
                            for l in range(384):
                                if(isinstance(curParseCSI[l], int)==False):
                                    print(curParseCSI[l]," is not int")
                                    isInt = False
                                    break
                            if isInt==False:
                                continue
                            csi_value.append([tsList[k]]+parseCSI(csiList[k]))
                            print("added ",k)
                        else:
                            csi_value.append([tsList[k]]+[0 for l in range(384)])
                            print("added ",k,'as 0s')
            else:
                for j in range(0,64):
                    if (6<=j<32 or 33<=j<59):
                        textX=[]
                        textY=[]
                        for k in csiIndices:
                            textX.append(tsList[k]/(10**6))
                            textY.append(rawCSItoAmp(parseCSI(x),128)[k][j])
                        ax2.plot(textX,gaussian_filter(textY,sigma=1), label='CSI subcarrier')
                print("added")
                
            # print(tsList[csiIdx])
            return False #if noPlot==True else ax0,ax1,ax2

    
        justCollect = True
        if(justCollect):
            collectLength=vidLength-startFrom
            # collectLength=8
            print('startFrom',startFrom)
            print('length',collectLength)
            for i in range(collectLength):
            # for i in range(0,collectLength-1):
            # for i in range(0,5):
                updatefig(i,noPlot=True)
        else:
            ani = animation.FuncAnimation(fig, updatefig, interval=1, blit=True,frames=vidLength-startFrom,repeat=False)
            custom_ylim = (-10, +40)
            plt.setp(ax2, ylim=custom_ylim,xlabel="Frame",ylabel="Amplitude(dB)")
            plt.show()

        print('saving',label)

        csi_value=np.array(csi_value)
        pose3D_value=np.array(pose3D_value)
        # pose3D_value=np.array(pose3D_value)
        print(csi_value.shape)
        print(pose3D_value.shape)
        np.savetxt(path+'data/CSI'+label+'.csv', csi_value ,delimiter=',',fmt='%1.6f')
        print('saved',path+'data/CSI'+label+'.csv')
        np.savetxt(path+'data/Pose3D'+label+'.csv', pose3D_value,delimiter=',',fmt='%1.6f')
        print('saved',path+'data/Pose3D'+label+'.csv')
