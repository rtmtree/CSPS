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
from datetime import datetime
import json
import imageio
import cv2
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from modules.parse_poses import parse_poses


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

path='drive/MyDrive/Project/'
# path=''
labels=['sleep8-12-2021end1000']
CSIendTime=['2021-12-09T10:00:00.000']
sleepIndxs = [0]
# timestampColName = 'local_timestamp'
timestampColName = 'real_timestamp'
timeperoid = 30
decimalShiftTs = 0

for fileIdx in range(len(labels)):
    # Organize sleep log file
    sleepFilePath = path+'raw_data/'+labels[fileIdx]+'.json'
    print("read sleep file",sleepFilePath)
    f = open(sleepFilePath)
    data = json.load(f)
    sleepSData = data['sleep'][sleepIndxs[fileIdx]]['levels']['data']

    # find sleep length

    #get first ts of sleep stage log
    firstsecSleep = sleepSData[0]['dateTime']
    print('first date/time in Sleep is',firstsecSleep)
    futc_time = datetime.strptime(firstsecSleep, "%Y-%m-%dT%H:%M:%S.%f")
    sleepFirstTs = (futc_time - datetime(1970, 1, 1)).total_seconds()
    print('first timestamp in Sleep is',sleepFirstTs)
    
    #get last ts of real world ts CSI log
    lastsecSleep = sleepSData[-1]['dateTime']
    print('last date/time in Sleep is',lastsecSleep)
    lutc_time = datetime.strptime(lastsecSleep, "%Y-%m-%dT%H:%M:%S.%f")
    sleepLastTs = (lutc_time - datetime(1970, 1, 1)).total_seconds() + sleepSData[-1]['seconds']
    print('last timestamp in Sleep is',sleepLastTs)
    
    vidLength=int((sleepLastTs-sleepFirstTs)/30)
    print('Sleep time Length',vidLength)
  
    # Organize CSI log file
    filePath = path+'raw_data/'+labels[fileIdx]+'.csv'
    print("read CSI file",filePath)
    colsName = ["type","role","mac","rssi","rate","sig_mode","mcs","bandwidth","smoothing","not_sounding","aggregation","stbc","fec_coding","sgi","noise_floor","ampdu_cnt","channel","secondary_channel","local_timestamp","ant","sig_len","rx_state","real_time_set","real_timestamp","len","CSI_DATA"]
    curFile = pd.read_csv(filePath,names=colsName) 

    #filter out the rows unvalid timestamp. CLEAN Data
    curFile[timestampColName] = curFile[timestampColName].astype('str')
    print("len file bf",len(curFile.index))
    curFile = curFile[curFile[timestampColName].str.match(r"[+-]?([0-9]*[.])?[0-9]+")==True]
    curFile = curFile[curFile[timestampColName].str.contains('[A-Za-z :]')==False]
    curFile = curFile[curFile[timestampColName].str.contains("[.]")==True]
    # curFile = curFile[curFile[timestampColName].str.contains(":")==False]
    
    print("len file af",len(curFile.index))
    curFile[timestampColName] = curFile[timestampColName].astype('float')
    curFile = curFile.sort_values(timestampColName)

    print(curFile)

    #find diffEpoch to sync time
    tsList=(list(x/(10**decimalShiftTs) for x in curFile[timestampColName]))
    #get last ts of real world ts CSI log
    csiRealLastTs=CSIendTime[fileIdx]
    print('last real world date/time in CSI is',csiRealLastTs)
    utc_time = datetime.strptime(csiRealLastTs, "%Y-%m-%dT%H:%M:%S.%f")
    realLastTs = (utc_time - datetime(1970, 1, 1)).total_seconds()
    print('last real world timestamp in CSI is',realLastTs)
    #get last ts of CSI log
    csiLastTs=tsList[-1]
    print('last self timestamp in CSI is',csiLastTs)
    diffEpoch = realLastTs - csiLastTs
    print("diffEpoch is",diffEpoch)

    #filter for usable CSI data
    # my_filter_address="7C:9E:BD:D2:D8:9C"
    my_filter_address="98:F4:AB:7D:DD:1D"
    curFile = curFile[(curFile['mac']==my_filter_address )]
    curFile = curFile[(curFile['len']==384 )]
    curFile = curFile[(curFile['stbc']==0 )]
    curFile = curFile[(curFile['rx_state']==0 )]
    curFile = curFile[(curFile['sig_mode']==1 )]
    curFile = curFile[(curFile['bandwidth']==1 )]
    curFile = curFile[(curFile['secondary_channel']==1 )]
    print("filtering CSI done")
    print("len CSI",len(curFile.index))
    # curCSI = curFile['CSI_DATA']
    # csiList = list(rawCSItoAmp(parseCSI(x),128) for x in curCSI)
    # csiList = list(x for x in curCSI)
    # curRSSI = curFile['rssi']
    # rssiList=list(x for x in curRSSI)

    if True: 
        ss_value=[]
        csi_value=[]
        def updatefig(i):
            global realLastTs
            global sleepSData
            global sleepFirstTs
            global curFile
            global diffEpoch

            sleepIdx=i+startFrom

            sleepTs = int((sleepIdx*30)+sleepFirstTs)
            print("sleepTs",sleepTs)
            if(sleepTs>realLastTs):
              print("Exceed realLastTs. DONE!")
              return False

            stage = 0
            for j in range(len(sleepSData)):
              # print(sleepSData[j])
              tutc_time = datetime.strptime(sleepSData[j]['dateTime'], "%Y-%m-%dT%H:%M:%S.%f")
              tepoch_time = int((tutc_time - datetime(1970, 1, 1)).total_seconds())
              # print(int(sleepTs))
              # print(int(tepoch_time))
              # print(int(sleepTs)>=int(tepoch_time))
              if((sleepTs)>=(tepoch_time)):
                if(sleepSData[j]['level']=="wake"):
                  stage=1
                elif(sleepSData[j]['level']=="rem"):
                  stage=2
                elif(sleepSData[j]['level']=="light"):
                  stage=3
                elif(sleepSData[j]['level']=="deep"):
                  stage=4
              else:
                # print("stage",stage)
                break
            if(stage==0):
              print("catch no sleep stage detected")
              return False
            ss_value.append([sleepTs]+[stage])

            if(True):
                # Setting the values for all axes.
                # csiIndices,parsedTimeInVid=imageIdx2csiIndicesPrecise(duration_in_sec,imageIdx,tsList,vidLength,lastsec)
                csiTsEnd = csiTs
                csiTs = (sleepTs - diffEpoch)-timeperoid
                # parse to microsecond
                csiTsEnd = (10**decimalShiftTs) * csiTsEnd
                csiTs = (10**decimalShiftTs) * csiTs
      
                print("csiTs",csiTs)
                print("csiTsEnd",csiTsEnd)
                
                # dataInPeriod = curFile[(curFile[timestampColName]>=csiTs)]
                # dataInPeriod = dataInPeriod[(dataInPeriod[timestampColName]<csiTsEnd)]

                dataInPeriod = curFile[(curFile[timestampColName]>=csiTs) & (curFile[timestampColName]<csiTsEnd)]
                
                print("len csidataInPeriod",len(dataInPeriod.index))
                tsInPeriod = list((x/(10**decimalShiftTs))+diffEpoch  for x in dataInPeriod[timestampColName])
                csiInPeriod = list(parseCSI(x) for x in dataInPeriod['CSI_DATA'])
                if(len(csiInPeriod)>0):
                    for k in range(len(csiInPeriod)):
                        curParseCSI=(csiInPeriod[k])
                        curParseTs = tsInPeriod[k]
                        if(k>0 and curParseCSI==csiInPeriod[k-1] and curParseTs==tsInPeriod[k-1]):
                          # print("duplicate CSI row found. SKIP")
                          continue
                        # print("adding ",curParseCSI)
                        if(curParseCSI!=False):
                            # print("len check")
                            # print(k,len(curParseCSI),curParseTs)
                            if(len(curParseCSI)!=384):
                                print("len not 384")
                                continue
                            # print("isFloat check")
                            isInt = True
                            for l in range(384):
                                if(isinstance(curParseCSI[l], int)==False):
                                    # print(curParseCSI[l]," is not int")
                                    isInt = False
                                    break
                            if isInt==False:
                                continue
                            csi_value.append([curParseTs]+curParseCSI)
                            # print("added ",len(curParseCSI))
                        else:
                            continue
                            # csi_value.append([curParseTs]+[0 for l in range(384)])
                            # print("added ",k,'as 0s')
                # print("====",i,"====")
                return False
        
        startFrom=0
        collectLength=vidLength-startFrom
        print('startFrom',startFrom)
        print('length',collectLength)
        for i in range(collectLength):
            updatefig(i)
            print("====",i,"/",collectLength,"====")
            
        print('saving',labels[fileIdx])

        csi_value=np.array(csi_value)
        ss_value=np.array(ss_value)
        print(csi_value.shape)
        print(ss_value.shape)
        if(True):
          pathSavedFileCSI = path+'data/CSI'+labels[fileIdx]+'.csv'
          pathSavedFileSleep = path+'data/SS'+labels[fileIdx]+'.csv'
          # pathSavedFileCSI = 'CSI'+labels[fileIdx]+'.csv'
          # pathSavedFileSleep = 'SS'+labels[fileIdx]+'.csv'
          fmt = '%1.6f,'+('%d,'*383)+'%d'
          np.savetxt(pathSavedFileCSI, csi_value ,delimiter=',',fmt=fmt)
          print('saved',pathSavedFileCSI)
          np.savetxt(pathSavedFileSleep, ss_value,delimiter=',',fmt='%d')
          print('saved',pathSavedFileSleep)
