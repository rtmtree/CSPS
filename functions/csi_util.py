import numpy as np 
from math import sqrt, atan2 ,isnan
import re

def parseCSI(csi):
    try:
        csi_string = re.findall(r"\[(.*)\]", csi)[0]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
        return csi_raw
    except:
        return False

def rawCSItoAmp(data,length=128):
    amplitudes = []
    for j in range(0,length,2):
        amplitudes.append(sqrt(data[j] ** 2 + data[j+1] ** 2))
    return amplitudes
def filterNullSC(data,length=64):
    amplitudes = []
    for j in range(0,length,1):
        if (6<=j<32 or 33<=j<59):
            amplitudes.append(data[j])
    return amplitudes
    
def featureEngineer(X):
    FE_X = X
    for i in range(len(FE_X)):
        for j in range(len(FE_X[i])-1,0,-1):
            FE_X[i][j]=FE_X[i][j]-FE_X[i][j-1]
        FE_X[i][0]=FE_X[i][0]-FE_X[i][0]
    return FE_X

def featureEngineerNorm(X):
    FE_X = X
    for i in range(len(FE_X)):
        for j in range(len(FE_X[i])):
            FE_X[i][j]=FE_X[i][j]-FE_X[i][0]
    return FE_X

def csiIndices_sec(startTime,endTime,csiList):
    csiIndices=[]
    # print("startTime",startTime)
    # print("endTime",endTime)
    for i in range(0,len(csiList)):
        if(csiList[i][0]>=startTime and csiList[i][0]<endTime):
            # print("addeed",csiList[i][0])
            csiIndices.append(i)
        elif(csiList[i][0]>=endTime):
            break
    return csiIndices
def poseIndices_sec(index,poseList,sec=1):
    endTime=poseList[index][0]
    startTime = endTime-sec
    poseIndices=[index]
    # print("startTime",startTime)
    # print("endTime",endTime)
    # for i in range(index,len(poseList)):
    #     if(poseList[i][0]<endTime):
    #         # print("addeed",poseList[i][0])
    #         poseIndices.append(i)
    #     else:
    #         break
    return poseIndices,startTime,endTime

def samplingCSI(csiList,csiIndices,poseList,poseIndices,paddingTo=30):
    simplingedCSIs=[]
    expectedTSs=[]
    for j in range(paddingTo):
        # print(poseIndices)
        # print(j)
        expectedTS=poseList[poseIndices[j]][0]
        # print("index",j,"expect",expectedTS)

        expectedTSs.append(expectedTS)

        csiIndicesExtended=[]
        if(csiIndices[0]!=0 or j!=0):
            csiIndicesExtended += [csiIndices[0]-1]
        else:
            simplingedCSIs.append( np.array(filterNullSC( rawCSItoAmp(   csiList[csiIndices[0]][1:]  )  ) )   )
            startIndex = 0
            continue

        csiIndicesExtended += csiIndices

        if(csiIndices[-1]!=len(csiList)-1):
            csiIndicesExtended += [csiIndices[-1]+1]
        # else:
        #     None
        
        if j==0:
            startIndex = csiIndicesExtended[0]
        for k in range(startIndex,csiIndicesExtended[-1]+1):
            if(k>=len(csiList) or csiList[k][0]>expectedTS):
                break
            else:
                startIndex=k
        # if startIndex TS matched expected TS

        if(csiList[startIndex][0]==expectedTS):
            simplingedCSIs.append( np.array( filterNullSC( rawCSItoAmp(   csiList[startIndex][1:]  )  ) )   )
            continue
        
        if(startIndex== len(csiList)-1  ):
            simplingedCSIs.append( np.array( filterNullSC( rawCSItoAmp(   csiList[startIndex][1:]  )  ) )   )
            continue

        endIndex = startIndex+1
        
        startCSI=filterNullSC( rawCSItoAmp(   csiList[startIndex][1:]  )  )
        endCSI=filterNullSC( rawCSItoAmp(   csiList[endIndex][1:]  )  )
        middleCSI=[]
        offsetX=csiList[endIndex][0]-csiList[startIndex][0]
        offsetXo=expectedTS -csiList[startIndex][0]
        for k in range(52):
            offsetY=endCSI[k]-startCSI[k]
            offsetYo= (offsetXo*offsetY) / offsetX

            middleCSI.append((startCSI[k]+offsetYo))

        simplingedCSIs.append( np.array(middleCSI)   )
    return simplingedCSIs,expectedTSs

def samplingCSISleep(csiList,csiIndices,poseList,sleepIndex,newCSILen,timeLen=30):
    simplingedCSIs=[]
    expectedTSs=[]
    startTS = poseList[sleepIndex][0]- timeLen
    for j in range(newCSILen):
        # print(poseIndices)
        # print(j)
        # print(poseList[poseIndices[0]][0])
        if(j<newCSILen-1):
            expectedTS=  startTS +  ( j * (poseList[sleepIndex][0] - startTS)/newCSILen  )
        else:
            expectedTS=poseList[sleepIndex][0]
        # print("index",j,"expect",expectedTS)

        expectedTSs.append(expectedTS)

        csiIndicesExtended=[]
        if(csiIndices[0]!=0 or j!=0):
            csiIndicesExtended += [csiIndices[0]-1]
        else:
            simplingedCSIs.append( np.array(filterNullSC( rawCSItoAmp(   csiList[csiIndices[0]][1:]  )  )  )   )
            startIndex = 0
            continue

        csiIndicesExtended += csiIndices

        if(csiIndices[-1]!=len(csiList)-1):
            csiIndicesExtended += [csiIndices[-1]+1]
        # else:
        #     None
        
        if j==0:
            startIndex = csiIndicesExtended[0]
        for k in range(startIndex,csiIndicesExtended[-1]+1):
            if(k>=len(csiList) or csiList[k][0]>expectedTS):
                break
            else:
                startIndex=k
        # if startIndex TS matched expected TS

        if(csiList[startIndex][0]==expectedTS):
            simplingedCSIs.append( np.array( filterNullSC( rawCSItoAmp(   csiList[startIndex][1:]  )  )    )   )
            continue
        
        if(startIndex== len(csiList)-1  ):
            simplingedCSIs.append( np.array( filterNullSC( rawCSItoAmp(   csiList[startIndex][1:]  )  )   )   )
            continue

        endIndex = startIndex+1
        
        startCSI=filterNullSC( rawCSItoAmp(   csiList[startIndex][1:]  )  ) 
        endCSI=filterNullSC( rawCSItoAmp(   csiList[endIndex][1:]  )  ) 
        middleCSI=[]
        offsetX=csiList[endIndex][0]-csiList[startIndex][0]
        offsetXo=expectedTS -csiList[startIndex][0]
        for k in range(len(startCSI)):
            offsetY=endCSI[k]-startCSI[k]
            offsetYo= (offsetXo*offsetY) / offsetX

            middleCSI.append((startCSI[k]+offsetYo))

        simplingedCSIs.append( np.array(middleCSI)   )
    return simplingedCSIs,expectedTSs

def samplingRSSISleep(csiList,csiIndices,poseList,sleepIndex,newCSILen,timeLen=30):
    simplingedCSIs=[]
    expectedTSs=[]
    startTS = poseList[sleepIndex][0]- timeLen
    for j in range(newCSILen):
        if(j<newCSILen-1):
            expectedTS=  startTS +  ( j * (poseList[sleepIndex][0] - startTS)/newCSILen  )
        else:
            expectedTS=poseList[sleepIndex][0]

        expectedTSs.append(expectedTS)

        csiIndicesExtended=[]
        if(csiIndices[0]!=0 or j!=0):
            csiIndicesExtended += [csiIndices[0]-1]
        else:
            simplingedCSIs.append( np.array(   csiList[csiIndices[0]][1:]  )  ) 
            startIndex = 0
            continue

        csiIndicesExtended += csiIndices

        if(csiIndices[-1]!=len(csiList)-1):
            csiIndicesExtended += [csiIndices[-1]+1]
        # else:
        #     None
        
        if j==0:
            startIndex = csiIndicesExtended[0]
        for k in range(startIndex,csiIndicesExtended[-1]+1):
            if(k>=len(csiList) or csiList[k][0]>expectedTS):
                break
            else:
                startIndex=k
        # if startIndex TS matched expected TS

        if(csiList[startIndex][0]==expectedTS):
            simplingedCSIs.append( np.array(    csiList[startIndex][1:]  )  ) 
            continue
        
        if(startIndex== len(csiList)-1  ):
            simplingedCSIs.append( np.array(   csiList[startIndex][1:]  )  ) 
            continue

        endIndex = startIndex+1
        
        startCSI= csiList[startIndex][1:]  
        endCSI=  csiList[endIndex][1:]  
        middleCSI=[]
        offsetX=csiList[endIndex][0]-csiList[startIndex][0]
        offsetXo=expectedTS -csiList[startIndex][0]
        for k in range((1)):
            offsetY=endCSI[k]-startCSI[k]
            offsetYo= (offsetXo*offsetY) / offsetX

            middleCSI.append((startCSI[k]+offsetYo))

        simplingedCSIs.append( np.array(middleCSI)   )
    return simplingedCSIs,expectedTSs

def imageIdx2csiIndices_timestamp(poseIdx,poseList,csiList,skipframe=1):
    timeInPose=poseList[poseIdx][0]
    if (poseIdx>0):
        prevTimeInPose=poseList[poseIdx-skipframe][0]
    else:
        prevTimeInPose=0

    csiIndices=[]
    for i in range(len(csiList)):
        if(prevTimeInPose < csiList[i][0] <= timeInPose):
            csiIndices.append(i)
    
    return csiIndices

def sleepIdx2csiIndices_timestamp(sleepIndex,sleepList,csiStartIdx,csiList,timeLen=30):
    timeInPose=sleepList[sleepIndex][0]
    prevTimeInPose=sleepList[sleepIndex][0]-timeLen
    csiIndices=[]
    if(isinstance(prevTimeInPose,np.int64 ) and isinstance(timeInPose,np.int64)):
        for i in range(csiStartIdx,len(csiList)):
            # print("csiList",type(csiList[i][0]))
            # print("timeInPose",type(timeInPose))
            if(csiList[i][0]>=timeInPose):
                break
            if(isinstance(csiList[i][0],np.float64)):
                if(prevTimeInPose <= csiList[i][0] and csiList[i][0] < timeInPose):
                    csiIndices.append(i)
    return csiIndices