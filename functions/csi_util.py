import numpy as np 
from math import sqrt, atan2 ,isnan

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
def poseIndices_sec(index,poseList,seqLen=30):
    sec = seqLen/30
    startTime = poseList[index][0]
    # endTime=startTime+(sec* (10**6) )
    endTime=poseList[index+seqLen-1][0]
    poseIndices=[i for i in range(index,index+seqLen)]
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


def imageIdx2csiIndices_timestamp(poseIdx,poseList,csiList,skipframe=1):
    timeInPose=poseList[poseIdx][0]
    print("timeInPose",timeInPose)
    if (poseIdx>0):
        prevTimeInPose=poseList[poseIdx-skipframe][0]
    else:
        prevTimeInPose=0
    print("prevTimeInPose",prevTimeInPose)
    csiIndices=[]
    for i in range(len(csiList)):
        if(prevTimeInPose < csiList[i][0] <= timeInPose):
            csiIndices.append(i)
    print()
    return csiIndices