import numpy as np
from math import sqrt, atan2, isnan
import re
from datetime import datetime, timedelta


def getSleepStageByTime(sleepTs, lastSleepTs, sleepSData):
    stage = -1
    for j in range(lastSleepTs, len(sleepSData)):
        # print(sleepSData[j])
        tutc_time = datetime.strptime(
            sleepSData[j]['dateTime'], "%Y-%m-%dT%H:%M:%S.%f")
        tepoch_time = int(
            (tutc_time - datetime(1970, 1, 1)).total_seconds())
        if((sleepTs) >= (tepoch_time)):
            if(sleepSData[j]['level'] == "wake"):
                stage = 0
            elif(sleepSData[j]['level'] == "rem"):
                stage = 1
            elif(sleepSData[j]['level'] == "light"):
                stage = 2
            elif(sleepSData[j]['level'] == "deep"):
                stage = 3
        else:
            lastSleepTs = 0 if j == 0 else j-1
            break
    return stage


def oneSS2Mat(focusStage, ss_value):
    curSleeps = False
    curSleeps=[0 for i in range(len(focusStage))]
    
    curIdx = (np.where(focusStage == ss_value))[0][0]
    curSleeps[curIdx]=1
    return curSleeps
def twoSS2Mat(focusStage, ss_value,ss_value2):
    curSleeps = False
    curSleeps=[0 for i in range(len(focusStage)*len(focusStage))]
    # if(ss_value==ss_value2):
    if(False):
        curSleeps[0] = 1
    else:
        curSleeps[(ss_value*len(focusStage)) + ss_value2] = 1
    return curSleeps

def mat2OneSS(mat,focusStage):
    maximum_test = np.max(mat)
    curPred = (np.where(mat == maximum_test))[0][0]
    return focusStage[curPred]


def mat2TwoSS(mat,focusStage):
    isNaN = all(isnan(l) for l in mat)
    if isNaN:
        return focusStage[0],focusStage[0]
    maximum_test = np.max(mat)
    curPred = (np.where(mat == maximum_test))[0][0]

    ss_value = int(curPred/len(focusStage))
    ss_value2 = curPred%len(focusStage)

    return focusStage[ss_value],focusStage[ss_value2]