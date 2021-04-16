import numpy as np 
import torch
from math import sqrt, atan2 ,isnan


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
def poseToPAM(pose):
    PAM3D = np.zeros(( 3, 19, 19))
    # x = pose[:,:,0]
    # y = pose[:,:,1]
    # z = pose[:,:,2]
    onePAM = np.zeros((1, 19, 19))
    for f in range(3):
        oneMat = pose[:,:,f]
        for i in range(19):
            for j in range(19):
                if i!=j:
                    onePAM[0,i,j] = oneMat[0,i]-oneMat[0,j]
                else:
                    onePAM[0,i,j] = oneMat[0,i]
        PAM3D[f,:,:] = onePAM

    return PAM3D

def PAMtoPose(PAM):
    pose3D = np.ones((1, 19, 3))
    for index in range(19):
        pose3D[0,index,0] = PAM[0,index,index]
        pose3D[0,index,1] = PAM[1,index,index]
        pose3D[0,index,2] = PAM[2,index,index]
    return pose3D
def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def calc_dists(preds, target, normalize):
    print(preds)
    print(target)
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dist, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1

# def accuracy(output, target, idxs, thr=0.5):
def accuracy(preds, gts, idxs):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    # preds   = get_preds(output)
    # gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]-1])
        if acc[i+1] >= 0:
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc

def getPCK(preds, gts,idx, frame=30,threshold=50):
    PCK=0
    for i in range(frame):
        curPreds=PAMtoPose(preds[i].reshape(3,19,19))[0]
        curGts=PAMtoPose(gts[i].reshape(3,19,19))[0]
        sumKP=0
        rt_shoulder= curGts[6]
        lf_hip= curGts[9]
        upper_limp = sqrt(
                sum([(rt_shoulder[0] - lf_hip[0])**2,(rt_shoulder[1] - lf_hip[1])**2,(rt_shoulder[2] - lf_hip[2])**2])
        )
        # print("upper_limp")
        # print(upper_limp)
        a = curPreds[idx]
        b = curGts[idx]
        dist_squared = sum([(a[0] - b[0])**2,(a[1] - b[1])**2,(a[2] - b[2])**2])
        # print("dist_squared/upper_limp")
        # print(dist_squared/upper_limp)
        if( upper_limp!=0 and (dist_squared/upper_limp )<=threshold):
            PCK+=1
    PCK=PCK/frame
    return PCK