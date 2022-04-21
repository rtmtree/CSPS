import numpy as np

X=np.array([1,2,3,4,5,6,7,8,9,20])
Y=np.array([4,5,6,7])


print(X)
print(Y)

print(X.shape[0])
print(int(X.shape[0]*0.9))
print(X[9:])