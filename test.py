import numpy as np

def euclidean_dist(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

print(euclidean_dist([1,2,3,20,4,6,2,5,54],[5,10,11,0,0,43,2,3,0]))