import numpy as np
import numpy as np
import scipy.io
from PIL import Image
import time
from matplotlib import image
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from sklearn.datasets import load_wine
from Prob1_1 import auto_run
# This program is to transform A into bidiagonal form utilizing householder

#A=np.random.rand(256,256)
def image_process(L):
    im = Image.open(L) 
    im = im.convert('L')
    img = np.array(im)
    img = img.astype(np.float64) / 256
    return img

#L = "D:\\Download\\test_images\\test_images\\256_256_hand.png"
#A = image_process(L)
#print(A)

A = np.array([
    [1.0,3.0,2.5,43,43,65],
    [2.3,4.5,2.5,23,52,1],
    [2.4,2.5,2.6,52,7,2]
])


# 第二问,求∑
def WilkinsonShift(X):
    sigma = (X[0,0]-X[1,1])/2
    # 如果这么计算是正确的，回头还要考虑加上 (abs(sigma) + np.sqrt((sigma**2)+(X[0,1]**2))=0 的情况
    if sigma > 0:
        ratio = X[1,1] - (X[0,1]**2)/(abs(sigma) + np.sqrt((sigma**2)+(X[0,1]**2)))
    else:
        ratio = X[1,1] + (X[0,1]**2)/(abs(sigma) + np.sqrt((sigma**2)+(X[0,1]**2)))
    return ratio

def QR_factorization(B,diff):
    T = B.T@B
    Yk = T
    eigenvalues = [0 for i in range(len(T))]
    eigenvector_matrix = np.identity(len(T))
    # deflation
    while len(Yk)>0:
        Q_sum = np.identity(len(Yk))
        if len(Yk) == 1:
            eigenvalues[0] = Yk[0,0]
            Qk = Yk
        else:
            n = np.shape(Yk)[0] #Yk的size
            # Wilkinson shift
            ratio = WilkinsonShift(Yk)
            flag = 1
            while flag:
                Yk_1 = Yk
                Qk, Rk = np.linalg.qr(Yk_1-ratio*np.identity(len(Yk_1)), mode="complete")
                Yk = Rk@Qk + ratio*np.identity(len(Yk_1))
                Q_sum = Q_sum@Qk
                if (np.abs(Yk[-1,-2])) <= 1e-8:
                    flag = 0
            eigenvalues[n-1] = Yk[-1,-1] # convergence完成后储存对应的eigenvalue 
        # 获取eigenvector
        Q = np.identity(len(T))
        Q[:len(Q_sum), :len(Q_sum)] = Q_sum
        eigenvector_matrix = eigenvector_matrix@Q
        # print(eigenvector_matrix)
        #deflation
        Yk = Yk[:-1, :-1] #更新Yk为上一个matrix的左上角
    eigenvalues = np.sqrt(eigenvalues)
    VB = eigenvector_matrix
    idx = np.argsort(eigenvalues)
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]
    VB = VB[:,idx[::-1]]
    UB = B@VB@np.diag((1/np.abs(eigenvalues)))
    UA = U.T@UB
    VA = VB.T@V.T
    if diff >= 0:
        return UA,eigenvalues,VA.T
    else:
        return VA.T,eigenvalues,UA

B,U,V,diff = auto_run(A) #Note: please change your code! add variable DIFF
UA,sig,VA=QR_factorization(B,diff) #输出的V是竖着的!!
#print(UA@np.diag(sig)@VA.T) 

