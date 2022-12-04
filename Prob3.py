#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# This program is to transform A into bidiagonal form utilizing householder

def householder(b_r,length): # b is a vector that we want to only preserve b1, length is the longtitude of b_r
    beta = np.linalg.norm(b_r)
    v = np.array([0.0 for i in range(length)])
    if (b_r[1:]==0).all():
        return np.identity(length), b_r[0]
    v[0] = (np.abs(b_r[0])+beta)*np.sign(b_r[0])
    v[1:] = b_r[1:]
    H = np.identity(length)-(2/(np.inner(v,v)))*np.outer(v,v.T)
    #print(H@b_r)
    return H,-1*np.sign(b_r[0])*beta  #np.identity(length)-(2/(np.linalg.norm(b_r)**2))*np.outer(b_r,b_r)

def Bidiagonalization(A):
    row,col = np.shape(A)
    diff = row - col
    H_all_U = np.identity(row)
    H_all_V = np.identity(col)
    for i in range(col-2):
        H1 = np.identity(row)
        H2 = np.identity(col)
        H_U, entry_u = householder(A[i:,i],row-i) #gen U
        A[i,i] = entry_u
        A[i+1:,i] = 0
        A[i:,i+1:] = np.matmul(H_U,A[i:,i+1:]) #update A utilizing U  列+1
        H1[i:,i:] = H_U
        H_all_U = H1@H_all_U
        #print(np.round(A,4))
        H_V, entry_v = householder(A[i,i+1:],col-i-1) # gen V
        A[i,i+1] = entry_v
        A[i,i+2:] = 0
        A[i+1:,i+1:] = A[i+1:,i+1:]@H_V #行-1
        H2[i+1:,i+1:] = H_V
        H_all_V = H_all_V@H2
    if diff == 0:
        H1 = np.identity(row)
        H_U, entry_u = householder(A[row-2:,col-2],2)
        A[row-2,row-2] = entry_u
        A[row-1,row-2] = 0.0
        A[row-2:,row-1:] = H_U@A[row-2:,row-1:] #lie-1
        H1[row-2:,row-2:] = H_U
        H_all_U = H1@H_all_U
        return A,H_all_U,H_all_V
    else:
        H1 = np.identity(row)
        H_U, entry_u = householder(A[col-2:,col-2],diff+2)
        A[col-2,col-2] = entry_u
        A[col-1:,col-2] = 0.0
        #print(A[col-1:,col-1],H_U)
        A[col-2:,col-1:] = H_U@A[col-2:,col-1:] #lie-1
        H1[col-2:,col-2:] = H_U
        H_all_U = H1@H_all_U
        H1 = np.identity(row)
        H_U, entry_u = householder(A[col-1:,col-1],diff+1)
        A[col-1,col-1] = entry_u
        A[col:,col-1] = 0.0  
        H1[col-1:,col-1:] = H_U
        H_all_U = H1@H_all_U
        return A[:col,:col],H_all_U,H_all_V

def auto_run(A):
    row,col = np.shape(A)
    diff = row - col
    if diff < 0 :
        result, U, V = Bidiagonalization(A.T)
        return result.T, V.T, U.T
    else:
        result,U,V = Bidiagonalization(A)
        return result, U, V

'''
A = np.array([
    [1.0,3.0,2.5,43,43,65],
    [2.3,4.5,2.5,23,52,1],
    [2.4,2.5,2.6,52,7,2]
])

A = np.random.rand(10000,5)
result,U,V = auto_run(A)
#print(np.round(result,4))
print(np.round(result,4)[:5,:5])
#print(np.round(U.T@result@V.T,5)) #verification
'''


# In[2]:


import numpy as np
import time
def special_cholesky(R,length): # R is tridagonal and symmetric, use accelarated method
    d = R.diagonal(0)
    a = R.diagonal(1)
    f = [np.sqrt(d[0])]
    b = []
    for k in range(1,length):
        l = a[k-1]/f[k-1]
        b.append(l)
        f.append (np.sqrt(d[k]-l**2))
    return f,b
    
def iter_in_each_row(B,length,type):
    Q_sum = np.identity(length)
    while np.abs(B[length-2,length-1]) >= 1e-12: #accelerated convergence. could use:(np.abs(B.diagonal(1)) >= 1e-12).any()  
        Q,R = np.linalg.qr(B.T, mode='complete')
        Q_sum = Q_sum@Q
        if type == 'special':
            dia,super_dia = special_cholesky(R@R.T,length)
            B = np.diag(dia)
            B[:length-1,1:length] += np.diag(super_dia)
        else:
            B = np.linalg.cholesky(R@R.T).T #use common Cholesky decomposition
    return B[:length-1,:length-1], B[length-1,length-1],Q_sum  #B, Q_sum 
        
def QR_iteration(B,type=None):
    C = np.array(B) #copy of B
    length,wid = np.shape(B)
    e_val = []
    next = B
    vector = np.identity(length)
    for i in range(length-1):
        Q_store = np.identity(length)
        next,eigen_val,Q_i = iter_in_each_row(next,length-i,type)
        e_val.append(eigen_val)
        Q_store[:length-i,:length-i] = Q_i
        vector = vector@Q_store
    e_val.append(next[0,0])
    value = np.array(e_val)
    value = value[::-1]
    U_B = C@vector@np.diag((1/value)) #calculate decomposition of B 
    sig = np.diag(value)
    return U_B,sig,vector


# In[3]:


import numpy as np
import imageio as iio
from pathlib import Path
from PIL import Image
from moviepy.video.fx import blackwhite
from moviepy.editor import *
from matplotlib import image
from matplotlib import pyplot as plt

def imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(arr, mode=mode) 
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp]) 
    return np.array(imnew)

video_folder = Path("/Users/yedou/Desktop/test_videos/640_360/")
video_to_open = video_folder / "rooster_01.mp4"
clip = VideoFileClip(str(video_to_open))

# 提取frames
frames = []
gray_frames = []

for t in range(50,251,50):
    frames.append(clip.get_frame(t/60)) #第“t/60s的截图”
for i in range(len(frames)):
    resized_im = np.array(Image.fromarray(frames[i]).resize((135,75))) #压缩
    im = Image.fromarray(resized_im)
    im = im.convert('L') #convert to gray scale
    gray_frames.append(np.array(im))

print(np.shape(gray_frames[1]))

#将array转换为vector
for i in range(len(gray_frames)):
    dim = np.shape(gray_frames[i])[0]*np.shape(gray_frames[i])[1]
    gray_frames[i] = gray_frames[i].reshape(dim,1)

#结合成A矩阵
A = np.hstack(gray_frames) / 256
print(A)
print(np.shape(A))
print(np.linalg.matrix_rank(A.T)) #查看A是否满秩


# In[4]:


import matplotlib.pyplot as plt
len,num = np.shape(A)

# u1 = np.linalg.svd(A)[0][:,0]
# s = np.linalg.svd(A)[1][0]
# v1 = np.linalg.svd(A)[2][0,:]
# # print(np.linalg.svd(A)[0][:,0])
# # print()
# # print(np.linalg.svd(A)[2][0,:])
# B_bz = s*v1[0]*u1

result, U_t, V_t = Bidiagonalization(A)
U_t = U_t[:5,:]
V_t = V_t[:,:5]
U_r, sig_r, V_r = QR_iteration(result)
u = U_t.T@U_r[:,0]
v = V_t@V_r[:,0]
sig = sig_r[0,0]
print(u)
print(sig)
print(v)

B_vec = sig*v[0]*u
B = B_vec.reshape(75,135)
plt.imshow(B)
plt.show()

