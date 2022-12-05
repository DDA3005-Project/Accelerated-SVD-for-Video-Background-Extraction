import numpy as np
from Prob1_1 import auto_run
import time
#This program is to implement a alternative QR iteration

A = np.array([
    [1.0,3.0,2.5,43,43,65],
    [2.3,4.5,2.5,23,52,1],
    [2.4,2.5,2.6,52,7,2]
])

#A = np.random.rand(216,216)
#print(np.linalg.matrix_rank(A))
#print(A)
#test_package = np.round(np.linalg.svd(A)[2],4)
#print(test_package)
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
            B = np.diag(dia,0)
            B += np.diag(super_dia,1)
        #print(R)
        #print(R@R.T)
        else:
            B = np.linalg.cholesky(R@R.T).T #use common Cholesky decomposition
    return B[:length-1,:length-1], B[length-1,length-1],Q_sum  #B, Q_sum 
        
def QR_iteration(B,U,V,diff,type=None):
    C = np.array(B) #copy of B
    length,wid = np.shape(B)
    e_val = []
    next = B
    vector = np.identity(length)
    #'''
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
    U_A = U.T@U_B
    V_A = V@vector 
    sig_A = np.diag(value)
    if diff >= 0:
        return U_A,sig_A,V_A
    else:
        return V_A,sig_A,U_A

#Implementation
B,U,V,diff = auto_run(A)
#print(np.linalg.svd(B)[2])
#print(np.round(np.linalg.svd(B)[0].T,1) == np.round(np.linalg.inv(np.linalg.svd(B)[0]),1))

timestart = time.time()
U_A,sig_A,V_A = QR_iteration(B,U,V,diff,'special') # eigenval and vector of BTB
timeend = time.time()
print(timeend-timestart)

timestart = time.time()
U_A,sig_A,V_A = QR_iteration(B,U,V,diff) # 输出的V是竖着的!!
#print(U_A@sig_A@V_A.T)
timeend = time.time()
print(timeend-timestart)