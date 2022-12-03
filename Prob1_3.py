import numpy as np
from Prob1_1 import Bidiagonalization
import time
#This program is to implement a alternative QR iteration
'''
A=np.array([
    [10.0,-52.3,-3.2],
    [-10.0,6.0,-2.0],
    [2.1,2.0,-4.0]
])
'''
A = np.random.rand(216,216)
#print(np.linalg.matrix_rank(A))
#print(A)
#print(np.linalg.svd(A)[1])

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
        #print(R)
        #print(R@R.T)
        else:
            B = np.linalg.cholesky(R@R.T).T #use common Cholesky decomposition
    return B[:length-1,:length-1], B[length-1,length-1],Q_sum  #B, Q_sum 
        
def QR_iteration(B,U,V,type=None):
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
    #'''
    #e_val,vector = iter_in_each_row(B,length)
    value = np.array(e_val)
    #value = e_val.diagonal(0)
    value = value[::-1]
    #print(value)
    U_B = C@vector@np.diag((1/value)) #calculate decomposition of B
    #print(np.round(U_B.T,1) == np.round(np.linalg.inv(U_B),1))
    #print(U_B@np.diag(np.sqrt(value))@vector.T)
    U_A = U.T@U_B
    V_A = V@vector 
    sig_A = np.diag(value)
    #print(value)
    return U_A,sig_A,V_A

#Implementation
B,U,V = Bidiagonalization(A)
#print(np.linalg.svd(B)[1])
#print(np.round(np.linalg.svd(B)[0].T,1) == np.round(np.linalg.inv(np.linalg.svd(B)[0]),1))
timestart = time.time()
U_A,sig_A,V_A = QR_iteration(B,U,V,'special') # eigenval and vector of BTB
timeend = time.time()
print(timeend-timestart)
timestart = time.time()
U_A,sig_A,V_A = QR_iteration(B,U,V) # eigenval and vector of BTB
timeend = time.time()
print(timeend-timestart)
#print(np.round(U_A.T,1) == np.round(np.linalg.inv(U_A),1))
#print(vector)
#print(vector@np.diag(value)@vector.T)
#print(U_B@np.diag(value)@vector.T)

#print(U_A)
#print(sig_A)
#print(V_A)
#print(U_A@sig_A@V_A.T) #verification