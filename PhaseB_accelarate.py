import numpy as np
from Prob1_1 import auto_run
import time
#This program is to implement a alternative QR iteration
'''
A = np.array([
    [1.0,3.0,2.5,43,43,65],
    [2.3,4.5,2.5,23,52,1],
    [2.4,2.5,2.6,52,7,2]
])
'''
A = np.random.rand(216,216)
#print(np.round(np.linalg.svd(A)[2].T,4))
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
    
def iter_in_each_row(eig_val,Q_all,total_length,B,type,total_start,total_end):
    length = len(B)
    Q_sum = np.identity(length)
    idx = []
    if length == 1:
        eig_val.append(B[0][0])
        return []
    elif length == 0:
        return []
    while len(idx) == 0: #accelerated convergence. could use:(np.abs(B.diagonal(1)) >= 1e-12).any()  
        Q,R = np.linalg.qr(B.T, mode='complete')
        Q_sum = Q_sum@Q
        if type == 'special':
            dia,super_dia = special_cholesky(R@R.T,length)
            B = np.diag(dia,0)
            B += np.diag(super_dia,1)
        else:
            B = np.linalg.cholesky(R@R.T).T #use common Cholesky 
        idx = np.argwhere(np.abs(B.diagonal(1)) < 1e-14)
    idx = idx[0,0]
    eig_val.append(B[idx,idx])
    Q_all[:,total_start:total_end+1] = Q_all[:,total_start:total_end+1]@Q_sum
    # Q_now = np.identity(len(Q_sum))
    #Q_now[total_start:total_end+1,total_start:total_end+1] += Q_sum
    iter_in_each_row(eig_val,Q_all,total_length,B[:idx,:idx],None,total_start,total_start+idx-1) #up
    iter_in_each_row(eig_val,Q_all,total_length,B[idx+1:,idx+1:],None,total_start+idx+1,total_end) #down 
    # if len(Q_up) != 0 and len(Q_down) != 0:
    #     Q_now[:len(Q_up)-1,:len(Q_up)-1] = Q_up
    #     Q_now[len(Q_sum)-len(Q_down):,len(Q_sum)-len(Q_down):] = Q_down
    # elif len(Q_up) == 0 and len(Q_down) != 0:
    #     Q_now[len(Q_sum)-len(Q_down):,len(Q_sum)-len(Q_down):] = Q_down
    # elif len(Q_up) != 0 and len(Q_down) == 0:
    #     Q_now[:len(Q_up)-1,:len(Q_up)-1] = Q_up
    # return Q_sum@Q_now

def QR_iteration(B,U,V,diff,type=None):
    eig_val = []
    Q_all = np.identity(len(A))
    C = np.array(B) #copy of B
    iter_in_each_row(eig_val,Q_all,len(C),B,None,0,len(C)-1)
    vector = Q_all
    eig_val = np.array(eig_val)
    eig_val.sort()
    value = eig_val[::-1]
    U_B = C@vector@np.diag((1/value)) #calculate decomposition of B
    U_A = U.T@U_B
    V_A = V@vector 
    sig_A = np.diag(value)
    if diff >= 0:
        return U_A,sig_A,V_A
    else:
        return V_A,sig_A,U_A

B,U,V,diff = auto_run(A)
timestart = time.time()
U_A,sig_A,V_A = QR_iteration(B,U,V,diff,'special') # eigenval and vector of BTB
timeend = time.time()
print(timeend-timestart)

timestart = time.time()
U_A,sig_A,V_A = QR_iteration(B,U,V,diff) # 输出的V是竖着的!!
timeend = time.time()
print(timeend-timestart)