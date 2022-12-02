import numpy as np
from Prob1_1 import Bidiagonalization
#This program is to implement a alternative QR iteration

A=np.array([
    [10.0,-52.3,-3.2],
    [-10.0,6.0,-2.0],
    [2.1,2.0,-4.0]
])
#print(np.linalg.svd(A))

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
    
def iter_in_each_row(B,length):
    Q_sum = np.identity(length)
    while (np.abs(B.diagonal(1)) >= 1e-12).all(): #accelerated convergence. could use:np.abs(B[0,1]) >= 1e-12
        Q,R = np.linalg.qr(B.T, mode='complete')
        Q_sum = Q_sum@Q
        dia,super_dia = special_cholesky(R@R.T,length)
        for i in range(length):
            B[i,i] = dia[i]
        for i in range(length-1):
            B[i,i+1] = super_dia[i]
    return B[:length-1,:length-1], B[length-1,length-1],Q_sum
        
def QR_iteration(B,U,V):
    C = np.array(B) #copy of B
    length,wid = np.shape(B)
    e_val = []
    next = B
    vector = np.identity(length)
    for i in range(length-1):
        Q_store = np.identity(length)
        next,eigen_val,Q_i = iter_in_each_row(next,length-i)
        e_val.append(eigen_val**2)
        Q_store[:length-i,:length-i] = Q_i
        vector = vector@Q_store
    e_val.append(next[0,0]**2)
    value = np.sqrt(np.array(e_val[::-1]))
    U_B = C@vector@np.diag((1/value)) #calculate decomposition of B
    #print(U_B@np.diag(np.sqrt(value))@vector.T)
    U_A = U.T@U_B  # need to improve **
    V_A = V@vector # need to imporve **
    sig_A = np.diag(value)
    return U_A,sig_A,V_A

#Implementation
B,U,V = Bidiagonalization(A)
#print(B)
U_A,sig_A,V_A = QR_iteration(B,U,V) # eigenval and vector of BTB
#print(vector)
#print(vector@np.diag(value)@vector.T)
#print(U_B@np.diag(value)@vector.T)

#print(U_A)
#print(sig_A)
#print(V_A)
print(U_A@sig_A@V_A.T) #verification