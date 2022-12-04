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
    row,col=np.shape(A)
    diff = row-col #find shape
    H_all_U = np.identity(row)
    H_all_V = np.identity(col)
    if diff >= -1 :
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
            #print(np.round(A,4))
        if diff == -1:
            return A,H_all_U,H_all_V
        elif diff == 0:
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
            #print(A)
            #print(A[col-1:,col-1],H_U)
            A[col-2:,col-1:] = H_U@A[col-2:,col-1:] #lie-1
            H1[col-2:,col-2:] = H_U
            H_all_U = H1@H_all_U
            H1 = np.identity(row)
            H_U, entry_u = householder(A[col-1:,col-1],diff+1)
            A[col-1,col-1] = entry_u
            A[col:,col-1] = 0.0
            A[col-1:,col] = H_U@A[col-1:,col] #lie-1
            H1[col-1:,col-1:] = H_U
            H_all_U = H1@H_all_U
            return A,H_all_U,H_all_V
    else:
        for i in range(row-1):
            H1 = np.identity(row)
            H2 = np.identity(col)
            H_U, entry_u = householder(A[i:,i],row-i) #gen U
            A[i,i] = entry_u
            A[i+1:,i] = 0
            A[i:,i+1:] = np.matmul(H_U,A[i:,i+1:]) #update A utilizing U  列-1
            #print(np.round(A,4))
            H1[i:,i:] = H_U
            H_all_U = H1@H_all_U
            H_V, entry_v = householder(A[i,i+1:],col-i-1) # gen V
            A[i,i+1] = entry_v
            A[i,i+2:] = 0
            A[i+1:,i+1:] = A[i+1:,i+1:]@H_V #行-1
            #print(np.round(A,4))
            H2[i+1:,i+1:] = H_V
            H_all_V = H_all_V@H2
        H2 = np.identity(col)
        H_V, entry_v = householder(A[row-1,row:],2)
        A[row-1,row:] = A[row-1,row:]@H_V
        H2[col-2:,col-2:] = H_V
        H_all_V = H_all_V@H2
        return A,H_all_U,H_all_V #return result, U,V(no need to do the transpose)

# 第二问,求∑
def WilkinsonShift(X):
    sigma = (X[0,0]-X[1,1])/2
    # 如果这么计算是正确的，回头还要考虑加上 (abs(sigma) + np.sqrt((sigma**2)+(X[0,1]**2))=0 的情况
    if sigma > 0:
        ratio = X[1,1] - (X[0,1]**2)/(abs(sigma) + np.sqrt((sigma**2)+(X[0,1]**2)))
    else:
        ratio = X[1,1] + (X[0,1]**2)/(abs(sigma) + np.sqrt((sigma**2)+(X[0,1]**2)))
    return ratio

def QR_factorization(B,U,V):
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
                if (np.abs(Yk[-1,-2])) <= 1e-14:
                    flag = 0
            eigenvalues[n-1] = Yk[-1,-1] # convergence完成后储存对应的eigenvalue 
        # 获取eigenvector
        Q = np.identity(len(T))
        Q[:len(Q_sum), :len(Q_sum)] = Q_sum
        eigenvector_matrix = eigenvector_matrix@Q
        # print(eigenvector_matrix)
        #deflation
        Yk = Yk[:-1, :-1] #更新Yk为上一个matrix的左上角
    eigenvalues = np.sqrt(np.abs(eigenvalues))
    VB = eigenvector_matrix
    idx = np.argsort(eigenvalues)
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]
    VB = VB[:,idx[::-1]]
    UB = B@VB@np.diag((1/eigenvalues))
    UA = U.T@UB
#     VA = VB.T@V.T
    VA = V@VB
    return UA,eigenvalues,VA

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
    sig_A = value
    #print(value)
    return U_A,sig_A,V_A