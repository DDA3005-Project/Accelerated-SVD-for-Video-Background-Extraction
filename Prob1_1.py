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

#A=np.random.rand(512,512)
'''
A = np.array([
    [1.0,3.0,2.5],
    [2.3,4.5,2.5],
    [2.4,2.5,2.6]
])
'''
#B = np.array(A)
#print(B)
#result,U,V = Bidiagonalization(A)
#print(np.round(result,4))
#print(np.round(U.T@result@V.T,5))