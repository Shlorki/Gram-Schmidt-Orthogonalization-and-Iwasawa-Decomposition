import numpy as np
import numpy.linalg as lin

def GSO(A):
    n = A[0].size
    B = np.zeros((n,n))
    
    B[0] = A[0]
    
    for i in range(1,n):
        B[i] = A[i]
        for j in range(i):
            B[i] += np.dot(A[i],B[j])/np.dot(B[j],B[j])*B[j]
    return B

def Iwasawa(A):
    n = A[0].size
    B = GSO(A)
    N = lin.inv(np.transpose(lin.solve(np.transpose(A),np.transpose(B))))
    
    D = np.zeros((n,n))
    for k in range(n):
        D[k][k] = np.sqrt(np.dot(B[k],B[k]))
    
    K = lin.solve(D,B)
    
    return N,D,K