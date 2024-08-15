import numpy as np

def l2norm(v):
    return np.sqrt(np.sum(np.square(v)))

def hessenberg(A, b, m):
    V = np.zeros((A.shape[0], m))
    H = np.zeros((m, m))
    V[:, 0] = b / l2norm(b)

    for i in range(m):
        v = np.dot(A, V[:, i])
        for j in range(i+1):
            H[j, i] = np.dot(V[:, j].T, v)
            v = v - H[j, i] * V[:, j]

        if i + 1 < m:
            H[i+1, i] = l2norm(v)
            if H[i+1, i] != 0:
                V[:, i+1] = v / H[i+1, i]
    
    return H

def _largest_l2norm_index(A):
    idx, val = 0, float("-inf")
    for i in range(A.shape[1]):
        n = l2norm(A[:, i]) 
        if n > val:
            idx = i
            val = n

    return idx

def qr_decomposition(A):
    Q = np.zeros(A.shape)
    A_p = np.copy(A)

    for i in range(A.shape[1]):
        # You could squentially go through the vectors in the matrix but getting the vector with
        # the largest l2norm minimises floating point error
        idx = _largest_l2norm_index(A_p)
        q = A_p[:, idx] / l2norm(A_p[:, idx])
        Q[:, i] = q

        for j in range(A.shape[1]):
            A_p[:, j] -= np.dot(q.T, A_p[:, j]) * q

    R = np.matmul(Q.T, A)

    return Q, R

def compute_eigenvalues(A, m=None, iterations=100):
    size = A.shape[0]
    if m == None:
        m = size

    # Just a random vector which forms the basis for the Krylov space
    b = np.random.uniform(-1, 1, (size))
    H = hessenberg(A, b, m)

    for _ in range(iterations):
        Q, R = qr_decomposition(H)
        H = np.matmul(R, Q)

    return H.diagonal()