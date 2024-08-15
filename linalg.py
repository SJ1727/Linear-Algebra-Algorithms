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

def qr_decomposition(A):
    Q = np.zeros(A.shape)
    A_p = np.copy(A)

    for i in range(A.shape[1]):
        q = A_p[:, i] / l2norm(A_p[:, i])
        Q[:, i] = q

        for j in range(A.shape[1]):
            A_p[:, j] -= np.dot(A_p[:, j].T, q) * q

    R = np.matmul(Q.T, A)

    return Q, R

def compute_eigenvalues(A, m=None, iterations=100):
    size = A.shape[0]
    if m == None:
        m = size

    # Just a random vector which forms the basis for the Krylov space
    b = np.random.uniform(-1, 1, (size))
    H = hessenberg(A, b, m)
    mu = 0

    for iteration in range(iterations):
        mu = H[-1, -1]
        Q, R = qr_decomposition(H - np.eye(H.shape[0]) * mu)
        H = R @ Q + np.eye(H.shape[0]) * mu
        
        if l2norm(np.tril(H)) < 1e-10:
            break

    return H.diagonal()