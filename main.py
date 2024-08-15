import numpy as np
from linalg import compute_eigenvalues

def main():
    SIZE = 10

    # Needs to be equal to or smaller that the size of the matrix
    # This gives the number of approxiamted eignevalues, the smaller the faster but less accurate
    # This gives the resulting size of the similar hessenberg matrix
    M = None

    # Number of iterations used to approxiamte the eigenvalues during the QR step
    # More iterations will take more time but be more accurate
    ITERATIONS = 50

    # The algorithm works with any matrix but for the sake of guaranteeing real eignenvalues 
    # I create a random positive definite matrix
    A = np.random.uniform(-10, 10, (SIZE, SIZE))
    A = np.matmul(A, A.T)

    eignevalues = compute_eigenvalues(A, m=M, iterations=ITERATIONS)
    sorted_eigenvalues = np.flip(np.sort(eignevalues))
    # Built in function is going to be obviously better and faster
    truth = np.flip(np.sort(np.linalg.eigvals(A)))[:M]

    print(np.round(sorted_eigenvalues, 5))
    print(np.round(truth, 5))
    
    print(f"MAE: {np.sum(np.abs((sorted_eigenvalues - truth))) / len(sorted_eigenvalues)}")

if __name__ == "__main__":
    main()