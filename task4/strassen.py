import numpy as np


def fast_exponent(A: np.array, pow: int, p=9):
    """
    Performs fast exponententiation by squaring
    @param: A: np.array -- given matrix
    @param: pow: int -- given exponent
    @param: p: int -- given integer s.t. matrices are over ring modulo p
    """
    n = A.shape[0]
    if (n & (n - 1)) != 0:  # check if n is a power of 2
        bit_length = n.bit_length()
        closest_power_of_two = int('1' + '0'*bit_length, 2)
        newA = np.identity((closest_power_of_two), dtype=int)
        newA[:n, :n] = A
        A = newA
    if pow == 1:
        return A
    if pow == 2:
        return strassen(A, A, p=p)
    else:
        if pow % 2 != 0:
            return strassen(A, fast_exponent(A, pow - 1, p))
        else:
            A_half = fast_exponent(A, pow // 2, p)
            return strassen(A_half, A_half)


def strassen(A: np.array, B: np.array, p=9):
    """
    Performs matrix multiplication using Stassen algorithm. Matrices are from Mat_n(Z_p)
    @param: A: np.array -- 1st matrix
    @param: B: np.array -- 2nd matrix
    @param: p: int -- integer(not necessary prime) such that given matrices are considered under ring modulo p
    """
    res = np.zeros_like(A, dtype=int)
    n = A.shape[0]
    if n == 2:
        res[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
        res[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
        res[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
        res[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
        return res % p
    else:
        k = n // 2
        A11, A12, A21, A22 = A[:k, :k], A[:k, k:], A[k:, :k], A[k:, k:]
        B11, B12, B21, B22 = B[:k, :k], B[:k, k:], B[k:, :k], B[k:, k:]
        M1 = strassen(A11 + A22, B11 + B22, p=p)
        M2 = strassen(A21 + A22, B11, p=p)
        M3 = strassen(A11, B12 - B22, p=p)
        M4 = strassen(A22, B21 - B11, p=p)
        M5 = strassen(A11 + A12, B22, p=p)
        M6 = strassen(A21 - A11, B11 + B12, p=p)
        M7 = strassen(A12 - A22, B21 + B22, p=p)
        res[:k, :k] = M1 + M4 - M5 + M7
        res[:k, k:] = M3 + M5
        res[k:, :k] = M2 + M4
        res[k:, k:] = M1 - M2 + M3 + M6
        return res % p


def read_matrix():
    """
    Reads matrix from standard input()
    """
    first_row = input().strip().split(" ")
    n = len(first_row)
    matrix = np.zeros((n, n), dtype=int)
    matrix[0, :] = np.array(list(map(int, first_row)))
    for i in range(1, n):
        row = input().strip().split(" ")
        matrix[i, :] = np.array(list(map(int, row)))
    return matrix, n


def print_array(A: np.array):
    """
    Pretty print an array
    @param: A: np.array: given array
    """
    (n, k) = A.shape
    for x in range(n):
        for y in range(k):
            print(A[x, y], end=' ')
        print()


def main():
    matrix, n = read_matrix()
    exp = fast_exponent(matrix, n)[:n, :n]
    print_array(exp)


if __name__ == '__main__':
    main()
