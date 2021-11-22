import numpy as np


def strassen(A: np.array, B: np.array, p=2):
    """
    Performs matrix multiplication using Stassen algorithm. Matrices are from Mat_n(Z_p)
    @param: A: np.array -- 1st matrix
    @param: B: np.array -- 2nd matrix
    @param: p: int -- integer(not necessary prime) such that given matrices are considered under ring modulo p
    """
    n = A.shape[0]
    res = np.zeros_like(A, dtype=np.int32)
    if n == 1:
        res[0, 0] = A[0, 0] * B[0, 0]
        return res % p
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
    matrix = np.zeros((n, n), dtype=np.int32)
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


def make_shape_square_of_two(matr: np.array, n=-1):
    """
    Returns square matrix which shape is a square of two
    @param matr: np.array -- given matrix
    @param n: int -- given shape; if n=-1, then it should be computed
    """
    real_shape = matr.shape
    if n == -1:
        n = matr.shape[0] if matr.shape[0] > matr.shape[1] else matr.shape[1]
        if (n & (n - 1)) != 0:
            bit_length = n.bit_length()
            closest_power_of_two = int('1' + '0'*bit_length, 2)
            new_matr = np.zeros(
                (closest_power_of_two, closest_power_of_two), dtype=int)
            new_matr[:real_shape[0], :real_shape[1]] = matr
            return new_matr
        else:
            return matr
    else:
        if (n & (n - 1)) != 0:
            bit_length = n.bit_length()
            closest_power_of_two = int('1' + '0'*bit_length, 2)
            new_matr = np.zeros(
                (closest_power_of_two, closest_power_of_two), dtype=int)
            new_matr[:real_shape[0], :real_shape[1]] = matr
            return new_matr
        else:
            new_matr = np.zeros(
                (n, n), dtype=int)
            new_matr[:real_shape[0], :real_shape[1]] = matr
            return new_matr


def inverse(matr: np.array):
    """
    Returns inverted matrix; computations are via gauss elimination over Z_2
    @param: matr: np.array
    """
    n = matr.shape[0]
    identity = np.identity(n, dtype=np.int32)
    matr = np.concatenate((matr, identity), axis=1)
    for i in range(n-1):
        ind_max = np.argmax(matr[i, :])
        pot = matr[:, i].copy()
        matr[:, i], matr[:, ind_max] = matr[:, ind_max], pot
        for j in range(i+1, n):
            matr[j, i:] = (matr[j, i:] + (-matr[j, i]) * matr[i, i:]) % 2
    for i in range(n-1, 0, -1):
        for j in range(i-1, -1, -1):
            matr[j, i:] = (matr[j, i:] + (-matr[j, i]) * matr[i, i:]) % 2
    return matr[:, n:]


def inverse_triangular_mod2(matr: np.array):
    """
    Finds inverse matrix for upper triangular matrix over Z_2
    @param: matr: np.array -- given matrix
    """
    n = matr.shape[0]
    if n == 1 or n == 2:
        return matr
    else:
        n_half = n // 2
        b = matr[:n_half, :n_half]
        c = matr[:n_half, n_half:]
        d = matr[n_half:, n_half:]
        res = np.zeros_like(matr)
        b_inv = inverse_triangular_mod2(b)
        d_inv = inverse_triangular_mod2(d)
        res[:n_half, :n_half] = b_inv
        res[n_half:, n_half:] = d_inv
        res[:n_half, n_half:] = matrix_mult(matrix_mult(b_inv, c), d_inv)
    return res


def matrix_mult(a: np.array, b: np.array, p=2):
    """
    Computes matrix multiplication for given matrices a and b
    @param: a: np.array -- first matrix
    @param: b: np.array -- second matrix
    @param: p: int -- such integer that all calculations are under Z/(pZ)
    """
    shape_a = a.shape
    shape_b = b.shape
    n = np.max([a.shape, b.shape])
    a = make_shape_square_of_two(a, int(n))
    b = make_shape_square_of_two(b, int(n))
    multiplication = strassen(a, b, p=p)
    return multiplication[:shape_a[0], :shape_b[1]] % p


def lup_decomposition(matrix: np.array, n: int, m: int):
    """
    Performs LUP-decomposition of given matrix from Z^{n x n}_2
    @param: matrix: np.array -- given matrix from Z^{n x n}_2
    @param: n: int -- number of rows
    @param: m: int -- number of columns
    """
    if n == 1:
        l = np.ones((1, 1), dtype=np.int32)
        try:
            nonzero_indx = np.nonzero(matrix)[1][0]
        except:
            nonzero_indx = 0
        perm = np.arange(m)
        perm[0] = nonzero_indx
        perm[nonzero_indx] = 0
        p = np.identity(m, dtype=np.int32)[:, perm]
        u = matrix_mult(matrix, p)
        return l, u, p
    else:
        n_half = n // 2
        b, c = matrix[:n_half], matrix[n_half:]
        l_1, u_1, p_1 = lup_decomposition(b, n_half, m)
        d = matrix_mult(c, p_1.T)
        e, f = u_1[:, :n_half], d[:, :n_half]
        e_inv = inverse_triangular_mod2(e)
        fe_inv = matrix_mult(f, e_inv)
        g = (d - matrix_mult(fe_inv, u_1)) % 2
        g_prime = g[:, -(m - n_half):]
        l_2, u_2, p_2 = lup_decomposition(g_prime, n-n_half, m - n_half)
        p_3 = np.identity(m, dtype=np.int32)
        p_3[-(m-n_half):, -(m-n_half):] = p_2
        h = matrix_mult(u_1, p_3.T)
        l = np.zeros((n, n), dtype=np.int32)
        l[:n_half, :n_half] = l_1
        l[n_half:, :n_half] = fe_inv
        l[n_half:, n_half:] = l_2
        u = np.zeros((n, m), dtype=np.int32)
        u[:n_half] = h
        u[n_half:, -(m-n_half):] = u_2
        p = matrix_mult(p_3, p_1)
        return l, u, p


def main():
    matrix, n = read_matrix()
    l, u, p = lup_decomposition(matrix, n, n)
    print("Answer")
    print_array(l)
    print('----')
    print_array(u)
    print('----')
    print_array(p)


if __name__ == '__main__':
    main()
