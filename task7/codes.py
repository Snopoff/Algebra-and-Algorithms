import numpy as np


def create_hadamard_matrix(n: int):
    """
    Creates nxn hadamard_matrix over Z_{n-1}
    @param: n: int -- hadamard matrix shape
    """
    p = n-1
    multiplicative_group_underlying_set = np.arange(1, p)
    quadratic_residue = np.unique(multiplicative_group_underlying_set**2 % p)

    def legendre_symbol(x):
        if x == 0:
            return 0
        if x in quadratic_residue:
            return 1
        return -1

    legendre_symbol = np.vectorize(legendre_symbol)
    hadamard = np.fromfunction(
        lambda i, j: legendre_symbol((i-j) % p), (p, p), dtype=int)

    np.fill_diagonal(hadamard, -1)
    hadamard = np.pad(hadamard, ((1, 0), (1, 0)), constant_values=1)
    return hadamard


def code_construction(n: int):
    """
    Constructs Bose Shrikhande codes of 2 type using Hadamard matrix
    @param: n: int -- Hadamard matrix shape = length of codes
    """
    hadamard = create_hadamard_matrix(n)
    hadamard[hadamard == -1] = 0
    codes = np.zeros((2*n, n), dtype=int)
    for i in range(n):
        codes[i] = hadamard[i, :]
        codes[n+i] = np.logical_xor(codes[i], 1).astype(np.int64)
    return codes


def main():
    n = int(input())
    codes = code_construction(n)
    for code in codes:
        print("".join(list(map(str, code))))


if __name__ == '__main__':
    main()
