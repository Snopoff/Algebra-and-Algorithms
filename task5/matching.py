from typing import List, Tuple
import numpy as np
from random import randint

PRIME_CONST = 27644437


def fast_exponent(x: np.int64, n: np.int64, mod=PRIME_CONST):
    """
    Performs fast exponent x^n in Z/(modZ)

    @param: x: np.int64 -- integer to exponent
    @param: n: np.int64 -- exponent
    @param: mod=PRIME_CONST -- integer s.t. we work in Z/(modZ)
    """
    if n == 1:
        return x
    if n == 2:
        return (x * x) % mod
    else:
        if n % 2 != 0:
            return (x * fast_exponent(x, n - 1, mod)) % mod
        else:
            x_new = fast_exponent(x, n // 2, mod)
            return (x_new * x_new) % mod


def gauss_elimination(matr: np.array, n_vertices: int, mod=PRIME_CONST):
    """
    Performs forward gauss elimination in Z/(modZ)

    @param: matr: np.array -- given square matrix
    @param: n_vertices:int -- number of vertices of a graph
    @param: mod=PRIME_CONST -- integer s.t. we work in Z/(modZ)
    """
    for i in range(n_vertices-1):
        ind_max = np.argmax(matr[i, :])
        pot = matr[:, i].copy()
        matr[:, i], matr[:, ind_max] = matr[:, ind_max], pot
        matr[i, :] = (fast_exponent(matr[i, i], (mod-2), mod)
                      * matr[i, :]) % mod
        for j in range(i+1, n_vertices):
            matr[j, i:] = (matr[j, i:] + (-matr[j, i]) * matr[i, i:]) % mod
    return matr


def create_random_Edmonds(pairings: List[Tuple[int, int]], n_vertices: int, mod=PRIME_CONST):
    """
    Creates Edmonds matrix and fulfil it with random integers

    @param: pairings: List[Tuple[int,int]] -- list of pairs
    @param: n_vertices: int -- number of vertices of a graph
    @param: mod=PRIME_CONST -- integer s.t. we work in Z/(modZ)
    """
    matr = np.zeros((n_vertices, n_vertices), dtype=np.int64)
    for pair in pairings:
        matr[pair] = randint(1, mod)
    return matr


def main():
    n = int(input())
    pairings = [None] * n
    for i in range(n):
        pairings[i] = tuple(map(int, input().strip().split(" ")))
    n_vertices = max(max(pairings)) + 1
    matr = create_random_Edmonds(pairings, n_vertices)
    print(matr)
    eliminated_matr = gauss_elimination(matr, n_vertices)
    diagonal = np.diagonal(eliminated_matr)
    determinant = np.prod(diagonal)
    print(eliminated_matr)
    if determinant == 0:
        print("no")
        print('real determinant = {}'.format(np.linalg.det(matr)))
        return 0
    print("yes")


if __name__ == '__main__':
    main()
