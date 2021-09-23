from math import log, ceil
from itertools import count


def parallel_prefix_computation(n: int):
    """
    Compute OR operation using parallel prefix

    Input: 
        n: int 
            the number of input elements   
    """
    res = ""
    height = ceil(log(n, 2))
    gate = count(start=n, step=1)
    for i in range(height):
        for j in range(2**i, n):
            curr_gate = next(gate)
            res += "GATE {} OR {} {}\n".format(curr_gate,
                                               j - 2**i, curr_gate - n + j)
    res += "OUTPUT 0 0\n"
    for i in range(1, n):
        res += "OUTPUT {} {}\n".format(i, n+2*(i-1))
    return res


if __name__ == "__main__":
    n = int(input())
    result = parallel_prefix_computation(n)
    print(result)
