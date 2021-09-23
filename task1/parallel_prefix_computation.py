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
    if n > 0:
        height = ceil(log(n, 2))
        gate = count(start=n, step=1)
        curr = list(range(n))
        outputs = [0]
        for i in range(height):
            threshold = 2**(i+1)
            prev = curr.copy()
            for j in range(2**i, n):
                real_width_index = j - 2**i
                curr[j] = next(gate)
                res += "GATE {} OR {} {}\n".format(curr[j],
                                                   prev[real_width_index], prev[j])
                if j < threshold:
                    outputs.append(curr[j])
        for ind, out in enumerate(outputs):
            res += "OUTPUT {} {}\n".format(ind, out)
    return res


if __name__ == "__main__":
    n = int(input())
    result = parallel_prefix_computation(n)
    print(result)
