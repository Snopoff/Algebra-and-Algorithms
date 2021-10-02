from itertools import count
from typing import Iterable
from random import randrange
import checker


def compressor(values: list):
    """
    3-2' compressor

    @param: values: list  -- list of all 3n values for 3 n-bit numbers
    """
    res = ""
    trippleN = len(values)
    n = int(trippleN / 3)
    gate = count(start=trippleN, step=1)
    outputs_x = []
    outputs_y = []
    for i in range(n):
        a, b, c = i, i+n, i+2*n
        or_gate = next(gate)
        res += "GATE {} OR {} {}\n".format(or_gate, a, b)
        and_gate = next(gate)
        res += "GATE {} AND {} {}\n".format(and_gate, a, b)
        not_c_gate = next(gate)
        res += "GATE {} NOT {}\n".format(not_c_gate, c)
        just_gate = next(gate)
        res += "GATE {} NOT {}\n".format(just_gate, and_gate)
        xor_a_b_gate = next(gate)
        res += "GATE {} AND {} {}\n".format(xor_a_b_gate, or_gate, just_gate)
        just_gate = next(gate)
        res += "GATE {} AND {} {}\n".format(just_gate, or_gate, c)
        maj_gate = next(gate)
        res += "GATE {} OR {} {}\n".format(maj_gate, and_gate, just_gate)
        outputs_y.append(maj_gate)
        just_gate = next(gate)
        res += "GATE {} NOT {}\n".format(just_gate, or_gate)
        just_gate = next(gate)
        res += "GATE {} OR {} {}\n".format(just_gate, and_gate, just_gate-1)
        just_gate = next(gate)
        res += "GATE {} AND {} {}\n".format(just_gate, just_gate-1, c)
        just_gate = next(gate)
        res += "GATE {} AND {} {}\n".format(just_gate,
                                            xor_a_b_gate, not_c_gate)
        xor_a_b_c_gate = next(gate)
        res += "GATE {} OR {} {}\n".format(xor_a_b_c_gate,
                                           just_gate, just_gate-1)
        outputs_x.append(xor_a_b_c_gate)

    const_gate = next(gate)
    res += "GATE {} AND {} {}\n".format(const_gate, c, not_c_gate)

    output = count(start=0, step=1)

    for i in range(len(outputs_x)):
        res += "OUTPUT {} {}\n".format(next(output), outputs_x[i])

    res += "OUTPUT {} {}\n".format(next(output), const_gate)
    res += "OUTPUT {} {}\n".format(next(output), const_gate)

    for i in range(len(outputs_y)):
        if i == len(outputs_x)-1:
            res += "OUTPUT {} {}".format(next(output), outputs_y[i])
        else:
            res += "OUTPUT {} {}\n".format(next(output), outputs_y[i])

    return res


def randbinary(n, force_leading_one=False):
    r = randrange(0, 2**n) | (2**n)
    r = list(map(int, bin(r)[3:]))
    if force_leading_one:
        r[-1] = 1
    return r


if __name__ == "__main__":
    n = int(input())
    a = randbinary(n)
    b = randbinary(n)
    c = randbinary(n)
    s = compressor(a+b+c)
    print(s)
    circuit = checker.read_circuit(s)
    b = checker.check_3_2_trick(circuit, 1)
    print(b)
