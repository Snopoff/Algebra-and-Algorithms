from itertools import count
from typing import Dict, Tuple, List, Set
from math import log2


def parse_function(gate: int, gates_dict: Dict[int, Tuple[int, int]], n: int) -> List[int]:
    """
    Parses gate parameters and returns elementary conjunction

    @param: gate: int -- given gate
    @param: gates_dict: Dict[int, Tuple[int, int]] -- dict of gates parameters
    @param: n: int -- the number of variables
    """
    if gate < 2 * n:
        return [gate]
    gate_params = gates_dict[gate]
    res = []
    for v in gate_params:
        if v < 2 * n:
            res.append(v)
        else:
            res.extend(parse_function(v, gates_dict, n))
    return res


def intersection_of_functions(gate1: int, gate2: int, gates_dict: Dict[int, Tuple[int, int]], n: int, negation=True) -> Set[int]:
    """
    Returns the intersection of two functions

    @param: gate1: int -- first gate
    @param: gate2: int -- second gate
    @param: gates_dict: Dict[int, Tuple[int, int]] -- dict of gates parameters
    @param: n: int -- the number of variables
    @param: negation: bool -- include negation or not
    """
    function1 = set(parse_function(gate1, gates_dict, n))
    print(function1)
    function2 = set(parse_function(gate2, gates_dict, n))
    if negation:
        function1 = function1.union(set([(v+n) % (2*n) for v in function1]))
    # function2 = function2.union((set([(v+n) % 2*n for v in function2])))
    print('f1 = {}\nf2 = {}\nintersection = {}\n'.format(
        function1, function2, function1 & function2))
    return function1 & function2


def multipole(n: int) -> str:
    """
    Performs caluclations of all binary functions of n variables

    @param: n: int -- number of binary variables
    """
    gate = count(start=n, step=1)
    gates_dict = {}
    func_dict = {}
    last_gate = n-1
    last_gates = [last_gate]
    res = ""
    for i in range(n):  # only NOT
        last_gate = next(gate)
        res += "GATE {} NOT {}\n".format(last_gate, i)
    last_gates.append(last_gate)
    outputs = list(range(2*n))
    for p in range(1, n):  # elementary conjunctions
        if p == 1:
            for i in range(n):
                for j in range(i+1, n):
                    last_gate = next(gate)
                    res += "GATE {} AND {} {}\n".format(last_gate, i, j)
                    gates_dict[last_gate] = (i, j)
                    func_dict[last_gate] = set([i, j])
                    last_gate = next(gate)
                    res += "GATE {} AND {} {}\n".format(last_gate, i+n, j)
                    gates_dict[last_gate] = (i+n, j)
                    func_dict[last_gate] = set([i+n, j])
                    last_gate = next(gate)
                    res += "GATE {} AND {} {}\n".format(last_gate, i, j+n)
                    gates_dict[last_gate] = (i, j+n)
                    func_dict[last_gate] = set([i, j+n])
                    last_gate = next(gate)
                    res += "GATE {} AND {} {}\n".format(last_gate, i+n, j+n)
                    gates_dict[last_gate] = (i+n, j+n)
                    func_dict[last_gate] = set([i+n, j+n])
            last_gates.append(last_gate)
        else:
            for i in range(last_gates[-2] + 1, last_gates[-1] + 1):
                for j in range(2*n):
                    intersection = intersection_of_functions(
                        i, j, gates_dict, n)
                    a = parse_function(i, gates_dict, n)
                    a.append(j)
                    if not intersection and set(a) not in func_dict.values():
                        print(a)
                        last_gate = next(gate)
                        if p == n-1:
                            res += "GATE {} AND {} {}\n".format(
                                last_gate, i, j)
                            outputs.append(last_gate)
                        gates_dict[last_gate] = (i, j)
                        func_dict[last_gate] = set(a)
            last_gates.append(last_gate)
    all_elementary_conjunctions = list(
        range(last_gates[-2] + 1, last_gates[-1] + 1))
    prev_functions = all_elementary_conjunctions
    for i in range(2, 2**n):  # Disjunctions
        if i == 2:
            for j, func in enumerate(prev_functions[:-1]):
                for conj in all_elementary_conjunctions[j+1:]:
                    intersection = func_dict[func] & func_dict[conj]
                    if len(intersection) == n-1:
                        continue
                    last_gate = next(gate)
                    res += "GATE {} OR {} {}\n".format(
                        last_gate, func, conj)
                    gates_dict[last_gate] = (func, conj)
                    func_dict[last_gate] = set([func, conj])
                    outputs.append(last_gate)
            last_gates.append(last_gate)
            prev_functions = list(
                range(last_gates[-2] + 1, last_gates[-1] + 1))
        else:
            this_loop_functions = []
            for f in range(len(prev_functions)):
                func = prev_functions[f]
                components = func_dict[func]
                for j in range(len(all_elementary_conjunctions)):
                    conj = all_elementary_conjunctions[j]
                    curr_function = func_dict[func].union(
                        set([conj]))
                    if conj not in components and curr_function not in this_loop_functions:
                        if i in [2**p for p in range(1, n)]:
                            curr_parsed_functions = [
                                func_dict[function] for function in curr_function]
                            curr_intersection = curr_parsed_functions[0].intersection(
                                *curr_parsed_functions[1:])
                            if len(curr_intersection) == n - int(log2(i)) and len(curr_intersection) == 1:
                                continue
                        last_gate = next(gate)
                        res += "GATE {} OR {} {}\n".format(
                            last_gate, func, conj)
                        gates_dict[last_gate] = (func, conj)
                        func_dict[last_gate] = curr_function
                        this_loop_functions.append(func_dict[last_gate])
                        outputs.append(last_gate)
            last_gates.append(last_gate)
            prev_functions = range(last_gates[-2] + 1, last_gates[-1] + 1)
    const_gate_0 = next(gate)
    outputs.append(const_gate_0)
    res += "GATE {} AND {} {}\n".format(const_gate_0, 0, n)
    const_gate_1 = next(gate)
    outputs.append(const_gate_1)
    res += "GATE {} OR {} {}\n".format(const_gate_1, 0, n)
    for i in range(2**(2**n)):
        res += "OUTPUT {} {}\n".format(i, i)
    # for i, o in enumerate(outputs):
    #    res += "OUTPUT {} {}\n".format(i, o)
    return res


if __name__ == "__main__":
    n = int(input())
    result = multipole(n)
    print(result)
