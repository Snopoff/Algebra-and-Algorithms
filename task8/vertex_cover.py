import numpy as np
from scipy.optimize import linprog


def intlinprog(c: np.array, A_ub=None, b_ub=None, bounds=None, options=None):
    """
    Integer linear programming: 
    minimize a linear objective function subject to linear equality and inequality constraints over Z
    @param: c: np.array -- The coefficients of the linear objective function to be minimized.
    @param: A_ub=None -- The inequality constraint matrix.
    @param: b_ub=None -- The inequality constraint vector.
    @param: bounds=None -- The sequence of bounds for the solution.
    @param: options=None -- different options for scipy.optimize.linprog.
    """
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=options)
    print(res)
    return np.rint(res.x).astype(bool)


def main():
    num_vertices = int(input())
    weights = np.zeros(num_vertices, dtype=int)
    for i in range(num_vertices):
        weights[i] = int(input())
    num_edges = int(input())
    edges = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        edges[i, :] = np.array(list(map(int, input().split(" "))))
    constraint_matrix = np.zeros((num_edges, num_vertices))
    for i in range(num_edges):
        constraint_matrix[i] = np.zeros(num_vertices, dtype=int)
        constraint_matrix[i][edges[i]] = -1
    constraint_rhs = -1 * np.ones(num_edges, dtype=int)
    bounds = (0, 1)
    options = {"sparse": True}
    res = intlinprog(weights, constraint_matrix,
                     constraint_rhs, bounds, options)
    vertex_cover = np.arange(num_vertices)[res]
    print(" ".join(list(map(str, vertex_cover))))


if __name__ == '__main__':
    main()
