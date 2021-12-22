import numpy as np


def fast_exponent(A: np.array, pow: int):
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
        return strassen(A, A)
    else:
        if pow % 2 != 0:
            return strassen(A, fast_exponent(A, pow - 1))
        else:
            A_half = fast_exponent(A, pow // 2)
            return strassen(A_half, A_half)


def strassen(A: np.array, B: np.array):
    """
    Performs matrix multiplication using Stassen algorithm. Matrices are from Mat_n(Z_p)
    @param: A: np.array -- 1st matrix
    @param: B: np.array -- 2nd matrix
    """
    res = np.zeros_like(A, dtype=int)
    n = A.shape[0]
    if n == 2:
        res[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
        res[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
        res[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
        res[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
        return res
    else:
        k = n // 2
        A11, A12, A21, A22 = A[:k, :k], A[:k, k:], A[k:, :k], A[k:, k:]
        B11, B12, B21, B22 = B[:k, :k], B[:k, k:], B[k:, :k], B[k:, k:]
        M1 = strassen(A11 + A22, B11 + B22)
        M2 = strassen(A21 + A22, B11)
        M3 = strassen(A11, B12 - B22)
        M4 = strassen(A22, B21 - B11)
        M5 = strassen(A11 + A12, B22)
        M6 = strassen(A21 - A11, B11 + B12)
        M7 = strassen(A12 - A22, B21 + B22)
        res[:k, :k] = M1 + M4 - M5 + M7
        res[:k, k:] = M3 + M5
        res[k:, :k] = M2 + M4
        res[k:, k:] = M1 - M2 + M3 + M6
        return res


def fast_matrix_mult(A: np.array, B: np.array):
    """
    Wrapper for strassen for matrices A, B s.t. A.shape = B.shape
    @param: A: np.array -- 1st matrix
    @param: B: np.array -- 2nd matrix
    """
    n = A.shape[0]
    if (n & (n - 1)) != 0:  # check if n is a power of 2
        bit_length = n.bit_length()
        closest_power_of_two = int('1' + '0'*bit_length, 2)
        newA = np.identity((closest_power_of_two), dtype=int)
        newB = np.identity((closest_power_of_two), dtype=int)
        newA[:n, :n] = A
        newB[:n, :n] = B
        mult = strassen(newA, newB)
        return mult[:n, :n]
    else:
        return strassen(A, B)


def create_adj_matrix(edges: np.array):
    """
    Creates adjacency matrix for given graph
    @param: edges: np.array -- edges of given graph
    """
    num_of_vertices = edges.max()
    adj = np.zeros((num_of_vertices+1, num_of_vertices+1), dtype=int)
    for edge in edges:
        i, j = edge
        adj[i, j] += 1
        adj[j, i] += 1
    return adj


def seidel(adj: np.array):
    """
    Finds the shortest paths between every pair of vertices in graph using Seidel algorithm.
    @param: adj: np.array -- adjacency matrix of given graph
    """
    n = adj.shape[0]
    log2n = int(np.ceil(np.log2(n)))
    reachabilities = [None]*log2n
    reachabilities[0] = adj
    for i in range(1, log2n):
        reachabilities[i] = fast_matrix_mult(
            reachabilities[i-1], reachabilities[i-1])
        reachabilities[i] = np.logical_or(
            reachabilities[i], reachabilities[i-1]).astype(int)
        np.fill_diagonal(reachabilities[i], 0)
    distances = [None]*log2n
    distances[-1] = reachabilities[-1]
    for x in range(log2n-2, -1, -1):
        prod = fast_matrix_mult(distances[x+1], reachabilities[x])
        dist = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                deg_j = np.sum(reachabilities[x][:, j])
                if prod[i, j] < deg_j * distances[x+1][i, j]:
                    dist[i, j] = 2 * distances[x+1][i, j] - 1
                else:
                    dist[i, j] = 2 * distances[x+1][i, j]
        distances[x] = dist
    return distances[0]


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


def bfs(edges: np.array, start_node: int):
    """
    Performs breadth-first search for given graph represented as an array of edges
    @param: edges: np.array -- representation of given graph
    @param: start_node: int -- start node of search
    """
    n = edges.shape[0]+1
    mark = [0]*n
    mark[start_node] = 1
    queue = [start_node]
    dist = [0]*n
    while queue != []:
        front = queue.pop(0)
        for edge in edges:
            if edge[0] == front:
                if mark[edge[1]] != 1:
                    dist[edge[1]] = dist[edge[0]] + 1
                    mark[edge[1]] = 1
                    queue.append(edge[1])
            if edge[1] == front:
                if mark[edge[0]] != 1:
                    dist[edge[0]] = dist[edge[1]] + 1
                    mark[edge[0]] = 1
                    queue.append(edge[0])
    return np.array(dist, dtype=int)


def hitting(adj: np.array, edges):
    """
    Finds the shortest paths between every pair of vertices in graph using Hitting set algorithm.
    @param: adj: np.array -- adjacency matrix of given graph
    """
    n = adj.shape[0]
    dist = np.copy(adj)
    k = np.floor(n ** (0.095)).astype(int)
    hit_size = np.ceil(n / k * (2 * np.log(n) + np.log(100))).astype(int)
    hit_set = [None] * hit_size
    for i in range(hit_size):
        hit_set[i] = np.random.randint(0, n)
    hit_set = list(set(hit_set))
    others = [vert for vert in range(0, n) if vert not in hit_set]
    reachabilities = [None] * k
    for i in range(k):
        reachabilities[i] = fast_exponent(adj, i+1)[:n, :n]
    for i in range(1, k):
        only_i = (
            i+1) * np.logical_and(np.logical_not(reachabilities[i-1]), reachabilities[i]).astype(int)
        dist += only_i

    for h in hit_set:
        h_bfs = bfs(edges, h)
        for k in range(n):
            dist[h, k] = h_bfs[k]

    for i in others:
        for j in others:
            dist[i, j] = np.infty
            for h in hit_set:
                d = dist[i, h] + dist[h, j]
                if d < dist[i, j]:
                    dist[i, j] = d

    return dist


def main():
    edges = []
    while True:
        try:
            s = input()
            if s == "":
                break
            edges.append(list(map(int, s.split(" "))))
        except EOFError:
            break
    edges = np.array(edges, dtype=int)
    adj = create_adj_matrix(edges)
    shortest_paths = hitting(adj, edges)
    unique, counts = np.unique(
        np.triu(shortest_paths, k=1), return_counts=True)
    res = np.asarray((unique[1:], counts[1:])).T
    print_array(res)


if __name__ == '__main__':
    main()
