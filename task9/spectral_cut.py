import numpy as np


def create_laplacian(edges: np.array):
    """
    Constructs the Laplacian matrix for given graph
    @param: edges: np.array -- given set of edges for given graph
    """
    num_vertices = np.max(edges)+1
    _, counts = np.unique(edges, return_counts=True)
    degree_matrix = np.diag(counts)
    adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
    for edge in edges:
        i, j = edge
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    return degree_matrix - adjacency_matrix


def num_of_bipartite_edges(vertices1: np.array, vertices2: np.array, edges: np.array):
    """
    Returns the number of edges between two disjoint parts of the graph
    @param: vertices1: np.array -- set of vertices in the first part of the graph
    @param: vertices2: np.array -- set of vertices in the second part of the graph
    @param: edges: np.array -- set of edges of the graph
    """
    num = 0
    for edge in edges:
        v1, v2 = edge
        if v1 in vertices1 and v2 in vertices2:
            num += 1
        elif v2 in vertices1 and v1 in vertices2:
            num += 1
    return num


def spectral_cut(laplacian: np.array, edges: np.array):
    """
    Spectral cut algorithm realization
    @param: laplacian: np.array -- given Laplacian matrix of the graph
    @param: edges: np.array -- set of edges of the graph
    """
    num_vertices = laplacian.shape[0]
    vertices = np.linspace(0, num_vertices-1, num_vertices, dtype=int)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fisher_number = sorted(eigenvalues)[1]
    vector = eigenvectors[:, np.where(eigenvalues == fisher_number)[0][0]]
    sorted_vertices = sorted(vertices, key=lambda x: -vector[x])
    cost = np.infty
    best_cut_set = None
    for i in range(1, num_vertices):
        if i < num_vertices // 2:
            cut_set = sorted_vertices[:i]
            complement_set = sorted_vertices[i:]
        else:
            cut_set = sorted_vertices[i:]
            complement_set = sorted_vertices[:i]
        num_of_edges = num_of_bipartite_edges(cut_set, complement_set, edges)
        val = num_of_edges * num_vertices / \
            (len(cut_set) * len(complement_set))
        if val < cost or val == cost and cut_set < best_cut_set:
            cost = val
            best_cut_set = cut_set
    return best_cut_set


def main():
    num_edges = int(input())
    edges = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        edges[i, :] = np.array(list(map(int, input().split(" "))))
    laplacian = create_laplacian(edges)
    best_cut_set = spectral_cut(laplacian, edges)
    print(" ".join(list(map(str, best_cut_set))))


if __name__ == '__main__':
    main()
