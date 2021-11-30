import spectral_cut as cut
import numpy as np
import unittest


class Test_TestSpectralCut(unittest.TestCase):
    def test_create_laplacian_1(self):
        edges = np.array([[0, 1], [0, 2]])
        res = cut.create_laplacian(edges)
        exp = np.array([
            [2, -1, -1],
            [-1, 1, 0],
            [-1, 0, 1]
        ])
        np.testing.assert_array_equal(exp, res)

    def test_create_laplacian_2(self):
        edges = np.array(
            [[5, 3], [3, 4], [3, 2], [2, 1], [4, 1], [4, 0], [1, 0]])
        res = cut.create_laplacian(edges)
        exp = np.array([
            [2, -1, 0, 0, -1, 0],
            [-1, 3, -1, 0, -1, 0],
            [0, -1, 2, -1, 0, 0],
            [0, 0, -1, 3, -1, -1],
            [-1, -1, 0, -1, 3, 0],
            [0, 0, 0, -1, 0, 1]
        ])
        np.testing.assert_array_equal(exp, res)


if __name__ == '__main__':
    unittest.main()
