import lup as l
import numpy as np
import unittest


class Test_TestLUPDecomposition(unittest.TestCase):
    def test_is_lower_triangular_1(self):
        matr = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        res = l.is_lower_triangular(matr)
        self.assertTrue(res)

    def test_is_lower_triangular_2(self):
        matr = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1]])
        res = l.is_lower_triangular(matr)
        self.assertFalse(res)

    def test_inverse_triangular_mod2_1(self):
        matr = np.identity(2)
        res = l.inverse_triangular_mod2(matr)
        np.testing.assert_array_equal(matr, res)

    def test_inverse_triangle_mod2_2(self):
        matr = np.array([[1, 0], [1, 1]])
        res = l.inverse_triangular_mod2(matr)
        np.testing.assert_array_equal(matr, res)

    def test_inverse_triangle_mod2_3(self):
        matr = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                        [1, 0, 1, 0], [0, 0, 0, 1]])
        res = l.inverse_triangular_mod2(matr)
        np.testing.assert_array_equal(matr, res)

    def test_inverse_triangle_mod2_4(self):
        matr = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ])
        res = l.inverse_triangular_mod2(matr)
        np.testing.assert_array_equal(
            (res @ matr) % 2, np.identity(5, dtype=np.int32))

    def test_inverse_triangle_mod2_5(self):
        matr = np.array([[1, 0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 1, 1],
                         [0, 1, 0, 1, 1, 0],
                         [0, 0, 0, 1, 1, 1],
                         [1, 1, 1, 0, 0, 1]])
        matr = np.tril(matr)
        res = l.inverse_triangular_mod2(matr)
        exp = np.linalg.inv(matr)
        print(exp)
        print(res)
        np.testing.assert_array_equal(
            (res @ matr) % 2, np.identity(6, dtype=np.int32))

    def test_inverse_triangle_mod2_6(self):
        matr = np.array([[1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                         [0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
                         [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                         [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
                         [0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]])
        matr = np.tril(matr)

        res = l.inverse_triangular_mod2(matr)
        np.testing.assert_array_equal(
            (res @ matr) % 2, np.identity(10, dtype=np.int32))

    def test_matrix_mult_1(self):
        matr1 = np.array([[1, 0], [0, 1]])
        matr2 = np.array([[1, 1], [0, 1]])
        res = l.matrix_mult(matr1, matr2)
        np.testing.assert_array_equal(matr2, res)

    def test_matrix_mult_2(self):
        matr1 = np.array([[1, 1], [0, 1]])
        matr2 = np.array([[1, 1], [0, 1]])
        exp = np.identity(2, dtype=np.int32)
        res = l.matrix_mult(matr1, matr2)
        np.testing.assert_array_equal(exp, res)

    def test_matrix_mult_3(self):
        matr1 = np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]])
        matr2 = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
        exp = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        res = l.matrix_mult(matr1, matr2)
        np.testing.assert_array_equal(exp, res)

    def test_matrix_mult_4(self):
        matr1 = np.array([[1, 1, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
        matr2 = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
        exp = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1]])
        res = l.matrix_mult(matr1, matr2)
        np.testing.assert_array_equal(exp, res)

    def test_matrix_mult_5(self):
        matr1 = np.random.randint(2, size=(2, 10))
        matr2 = np.random.randint(2, size=(10, 2))
        exp = (matr1 @ matr2) % 2
        res = l.matrix_mult(matr1, matr2)
        np.testing.assert_array_equal(exp, res)

    def test_matrix_mult_6(self):
        matr1 = np.random.randint(2, size=(10, 10))
        matr2 = np.random.randint(2, size=(10, 10))
        exp = (matr1 @ matr2) % 2
        res = l.matrix_mult(matr1, matr2)
        np.testing.assert_array_equal(exp, res)

    def test_matrix_mult_7(self):
        matr1 = np.random.randint(2, size=(100, 100))
        matr2 = np.random.randint(2, size=(100, 100))
        exp = (matr1 @ matr2) % 2
        res = l.matrix_mult(matr1, matr2)
        np.testing.assert_array_equal(exp, res)

    def test_lup_1(self):
        matr = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1]])
        res_l, res_u, res_p = l.lup_decomposition(matr, 3, 3)
        res = (res_l @ res_u @ res_p) % 2
        np.testing.assert_array_equal(matr, res)

    def test_lup_2(self):
        matr = np.array([[0, 0, 1, 1], [0, 1, 1, 0],
                        [1, 0, 0, 1], [1, 1, 0, 1]])
        res_l, res_u, res_p = l.lup_decomposition(
            matr, matr.shape[0], matr.shape[1])
        res = (res_l @ res_u @ res_p) % 2
        np.testing.assert_array_equal(matr, res)

    def test_lup_3(self):
        matr = np.array([[1, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 1, 1, 0],
                         [0, 1, 0, 1, 1]])
        print("\n{}".format(matr))
        res_l, res_u, res_p = l.lup_decomposition(
            matr, matr.shape[0], matr.shape[1])
        res = (res_l @ res_u @ res_p) % 2
        np.testing.assert_array_equal(matr, res)

    def test_lup_4(self):
        matr = np.array([[0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1],
                         [0, 1, 0, 1, 1, 0],
                         [0, 0, 0, 1, 0, 1],
                         [1, 1, 1, 0, 0, 1]])
        res_l, res_u, res_p = l.lup_decomposition(
            matr, matr.shape[0], matr.shape[1])
        res = (res_l @ res_u @ res_p) % 2
        print("\n{}".format(res))
        np.testing.assert_array_equal(matr, res)

    def test_lup_5(self):
        matr = np.array([[1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                         [0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
                         [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                         [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
                         [0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]])
        res_l, res_u, res_p = l.lup_decomposition(
            matr, matr.shape[0], matr.shape[1])
        res = (res_l @ res_u @ res_p) % 2
        print("\n{}".format(res))
        np.testing.assert_array_equal(matr, res)


if __name__ == '__main__':
    unittest.main()
