import parallel_prefix_computation as ppc
import unittest


class Test_TestParallelPrefixComputation(unittest.TestCase):
    def test_2(self):
        exp = "GATE 2 OR 0 1\nOUTPUT 0 0\nOUTPUT 1 2\n"
        res = ppc.parallel_prefix_computation(2)
        self.assertEqual(res, exp)

    def test_3(self):
        exp = "GATE 3 OR 0 1\nGATE 4 OR 1 2\nGATE 5 OR 0 4\n\
OUTPUT 0 0\nOUTPUT 1 3\nOUTPUT 2 5\n"
        res = ppc.parallel_prefix_computation(3)
        print(res)
        self.assertEqual(res, exp)


if __name__ == '__main__':
    unittest.main()
