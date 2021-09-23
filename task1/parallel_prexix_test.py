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

    def test_5(self):
        exp = "GATE 5 OR 0 1\nGATE 6 OR 1 2\nGATE 7 OR 2 3\n\
GATE 8 OR 3 4\nGATE 9 OR 0 6\nGATE 10 OR 5 7\nGATE 11 OR 6 8\n\
GATE 12 OR 0 11\nOUTPUT 0 0\nOUTPUT 1 5\nOUTPUT 2 9\nOUTPUT 3 10\nOUTPUT 4 12\n"
        res = ppc.parallel_prefix_computation(5)
        print(res)
        self.assertEqual(res, exp)


if __name__ == '__main__':
    unittest.main()
