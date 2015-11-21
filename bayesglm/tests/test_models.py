import unittest


# Here's our "unit".
def is_odd(n):
    return n % 2 == 1


# Here's our "unit tests".
class IsOddTests(unittest.TestCase):

    def testOne(self):
        self.assertTrue(is_odd(1))

    def testTwo(self):
        self.assertFalse(is_odd(2))


if __name__ == '__main__':
    unittest.main()