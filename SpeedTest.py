import numpy as np
import time
import sys

class NumericalPythonTipsTest():

    def __init__(self):
        num = 10000000
        self.x = np.linspace(0, 1, num=num)
        self.y = np.random.rand(num)*100
        self.dydx = np.empty(num)

        self.n = 25000
        self.A = np.zeros((self.n, self.n))
        self.a1 = np.random.rand(self.n) * 100
        self.a2 = np.random.rand(self.n-1) * 100
        self.a3 = np.random.rand(self.n-1) * 100

    """
    中央差分(for)
    """
    def test_for_derivative(self):
        dydx = self.dydx
        x = self.x
        y = self.y
        tick = time.time()
        for i in range(1, len(x)-1):
            dydx[i] = (y[i+1] - y[i-1])/(x[i+1] - x[i-1])
        tock = time.time()
        print('%s: %.06f[s]' % (sys._getframe().f_code.co_name, tock-tick))

    """
    中央差分(numpy)
    """
    def test_numpy_derivative(self):
        dydx = self.dydx
        x = self.x
        y = self.y
        tick = time.time()
        dxdy = (y[1:] - y[:-1])/(x[1:] - x[:-1])
        tock = time.time()
        print('%s: %.06f[s]' % (sys._getframe().f_code.co_name, tock-tick))

    """
    三重対角行列(for)
    """
    def test_for_matrix(self):
        A = self.A
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        n = self.n

        tick = time.time()
        for i in range(0, n-1):
            A[i, i] = a1[i]
            A[i+1, i] = a2[i]
            A[i, i+1] = a3[i]
        A[n-1,n-1] = a1[n-1]
        tock = time.time()
        print('%s: %.06f[s]' % (sys._getframe().f_code.co_name, tock-tick))

    """
    三重対角行列(diag)
    """
    def test_diag_martirx(self):
        n = self.n
        A = self.A
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3

        tick = time.time()
        A = np.diag(a1) + np.diag(a2, k=-1) + np.diag(a3, k=1)
        tock = time.time()
        print('%s: %.06f[s]' % (sys._getframe().f_code.co_name, tock-tick))

    """
    三重対角行列(numpy)
    """
    def test_numpy_martirx(self):
        n = self.n
        A = self.A
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3

        tick = time.time()
        i = np.arange(0, n-1)
        A[i, i] = a1[i]
        A[i+1, i] = a2[i]
        A[i, i+1] = a3[i]
        A[n-1,n-1] = a1[n-1]
        tock = time.time()
        print('%s: %.06f[s]' % (sys._getframe().f_code.co_name, tock-tick))

    """
    for文
    """
    def test_for_loop(self):
        s = 0
        tick = time.time()
        for i in range(1, 100000001):
            s += i
        print(s)
        tock = time.time()
        print('%s: %.06f[s]' % (sys._getframe().f_code.co_name, tock-tick))

    """
    sum
    """
    def test_sum(self):
        s = 0
        tick = time.time()
        s = sum(range(1, 100000001))
        print(s)
        tock = time.time()
        print('%s: %.06f[s]' % (sys._getframe().f_code.co_name, tock-tick))

    """
    numpy sum
    """
    def test_numpy_sum(self):
        s = 0
        tick = time.time()
        a = np.arange(1, 100000001, dtype=np.int64)
        print(a.sum())
        tock = time.time()
        print('%s: %.06f[s]' % (sys._getframe().f_code.co_name, tock-tick))

if __name__ == '__main__':
    test = NumericalPythonTipsTest()
    #test.test_for_derivative()
    #test.test_numpy_derivative()
    #test.test_for_matrix()
    #test.test_diag_martirx()
    #test.test_numpy_martirx()
    test.test_for_loop()
    test.test_sum()
    test.test_numpy_sum()