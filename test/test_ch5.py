import math
import os
import sys
sys.path.append(os.pardir)

import pytest
import sympy as sp

import numana.chapter5 as ch5

class TestNewtonCotes(object):
    def setup(self):
        self.x = sp.Symbol('x')

    def outputNewtonCotes(self, f: sp.Function, a: float, b: float):
        nc = ch5.NewtonCotes(f)
        result = nc(a, b)
        print("The function is \033\13331mf(x) = {}\033\1330m and the interval is \033\13331m[{}, {}]\033\1330m".format(nc.symbol_f, a, b))
        for i in range(7):
            print("Using \033\13331m{0}\033\1330m order Newton-Cotes formula, the result is: \033\13334m[{1}]\033\1330m".format(i + 1, result[i]))
        print("Using symbolic integration formula, the result is: \033\13334m[{}]\033\1330m".format(result[-1]))
  
    def testNewtonCotes(self):
        self.outputNewtonCotes(sp.log(self.x), 1.0, 2.0)
        self.outputNewtonCotes(self.x ** 2,    0.0, 1.0)
        self.outputNewtonCotes(sp.cos(self.x), 0.0, math.pi / 2.0)
        self.outputNewtonCotes(sp.exp(self.x), 0.0, 1.0)

class TestCompositeNewtonCotes(object):
    def setup(self):
        self.x = sp.Symbol('x')

    def outputCompositeNewtonCotes(self, f: sp.Function, a: float, b: float, m: int):
        nc = ch5.CompositeNewtonCotes(f)
        result = nc(a, b, m)
        print("The function is \033\13331mf(x) = {}\033\1330m and the interval is \033\13331m[{}, {}]\033\1330m, we divide the it into \033\13331m[{}]\033\1330m intervals".format(nc.symbol_f, a, b, m))
        print("Using \033\13331mComposite Trapezoid Rule\033\1330m, the result is: \033\13334m[{}]\033\1330m".format(result[0]))
        print("Using \033\13331mComposite Midpoint  Rule\033\1330m, the result is: \033\13334m[{}]\033\1330m".format(result[1]))
        print("Using \033\13331mComposite Simpson's Rule\033\1330m, the result is: \033\13334m[{}]\033\1330m".format(result[2]))
        print("Using symbolic integration    , the result is: \033\13334m[{}]\033\1330m".format(result[-1]))
  
    def testCompositeNewtonCotes(self):
        self.outputCompositeNewtonCotes(sp.log(self.x), 1.0, 2.0, 4)
        self.outputCompositeNewtonCotes(sp.sin(self.x) / self.x, 0.0, 1.0, 4)
        # (1), (2), (3)
        self.outputCompositeNewtonCotes(self.x ** 2, 0.0, 1.0, 1)
        self.outputCompositeNewtonCotes(self.x ** 2, 0.0, 1.0, 2)
        self.outputCompositeNewtonCotes(self.x ** 2, 0.0, 1.0, 4)
        self.outputCompositeNewtonCotes(sp.cos(self.x), 0.0, math.pi / 2.0, 1)
        self.outputCompositeNewtonCotes(sp.cos(self.x), 0.0, math.pi / 2.0, 2)
        self.outputCompositeNewtonCotes(sp.cos(self.x), 0.0, math.pi / 2.0, 4)
        self.outputCompositeNewtonCotes(sp.exp(self.x), 0.0, 1.0, 1)
        self.outputCompositeNewtonCotes(sp.exp(self.x), 0.0, 1.0, 2)
        self.outputCompositeNewtonCotes(sp.exp(self.x), 0.0, 1.0, 4)
        # (4)
        self.outputCompositeNewtonCotes(self.x * sp.exp(self.x), 0.0, 1.0, 1)
        self.outputCompositeNewtonCotes(self.x * sp.exp(self.x), 0.0, 1.0, 2)
        self.outputCompositeNewtonCotes(self.x * sp.exp(self.x), 0.0, 1.0, 4)
        # (5)
        self.outputCompositeNewtonCotes(1 / (1 + self.x ** 2), 0.0, 1.0, 1)
        self.outputCompositeNewtonCotes(1 / (1 + self.x ** 2), 0.0, 1.0, 2)
        self.outputCompositeNewtonCotes(1 / (1 + self.x ** 2), 0.0, 1.0, 4)
        # (6)
        self.outputCompositeNewtonCotes(self.x * sp.cos(self.x), 0.0, math.pi, 1)
        self.outputCompositeNewtonCotes(self.x * sp.cos(self.x), 0.0, math.pi, 2)
        self.outputCompositeNewtonCotes(self.x * sp.cos(self.x), 0.0, math.pi, 4)

        # (7)
        self.outputCompositeNewtonCotes(sp.sin(self.x) / self.x, 0.0, math.pi / 2.0, 16)
        self.outputCompositeNewtonCotes(sp.sin(self.x) / self.x, 0.0, math.pi / 2.0, 32)
        self.outputCompositeNewtonCotes((sp.exp(self.x) - 1) / sp.sin(self.x), 0.0, math.pi / 2.0, 16)
        self.outputCompositeNewtonCotes((sp.exp(self.x) - 1) / sp.sin(self.x), 0.0, math.pi / 2.0, 32)
        self.outputCompositeNewtonCotes(sp.atan(self.x) / self.x, 0.0, 1 / 2.0, 16)
        self.outputCompositeNewtonCotes(sp.atan(self.x) / self.x, 0.0, 1 / 2.0, 32)

class TestRomberg(object):
    def setup(self):
        self.x = sp.Symbol('x')

    def outputRomberg(self, f: sp.Function, a: float, b: float, m: int):
        romberg = ch5.Romberg(f)
        result = romberg(a, b, m)
        print("The function is \033\13331mf(x) = {}\033\1330m and the interval is \033\13331m[{}, {}]\033\1330m, line of Romberg table is \033\13331m[{}]\033\1330m.".format(romberg.symbol_f, a, b, m))
        print("Using \033\13331mRomberg integration\033\1330m , the result is: \033\13334m[{}]\033\1330m".format(result[0]))
        print("Using symbolic integration, the result is: \033\13334m[{}]\033\1330m".format(result[-1]))

    def testRomberg(self):
        self.outputRomberg(sp.log(self.x), 1.0, 2.0, 4)
        # (1)
        self.outputRomberg(self.x ** 2, 0.0, 1.0, 3)
        self.outputRomberg(sp.cos(self.x), 0.0, math.pi / 2.0, 3)
        self.outputRomberg(sp.exp(self.x), 0.0, 1.0, 3)
        # (2)
        self.outputRomberg(self.x * sp.exp(self.x), 0.0, 1.0, 3)
        self.outputRomberg(1.0 / (1.0 + self.x ** 2), 0.0, 1.0, 3)
        self.outputRomberg(self.x * sp.cos(self.x), 0.0, math.pi, 3)
        # (1)
        self.outputRomberg(self.x / sp.sqrt(self.x ** 2 + 9), 0.0, 4.0, 5)
        self.outputRomberg(self.x ** 3 / sp.sqrt(self.x ** 2 + 1), 0.0, 1.0, 5)
        self.outputRomberg(self.x * sp.exp(self.x), 0.0, 1.0, 5)
        self.outputRomberg(self.x ** 2 * sp.log(self.x), 1.0, 3.0, 5)
        self.outputRomberg(self.x ** 2 * sp.sin(self.x), 0.0, math.pi, 5)
        self.outputRomberg(self.x ** 3 / sp.sqrt(self.x ** 4 - 1), 2.0, 3.0, 5)
        self.outputRomberg(1 / sp.sqrt(self.x ** 2 + 4), 0.0, 2 * math.sqrt(3), 5)
        self.outputRomberg(self.x / sp.sqrt(self.x ** 4 + 1), 0.0, 1.0, 5)

class TestGaussLegendre(object):
    def setup(self):
        self.x = sp.Symbol('x')

    def outputGaussLegendre(self, f: sp.Function, a: float, b: float):
        gl = ch5.GaussLegendre(f)
        result = gl(a, b)
        print("The function is \033\13331mf(x) = {}\033\1330m and the interval is \033\13331m[{}, {}]\033\1330m.".format(gl.symbol_f, a, b))
        for i in range(3):
            print("Using \033\13331m{0}\033\1330m order Gauss-Legendre formula, the result is: \033\13334m[{1}]\033\1330m".format(i + 2, result[i]))
        print("Using symbolic integration, the result is: \033\13334m[{}]\033\1330m".format(result[-1]))

    def testGaussLegendre(self):
        self.outputGaussLegendre(sp.exp(-self.x ** 2 / 2), -1.0, 1.0)
        self.outputGaussLegendre(sp.log(self.x), 1.0, 2.0)
        # (1) (2) (3)
        self.outputGaussLegendre((self.x ** 3 + 2 * self.x), -1.0, 1.0)
        self.outputGaussLegendre(self.x ** 4, -1.0, 1.0)
        self.outputGaussLegendre(sp.exp(self.x), -1.0, 1.0)
        self.outputGaussLegendre(sp.cos(sp.pi * self.x), -1.0, 1.0)
        # (4) (5)
        self.outputGaussLegendre(self.x / sp.sqrt(self.x ** 2 + 9), 0.0, 4.0)
        self.outputGaussLegendre(self.x ** 3 / sp.sqrt(self.x ** 2 + 1), 0.0, 1.0)
        self.outputGaussLegendre(self.x * sp.exp(self.x), 0.0, 1.0)
        self.outputGaussLegendre(self.x ** 2 * sp.log(self.x), 1.0, 3.0)
        # (6)
        self.outputGaussLegendre((self.x ** 3 + 2 * self.x), 0.0, 1.0)
        self.outputGaussLegendre(sp.log(self.x), 1.0, 4.0)
        self.outputGaussLegendre(self.x ** 5, -1.0, 2.0)
        self.outputGaussLegendre(sp.exp(-self.x ** 2 / 2), -3.0, 3.0)

if __name__ == "__main__":
    # pytest.main(["-s", "test_ch5.py::TestNewtonCotes::testNewtonCotes"])
    # pytest.main(["-s", "test_ch5.py::TestCompositeNewtonCotes::testCompositeNewtonCotes"])
    # pytest.main(["-s", "test_ch5.py::TestRomberg::testRomberg"])
    pytest.main(["-s", "test_ch5.py::TestGaussLegendre::testGaussLegendre"])
