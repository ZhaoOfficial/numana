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

    def output_newton_cotes(self, f: sp.Function, a: float, b: float):
        nc = ch5.NewtonCotes(f)
        result = nc(a, b)
        print("The function is \033\13331mf(x) = {}\033\1330m and the interval is \033\13331m[{}, {}]\033\1330m".format(nc.symbol_f, a, b))
        for i in range(7):
            print("Using \033\13331m{0}\033\1330m order Newton-Cotes formula, the result is: \033\13334m[{1}]\033\1330m".format(i + 1, result[i]))
        print("Using symbolic integration formula, the result is: \033\13334m[{}]\033\1330m".format(result[-1]))
  
    def test_newton_cotes(self):
        self.output_newton_cotes(sp.log(self.x), 1.0, 2.0)
        self.output_newton_cotes(self.x ** 2,    0.0, 1.0)
        self.output_newton_cotes(sp.cos(self.x), 0.0, math.pi / 2.0)
        self.output_newton_cotes(sp.exp(self.x), 0.0, 1.0)

class TestCompositeNewtonCotes(object):
    def setup(self):
        self.x = sp.Symbol('x')

    def output_newton_cotes(self, f: sp.Function, a: float, b: float, m: int):
        nc = ch5.CompositeNewtonCotes(f)
        result = nc(a, b, m)
        print("The function is \033\13331mf(x) = {}\033\1330m and the interval is \033\13331m[{}, {}]\033\1330m, we divide the it into \033\13331m[{}]\033\1330m intervals".format(nc.symbol_f, a, b, m))
        print("Using \033\13331mComposite Trapezoid Rule\033\1330m, the result is: \033\13334m[{}]\033\1330m".format(result[0]))
        print("Using \033\13331mComposite Midpoint  Rule\033\1330m, the result is: \033\13334m[{}]\033\1330m".format(result[1]))
        print("Using \033\13331mComposite Simpson's Rule\033\1330m, the result is: \033\13334m[{}]\033\1330m".format(result[2]))
        print("Using symbolic integration    , the result is: \033\13334m[{}]\033\1330m".format(result[-1]))
  
    def test_composite_newton_cotes(self):
        self.output_newton_cotes(sp.log(self.x), 1.0, 2.0, 4)
        self.output_newton_cotes(sp.sin(self.x) / self.x, 0.0, 1.0, 4)
        # (1), (2), (3)
        self.output_newton_cotes(self.x ** 2, 0.0, 1.0, 1)
        self.output_newton_cotes(self.x ** 2, 0.0, 1.0, 2)
        self.output_newton_cotes(self.x ** 2, 0.0, 1.0, 4)
        self.output_newton_cotes(sp.cos(self.x), 0.0, math.pi / 2.0, 1)
        self.output_newton_cotes(sp.cos(self.x), 0.0, math.pi / 2.0, 2)
        self.output_newton_cotes(sp.cos(self.x), 0.0, math.pi / 2.0, 4)
        self.output_newton_cotes(sp.exp(self.x), 0.0, 1.0, 1)
        self.output_newton_cotes(sp.exp(self.x), 0.0, 1.0, 2)
        self.output_newton_cotes(sp.exp(self.x), 0.0, 1.0, 4)
        # (4)
        self.output_newton_cotes(self.x * sp.exp(self.x), 0.0, 1.0, 1)
        self.output_newton_cotes(self.x * sp.exp(self.x), 0.0, 1.0, 2)
        self.output_newton_cotes(self.x * sp.exp(self.x), 0.0, 1.0, 4)
        # (5)
        self.output_newton_cotes(1 / (1 + self.x ** 2), 0.0, 1.0, 1)
        self.output_newton_cotes(1 / (1 + self.x ** 2), 0.0, 1.0, 2)
        self.output_newton_cotes(1 / (1 + self.x ** 2), 0.0, 1.0, 4)
        # (6)
        self.output_newton_cotes(self.x * sp.cos(self.x), 0.0, math.pi, 1)
        self.output_newton_cotes(self.x * sp.cos(self.x), 0.0, math.pi, 2)
        self.output_newton_cotes(self.x * sp.cos(self.x), 0.0, math.pi, 4)

        # (7)
        self.output_newton_cotes(sp.sin(self.x) / self.x, 0.0, math.pi / 2.0, 16)
        self.output_newton_cotes(sp.sin(self.x) / self.x, 0.0, math.pi / 2.0, 32)
        self.output_newton_cotes((sp.exp(self.x) - 1) / sp.sin(self.x), 0.0, math.pi / 2.0, 16)
        self.output_newton_cotes((sp.exp(self.x) - 1) / sp.sin(self.x), 0.0, math.pi / 2.0, 32)
        self.output_newton_cotes(sp.atan(self.x) / self.x, 0.0, 1 / 2.0, 16)
        self.output_newton_cotes(sp.atan(self.x) / self.x, 0.0, 1 / 2.0, 32)

class TestRomberg(object):
    def setup(self):
        self.x = sp.Symbol('x')

    def output_romberg(self, f: sp.Function, a: float, b: float, m: int):
        romberg = ch5.Romberg(f)
        result = romberg(a, b, m)
        print("The function is \033\13331mf(x) = {}\033\1330m and the interval is \033\13331m[{}, {}]\033\1330m, line of Romberg table is \033\13331m[{}]\033\1330m.".format(romberg.symbol_f, a, b, m))
        print("Using \033\13331mRomberg integration\033\1330m , the result is: \033\13334m[{}]\033\1330m".format(result[0]))
        print("Using symbolic integration, the result is: \033\13334m[{}]\033\1330m".format(result[-1]))

    def test_romberg(self):
        self.output_romberg(sp.log(self.x), 1.0, 2.0, 4)
        # (1)
        self.output_romberg(self.x ** 2, 0.0, 1.0, 3)
        self.output_romberg(sp.cos(self.x), 0.0, math.pi / 2.0, 3)
        self.output_romberg(sp.exp(self.x), 0.0, 1.0, 3)
        # (2)
        self.output_romberg(self.x * sp.exp(self.x), 0.0, 1.0, 3)
        self.output_romberg(1.0 / (1.0 + self.x ** 2), 0.0, 1.0, 3)
        self.output_romberg(self.x * sp.cos(self.x), 0.0, math.pi, 3)
        # (1)
        self.output_romberg(self.x / sp.sqrt(self.x ** 2 + 9), 0.0, 4.0, 5)
        self.output_romberg(self.x ** 3 / sp.sqrt(self.x ** 2 + 1), 0.0, 1.0, 5)
        self.output_romberg(self.x * sp.exp(self.x), 0.0, 1.0, 5)
        self.output_romberg(self.x ** 2 * sp.log(self.x), 1.0, 3.0, 5)
        self.output_romberg(self.x ** 2 * sp.sin(self.x), 0.0, math.pi, 5)
        self.output_romberg(self.x ** 3 / sp.sqrt(self.x ** 4 - 1), 2.0, 3.0, 5)
        self.output_romberg(1 / sp.sqrt(self.x ** 2 + 4), 0.0, 2 * math.sqrt(3), 5)
        self.output_romberg(self.x / sp.sqrt(self.x ** 4 + 1), 0.0, 1.0, 5)


if __name__ == "__main__":
    # pytest.main(["-s", "test_ch5.py::TestNewtonCotes::test_newton_cotes"])
    # pytest.main(["-s", "test_ch5.py::TestCompositeNewtonCotes::test_composite_newton_cotes"])
    pytest.main(["-s", "test_ch5.py::TestRomberg::test_romberg"])
