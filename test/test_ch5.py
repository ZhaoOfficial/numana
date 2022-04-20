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
        print("Using \033\13331m1\033\1330m order Newton-Cotes formula, the result is: \033\13334m[{}]\033\1330m".format(result[0]))
        print("Using \033\13331m2\033\1330m order Newton-Cotes formula, the result is: \033\13334m[{}]\033\1330m".format(result[1]))
        print("Using \033\13331m3\033\1330m order Newton-Cotes formula, the result is: \033\13334m[{}]\033\1330m".format(result[2]))
        print("Using \033\13331m4\033\1330m order Newton-Cotes formula, the result is: \033\13334m[{}]\033\1330m".format(result[3]))
        print("Using symbolic integration, the result is: \033\13334m[{}]\033\1330m".format(result[4]))

    def test_newton_cotes(self):
        self.output_newton_cotes(sp.log(self.x), 1.0, 2.0)
        self.output_newton_cotes(self.x ** 2,    0.0, 1.0)
        self.output_newton_cotes(sp.cos(self.x), 0.0, math.pi / 2.0)
        self.output_newton_cotes(sp.exp(self.x), 0.0, 1.0)

class TestCompositeNewtonCotes(object):
    def setup(self):
        pass

    def test_composite_newton_cotes(self):
        pass

if __name__ == "__main__":
    pytest.main(["-s", "test_ch5.py::TestNewtonCotes::test_newton_cotes"])