from typing import Tuple
import sys

import numpy as np
import sympy as sp
import sympy.abc

class Solver(object):
    epsilon = sys.float_info.epsilon

    def __init__(self, f: sp.Function):
        self.symbol_f = f
        self.numeric_f = sp.lambdify(sympy.abc.x, f, "numpy")

    def symbolic(self) -> list:
        return sp.solve(self.symbol_f, sympy.abc.x)

    def bisection(self, a: float, b: float) -> float:
        """Using bisection method to find a root of funtion `f` in the interval `[a, b]`."""
        a, b = np.asarray(a), np.asarray(b)
        self.__checkEndpoint(a, b)
        return self._bisectionCore(a, b)

    def _bisectionCore(self, a: float, b: float) -> float:
        """Left endpoint is the anchor."""
        fa = self.numeric_f(a)
        tolerance = 2.0 * self.epsilon * max(1, abs(a))

        while abs(b - a) > tolerance:
            c = (a + b) / 2.0
            fc = self.numeric_f(c)
            if fc == 0.0:
                return c
            if fa * fc < 0:
                b = c
            else:
                a, fa = c, fc
        return (a + b) / 2

    def fixedPointIteration(self, a: float) -> float:
        """Using fixed point iteration to find the root of `f`."""
        a = np.asarray(a)
        b = self.numeric_f(a)
        tolerance = 2.0 * self.epsilon * max(1.0, abs(a))

        for _ in range(1000):
            if abs(b - a) > tolerance:
                a, b = b, self.numeric_f(b)
            else:
                break

        return b

    def secant(self, a: float, b: float) -> float:
        """Secant method (an improvement of Newton's method) for finding a root near `a` and `b`."""
        a, b = np.asarray(a), np.asarray(b)
        self.checkEndpoint(a, b)
        return self.__checkEndpoint(a, b)

    def _secantCore(self, a: float, b: float) -> float:
        fa, fb = self.numeric_f(a), self.numeric_f(b)
        tolerance = 2.0 * self.epsilon * max(1.0, abs(a))

        while abs(b - a) > tolerance:
            c = b - (fb * (b - a)) / (fb - fa)
            a, b = b, c
            fa, fb = fb, self.numeric_f(c)
        return b

    def __checkEndpoint(self, a: float, b: float) -> None:
        fa, fb = self.numeric_f(a), self.numeric_f(b)
        assert fa * fb < 0, "[{}, {}] is not a proper interval".format(a, b)
