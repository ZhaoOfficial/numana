import math
from typing import Tuple

import numpy as np
import sympy as sp

class NewtonCotes(object):
    """
    Newton-Cotes methods for numerical integration.
    @param `f`: the function to evaluate on some intervals.
    """
    def __init__(self, f: sp.Function):
        self.x = sp.Symbol('x')
        self.symbol_f = f
        self.numeric_f = sp.lambdify(self.x, f, "numpy")

    def __call__(self, a: float, b: float) -> Tuple[float]:
        """Integration on `[a, b]`."""

        assert not (math.isinf(a) or math.isnan(a)), "Invalid interval."
        assert not (math.isinf(b) or math.isnan(b)), "Invalid interval."

        return (
            self._firstOrder(a, b),
            self._secondOrder(a, b),
            self._thirdOrder(a, b),
            self._forthOrder(a, b),
            self._fifthOrder(a, b),
            self._sixthOrder(a, b),
            self._seventhOrder(a, b),
            sp.integrate(self.symbol_f, (self.x, a, b)),
        )

    def _firstOrder(self, a: float, b: float) -> float:
        """Trapezoid Rule."""
        h = (b - a)
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(b)
        return h * (y0 + y1) / 2.0

    def _secondOrder(self, a: float, b: float) -> float:
        """Simpson's Rule."""
        h = (b - a) / 2.0
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(a + h)
        y2 = self.numeric_f(b)
        return h * (y0 + 4.0 * y1 + y2) / 3.0

    def _thirdOrder(self, a: float, b: float) -> float:
        """Simpson's 3/8 Rule"""
        h = (b - a) / 3.0
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(a + h)
        y2 = self.numeric_f(b - h)
        y3 = self.numeric_f(b)
        return h * (y0 + 3.0 * (y1 + y2) + y3) * 3.0 / 8.0

    def _forthOrder(self, a: float, b: float) -> float:
        """Boole's Rule"""
        h = (b - a) / 4.0
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(a + h)
        y2 = self.numeric_f(a + 2.0 * h)
        y3 = self.numeric_f(b - h)
        y4 = self.numeric_f(b)
        return h * (7.0 * (y0 + y4) + 32.0 * (y1 + y3) + 12.0 * y2) * 2.0 / 45.0

    def _fifthOrder(self, a: float, b: float) -> float:
        h = (b - a) / 5.0
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(a + h)
        y2 = self.numeric_f(a + 2.0 * h)
        y3 = self.numeric_f(b - 2.0 * h)
        y4 = self.numeric_f(b - h)
        y5 = self.numeric_f(b)
        return h * (19.0 * (y0 + y5) + 75.0 * (y1 + y4) + 50.0 * (y2 + y3)) * 5.0 / 288.0

    def _sixthOrder(self, a: float, b: float) -> float:
        h = (b - a) / 6.0
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(a + h)
        y2 = self.numeric_f(a + 2.0 * h)
        y3 = self.numeric_f(a + 3.0 * h)
        y4 = self.numeric_f(b - 2.0 * h)
        y5 = self.numeric_f(b - h)
        y6 = self.numeric_f(b)
        return h * (41.0 * (y0 + y6) + 216.0 * (y1 + y5) + 27.0 * (y2 + y4) + 272.0 * y3) / 140.0

    def _seventhOrder(self, a: float, b: float) -> float:
        h = (b - a) / 7.0
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(a + h)
        y2 = self.numeric_f(a + 2.0 * h)
        y3 = self.numeric_f(a + 3.0 * h)
        y4 = self.numeric_f(b - 3.0 * h)
        y5 = self.numeric_f(b - 2.0 * h)
        y6 = self.numeric_f(b - h)
        y7 = self.numeric_f(b)
        return h * (751.0 * (y0 + y7) + 3577.0 * (y1 + y6) + 1323.0 * (y2 + y5) + 2989.0 * (y3 + y4)) * 7.0 / 17280.0

class CompositeNewtonCotes(object):
    """
    Composite Newton-Cotes methods for numerical integration.
    @param `f`: the function to evaluate on some intervals.
    """
    def __init__(self, f: sp.Function):
        self.x = sp.Symbol('x')
        self.symbol_f = f
        self.numeric_f = sp.lambdify(self.x, f, "numpy")

    def __call__(self, a: float, b: float, m: int) -> Tuple[float]:
        """Divide `[a, b]` into `m` segments."""

        assert not (math.isinf(a) or math.isnan(a)), "Invalid interval."
        assert not (math.isinf(b) or math.isnan(b)), "Invalid interval."
        assert m > 0, "Invalid number of intervals."

        return (
            self._trapezoid(a, b, m),
            self._midpoint(a, b, m),
            self._simpson(a, b, m),
            sp.integrate(self.symbol_f, (self.x, a, b)),
        )

    def _trapezoid(self, a: float, b: float, m: int) -> float:
        """Composite Trapezoid Rule."""
        h = (b - a) / m
        x = np.linspace(a, b, m + 1)
        y = self.numeric_f(x)
        return (y[0] + y[-1] + 2.0 * np.sum(y[1:-1])) * h * 0.5

    def _midpoint(self, a: float, b: float, m: int) -> float:
        """Composite Midpoint Rule."""
        h = (b - a) / m
        h2 = h / 2.0
        x = np.linspace(a + h2, b - h2, m)
        y = self.numeric_f(x)
        return np.sum(y) * (b - a) / m

    def _simpson(self, a: float, b: float, m: int) -> float:
        """Composite Simpson's Rule."""
        h = (b - a) / (2.0 * m)
        x = np.linspace(a, b, 2 * m + 1)
        y = self.numeric_f(x)
        return (y[0] + y[-1] + 2.0 * np.sum(y[1::2]) + 2.0 * np.sum(y[1:-1])) * h / 3.0

class Romberg(object):
    """
    Romberg methods for numerical integration.
    @param `f`: the function to evaluate on some intervals.
    """
    def __init__(self, f: sp.Function):
        self.x = sp.Symbol('x')
        self.symbol_f = f
        self.numeric_f = sp.lambdify(self.x, f, "numpy")

    def __call__(self, a: float, b: float, m: int) -> Tuple[float]:
        """Romberg integration on `[a, b]` with `m` lines of romberg table."""

        assert not (math.isinf(a) or math.isnan(a)), "Invalid interval."
        assert not (math.isinf(b) or math.isnan(b)), "Invalid interval."
        assert m > 0, "Invalid number of lines."

        R = np.zeros((m, m), dtype=float)

        h = (b - a) / 2.0
        R[0, 0] = (self.numeric_f(a) + self.numeric_f(b)) * h
        for i in range(1, m):
            R[i, 0] = (R[i - 1, 0] / 2.0) + h * np.sum(self.numeric_f(np.linspace(a + h, b - h, 2 ** (i - 1))))
            for j in range(1, i + 1):
                R[i, j] = (4 ** j * R[i, j - 1] - R[i - 1, j - 1]) / (4 ** j - 1)
            h /= 2.0

        return (R[-1, -1], sp.integrate(self.symbol_f, (self.x, a, b)))
