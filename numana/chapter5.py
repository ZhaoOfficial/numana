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
        h = (b - a)
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(b)
        return h * (y0 + y1) / 2.0

    def _secondOrder(self, a: float, b: float) -> float:
        h = (b - a) / 2.0
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(a + h)
        y2 = self.numeric_f(b)
        return h * (y0 + 4.0 * y1 + y2) / 3.0

    def _thirdOrder(self, a: float, b: float) -> float:
        h = (b - a) / 3.0
        y0 = self.numeric_f(a)
        y1 = self.numeric_f(a + h)
        y2 = self.numeric_f(b - h)
        y3 = self.numeric_f(b)
        return h * (y0 + 3.0 * (y1 + y2) + y3) * 3.0 / 8.0

    def _forthOrder(self, a: float, b: float) -> float:
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
    @param
        `f`: the function to evaluate on some intervals.
    """
    def __init__(self, f: sp.Function):
        self.x = sp.Symbol('x')
        self.symbol_f = f
        self.numeric_f = sp.lambdify(self.x, f, "numpy")
