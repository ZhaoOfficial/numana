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
            sp.integrate(self.symbol_f, (self.x, a, b))
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
