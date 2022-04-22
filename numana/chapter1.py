from typing import Tuple

import numpy as np

class Solver(object):
    def __init__(self, f):
        self.f = f

    def bisection(self, initial: Tuple[float, float]) -> float:
        a, b = map(float, initial)
        fa, fb = self.f(a), self.f(b)
        assert fa * fb < 0, "{} is not a proper interval".format(initial)

        return self._bisectionCore(a, b)

    def _bisectionCore(self, a, b):
        fa, fb = self.f(a), self.f(b)
        tolerance = 2 * np.finfo(float).eps * max(1, abs(a))
        while abs(b - a) > tolerance:
            c = (a + b) / 2
            fc = self.f(c)
            if fc == 0.0:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return (a + b) / 2
