from typing import Iterable, Optional, Union

import numpy as np

def nest(
    x: Union[float, Iterable],
    coefficients: Iterable,
    base_points: Optional[Iterable] = None
) -> float:
    """
    @param `x`: the point to evaluate.
    @param `coefficients`: coefficients array from lower to higher order.
    @param `base_points`: base points in evaluatation.
    @return: the value at point `x`.
    """
    x, y = np.asarray(x), coefficients[-1]
    coeffs = np.asarray(coefficients[-2::-1])
    if base_points is None:
        base_points = np.zeros_like(coeffs)
    else:
        base_points = np.asarray(base_points)

    for i in range(coeffs.shape[0]):
        y = y * (x + base_points[i]) + coeffs[i]

    return y

def quadratic(a: float, b: float, c: float) -> tuple:
    """Solve the equation `ax^2 + bx + c = 0`."""

    Delta = b * b - 4.0 * a * c
    solution = (None, None)
    if Delta == 0.0:
        solution = (-b / (2.0 * a), -b / (2.0 * a))
    elif Delta > 0.0:
        if b > 0.0:
            solution = (-(b + Delta) / (2.0 * a), -(2.0 * c) / (b + Delta))
        else:
            solution = ((Delta - b) / (2.0 * a), (2.0 * c) / (Delta - b))

    return solution

def dec2bin(deci: Union[float, str], fraction_len: int = 10) -> str:
    """
    Convert a decimal formed number into binary form

    @param `deci`: the decimal number
    @param `fraction_len`: the length of fraction part
    @return: the binary number
    """

    num = str(deci)
    decimal_point = num.find('.')
    # if there is a decimal point
    if decimal_point != -1:
        integer = int(num[:decimal_point])
        fraction = float(num[decimal_point:])
    else:
        integer = int(num)
        fraction = 0.0

    # integer part
    result = []
    while integer != 0:
        result.append(str(integer & 1))
        integer >>= 1
    result.reverse()
    if result == []:
        result.append('0')

    # fraction part
    if fraction != 0.0:
        result.append('.')
    while fraction != 0.0 and fraction_len > 0:
        fraction *= 2.0
        if fraction >= 1.0:
            fraction -= 1.0
            result.append('1')
        else:
            result.append('0')
        fraction_len -= 1

    result = ''.join(result)
    return result

def bin2dec(bina: str, recur_pos: tuple = None) -> float:
    """
    Convert a binary formed number into decimal form

    @param `bina`: the binary number
    @param `recur_pos`: the start and the end of the recurring fraction after the binary point
    @return: the decimal number
    """

    num = str(bina)
    binary_point = num.find('.')
    if recur_pos != None:
        start, end = recur_pos
        x = num[:binary_point] + num[binary_point + 1:]
        y = num[:binary_point] + num[binary_point + 1: binary_point + start + 1]
        x_coefficient = list(map(int, x))
        x_coefficient.reverse()
        x_result = nest(2, x_coefficient)
        y_coefficient = list(map(int, y))
        y_coefficient.reverse()
        y_result = nest(2, y_coefficient)

        x_scale = 2 << end
        y_scale = 1 << start
        return (x_result - y_result) / (x_scale - y_scale)
    else:
        if binary_point != -1:
            integer = num[:binary_point]
            fraction = '0' + num[binary_point + 1:]
        else:
            integer = num
            fraction = '0'

        integer_coefficient = list(map(int, integer))
        integer_coefficient.reverse()
        result = nest(2, integer_coefficient)

        fraction_coefficient = list(map(int, fraction))
        result += nest(0.5, fraction_coefficient)

    return result
