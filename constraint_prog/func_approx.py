#!/usr/bin/env python3
# Copyright (C) 2021, Miklos Maroti
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Set, Dict, List, Callable

import numpy
import sympy
from scipy import optimize


def get_symbols(expr: sympy.Expr) -> Set[str]:
    """
    Returns all symbols of this expression in a set.
    """
    symbols = set()

    def traverse(e: sympy.Expr):
        if isinstance(e, float) or isinstance(e, int):
            return
        if e.func == sympy.Symbol:
            symbols.add(e.name)
        for a in e.args:
            traverse(a)

    traverse(expr)
    return symbols


def evaluate(expr: sympy.Expr, input_data: Dict[str, numpy.ndarray]) -> numpy.ndarray:
    """
    Evaluates the given expression with the input data and returns the output.
    All numpy array must be of the same size or at least broadcastable.
    """
    if (isinstance(expr, float) or isinstance(expr, int)
            or expr.func == sympy.Float
            or expr.func == sympy.Integer
            or expr.func == sympy.core.numbers.Rational
            or expr.func == sympy.core.numbers.NegativeOne
            or expr.func == sympy.core.numbers.Zero
            or expr.func == sympy.core.numbers.One
            or expr.func == sympy.core.numbers.Pi
            or expr.func == sympy.core.numbers.Half):
        return numpy.full((), float(expr))
    elif expr.func == sympy.Symbol:
        return input_data[expr.name]
    elif expr.func == sympy.Add:
        value = evaluate(expr.args[0], input_data)
        for arg in expr.args[1:]:
            value = value + evaluate(arg, input_data)
        return value
    elif expr.func == sympy.Mul:
        value = evaluate(expr.args[0], input_data)
        for arg in expr.args[1:]:
            value = value * evaluate(arg, input_data)
        return value
    elif expr.func == sympy.Pow:
        assert len(expr.args) == 2
        value0 = evaluate(expr.args[0], input_data)
        value1 = float(expr.args[1])
        return numpy.power(value0, value1)
    else:
        raise ValueError(
            "Unknown symbolic expression " + str(type(expr)))


def approximate(func: sympy.Expr, input_data: Dict[str, numpy.ndarray],
                output_data: numpy.ndarray) -> Dict[str, float]:
    """
    Takes a expression with input and parameter variables, an input 
    data set of shape [num_points, input_vars] and an output data
    of shape [num_points]. Returns the mapping of parameter values
    to floats that minimizes the square error of the calculated and
    specified outputs.
    """
    symbols = get_symbols(func)
    param_vars = list(symbols - set(input_data.keys()))

    # common shape
    shape = numpy.broadcast(*input_data.values()).shape

    class Function(Callable):
        def __call__(self, params: List[float]):
            assert len(params) == len(param_vars)
            func2 = func.subs({var: params[idx] for idx, var in enumerate(param_vars)})
            output_data2 = evaluate(func2, input_data)
            return output_data2 - output_data

    class Jacobian(Callable):
        def __init__(self):
            self.diffs = [func.diff(var) for var in param_vars]

        def __call__(self, params: List[float]):
            assert len(params) == len(param_vars)
            subs = {var: params[idx] for idx, var in enumerate(param_vars)}
            diffs = [evaluate(diff.subs(subs), input_data) for diff in self.diffs]
            diffs = [numpy.broadcast_to(diff, shape) for diff in diffs]
            return numpy.array(diffs).transpose()

    init = [0.0] * len(param_vars)
    result = optimize.least_squares(
        fun=Function(),
        x0=init,
        jac=Jacobian(),
    )

    if not result.success:
        print("WARNING: apprixmation failed with cost", result.cost)

    params = {var: result.x[idx] for idx, var in enumerate(param_vars)}
    return params, result.cost


if __name__ == '__main__':
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    x = sympy.Symbol('x')
    f = a * x + b

    id = {
        'x': numpy.array([1.0, 2.0, 3.0, 4.0])
    }
    od = numpy.array([3.0, 5.0, 7.0, 9.0])

    print(approximate(f, id, od))
