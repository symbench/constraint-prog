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

from typing import Set, Dict, Callable, List

import numpy
import sympy
import scipy.optimize

from constraint_prog.point_cloud import PointCloud


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


def approximate(func: sympy.Expr, points: 'PointCloud',
                output: str = 'result') -> Dict[str, float]:
    """
    Takes a expression with input and parameter variables, a point cloud with
    values for the input variables and a name for the output variable. The 
    method tries to find the best parameter values for the non-input variables
    that best approximates the output value from the point cloud.
    """
    symbols = get_symbols(func)
    param_vars = sorted(symbols - set(points.float_vars))
    input_vars = sorted(symbols.intersection(points.float_vars))

    assert output in points.float_vars

    class Func(Callable):
        def __call__(self, params: List[float]):
            assert len(params) == len(param_vars)

            sub = {var: params[idx] for idx, var in enumerate(param_vars)}
            points2 = points.evaluate(['result'], [func.subs(sub)])
            result = (points2['result'] - points[output]).numpy()
            return result

    init = [1.0] * len(param_vars)
    result = scipy.optimize.least_squares(
        Func(),
        init,
        diff_step=1e-5)

    if result.success:
        print("INFO: approximation is successful with cost", result.cost)
        print("INFO:", result.message)
    else:
        print("WARNING: apprixmation failed with cost", result.cost)

    return result.x


if __name__ == '__main__':
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    x = sympy.Symbol('x')
    f = a * x + b

    data = PointCloud(['x', 'y'])
    data.append({'x': 1, 'y': 3})
    data.append({'x': 2, 'y': 5})
    data.append({'x': 3, 'y': 7})
    data.append({'x': 4, 'y': 9})

    print(approximate(f, data, 'y'))
