#!/usr/bin/env python3
# Copyright (C) 2022, Miklos Maroti
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

import sympy
import torch

from constraint_prog.point_cloud import PointCloud, PointFunc
from constraint_prog.sympy_func import ParetoFunc, pareto_func


def create_pareto_func():
    # variables
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

    constraints = PointFunc({
        "radius": x**2 + y**2 <= 1,
    })

    bounds = {
        "x": (-2.0, 2.0),
        "y": (-2.0, 2.0),
    }

    num_points = 1000
    points = PointCloud.generate(bounds, num_points)

    for _ in range(5):
        points.add_mutations(0.1, num_points)
        points = points.newton_raphson(constraints, bounds)

        errors = constraints(points)
        points = points.prune_by_tolerances(errors, 1e-3)
        points = points.prune_pareto_front2({"x": 1.0, "y": 1.0})
        points = points.prune_close_points2({"x": 0.01, "y": 0.01})

    # points.plot2d("x", "y")
    return pareto_func("within_quater_circle", points.projection(["x", "y"]), [1.0, 1.0])


def use_pareto_func(pareto_func):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

    constraints = PointFunc({
        "diag": x + y >= 0,
        "within": pareto_func(x, y),
    })

    print(constraints)

    bounds = {
        "x": (-2.0, 2.0),
        "y": (-2.0, 2.0),
    }

    num_points = 1000
    points = PointCloud.generate(bounds, num_points)

    for _ in range(5):
        points.add_mutations(0.1, num_points)
        points = points.newton_raphson(constraints, bounds)

        errors = constraints(points)
        points = points.prune_by_tolerances(errors, 1e-3)
        points = points.prune_close_points2({"x": 0.01, "y": 0.01})

        print(points.num_points)

    points.plot2d("x", "y")


if __name__ == '__main__':
    within_quater_circle = create_pareto_func()
    within_quater_circle.cloud.print_info()

    if False:
        u = sympy.Symbol("u")
        print(within_quater_circle(1.0, u))
        print(within_quater_circle(1.0, 1.0))
        print(within_quater_circle(0.2, 0.9))
        print(within_quater_circle(-1.0, 1.0))

    use_pareto_func(within_quater_circle)
