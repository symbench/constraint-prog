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

import sympy

from constraint_prog.point_cloud import PointCloud, PointFunc


def run():
    """
    put two circles inside a rectangle, while maximize the distance between
    the circles, and minimize the area of rectangle.
    """

    # symbols
    x0 = sympy.Symbol("x0")  # rectangle
    y0 = sympy.Symbol("y0")
    x1 = sympy.Symbol("x1")  # unit circle 1
    y1 = sympy.Symbol("y1")
    x2 = sympy.Symbol("x2")  # unit circle 1
    y2 = sympy.Symbol("y2")

    # bounds
    bounds = {
        "x0": (2.0, 5.0),
        "y0": (2.0, 5.0),
        "x1": (1.0, 4.0),
        "y1": (1.0, 4.0),
        "x2": (1.0, 4.0),
        "y2": (1.0, 4.0),
    }

    # derived quantities
    derived = PointFunc({
        "area": x0 * y0,  # rectange area
        "dist": (x1 - x2) ** 2 + (y1 - y2) ** 2,  # distance squared
    })

    # circles lay inside rectangle
    constraints = PointFunc({
        "equ1": x1 + 1 <= x0,
        "equ2": y1 + 1 <= y0,
        "equ3": x2 + 1 <= x0,
        "equ4": y2 + 1 <= y0,
    })

    # generate random points
    num = 1000
    points = PointCloud.generate(bounds, num)

    for step in range(50):
        points.add_mutations(0.1, num)

        points = points.newton_raphson(constraints, bounds, num_iter=10)
        # points.print_info()

        points = points.prune_by_tolerances(constraints(points), 0.01)
        # points.print_info()

        points = points.extend(derived(points))
        # points.print_info()

        directions = [0.0] * points.num_float_vars
        directions[points.float_vars.index("area")] = -1.0
        directions[points.float_vars.index("dist")] = 1.0

        points = points.prune_pareto_front(directions)

        if step % 10 == 9:
            points.plot2d("area", "dist")


if __name__ == '__main__':
    run()
