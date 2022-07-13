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

import numpy
from matplotlib import pyplot
import sympy

from constraint_prog.point_cloud import PointCloud, PointFunc


def get_constraints(balls=4):

    bounds = dict()
    for i in range(balls):
        bounds['x{}'.format(i)] = (0.0, 1.0)
        bounds['y{}'.format(i)] = (0.0, 1.0)
        bounds['r{}'.format(i)] = (0.0, 1.0)

    constraints = dict()

    for i in range(balls):
        xi = sympy.Symbol('x{}'.format(i))
        yi = sympy.Symbol('y{}'.format(i))
        ri = sympy.Symbol('r{}'.format(i))

        constraints['bnd1{}'.format(i)] = xi >= ri
        constraints['bnd2{}'.format(i)] = yi >= ri
        constraints['bnd3{}'.format(i)] = 1 - xi >= ri
        constraints['bnd4{}'.format(i)] = 1 - yi >= ri

    for i in range(balls - 1):
        xi = sympy.Symbol('x{}'.format(i))
        yi = sympy.Symbol('y{}'.format(i))
        ri = sympy.Symbol('r{}'.format(i))
        for j in range(i + 1, balls):
            xj = sympy.Symbol('x{}'.format(j))
            yj = sympy.Symbol('y{}'.format(j))
            rj = sympy.Symbol('r{}'.format(j))
            constraints['dis{}{}'.format(i, j)] = \
                (xi - xj) ** 2 + (yi - yj) ** 2 >= (ri + rj) ** 2

    rx = 0.0
    ry = 0.0
    rs = 0.0
    for i in range(balls // 2):
        xi = sympy.Symbol('x{}'.format(i))
        yi = sympy.Symbol('y{}'.format(i))
        ri = sympy.Symbol('r{}'.format(i))
        rx = rx + xi * ri ** 2
        ry = ry + yi * ri ** 2
        rs = rs + ri ** 2

    bx = 0.0
    by = 0.0
    bs = 0.0
    for i in range(balls // 2, balls):
        xi = sympy.Symbol('x{}'.format(i))
        yi = sympy.Symbol('y{}'.format(i))
        ri = sympy.Symbol('r{}'.format(i))
        bx = bx + xi * ri ** 2
        by = by + yi * ri ** 2
        bs = bs + ri ** 2

    constraints['cnt1'] = bx * rs <= (rx + 0.01 * rs) * bs
    constraints['cnt2'] = by * rs <= (ry + 0.01 * rs) * bs
    constraints['cnt3'] = rx * bs <= (bx + 0.01 * bs) * rs
    constraints['cnt4'] = ry * bs <= (by + 0.01 * bs) * rs

    reports = {
        'ox': bx + rx,
        'oy': by + ry,
        'os': bs + rs,
    }

    return bounds, constraints, reports


def main():
    bounds, constraints, reports = get_constraints(4)

    if True:
        for key, val in bounds.items():
            print(key + ':', val)

        for key, val in constraints.items():
            print(key + ':', val)

        for key, val in reports.items():
            print(key + ':', val)

    constraints = PointFunc(constraints)
    reports = PointFunc(reports)

    num = 10000
    points = PointCloud.generate(bounds, num)

    for _ in range(50):
        points.add_mutations(0.01, num)

        points = points.newton_raphson(constraints, bounds)

        errors = constraints(points)
        points = points.prune_by_tolerances(errors, 1e-5)

        points = points.extend(reports(points, False))
        # points = points.extend(constraints(points))

        points = points.prune_close_points2(
            {
                "ox": 0.005,
                "oy": 0.005,
                "os": 0.005,
            })

        print(points.num_points)
        print(points.row(0))

    points.plot3d("ox", "oy", "os")


if __name__ == '__main__':
    main()
