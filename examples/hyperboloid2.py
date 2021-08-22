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

import numpy
from matplotlib import pyplot
import sympy
import torch

from constraint_prog.point_cloud import PointCloud, PointFunc


def main_pareto():
    dim = 10
    num = 1000  # number of simultaneous designs

    # create the constraint expressions
    expr1 = 1
    expr2 = 0
    for i in range(dim):
        symb = sympy.Symbol("x" + str(i))
        expr1 = expr1 * symb
        expr2 = expr2 + symb
    constraints = PointFunc({
        "prod_err": expr1 >= 1.0,
        "sum_err": expr2 <= dim + 1.0,
    })
    print(constraints)

    # generate random data points in bounding box
    bounds = {var: (0.0, 4.0) for var in constraints.input_names}
    points = PointCloud.generate(bounds, num)
    print("random designs:", points.float_data.shape)

    # repeatedly mutate designs to get closer to pareto front
    for _ in range(10):
        # mutate the existing points
        points.add_mutations(0.1, num)
        # points.plot2d("x0", "x1")

        # solve the constraints and update points
        points = points.newton_raphson(constraints, bounds)
        # points.plot2d("x0", "x1")

        # calculate the errors and prune points
        errors = constraints(points)
        points = points.prune_by_tolerances(errors, 1e-5)
        # points.plot2d("x0", "x1")

        # prune pareto front and plot it again
        points = points.prune_pareto_front([-1, -1] + [0] * (dim - 2))
        # points.plot2d("x0", "x1")

    print("final pareto designs:", points.float_data.shape)
    print(points.float_data[:5, :].numpy())
    points.plot2d("x0", "x1")


def main_nr_dist():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

    constraints = PointFunc({
        "prod_err": x * y >= 1,
        "sum_err": x + y <= 3,
    })

    steps = 40

    # generate mesh grid data
    points1 = PointCloud.mesh_grid({
        "x": (-1.0, 5.0, steps),
        "y": (-1.0, 5.0, steps),
    })

    # solve the constraints and update points
    bounds = {var: (0.0, 4.0) for var in constraints.input_names}
    points2 = points1.newton_raphson(constraints, bounds, num_iter=10)

    xs = points1["x"].numpy()
    ys = points1["y"].numpy()
    zs = numpy.sqrt((points2["x"].numpy() - xs) ** 2 + (points2["y"].numpy() - ys) ** 2)

    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(
        xs.reshape(steps, steps),
        ys.reshape(steps, steps),
        zs.reshape(steps, steps))
    pyplot.show()


if __name__ == '__main__':
    main_pareto()
    # main_nr_dist()
