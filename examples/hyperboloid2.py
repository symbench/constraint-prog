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

import torch

from constraint_prog.point_cloud import PointCloud
from constraint_prog.newton_raphson import newton_raphson


def constraints(designs: torch.Tensor) -> torch.Tensor:
    """
    Takes a tensor of shape [*,dim] and returns a tensor of shape [*,2].
    We use direct pytorch computation, could have used sympy too.
    """
    dim = designs.shape[-1]
    zero = torch.zeros((), dtype=designs.dtype, device=designs.device)
    err1 = 1.0 - torch.prod(designs, dim=-1)
    err2 = torch.sum(designs, dim=-1) - (dim + 1.0)
    # the error values should be negative, so we clamp it
    return torch.stack((err1, err2), dim=-1).clamp_min(0.0)


def main():
    dim = 10
    num = 1000  # number of simultaneous designs

    # generate random data points in bounding box
    variables = ["x" + str(i) for i in range(dim)]
    minimums = [0.0] * dim
    maximums = [4.0] * dim
    points = PointCloud.generate({var: (0.0, 4.0) for var in variables}, num)
    print("random designs:", points.float_data.shape)

    # manually call the newton raphson optimization
    points.float_data = newton_raphson(
        constraints,
        points.float_data,
        num_iter=10,
        bounding_box=torch.tensor([minimums, maximums], dtype=torch.float32),
        method="mmclip")

    # manually calculate the final errors
    errors = PointCloud(["prod_err", "sum_err"], constraints(points.float_data))
    points = points.prune_by_tolerances(errors, [1e-5, 1e-5])
    print("feasible designs: ", points.float_data.shape)

    # print the first 5 solutions and plot x0, x1 coords
    print(points.float_data[:5, :].numpy())
    points.plot2d(0, 1)

    # prune pareto front and plot it again
    points = points.prune_pareto_front([-1, -1] + [0] * (dim - 2))
    print("pareto designs:", points.float_data.shape)
    print(points.float_data[:5, :].numpy())
    points.plot2d(0, 1)

    # repeatedly mutate designs to get closer to pareto front
    for _ in range(10):
        # mutate the existing points
        points.add_mutations([0.1] * dim, num)
        # points.plot2d(0, 1)

        # manually call the newton raphson optimization
        points.float_data = newton_raphson(
            constraints,
            points.float_data,
            num_iter=10,
            bounding_box=torch.tensor([minimums, maximums], dtype=torch.float32),
            method="mmclip")

        # manually calculate the final errors
        errors = PointCloud(["prod_err", "sum_err"], constraints(points.float_data))
        points = points.prune_by_tolerances(errors, [1e-5, 1e-5])
        # points.plot2d(0, 1)

        # prune pareto front and plot it again
        points = points.prune_pareto_front([-1, -1] + [0] * (dim - 2))
        # points.plot2d(0, 1)

    print("final pareto designs:", points.float_data.shape)
    print(points.float_data[:5, :].numpy())
    points.plot2d(0, 1)


if __name__ == '__main__':
    main()
