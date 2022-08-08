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

import os
import sympy
import torch

from constraint_prog.point_cloud import PointCloud, PointFunc
from constraint_prog.sympy_func import neural_func


def symbolic_constraints():
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

    for _ in range(10):
        points.add_mutations(0.1, num_points)

        points = points.newton_raphson(constraints, bounds)

        errors = constraints(points)
        points = points.prune_by_tolerances(errors, 1e-3)

        points = points.prune_close_points2(
            {
                "x": 0.01,
                "y": 0.01,
            })

        print(points.num_points)

    points.plot2d("x", "y")


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'neuralnet.pt')


class MyNetwork(torch.nn.Module):
    """
    This is the model we want to train and use in the constraint solver.
    Here we are going to approximate the x^2 + y^2 function.
    """

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(2, 20)
        self.fc2 = torch.nn.Linear(20, 1)

    def forward(self, xy):
        tmp = self.fc1(xy)
        tmp = torch.tanh(tmp)
        tmp = self.fc2(tmp)
        return tmp


def neuralnet_training():
    network = MyNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    batchsize = 1000

    for step in range(5000):
        # random points in [-2, 2] interval
        inputs = torch.rand(size=(batchsize, 2)) * 4.0 - 2.0
        labels = torch.sqrt(inputs[:, 0:1] ** 2 + inputs[:, 1:2] ** 2)

        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(step, loss.item())

    print("Saving model to {}".format(MODEL_PATH))
    torch.save(network, MODEL_PATH)


def neuralnet_testing():
    my_function = neural_func("my_function", 2, torch.load(MODEL_PATH))

    print(my_function(1.0, 1.0))

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    print(my_function(sympy.Symbol("x"), 1.0))

    constraints = PointFunc({
        "radius": my_function(x, y) <= 1,
    })

    print(constraints)
    print(constraints.input_names, constraints.output_names)

    bounds = {
        "x": (-2.0, 2.0),
        "y": (-2.0, 2.0),
    }

    points = PointCloud.generate(bounds, 10)
    output = constraints(points)
    output.print_info()


def neuralnet_constraints():
    # define my new function using the saved model
    my_function = neural_func("my_function", 2, torch.load(MODEL_PATH))

    # variables
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

    constraints = PointFunc({
        "radius": my_function(x, y) <= 1,
    })

    bounds = {
        "x": (-2.0, 2.0),
        "y": (-2.0, 2.0),
    }

    num_points = 1000
    points = PointCloud.generate(bounds, num_points)

    for _ in range(10):
        points.add_mutations(0.1, num_points)

        points = points.newton_raphson(constraints, bounds)

        errors = constraints(points)
        points = points.prune_by_tolerances(errors, 1e-3)

        points = points.prune_close_points2(
            {
                "x": 0.01,
                "y": 0.01,
            })

        print(points.num_points)

    points.plot2d("x", "y")


if __name__ == '__main__':
    symbolic_constraints()
    # neuralnet_training()
    # neuralnet_testing()
    neuralnet_constraints()
