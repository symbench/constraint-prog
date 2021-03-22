#!/usr/bin/env python3
# Copyright (C) 2021, Zsolt Vizi
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

import argparse
from typing import Callable
from inspect import getmembers
import json
import os

import numpy as np
import sympy
import torch

from constraint_prog import uuv_equations
from constraint_prog.sympy_func import SympyFunc
from constraint_prog.gradient_descent import gradient_descent
from constraint_prog.newton_raphson import newton_raphson


class Explorer:
    def __init__(self, problem_json: str, is_cuda_used: bool, max_points: int,
                 n_iter: int, output_dir: str, method: str,
                 newton_eps: float, gradient_lr: float):
        self.is_cuda_used = is_cuda_used
        self.max_points = max_points
        self.n_iter = n_iter
        self.output_dir = output_dir
        self.method = method

        if self.method == "newton":
            self.eps = newton_eps
        elif self.method == 'gradient':
            self.learning_rate = gradient_lr

        with open(problem_json) as f:
            self.json_content = json.loads(f.read())
        self.equations = None
        self.get_equations()
        self.func = SympyFunc(self.equations)

        constraints = self.json_content["constraints"]
        self.input_min = torch.Tensor([[constraints[x]["min"]
                                        for x in sorted(constraints.keys())]])
        self.input_max = torch.Tensor([[constraints[x]["max"]
                                        for x in sorted(constraints.keys())]])
        self.input_res = torch.Tensor([constraints[x]["resolution"]
                                       for x in sorted(constraints.keys())])

        self.tolerance = self.json_content["eps"]

    def get_equations(self):
        eqns = []
        if self.json_content["eqns"] == "uuv":
            members = getmembers(uuv_equations)
            for member in members:
                if isinstance(member[1], sympy.Eq):
                    eqns.append("uuv_equations." + member[0])
            eqns_str = ', '.join(eqns)
            self.equations = eval('[' + eqns_str + ']')

            # Alternative way: direct construction by name
            # from uuv_equations import hull_length_equation, hull_thickness_equation, mission_duration_equation
            # self.equations = [hull_length_equation, hull_thickness_equation, mission_duration_equation]

    def print_equations(self):
        print("Loaded equations:")
        for eq in self.equations:
            print(eq)
        print("Variable names:", ', '.join(self.func.input_names))

    def run(self):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.is_cuda_used else "cpu")
        input_data = self.generate_input().to(device)

        print("Running {} method on {} many design points".format(
            self.method, input_data.shape[0]))
        output_data = None
        if self.method == "newton":
            output_data = newton_raphson(func=self.func, input_data=input_data,
                                         num_iter=self.n_iter, epsilon=self.eps)
        elif self.method == "gradient":
            output_data = gradient_descent(f=self.func, in_data=input_data,
                                           it=self.n_iter, lrate=self.learning_rate,
                                           device=device)

        output_data = self.check_tolerance(output_data)
        print("After checking with {} tolerance we have {} designs".format(
            self.tolerance, output_data.shape[0]))

        if False:
            output_data = self.remove_close_points(output_data)
            print("After pruning with {} tolerance we have {} designs".format(
                self.tolerance, output_data.shape[0]))

        file_name = os.path.join(os.path.abspath(
            self.output_dir), "output_data.npz")
        print("Saving generated design points to:", file_name)
        np.savez_compressed(file_name, data=output_data)
        # self.load_npz(file_name)

    def load_npz(self, file_name: str):
        saved_out = np.load(file_name)["data"]
        print(saved_out.shape)

    def generate_input(self):
        """Generates input data with uniform distribution."""
        sample = torch.rand(size=(self.max_points, self.func.input_size))
        sample *= (self.input_max - self.input_min)
        sample += self.input_min
        return sample

    def check_tolerance(self, samples: torch.tensor):
        """
        Evaluates the given list fo designs and returns a new list for
        which the 2-norm of errors is less than the tolerance.
        """
        equation_output = self.func(samples)
        good_point_idx = torch.norm(equation_output, dim=1) < self.tolerance
        return samples[good_point_idx]

    def remove_close_points(self, output_data: torch.tensor):
        output_data_tile = torch.tile(
            output_data, (output_data.shape[0], 1, 1))
        output_data_1kn = output_data.reshape(
            (1, output_data.shape[0], output_data.shape[1]))
        output_data_k1n = output_data.reshape(
            (output_data.shape[0], 1, output_data.shape[1]))
        output_data_compare = output_data_1kn ** 2 - 2 * \
            output_data_tile * output_data_k1n + output_data_k1n ** 2

        comparison = torch.tensor(
            output_data_compare > (self.input_res.reshape(
                (1, 1, self.input_res.shape[0])) ** 2)
        )
        difference_matrix = torch.sum(comparison.to(int), dim=2)
        indices = np.array([[(i, j) for j in range(output_data.shape[0])]
                            for i in range(output_data.shape[0])])

        close_points_bool_idx = torch.tensor(difference_matrix == 0)
        close_point_idx = indices[close_points_bool_idx.numpy()]
        close_and_different_points_idx = indices[close_point_idx[:, 0]
                                                 != close_point_idx[:, 1]]

        # TODO: from close and different point sets remove all except one
        return output_data


def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('problem_json', type=str,
                        help='Path to the problem JSON file used for exploration')
    parser.add_argument('--method', type=str, choices=["gradient", "newton"],
                        default="newton", help='Method to run')
    parser.add_argument('--output-dir', metavar='DIR', type=str,
                        default=os.getcwd(),
                        help='Path to output directory')
    parser.add_argument('--cuda', action='store_true',
                        help='Flag for enabling CUDA for calculations')
    parser.add_argument('--iter', type=int, metavar='NUM', default=10,
                        help='Number of iterations in the solver')
    parser.add_argument('--max-points', type=int, metavar='NUM', default=1000,
                        help='Maximal number of points generated for exploration')
    parser.add_argument('--eps', type=float, metavar='NUM', default=0.1,
                        help='Epsilon value for Newton-Raphson method')
    parser.add_argument('--lrate', type=int, metavar='NUM', default=0.1,
                        help='Learning rate value for gradient descent method')
    parser.add_argument('--print-equs', action='store_true',
                        help='Print the loaded equations')
    args = parser.parse_args(args)

    explorer = Explorer(problem_json=args.problem_json,
                        is_cuda_used=args.cuda,
                        max_points=args.max_points,
                        n_iter=args.iter,
                        output_dir=args.output_dir,
                        method=args.method,
                        newton_eps=args.eps,
                        gradient_lr=args.lrate)
    if args.print_equs:
        explorer.print_equations()
    explorer.run()


if __name__ == '__main__':
    main()
