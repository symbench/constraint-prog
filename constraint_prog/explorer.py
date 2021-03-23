#!/usr/bin/env python3
# Copyright (C) 2021, Zsolt Vizi, Miklos Maroti
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
import csv
from inspect import getmembers
import importlib.util
import json
import os
from typing import List

import numpy as np
import sympy
import torch

from constraint_prog.sympy_func import SympyFunc
from constraint_prog.gradient_descent import gradient_descent
from constraint_prog.newton_raphson import newton_raphson


class Explorer:
    def __init__(self, problem_json: str, output_dir: str,
                 cuda_enable: bool, to_csv: bool,
                 max_points: int, method: str,
                 newton_eps: float, newton_iter: int,
                 gradient_lr: float, gradient_iter: int):
        self.problem_json = problem_json
        self.output_dir = output_dir
        self.cuda_enable = cuda_enable
        self.to_csv = to_csv
        self.max_points = max_points
        self.method = method

        self.newton_eps = newton_eps
        self.newton_iter = newton_iter

        self.gradient_lr = gradient_lr
        self.gradient_iter = gradient_iter

        with open(problem_json) as f:
            self.json_content = json.loads(f.read())
        self.equations = None
        self.get_equations()
        self.func = SympyFunc(self.equations)
        # print(self.func.input_names)

        constraints = self.json_content["constraints"]
        # disregard entries that start with a dash
        constraints = {key: val for (key, val) in constraints.items()
                       if not key.startswith('-')}

        assert sorted(constraints.keys()) == self.func.input_names

        self.input_min = torch.Tensor([[constraints[var]["min"]
                                        for var in self.func.input_names]])
        self.input_max = torch.Tensor([[constraints[var]["max"]
                                        for var in self.func.input_names]])
        self.input_res = torch.Tensor([constraints[var]["resolution"]
                                       for var in self.func.input_names])

        self.tolerance = self.json_content["eps"]

    def get_equations(self):
        path = os.path.join(
            os.path.dirname(self.problem_json), self.json_content["source"])
        print("Loading python file:", path)
        if not os.path.exists(path):
            raise FileNotFoundError()

        spec = importlib.util.spec_from_file_location("equmod", path)
        equmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(equmod)
        members = getmembers(equmod)

        self.equations = []
        for name in self.json_content["equations"]:
            # disregard entries that start with a dash
            if name.startswith('-'):
                continue
            for member in members:
                if member[0] == name:
                    assert isinstance(member[1], sympy.Eq)
                    self.equations.append(member[1])
                    break
            else:
                raise ValueError("equation " + name + " not found")

    def print_equations(self):
        print("Loaded equations:")
        for eq in self.equations:
            print(eq)
        print("Variable names:", ', '.join(self.func.input_names))

    def run(self):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.cuda_enable else "cpu")
        input_data = self.generate_input().to(device)

        print("Running {} method on {} many design points".format(
            self.method, input_data.shape[0]))
        output_data = None
        if self.method == "newton":
            output_data = newton_raphson(func=self.func,
                                         input_data=input_data,
                                         num_iter=self.newton_iter,
                                         epsilon=self.newton_eps)
        elif self.method == "gradient":
            output_data = gradient_descent(f=self.func,
                                           in_data=input_data,
                                           it=self.gradient_iter,
                                           lrate=self.gradient_lr,
                                           device=device)

        output_data = self.check_tolerance(output_data)
        print("After checking with {} tolerance we have {} designs".format(
            self.tolerance, output_data.shape[0]))

        output_data = self.prune_close_points2(output_data)
        print("After pruning close points we have {} designs".format(
            output_data.shape[0]))

        if False:
            output_data = self.prune_bounding_box(output_data)
            print("After bounding box pruning we have {} designs".format(
                output_data.shape[0]))

        self.save_data(output_data)

    def save_data(self, samples: torch.tensor) -> None:
        filename = os.path.splitext(os.path.basename(self.problem_json))[0] + \
            (".csv" if self.to_csv else ".npz")
        file_name = os.path.join(os.path.abspath(self.output_dir), filename)
        print("Saving generated design points to:", file_name)
        if self.to_csv:
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.func.input_names)
                writer.writerows(samples.numpy())
        else:
            np.savez_compressed(file_name,
                                sample_vars=self.func.input_names,
                                sample_data=samples)

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

    def prune_close_points(self, output_data: torch.tensor):
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

    def prune_close_points2(self, samples: torch.tensor):
        assert samples.ndim == 2 and samples.shape[1] == len(self.input_res)
        rounded = samples / self.input_res.reshape((1, samples.shape[1]))
        rounded = rounded.round().type(torch.int64)

        # hash based filtering is not unique, but good enough
        randcoef = torch.randint(-10000000, 10000000, (1, samples.shape[1]))
        hashnums = (rounded * randcoef).sum(dim=1)

        # this is slow, but good enough
        selected = torch.zeros(samples.shape[0], dtype=bool)
        hashset = set()
        for idx in range(samples.shape[0]):
            value = int(hashnums[idx])
            if value not in hashset:
                hashset.add(value)
                selected[idx] = True

        return samples[selected]

    def prune_bounding_box(self, samples: torch.tensor):
        assert samples.ndim == 2
        selected = torch.logical_and(samples >= self.input_min,
                                     samples <= self.input_max)
        selected = selected.all(dim=1)
        return samples[selected]


def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('problem_json', type=str,
                        help='Path to the problem JSON file used for exploration')
    parser.add_argument('--output-dir', metavar='DIR', type=str,
                        default=os.getcwd(),
                        help='Path to output directory')
    parser.add_argument('--cuda', action='store_true',
                        help='Flag for enabling CUDA for calculations')
    parser.add_argument('--max-points', type=int, metavar='NUM', default=1000,
                        help='Maximal number of points generated for exploration')
    parser.add_argument('--method', type=str, choices=["gradient", "newton"],
                        default="newton", help='Method to run')
    parser.add_argument('--newton-eps', type=float, metavar='NUM', default=0.01,
                        help='Epsilon value for newton method')
    parser.add_argument('--newton-iter', type=int, metavar='NUM', default=10,
                        help='Number of iterations for the newton method')
    parser.add_argument('--gradient-lr', type=int, metavar='NUM', default=0.1,
                        help='Learning rate value for gradient method')
    parser.add_argument('--gradient-iter', type=int, metavar='NUM', default=100,
                        help='Number of iterations for the gradient method')
    parser.add_argument('--print-equs', action='store_true',
                        help='Print the loaded equations')
    parser.add_argument('--csv', action='store_true',
                        help='Store samples into .csv (instead of .npz)')
    args = parser.parse_args(args)

    explorer = Explorer(problem_json=args.problem_json,
                        cuda_enable=args.cuda,
                        to_csv=args.csv,
                        max_points=args.max_points,
                        output_dir=args.output_dir,
                        method=args.method,
                        newton_eps=args.newton_eps,
                        newton_iter=args.newton_iter,
                        gradient_lr=args.gradient_lr,
                        gradient_iter=args.gradient_iter)
    if args.print_equs:
        explorer.print_equations()
    explorer.run()


if __name__ == '__main__':
    main()
