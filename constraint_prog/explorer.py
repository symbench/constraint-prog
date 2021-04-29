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
from copy import deepcopy
import csv
import json
import os

import numpy as np
import torch

from constraint_prog.sympy_func import SympyFunc, Scaler
from constraint_prog.gradient_descent import gradient_descent
from constraint_prog.newton_raphson import newton_raphson
from constraint_prog.problem_loader import ProblemLoader


class Explorer:
    def __init__(self, problem_json: str, output_dir: str,
                 cuda_enable: bool, to_csv: bool,
                 max_points: int, method: str,
                 newton_eps: float, newton_iter: int, newton_bbox: str,
                 gradient_lr: float, gradient_iter: int):
        self.problem_json = problem_json
        self.output_dir = output_dir
        self.to_csv = to_csv
        self.max_points = max_points
        self.method = method

        self.newton_eps = newton_eps
        self.newton_iter = newton_iter
        self.newton_bbox = newton_bbox

        self.gradient_lr = gradient_lr
        self.gradient_iter = gradient_iter

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and cuda_enable else "cpu")

        with open(problem_json) as f:
            self.json_content = json.loads(f.read())

        problem = ProblemLoader(problem_json=problem_json)
        self.equations = problem.equations
        self.expressions = problem.expressions
        self.fixed_values = problem.fixed_values
        constraints = problem.constraints

        self.func = SympyFunc(
            expressions=[equ["expr"] for equ in self.equations.values()],
            device=self.device)

        # disregard entries that are unused in any equations
        for unused_key in np.setdiff1d(sorted(constraints.keys()), self.func.input_names):
            del constraints[unused_key]

        assert sorted(constraints.keys()) == self.func.input_names
        self.constraints_min = torch.tensor(
            [[constraints[var]["min"] for var in self.func.input_names]],
            device=self.device)
        self.constraints_max = torch.tensor(
            [[constraints[var]["max"] for var in self.func.input_names]],
            device=self.device)
        self.constraints_res = torch.tensor(
            [constraints[var]["resolution"]for var in self.func.input_names],
            device=self.device)

        self.tolerances = torch.tensor(
            [equ["tolerance"] for equ in self.equations.values()],
            device=self.device)
        self.tolerances *= self.json_content["global_tolerance"]
        self.scaled_func = Scaler(self.func, 1.0 / self.tolerances)

    def print_equations(self) -> None:
        print("Loaded equations:")
        for eq in self.equations:
            print(eq)
        print("Variable names:", ', '.join(self.func.input_names))

    def run(self) -> None:
        input_data = self.generate_input().to(self.device)

        print("Running {} method on {} many design points".format(
            self.method, input_data.shape[0]))
        output_data = None
        if self.method == "newton":
            bounding_box = torch.cat(
                (self.constraints_min, self.constraints_max))
            output_data = newton_raphson(func=self.scaled_func,
                                         input_data=input_data,
                                         num_iter=self.newton_iter,
                                         epsilon=self.newton_eps,
                                         bounding_box=bounding_box,
                                         method=self.newton_bbox)
        elif self.method == "gradient":
            output_data = gradient_descent(f=self.scaled_func,
                                           in_data=input_data,
                                           it=self.gradient_iter,
                                           lrate=self.gradient_lr,
                                           device=self.device)

        output_data = self.check_tolerance(output_data)
        print("After checking tolerances we have {} designs".format(
            output_data.shape[0]))

        output_data = self.prune_close_points2(output_data)
        print("After pruning close points we have {} designs".format(
            output_data.shape[0]))

        output_data = self.prune_bounding_box(output_data)
        print("After bounding box pruning we have {} designs".format(
            output_data.shape[0]))

        self.save_data(output_data)

    def save_data(self, samples: torch.tensor) -> None:
        filename = os.path.splitext(os.path.basename(self.problem_json))[0] + \
            (".csv" if self.to_csv else ".npz")
        file_name = os.path.join(os.path.abspath(self.output_dir), filename)
        print("Saving generated design points to:", file_name)

        # Append fixed values to samples
        sample_vars = deepcopy(self.func.input_names)
        sample_vars.extend(list(self.fixed_values.keys()))
        sample_vars.extend(list(self.expressions.keys()))
        sample_data = samples.detach().cpu().numpy()
        columns_fixed_values = np.repeat(
            np.array([list(self.fixed_values.values())]),
            sample_data.shape[0],
            axis=0
        )
        sample_data = np.concatenate(
            (sample_data, columns_fixed_values), axis=1
        )

        for name, expr in self.expressions.items():
            try:
                columns_expressions = self.func.evaluate(
                    [expr], samples, equs_as_float=False).detach().cpu().numpy()
            except ValueError as err:
                raise Exception("Expression " + name + " cannot be evaluated: " + str(err))
            sample_data = np.concatenate((sample_data, columns_expressions), axis=1)

        if self.to_csv:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(sample_vars)
                writer.writerows(sample_data)
        else:
            np.savez_compressed(file_name,
                                sample_vars=sample_vars,
                                sample_data=sample_data)

    def generate_input(self) -> torch.tensor:
        """Generates input data with uniform distribution."""
        sample = torch.rand(
            size=(self.max_points, self.func.input_size),
            device=self.device)
        sample *= (self.constraints_max - self.constraints_min)
        sample += self.constraints_min
        return sample

    def check_tolerance(self, samples: torch.tensor) -> torch.tensor:
        """
        Evaluates the given list of designs and returns a new list for
        which the absolute errors is less than the tolerance.
        """
        equation_output = self.func(samples)
        good_point_idx = (equation_output.abs() < self.tolerances).all(dim=-1)
        return samples[good_point_idx]

    def prune_close_points2(self, samples: torch.tensor) -> torch.tensor:
        assert samples.ndim == 2 and samples.shape[1] == len(self.constraints_res)
        rounded = samples / self.constraints_res.reshape((1, samples.shape[1]))
        rounded = rounded.round().type(torch.int64)

        # hash based filtering is not unique, but good enough
        randcoef = torch.randint(
            -10000000, 10000000, (1, samples.shape[1]),
            device=self.device)
        hashnums = (rounded * randcoef).sum(dim=1).cpu()

        # this is slow, but good enough
        selected = torch.zeros(samples.shape[0]).bool()
        hashset = set()
        for idx in range(samples.shape[0]):
            value = int(hashnums[idx])
            if value not in hashset:
                hashset.add(value)
                selected[idx] = True

        return samples[selected.to(samples.device)]

    def prune_bounding_box(self, samples: torch.tensor) -> torch.tensor:
        assert samples.ndim == 2
        selected = torch.logical_and(samples >= self.constraints_min,
                                     samples <= self.constraints_max)
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
    parser.add_argument('--newton-bbox', type=str, default="minmax",
                        choices=["none", "clip", "minmax", "mmclip"],
                        help='Bounding box calculation method')
    parser.add_argument('--gradient-lr', type=float, metavar='NUM', default=0.1,
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
                        newton_bbox=args.newton_bbox,
                        newton_iter=args.newton_iter,
                        gradient_lr=args.gradient_lr,
                        gradient_iter=args.gradient_iter)
    if args.print_equs:
        explorer.print_equations()
    explorer.run()


if __name__ == '__main__':
    main()
