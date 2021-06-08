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
import importlib.util
from inspect import getmembers
import json
import os

import numpy as np
import torch
import sympy

from constraint_prog.sympy_func import SympyFunc, Scaler
from constraint_prog.gradient_descent import gradient_descent
from constraint_prog.newton_raphson import newton_raphson
from constraint_prog.point_cloud import PointCloud


class ProblemLoader:
    def __init__(self, problem_json):
        self.problem_json = problem_json
        with open(self.problem_json) as f:
            self.json_content = json.loads(f.read())

        self.equations = dict()
        self.expressions = dict()
        self.get_equations_and_expressions()

        self.fixed_values = dict()
        self.constraints = self.process_constraints()

    def process_constraints(self) -> dict:
        # disregard entries that start with a dash
        constraints = {key: val
                       for (key, val) in self.json_content["variables"].items()
                       if not key.startswith('-')}

        # collect fixed values and substitute them into the equations
        self.fixed_values = {key: val["min"]
                             for (key, val) in constraints.items()
                             if val["min"] == val["max"]}
        for key, val in self.fixed_values.items():
            print("Fixing {} to {}".format(key, val))
            for val2 in self.equations.values():
                val2["expr"] = val2["expr"].subs(sympy.Symbol(key), val)
            self.expressions = {key2: val2.subs(sympy.Symbol(key), val)
                                for (key2, val2) in self.expressions.items()}
            del constraints[key]

        return constraints

    def get_equations_and_expressions(self) -> None:
        path = os.path.join(
            os.path.dirname(self.problem_json), self.json_content["source"])
        print("Loading python file:", path)
        if not os.path.exists(path):
            raise FileNotFoundError()

        spec = importlib.util.spec_from_file_location("equmod", path)
        equmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(equmod)
        members = getmembers(equmod)

        def find_member(name_):
            for member_ in members:
                if member_[0] == name_:
                    return member_[1]
            return None

        self.equations.clear()
        for (name, conf) in self.json_content["equations"].items():
            if name.startswith('-'):
                continue
            member = find_member(name)
            if member is None:
                raise ValueError("equation " + name + " not found")
            assert (isinstance(member, sympy.Eq) or
                    isinstance(member, sympy.LessThan) or
                    isinstance(member, sympy.StrictLessThan) or
                    isinstance(member, sympy.GreaterThan) or
                    isinstance(member, sympy.StrictGreaterThan))

            self.equations[name] = {
                "expr": member,
                "tolerance": conf["tolerance"]
            }

        self.expressions.clear()
        for name in self.json_content["expressions"]:
            if name.startswith('-'):
                continue
            member = find_member(name)
            if member is None:
                raise ValueError("expression " + name + " not found")
            self.expressions[name] = member


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
        self.constraints_min = [constraints[var]["min"] for var in self.func.input_names]
        self.constraints_max = [constraints[var]["max"] for var in self.func.input_names]
        self.constraints_res = [constraints[var]["resolution"] for var in self.func.input_names]

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
        """
        Runs exploration w.r.t. the given constraints
        """
        # Generate sample points
        sample_data = PointCloud.generate(float_vars=self.func.input_names,
                                          minimums=self.constraints_min,
                                          maximums=self.constraints_max,
                                          num_points=self.max_points)
        print("Running {} method on {} many design points".format(
            self.method, sample_data.num_points))

        # Run solver
        output_data = self.run_solver(
            samples=sample_data)

        # Prune by tolerances
        magnitudes = output_data.evaluate(
            variables=list(self.equations.keys()),
            expressions=[equ["expr"] for equ in self.equations.values()])
        tolerances = list(self.tolerances.detach().cpu().numpy())
        output_data = output_data.prune_by_tolerances(
            magnitudes=magnitudes,
            tolerances=tolerances)
        print("After checking tolerances we have {} designs".format(
            output_data.num_points))

        # Prune close points
        output_data = output_data.prune_close_points(
            resolutions=self.constraints_res)
        print("After pruning close points we have {} designs".format(
            output_data.num_points))

        # Prune bounding box
        output_data = output_data.prune_bounding_box(
            minimums=self.constraints_min,
            maximums=self.constraints_max)
        print("After bounding box pruning we have {} designs".format(
            output_data.num_points))

        # Save results
        filename = os.path.splitext(os.path.basename(self.problem_json))[0] + \
            (".csv" if self.to_csv else ".npz")
        file_name = os.path.join(os.path.abspath(self.output_dir), filename)
        print("Saving generated design points to:", file_name)

        output_data = self.extend_output_data(output_data=output_data)
        output_data.save(filename=file_name)

    def run_solver(self, samples: PointCloud) -> PointCloud:
        """
        Runs selected solver method
        :param PointCloud samples: generated sample for the exploration
        :return PointCloud: output of the iterative solver
        """
        input_data = samples.float_data.to(self.device)
        output_data = None
        if self.method == "newton":
            bounding_box = torch.cat(
                (torch.tensor([self.constraints_min], device=self.device),
                 torch.tensor([self.constraints_max], device=self.device)))
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

        return PointCloud(float_vars=self.func.input_names,
                          float_data=output_data)

    def extend_output_data(self, output_data: PointCloud) -> PointCloud:
        """
        Extends output data by fixed values and evaluation of given expressions
        :param PointCloud output_data: pruned and finalized result of the exploration
        :return PointCloud: extended output data
        """
        # Append fixed values to samples
        samples = output_data.float_data
        float_vars = list(self.func.input_names)
        float_vars.extend(self.fixed_values.keys())
        float_vars.extend(self.expressions.keys())
        float_data = samples.detach().cpu().numpy()
        columns_fixed_values = np.repeat(
            np.array([list(self.fixed_values.values())]),
            float_data.shape[0],
            axis=0
        )
        float_data = np.concatenate(
            (float_data, columns_fixed_values), axis=1
        )
        for name, expr in self.expressions.items():
            try:
                columns_expressions = self.func.evaluate(
                    [expr], samples, equs_as_float=False).detach().cpu().numpy()
            except ValueError as err:
                raise Exception("Expression " + name + " cannot be evaluated: " + str(err))
            float_data = np.concatenate((float_data, columns_expressions), axis=1)
        return PointCloud(float_vars=float_vars,
                          float_data=torch.tensor(float_data, device=self.device))


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
