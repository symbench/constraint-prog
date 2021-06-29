#!/usr/bin/env python3
# Copyright (C) 2021, Miklos Maroti, Zsolt Vizi
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

from collections import Counter
import csv
import math
import os
from typing import Dict, List, Union, Tuple

import matplotlib.pyplot as plt
import numpy
import sympy
import torch

from constraint_prog.sympy_func import SympyFunc
from constraint_prog.newton_raphson import newton_raphson


class PointCloud:
    def __init__(self, float_vars: List[str], float_data: torch.Tensor,
                 string_vars: List[str] = None, string_data: numpy.ndarray = None):
        """
        Creates a point cloud with num_vars many named coordinates.
        The shape of the float_data must be [num_points, num_vars].
        """
        assert float_data.ndim == 2
        assert float_data.shape[1] == len(float_vars)

        self.float_vars = list(float_vars)
        self.float_data = float_data

        if string_vars is None:
            assert string_data is None
            self.string_vars = []
            self.string_data = numpy.empty(shape=(float_data.shape[0], 0), dtype=str)
        else:
            assert string_data.ndim == 2
            assert string_data.shape[0] == float_data.shape[0]
            assert string_data.shape[1] == len(string_vars)
            self.string_vars = list(string_vars)
            self.string_data = string_data

        # check whether all variables are unique
        total_vars = self.float_vars + self.string_vars
        assert len(numpy.unique(total_vars)) == len(total_vars)

    @property
    def num_float_vars(self):
        return len(self.float_vars)

    @property
    def num_string_vars(self):
        return len(self.string_vars)

    @property
    def num_points(self):
        assert self.string_data.shape[0] == self.float_data.shape[0]
        return self.float_data.shape[0]

    @property
    def device(self):
        return self.float_data.device

    @property
    def dictionary(self):
        result = dict()
        for idx, var in enumerate(self.float_vars):
            result[var] = self.float_data[:, idx]
        for idx, var in enumerate(self.string_vars):
            result[var] = self.string_data[:, idx]
        return result

    def __getitem__(self, var: str) -> Union[torch.Tensor, numpy.ndarray]:
        """
        Implements the indexing operator so that the point cloud can be used
        as a dictionary.
        """
        if var in self.string_vars:
            idx = self.string_vars.index(var)
            return self.string_data[:, idx]
        else:
            idx = self.float_vars.index(var)
            return self.float_data[:, idx]

    def __setitem__(self, var: str, data: Union[float, str, torch.Tensor, numpy.ndarray]):
        """
        Implements the indexing operator so that the point cloud can be used
        as a dictionary and a full float or string column can be updated.
        """
        if isinstance(data, str):
            data = numpy.full((self.num_points, ), data)
        elif isinstance(data, float):
            data = torch.full((self.num_points, ), data, dtype=torch.float32, device=self.device)

        # https://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
        if type(data).__module__ == numpy.__name__:
            assert var not in self.float_vars
            data = numpy.expand_dims(data, axis=1)
            if var in self.string_vars:
                idx = self.string_vars.index(var)
                layers = (self.string_data[:, :idx], data, self.string_data[:, idx+1:])
            else:
                self.string_vars.append(var)
                layers = (self.string_data, data)
            # we have to recreate the array because of fixed string sizes
            self.string_data = numpy.concatenate(layers, axis=1)
        elif isinstance(data, torch.Tensor):
            assert var not in self.string_vars
            if var in self.float_vars:
                idx = self.float_vars.index(var)
                self.float_data[:, idx] = data
            else:
                self.float_vars.append(var)
                self.float_data = torch.cat(
                    (self.float_data, torch.unsqueeze(data, dim=-1)),
                    dim=1)

    def print_info(self):
        print("float shape: {}, string shape: {}".format(
            list(self.float_data.shape), list(self.string_data.shape)))
        print("float names: {}, string names: {}".format(
            ', '.join(self.float_vars), ', '.join(self.string_vars)))

    def save(self, filename: str):
        """
        Saves this point cloud to the given filename. The filename extension
        must be either csv or npz.
        """
        ext = os.path.splitext(filename)[1]
        if ext == ".csv":
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                header = []
                header.extend(self.float_vars)
                header.extend(self.string_vars)
                data = numpy.append(self.float_data.detach().float().cpu().numpy(),
                                    self.string_data,
                                    axis=1)

                writer.writerow(header)
                writer.writerows(data)
        elif ext == ".npz":
            numpy.savez_compressed(file=filename,
                                   float_vars=self.float_vars,
                                   float_data=self.float_data.detach().float().cpu().numpy(),
                                   string_vars=self.string_vars,
                                   string_data=self.string_data)
        else:
            raise ValueError("invalid filename extension")

    @staticmethod
    def load(filename: str, delimiter=',') -> 'PointCloud':
        """
        Loads the data from the given csv or npz file.
        """
        ext = os.path.splitext(filename)[1]
        if ext == ".csv":
            data = numpy.loadtxt(fname=filename, delimiter=delimiter, dtype=str)

            string_vars, string_data, float_vars, float_data = [], [], [], []
            for header, col in zip(data[0, :], data[1:, :].T):
                try:
                    float_data.append(col.astype(numpy.float32))
                    float_vars.append(header)
                except ValueError:
                    string_data.append(col)
                    string_vars.append(header)
            float_data = torch.tensor(numpy.array(float_data).T, dtype=torch.float32)

            if len(string_vars) < 1:
                string_vars, string_data = None, None
            else:
                string_data = numpy.array(string_data)

            return PointCloud(float_vars=float_vars,
                              float_data=float_data,
                              string_vars=string_vars,
                              string_data=string_data)
        elif ext == ".npz":
            data = numpy.load(file=filename, allow_pickle=False)

            # Enable backwards compatibility for data files of previous version
            if "sample_vars" in data.files:
                float_vars = list(data["sample_vars"])
                float_data = torch.tensor(data["sample_data"], dtype=torch.float32)
                string_vars, string_data = None, None
            else:
                float_vars = list(data["float_vars"])
                float_data = torch.tensor(data["float_data"], dtype=torch.float32)
                string_vars = list(data["string_vars"])
                string_data = data["string_data"]

            return PointCloud(float_vars=float_vars,
                              float_data=float_data,
                              string_vars=string_vars,
                              string_data=string_data)
        else:
            raise ValueError("invalid filename extension")

    @staticmethod
    def generate(bounds: Dict[str, Tuple[float, float]],
                 num_points: int, device='cpu') -> 'PointCloud':
        """
        Creates a new point cloud with the given number of points and
        bounding box with uniform distribution.
        """

        minimums = torch.tensor([val[0] for val in bounds.values()],
                                dtype=torch.float32, device=device)
        maximums = torch.tensor([val[1] for val in bounds.values()],
                                dtype=torch.float32, device=device)
        float_data = torch.rand((num_points, len(bounds)),
                                dtype=torch.float32, device=device)
        float_data = float_data * (maximums - minimums) + minimums
        return PointCloud(float_vars=bounds.keys(),
                          float_data=float_data)

    @staticmethod
    def mesh_grid(ranges: Dict[str, Tuple[float, float, int]],
                  device='cpu') -> 'PointCloud':
        """
        Creates a new mesh grid point cloud with the given parameters.
        """

        coords = [torch.linspace(a, b, c, dtype=torch.float32, device=device)
                  for (a, b, c) in ranges.values()]
        coords = torch.meshgrid(*coords)
        coords = [torch.flatten(c) for c in coords]
        coords = torch.stack(coords, dim=-1)

        return PointCloud(float_vars=ranges.keys(), float_data=coords)

    def newton_raphson(self, func: 'PointFunc',
                       bounds: Dict[str, Tuple[float, float]],
                       num_iter: int = 10, epsilon: float = 1e-3) \
            -> 'PointCloud':
        """
        Performs the Newton-Raphson optimization on the point cloud where
        the give func specifies the constraints that need to become zero,
        and returns a new point cloud with the result.
        """
        input_data = torch.empty((self.num_points, len(func.input_names)),
                                 dtype=torch.float32, device=self.device)
        bounding_box = torch.empty((2, len(func.input_names)),
                                   dtype=torch.float32, device=self.device)
        for idx, var in enumerate(func.input_names):
            input_data[:, idx] = self[var]
            bound = bounds.get(var, (-math.inf, math.inf))
            bounding_box[0, idx] = bound[0]
            bounding_box[1, idx] = bound[1]

        input_data = newton_raphson(func, input_data,
                                    num_iter=num_iter, epsilon=epsilon,
                                    bounding_box=bounding_box, method="mmclip")

        return PointCloud(func.input_names, input_data)

    def add_mutations(self, stddev: List[float], num_points: int):
        """
        Take random elements from the point cloud and add random mutations
        to the given coordinates so that the total number of points is
        num_points.
        """
        assert len(stddev) == self.num_float_vars
        count = num_points - self.num_points
        if count <= 0:
            return

        indices = torch.randint(0, self.num_points, (count, ), device=self.device)
        mutation = torch.randn((count, self.num_float_vars),
                               dtype=torch.float32, device=self.device)
        mutation = mutation * torch.tensor([stddev],
                                           dtype=torch.float32, device=self.device)
        new_data = self.float_data[indices] + mutation
        self.float_data = torch.cat((self.float_data, new_data), dim=0)

        indices = indices.numpy()
        new_data = self.string_data[indices]
        self.string_data = numpy.concatenate((self.string_data, new_data), axis=0)

    def to_device(self, device="cpu"):
        """
        Moves the sample data to the given device.
        """
        self.float_data = self.float_data.to(device)

    def prune_close_points(self, resolutions: List[float], keep=1) -> 'PointCloud':
        """
        Divides all variables with the given resolution and keeps at most keep
        many elements from each small rectangle in a new point cloud. If a
        resolution value is zero, then those variables do not participate in
        the decisions, so basically we project down only to those variables
        where the resolution is positive. The resolution list must be of shape
        [num_vars].
        """
        assert keep >= 1 and len(resolutions) == self.num_float_vars

        resolutions = torch.tensor(resolutions, dtype=torch.float32)
        multiplier = resolutions.clamp(min=0.0)
        indices = multiplier > 0.0
        multiplier[indices] = 1.0 / multiplier[indices]
        multiplier = multiplier.to(self.device)

        rounded = (self.float_data * multiplier).round().type(torch.int64)
        multiplier = None

        # hash based filtering is not unique, but good enough
        randcoef = torch.randint(-10000000, 10000000, (self.num_float_vars, ),
                                 device=self.device)
        hashnums = (rounded * randcoef).sum(dim=1).cpu()
        rounded = None

        # this is slow, but good enough
        float_data = self.float_data.cpu()
        selected = torch.zeros(float_data.shape[0]).bool()
        counter = Counter()
        for idx in range(float_data.shape[0]):
            value = int(hashnums[idx])
            if counter[value] < keep:
                counter[value] += 1
                selected[idx] = True

        float_data = float_data[selected].to(self.device)
        return PointCloud(float_vars=self.float_vars,
                          float_data=float_data,
                          string_vars=self.string_vars,
                          string_data=self.string_data[selected])

    def prune_bounding_box(self, minimums: List[float], maximums: List[float]) -> 'PointCloud':
        """
        Returns those points that lie in the specified bounding box in a new
        point cloud. The shape of the minimums and maximums lists must be of
        shape [num_vars]. If no bound is necessary, then use -inf or inf for
        that value.
        """
        assert len(minimums) == self.num_float_vars and len(maximums) == self.num_float_vars

        minimums = torch.tensor(minimums, dtype=torch.float32, device=self.device)
        maximums = torch.tensor(maximums, dtype=torch.float32, device=self.device)
        sel1 = self.float_data >= minimums
        sel2 = self.float_data <= maximums
        sel3 = torch.logical_and(sel1, sel2).all(dim=1)

        return PointCloud(float_vars=self.float_vars,
                          float_data=self.float_data[sel3],
                          string_vars=self.string_vars,
                          string_data=self.string_data[sel3])

    def prune_by_tolerances(self, errors: 'PointCloud',
                            tolerances: Union[Dict[str, float], float]) -> 'PointCloud':
        """
        Returns those points where the given error magnitudes are smaller 
        in absolute value than the given tolerances. The tolerances 
        dictionary must contain a value for each error magnitude names or
        it must be a float value which is applied for all errors.
        """
        assert self.num_points == errors.num_points
        if isinstance(tolerances, float):
            tolerances = [tolerances for _ in errors.float_vars]
        else:
            tolerances = [tolerances[var] for var in errors.float_vars]
        tolerances = torch.tensor(tolerances, dtype=torch.float32, device=self.device)
        sel = (errors.float_data.abs() <= tolerances).all(dim=1)
        return PointCloud(float_vars=self.float_vars,
                          float_data=self.float_data[sel],
                          string_vars=self.string_vars,
                          string_data=self.string_data[sel])

    def prune_pareto_front(self, directions: List[float]) -> 'PointCloud':
        """
        Removes all points that are dominated by a better solution on the Pareto
        front. The directions list specifies the direction of each variables, and
        must be of shape [num_vars]. If the direction is negative, then we prefer
        a smaller value all other values being equal (e.g. price). If the direction
        is positive, then we prefer a larger value (e.g. speed). If the direction
        is zero, then that variable does not participate in the Pareto front
        calculation, but their values are kept for selected points.
        """
        assert len(directions) == self.num_float_vars

        # gather data for minimization in all coordinates
        float_data = []
        for idx in range(self.num_float_vars):
            if directions[idx] == 0.0:
                continue
            else:
                data = self.float_data[:, idx]
                float_data.append(data if directions[idx] < 0.0 else -data)
        assert float_data
        float_data = torch.stack(float_data, dim=1)

        # TODO: find a faster algorithm than this quadratic
        selected = torch.ones(self.num_points, dtype=bool)
        for idx in range(self.num_points):
            test1 = (float_data[:idx, :] <= float_data[idx]).all(dim=1)
            test2 = (float_data[:idx, :] != float_data[idx]).any(dim=1)
            test3 = torch.logical_and(test1, test2).any().item()
            test4 = (float_data[idx + 1:, :] <= float_data[idx]).all(dim=1).any().item()
            selected[idx] = not (test3 or test4)

        return PointCloud(float_vars=self.float_vars,
                          float_data=self.float_data[selected],
                          string_vars=self.string_vars,
                          string_data=self.string_data[selected])

    def get_pareto_distance(self, directions: List[float],
                            points: torch.Tensor) -> torch.Tensor:
        """
        Calculates the distance of the given points to the pareto front. The
        shape of the points tensor must be [*,num_vars], and the returned
        tensor is of shape [*]. The returned distances is zero in the dominated
        (feasible) region, and positive in the non-dominated region. The meaning
        of the directions is exactly as in the prune_pareto_dominated method.
        """
        assert len(directions) == self.num_float_vars
        assert points.shape[-1] == self.num_float_vars

        # gather data for minimization in all coordinates
        float_data = []
        points2 = points.to(self.device).unbind(dim=-1)
        points = []
        for idx in range(self.num_float_vars):
            if directions[idx] == 0.0:
                continue
            else:
                data = self.float_data[:, idx]
                float_data.append(data if directions[idx] < 0.0 else -data)
                data = points2[idx]
                points.append(data if directions[idx] < 0.0 else -data)
        assert float_data
        float_data = torch.stack(float_data, dim=1)
        points = torch.stack(points, dim=-1)

        # calculate minimal distance
        points = points.reshape(points.shape[:-1] + (1, points.shape[-1]))
        float_data = (float_data - points).clamp(min=0.0)
        float_data = float_data.pow(2.0).sum(dim=-1).min(dim=-1).values
        return float_data.sqrt()

    def evaluate(self, variables: List[str],
                 expressions: List[sympy.Expr]) -> 'PointCloud':
        """
        Evaluates the given list of expressions on the current points
        and returns a new point cloud with these values. The length of
        the variables list must match that of the expressions. The free
        symbols of the expressions must be among the variables of this
        point cloud.
        """
        assert len(variables) == len(expressions)
        func = SympyFunc(expressions, device=self.device)
        assert all(var in self.float_vars for var in func.input_names)

        # TODO: speed it up, remove stack from here and unbind from SympyFunc
        input_data = []
        for var in func.input_names:
            idx = self.float_vars.index(var)
            input_data.append(self.float_data[:, idx])
        input_data = torch.stack(input_data, dim=1)

        return PointCloud(float_vars=variables,
                          float_data=func(input_data),
                          string_vars=self.string_vars,
                          string_data=self.string_data)

    def extend(self, other: 'PointCould') -> 'PointCloud':
        """
        Extends this point cloud with new columns from the other.
        The number of points in the two cloud must match.
        """
        assert self.num_points == other.num_points
        float_vars = list(self.float_vars) + other.float_vars
        float_data = torch.cat((self.float_data, other.float_data), dim=1)
        string_vars = list(self.string_vars) + other.string_vars
        string_data = numpy.concatenate((self.string_data, other.string_data), axis=1)

        return PointCloud(float_vars=float_vars,
                          float_data=float_data,
                          string_vars=string_vars,
                          string_data=string_data)

    def projection(self, variables: List[str]) -> 'PointCloud':
        """
        Returns the projection of this point cloud to the specified set of
        variables. The elements of the variables list must be names of the
        variables for which the projection is applied.
        """
        float_data = [self.float_data[:, self.float_vars.index(var)]
                      for var in variables]
        float_data = torch.stack(float_data, dim=1)
        return PointCloud(float_vars=variables,
                          float_data=float_data,
                          string_vars=self.string_vars,
                          string_data=self.string_data)

    def plot2d(self, idx1: int, idx2: int, point_size: float = 5.0):
        """
        Plots the 2d projection of the point cloud to the given coordinates.
        """
        assert 0 <= idx1 < self.num_float_vars and 0 <= idx2 < self.num_float_vars
        fig, ax1 = plt.subplots()
        ax1.scatter(
            self.float_data[:, idx1].numpy(),
            self.float_data[:, idx2].numpy(),
            s=point_size)
        ax1.set_xlabel(self.float_vars[idx1])
        ax1.set_ylabel(self.float_vars[idx2])
        plt.show()


class PointFunc(object):
    def __init__(self, exprs: Dict[str, sympy.Expr]):
        self.exprs = exprs
        self.func = SympyFunc(exprs.values())

    @property
    def input_names(self):
        """
        Returns the list of variable names for the input tensor.
        """
        return self.func.input_names

    @property
    def output_names(self):
        """
        Returns the list of variable names for the output tensor.
        """
        return self.exprs.keys()

    def __call__(self, points: Union['PointCloud', torch.Tensor]) \
            -> Union['PointCloud', torch.Tensor]:
        """
        Evaluates the given list of expressions for the given point
        cloud or tensor and returns the result in the same format.
        """
        if isinstance(points, torch.Tensor):
            assert points.ndim == 2 and points.shape[1] == len(self.input_names)
            input_data = points
        else:
            input_data = []
            for var in self.input_names:
                input_data.append(points[var])
            input_data = torch.stack(input_data, dim=-1)

        self.func.device = points.device  # works for both input types
        output_data = self.func(input_data)

        assert output_data.shape[-1] == len(self.output_names)

        if isinstance(points, torch.Tensor):
            return output_data
        else:
            return PointCloud(self.output_names, output_data)

    def __repr__(self):
        return self.exprs.__repr__()
