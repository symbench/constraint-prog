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
from typing import Dict, List, Union, Tuple, Optional

import matplotlib.pyplot as plt
import numpy
import sympy
import torch

from constraint_prog.sympy_func import SympyFunc
from constraint_prog.newton_raphson import newton_raphson


class PointCloud:
    def __init__(self, float_vars: Optional[List[str]],
                 float_data: Optional[torch.Tensor] = None,
                 string_vars: Optional[List[str]] = None,
                 string_data: Optional[numpy.ndarray] = None):
        """
        Creates a point cloud with num_vars many named coordinates.
        The shape of the float_data must be [num_points, len(float_vars)], and
        the shape of the string_data must be [num_points, len(string_vars)].
        """

        if float_vars is None:
            assert float_data is None and string_data.ndim == 2
            self.float_vars = []
            self.float_data = torch.empty(
                (string_data.shape[0], 0), dtype=torch.float32)
        else:
            assert float_data.ndim == 2
            assert float_data.shape[1] == len(float_vars)
            self.float_vars = list(float_vars)
            self.float_data = float_data

        if string_vars is None:
            assert string_data is None
            self.string_vars = []
            self.string_data = numpy.empty(
                shape=(float_data.shape[0], 0), dtype=str)
        else:
            assert string_data.ndim == 2
            assert string_data.shape[1] == len(string_vars)
            self.string_vars = list(string_vars)
            self.string_data = string_data

        assert self.string_data.shape[0] == self.float_data.shape[0]

        # check whether all variables are unique
        total_vars = self.float_vars + self.string_vars
        if len(numpy.unique(total_vars)) != len(total_vars):
            elems = list(total_vars)
            for e in numpy.unique(total_vars):
                elems.remove(e)
            raise ValueError("multiple keys: " + ", ".join(elems))

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

    def row(self, index: int) -> Dict[str, Union[float, str]]:
        """
        Returns a dictionary containing the given design from the
        point cloud.
        """
        assert 0 <= index < self.num_points
        result = dict()
        for idx, var in enumerate(self.float_vars):
            result[var] = self.float_data[index, idx].item()
        for idx, var in enumerate(self.string_vars):
            result[var] = str(self.string_data[index, idx])
        return result

    def __getitem__(self, var: str) -> Union[torch.Tensor, numpy.ndarray]:
        """
        Implements the indexing operator so that the point cloud can be used
        as a dictionary returning a full column for the given variable.
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
            data = torch.full((self.num_points, ), data,
                              dtype=torch.float32, device=self.device)

        # https://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
        if type(data).__module__ == numpy.__name__:
            assert var not in self.float_vars
            data = numpy.expand_dims(data, axis=1)
            if var in self.string_vars:
                idx = self.string_vars.index(var)
                layers = (self.string_data[:, :idx],
                          data, self.string_data[:, idx+1:])
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
            ', '.join(sorted(self.float_vars)),
            ', '.join(sorted(self.string_vars))))

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
            # TODO: find a better way to get the dtype and its max size
            data = numpy.loadtxt(
                fname=filename, delimiter=delimiter, dtype="<U99")

            string_vars, string_data, float_vars, float_data = [], [], [], []
            for header, col in zip(data[0, :], data[1:, :].T):
                try:
                    float_data.append(col.astype(numpy.float32))
                    float_vars.append(header)
                except ValueError as err:
                    # print(err)
                    string_data.append(col)
                    string_vars.append(header)

            print("Float variables:", ", ".join(sorted(float_vars)))
            print("String variables:", ", ".join(sorted(string_vars)))

            if not string_vars:
                string_vars, string_data = None, None
            else:
                string_data = numpy.array(string_data).transpose()

            if not float_vars:
                float_vars, float_data = None, None
            else:
                float_data = torch.tensor(numpy.array(
                    float_data).T, dtype=torch.float32)

            count = torch.isnan(float_data).sum().item()
            if count:
                print("WARNING: input data countains",
                      count, "many NAN entries")

            count = torch.isinf(float_data).sum().item()
            if count:
                print("WARNING: input data countains",
                      count, "many INF entries")

            return PointCloud(float_vars=float_vars,
                              float_data=float_data,
                              string_vars=string_vars,
                              string_data=string_data)
        elif ext == ".npz":
            data = numpy.load(file=filename, allow_pickle=False)

            # Enable backwards compatibility for data files of previous version
            if "sample_vars" in data.files:
                float_vars = list(data["sample_vars"])
                float_data = torch.tensor(
                    data["sample_data"], dtype=torch.float32)
                string_vars, string_data = None, None
            else:
                float_vars = list(data["float_vars"])
                float_data = torch.tensor(
                    data["float_data"], dtype=torch.float32)
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
        the given func specifies the constraints that need to become zero,
        and returns a new point cloud with the result.
        """
        assert bounds.keys() <= set(self.float_vars)
        input_data = torch.empty((self.num_points, len(func.input_names)),
                                 dtype=torch.float32, device=self.device)
        bounding_box = torch.empty((2, len(func.input_names)),
                                   dtype=torch.float32, device=self.device)
        for idx, var in enumerate(func.input_names):
            input_data[:, idx] = self[var]
            bound = bounds.get(var, (-math.inf, math.inf))
            assert isinstance(bound[0], float) and isinstance(bound[1], float)
            bounding_box[0, idx] = bound[0]
            bounding_box[1, idx] = bound[1]

        input_data = newton_raphson(func, input_data,
                                    num_iter=num_iter, epsilon=epsilon,
                                    bounding_box=bounding_box, method="mmclip")

        return PointCloud(func.input_names, input_data)

    def add_mutations(self, stddevs: Union[Dict[str, float], float],
                      num_points: int, multiplier: float = 1.0):
        """
        Take random elements from the point cloud and add random mutations
        to the given coordinates so that the total number of points is
        num_points. If a coordinate is not listed in the dictionary, then
        that value will not change.
        """
        count = num_points - self.num_points
        if count <= 0 or self.num_points <= 0:
            return

        if isinstance(stddevs, float):
            stddevs = torch.full((self.num_float_vars, ), stddevs * multiplier,
                                 dtype=torch.float32, device=self.device)
        else:
            assert stddevs.keys() <= set(self.float_vars)
            stddevs = torch.tensor(
                [stddevs.get(var, 0.0) *
                 multiplier for var in self.float_vars],
                dtype=torch.float32, device=self.device)

        indices = torch.randint(
            0, self.num_points, (count, ), device=self.device)
        mutation = torch.randn((count, self.num_float_vars),
                               dtype=torch.float32, device=self.device)
        new_data = self.float_data[indices] + mutation * stddevs
        self.float_data = torch.cat((self.float_data, new_data), dim=0)

        new_data = self.string_data[indices.numpy()]
        self.string_data = numpy.concatenate(
            (self.string_data, new_data), axis=0)

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
                          string_data=self.string_data[selected.numpy()])

    def prune_close_points2(self, resolutions: Union[float, Dict[str, float]],
                            keep=1) -> 'PointCloud':
        if isinstance(resolutions, float):
            resolutions = [resolutions] * self.num_float_vars
        else:
            resolutions = [resolutions.get(var, 0.0)
                           for var in self.float_vars]
        return self.prune_close_points(resolutions, keep)

    def prune_bounding_box(self, bounds: Dict[str, Tuple[float, float]]) -> 'PointCloud':
        """
        Returns those points that lie in the specified bounding box. If a specific
        variable is not listed in the dictionary, then that value will not be used
        in the pruning process.
        """
        assert bounds.keys() <= set(self.float_vars)
        minimums = torch.tensor(
            [bounds[var][0] if var in bounds else -
                math.inf for var in self.float_vars],
            dtype=torch.float32, device=self.device)
        maximums = torch.tensor(
            [bounds[var][1] if var in bounds else math.inf for var in self.float_vars],
            dtype=torch.float32, device=self.device)

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
        tolerances = torch.tensor(
            tolerances, dtype=torch.float32, device=self.device)
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

        pareto_data = torch.empty_like(float_data)
        pareto_idxs = torch.empty(self.num_points, dtype=int)
        pareto_count = 0

        for idx in range(self.num_points):
            test = (pareto_data[:pareto_count] <= float_data[idx]).all(dim=1)
            if test.any().item():
                continue

            test = (pareto_data[:pareto_count] < float_data[idx]).any(dim=1)
            count = test.count_nonzero().item()
            if count < pareto_count:
                pareto_data[:count] = pareto_data[:pareto_count][test]
                pareto_idxs[:count] = pareto_idxs[:pareto_count][test]
                pareto_count = count

            pareto_idxs[pareto_count] = idx
            pareto_data[pareto_count] = float_data[idx]
            pareto_count += 1

        selected = torch.zeros(self.num_points, dtype=bool)
        for idx in pareto_idxs[:pareto_count]:
            selected[idx] = True

        return PointCloud(float_vars=self.float_vars,
                          float_data=self.float_data[selected],
                          string_vars=self.string_vars,
                          string_data=self.string_data[selected])

    def prune_pareto_front2(self, directions: Dict[str, float]) -> 'PointCloud':
        """
        The same functionality as above, but the directions are specified by
        a dictionary.
        """
        dirs = [0.0] * self.num_float_vars
        for var, val in directions.items():
            assert var in self.float_vars
            idx = self.float_vars.index(var)
            dirs[idx] = val
        return self.prune_pareto_front(dirs)

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
        if input_data:
            input_data = torch.stack(input_data, dim=1)
        else:
            input_data = torch.empty((self.num_points, 0),
                                     dtype=torch.float32, device=self.device)

        return PointCloud(float_vars=variables,
                          float_data=func(input_data),
                          string_vars=self.string_vars,
                          string_data=self.string_data)

    def extend(self, other: 'PointCloud') -> 'PointCloud':
        """
        Extends this point cloud with new columns from the other.
        The number of points in the two cloud must match.
        """
        assert self.num_points == other.num_points
        assert set(self.float_vars).isdisjoint(set(other.float_vars))
        assert set(self.string_vars).isdisjoint(set(other.string_vars))
        float_vars = list(self.float_vars) + other.float_vars
        float_data = torch.cat((self.float_data, other.float_data), dim=1)
        string_vars = list(self.string_vars) + other.string_vars
        string_data = numpy.concatenate(
            (self.string_data, other.string_data), axis=1)

        return PointCloud(float_vars=float_vars,
                          float_data=float_data,
                          string_vars=string_vars,
                          string_data=string_data)

    def concat(self, other: 'PointCloud') -> 'PointCloud':
        """
        Concatenates this point cloud with the other and returns a new point
        cloud that has the data from both. The float and string variables must
        be the same in the two datasets.
        """
        assert self.float_vars == other.float_vars
        assert self.string_vars == other.string_vars
        return PointCloud(
            float_vars=self.float_vars,
            float_data=torch.cat((self.float_data, other.float_data), dim=0),
            string_vars=self.string_vars,
            string_data=numpy.concatenate(
                (self.string_data, other.string_data), axis=0)
        )

    def append(self, row: Dict[str, Union[float, str]]):
        """
        Adds a new row to the point cloud. The row dictionary must contain
        a value for each variables.
        """
        float_row = [float(row[var]) for var in self.float_vars]
        float_row = torch.tensor([float_row], dtype=torch.float32)
        self.float_data = torch.cat((self.float_data, float_row), axis=0)

        string_row = [str(row[var]) for var in self.string_vars]
        string_row = numpy.array([string_row], dtype=str)
        self.string_data = numpy.concatenate(
            (self.string_data, string_row), axis=0)

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

    def plot2d(self, var1: str, var2: str, point_size: float = 50.0,
               highlight: List[str] = None):
        """
        Plots the 2d projection of the point cloud to the given coordinates.
        """
        assert var1 in self.float_vars and var2 in self.float_vars

        fig, ax1 = plt.subplots()

        if highlight:
            displayed = numpy.zeros(shape=(self.num_points,), dtype=bool)
            for val in highlight:
                selected = numpy.zeros(shape=(self.num_points,), dtype=bool)
                for var in self.string_vars:
                    selected[self[var] == val] = True

                ax1.scatter(
                    x=self[var1].numpy()[selected],
                    y=self[var2].numpy()[selected],
                    label=val,
                    s=point_size)

                displayed = numpy.logical_or(displayed, selected)

            selected = numpy.logical_not(displayed)
            ax1.scatter(
                x=self[var1].numpy()[selected],
                y=self[var2].numpy()[selected],
                label="Everything else",
                s=point_size)

        else:
            ax1.scatter(
                x=self[var1].numpy(),
                y=self[var2].numpy(),
                s=point_size)

        ax1.set_xlabel(var1)
        ax1.set_ylabel(var2)
        # ax1.set_xlabel("Propeller thrust / eletrical power (N/W)")
        # ax1.set_ylabel("Motor and propeller mass (kg)")
        # ax1.set_title("Thrust / power vs mass Pareto-front")

        ax1.grid()
        # ax1.legend()

        plt.show()

    def plot3d(self, var1: str, var2: str, var3: str,
               point_size: float = 50.0, highlight: List[str] = None):
        """
        Plots the 3d projection of the point cloud to the given coordinates.
        """
        assert var1 in self.float_vars and var2 in self.float_vars and var3 in self.float_vars

        if highlight:
            colors = numpy.full(shape=(self.num_points,),
                                fill_value="blue", dtype=str)
            for var in self.string_vars:
                for val in highlight:
                    colors[self[var] == val] = "red"
        else:
            colors = None

        fig = plt.figure()
        ax1 = fig.add_subplot(projection='3d')
        ax1.scatter(
            self[var1].numpy(),
            self[var2].numpy(),
            self[var3].numpy(),
            # c=colors,
            c=self[var3].numpy(),
            cmap="jet",
            s=point_size)

        ax1.set_xlabel(var1)
        ax1.set_ylabel(var2)
        ax1.set_zlabel(var3)
        # ax1.set_xlabel("Motor and propeller mass (kg)")
        # ax1.set_ylabel("Eletrical power (W)")
        # ax1.set_zlabel("Propeller thrust (N)")
        # ax1.set_title("Mass vs power vs thrust Pareto-front")

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

    def __call__(self, points: Union['PointCloud', torch.Tensor],
                 equs_as_float: bool = True) \
            -> Union['PointCloud', torch.Tensor]:
        """
        Evaluates the given list of expressions for the given point
        cloud or tensor and returns the result in the same format.
        """
        if isinstance(points, torch.Tensor):
            assert points.ndim == 2 and points.shape[1] == len(
                self.input_names)
            input_data = points
        else:
            input_data = []
            for var in self.input_names:
                input_data.append(points[var])
            input_data = torch.stack(input_data, dim=-1)

        self.func.device = points.device  # works for both input types
        output_data = self.func(input_data, equs_as_float=equs_as_float)

        assert output_data.shape[-1] == len(self.output_names)

        if isinstance(points, torch.Tensor):
            return output_data
        else:
            return PointCloud(self.output_names, output_data)

    def __repr__(self):
        return self.exprs.__repr__()


def run_pareto_front(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', type=str,  nargs="+", metavar='FILE',
                        help='a CSV or NPZ file to read')
    parser.add_argument('--info', action='store_true',
                        help="prints out the variables of the data")
    parser.add_argument('--pos', type=str, nargs="*", metavar='VAR', default=[],
                        help='selects positive variables')
    parser.add_argument('--neg', type=str, nargs="*", metavar='VAR', default=[],
                        help='selects positive variables')
    parser.add_argument('--min', type=str, nargs="*", metavar='NV', default=[],
                        help='prune by minimum values, (name, value) pairs')
    parser.add_argument('--max', type=str, nargs="*", metavar='NV', default=[],
                        help='prune by maximum values, (name, value) pairs')
    parser.add_argument('--save', type=str, metavar='FILE',
                        help='save the pruned dataset to this file')
    args = parser.parse_args(args)

    points = None
    for file in args.file:
        print("Reading", file)
        if points is None:
            points = PointCloud.load(file)
        else:
            points = points.concat(PointCloud.load(file))

    print("Loaded", points.num_points, "designs")

    if args.info:
        points.print_info()

    if args.min or args.max:
        if len(args.max) % 2 != 0:
            raise ValueError(
                "you must have an event number of parameters to --max")

        if len(args.min) % 2 != 0:
            raise ValueError(
                "you must have an event number of parameters to --min")

        bounds = dict()

        for idx in range(0, len(args.min), 2):
            var = args.min[idx]
            if var not in points.float_vars:
                raise ValueError("invalid variable: " + var)
            val = float(args.min[idx + 1])
            bounds.setdefault(var, [-math.inf, math.inf])[0] = val

        for idx in range(0, len(args.max), 2):
            var = args.max[idx]
            if var not in points.float_vars:
                raise ValueError("invalid variable: " + var)
            val = float(args.max[idx + 1])
            bounds.setdefault(var, [-math.inf, math.inf])[1] = val

        print("Pruning by bounding box, please wait...")
        points = points.prune_bounding_box(bounds)
        print("After prooning we have", points.num_points, "designs")

    if args.pos or args.neg:
        for var in args.pos:
            if var not in points.float_vars:
                raise ValueError("invalid variable: " + var)
        for var in args.neg:
            if var not in points.float_vars:
                raise ValueError("invalid variable: " + var)

        dirs = []
        for var in points.float_vars:
            if var in args.pos:
                dirs.append(1.0)
            elif var in args.neg:
                dirs.append(-1.0)
            else:
                dirs.append(0)

        print("Positive variables:", ", ".join(args.pos))
        print("Negative variables:", ", ".join(args.neg))
        print("Pruning to the pareto front, please wait...")
        points = points.prune_pareto_front(dirs)
        print("After pruning we have", points.num_points, "designs")

    if args.save:
        print("Writing", args.save)
        points.save(args.save)


def run_plot(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', type=str,  nargs="+", metavar='FILE',
                        help='a CSV or NPZ file to read')
    parser.add_argument('--info', action='store_true',
                        help="prints out the variables of the data")
    parser.add_argument('--var', type=str, nargs="*", metavar='VAR', default=[],
                        help='selects variables to be plotted')
    parser.add_argument('--highlight', type=str, nargs="*", metavar='NAME', default=[],
                        help='highlight points that have this string value')
    args = parser.parse_args(args)

    points = None
    for file in args.file:
        print("Reading", file)
        if points is None:
            points = PointCloud.load(file)
        else:
            points = points.concat(PointCloud.load(file))

    print("Loaded", points.num_points, "designs")

    if args.info:
        points.print_info()

    for var in args.var:
        assert var in points.float_vars

    if len(args.var) == 2:
        points.plot2d(args.var[0], args.var[1], highlight=args.highlight)
    elif len(args.var) == 3:
        points.plot3d(args.var[0], args.var[1], args.var[2],
                      highlight=args.highlight)
    else:
        print("Invalid number of variables selected")
