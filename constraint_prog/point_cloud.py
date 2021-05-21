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

from collections import Counter
import csv
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sympy
import torch

from constraint_prog.sympy_func import SympyFunc


class PointCloud:
    def __init__(self, float_vars: List[str], float_data: torch.Tensor,
                 string_vars: List[str] = None, string_data: np.ndarray = None):
        """
        Creates a point cloud with num_vars many named coordinates.
        The shape of the float_data must be [num_points, num_vars].
        """
        assert float_data.ndim == 2
        assert float_data.shape[1] == len(float_vars)
        self.float_vars = float_vars
        self.float_data = float_data

        if string_vars is None:
            assert string_data is None
            self.string_vars = []
            self.string_data = np.empty(shape=(float_data.shape[0], 0), dtype=str)
        else:
            assert string_data.ndim == 2
            assert string_data.shape[0] == float_data.shape[0]
            assert string_data.shape[1] == len(string_vars)
            self.string_vars = string_vars
            self.string_data = string_data

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
                data = np.append(self.float_data.detach().float().cpu().numpy(),
                                 self.string_data,
                                 axis=1)

                writer.writerow(header)
                writer.writerows(data)
        elif ext == ".npz":
            np.savez_compressed(file=filename,
                                float_vars=self.float_vars,
                                float_data=self.float_data.detach().float().cpu().numpy(),
                                string_vars=self.string_vars,
                                string_data=self.string_data)
        else:
            raise ValueError("invalid filename extension")

    @staticmethod
    def load(filename: str) -> 'PointCloud':
        """
        Loads the data from the given csv or npz file.
        """
        ext = os.path.splitext(filename)[1]
        if ext == ".csv":
            data = np.loadtxt(fname=filename, delimiter=',', dtype=str)

            string_vars, string_data, float_vars, float_data = [], [], [], []
            for header, col in zip(data[0, :], data[1:, :].T):
                try:
                    float_data.append(col.astype(np.float32))
                    float_vars.append(header)
                except ValueError:
                    string_data.append(col)
                    string_vars.append(header)
            float_data = torch.tensor(np.array(float_data).T)

            if len(string_vars) < 1:
                string_vars, string_data = None, None
            else:
                string_data = np.array(string_data)

            return PointCloud(float_vars=float_vars,
                              float_data=float_data,
                              string_vars=string_vars,
                              string_data=string_data)
        elif ext == ".npz":
            data = np.load(file=filename, allow_pickle=False)

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
    def generate(float_vars: List[str],
                 minimums: List[float], maximums: List[float],
                 num_points: int, device='cpu') -> 'PointCloud':
        """
        Creates a new point cloud with the given number of points and
        bounding box with uniform distribution.
        """
        assert len(float_vars) == len(minimums)
        assert len(float_vars) == len(maximums)

        minimums = torch.tensor(minimums, device=device)
        maximums = torch.tensor(maximums, device=device)
        float_data = torch.rand(size=(num_points, len(float_vars)),
                                device=device)
        float_data = float_data * (maximums - minimums) + minimums
        return PointCloud(float_vars=float_vars,
                          float_data=float_data)

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

        resolutions = torch.tensor(list(resolutions))
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
                          string_data=self.string_data)

    def prune_bounding_box(self, minimums: List[float], maximums: List[float]) -> 'PointCloud':
        """
        Returns those points that lie in the specified bounding box in a new
        point cloud. The shape of the minimums and maximums lists must be of
        shape [num_vars]. If no bound is necessary, then use -inf or inf for
        that value.
        """
        assert len(minimums) == self.num_float_vars and len(maximums) == self.num_float_vars

        sel1 = self.float_data >= torch.tensor(minimums, device=self.device)
        sel2 = self.float_data <= torch.tensor(maximums, device=self.device)
        sel3 = torch.logical_and(sel1, sel2).all(dim=1)

        return PointCloud(float_vars=self.float_vars,
                          float_data=self.float_data[sel3],
                          string_vars=self.string_vars,
                          string_data=self.string_data)

    def prune_by_tolerances(self, magnitudes: 'PointCloud',
                            tolerances: List[float]) -> 'PointCloud':
        """
        Returns those points where the given magnitudes are smaller in absolute
        value than the given tolerances. The length of tolerances must be
        the number of variables of magnitudes, and the the number of points
        of magniteds must match that of this point cloud.
        """
        assert len(tolerances) == magnitudes.num_float_vars
        assert self.num_points == magnitudes.num_points
        sel = magnitudes.float_data.abs() <= torch.tensor(tolerances, device=self.device)
        return PointCloud(float_vars=self.float_vars,
                          float_data=self.float_data[sel.all(dim=1)],
                          string_vars=self.string_vars,
                          string_data=self.string_data)

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
                          float_data=self.float_data[selected, :],
                          string_vars=self.string_vars,
                          string_data=self.string_data)

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

        input_data = []
        for var in func.input_names:
            idx = self.float_vars.index(var)
            input_data.append(self.float_data[:, idx])
        input_data = torch.stack(input_data, dim=1)

        return PointCloud(float_vars=variables,
                          float_data=func(input_data),
                          string_vars=self.string_vars,
                          string_data=self.string_data)

    def projection(self, variables: List[int]) -> 'PointCloud':
        """
        Returns the projection of this point cloud to the specified set of
        variables. The elements of the variables list must be between 0
        and num_vars - 1.
        """
        float_vars = [self.float_vars[idx] for idx in variables]
        float_data = [self.float_data[:, idx] for idx in variables]
        float_data = torch.stack(float_data, dim=1)
        return PointCloud(float_vars=float_vars,
                          float_data=float_data)

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


if __name__ == '__main__':
    if False:
        points = PointCloud.load(filename='elliptic_curve.npz')
        points.print_info()
        points.plot2d(0, 1)
        points = points.prune_close_points(resolutions=[0.1, 0.0, 0.1],
                                           keep=10)
        points.print_info()
        points = points.prune_bounding_box(minimums=[-1, -1, -1],
                                           maximums=[1, 2, 3])
        points.print_info()

    points = PointCloud.load(filename='elliptic_curve.npz')
    # points = PointCloud.generate(["x", "y"], [0, 0], [1, 1], 1000)
    points.print_info()

    points2 = PointCloud(float_vars=points.float_vars,
                         float_data=points.float_data)

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    points = points.evaluate(variables=["x", "y"],
                             expressions=[x, y])
    # points.print_info()
    points = points.prune_pareto_front(directions=[-1.0, -1.0])
    points.print_info()
    points.plot2d(0, 1)

    xpos = torch.linspace(-2.0, 2.0, 50)
    ypos = torch.linspace(-2.0, 0.0, 50)
    mesh = torch.stack(torch.meshgrid(xpos, ypos), dim=-1)
    dist = points.get_pareto_distance(directions=[-1, -1], points=mesh)

    fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    ax1.plot_surface(
        mesh[:, :, 0].numpy(),
        mesh[:, :, 1].numpy(),
        dist.numpy())
    plt.show()
