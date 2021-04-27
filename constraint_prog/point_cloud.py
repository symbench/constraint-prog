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

from typing import List

import csv
import os
import numpy as np
import torch
from collections import Counter


class PointCloud:
    def __init__(self, sample_vars: List[str], sample_data: torch.Tensor):
        """
        Creates a point cloud with num_vars many named coordinates.
        The shape of the sample_data must be [num_points, num_vars].
        """
        assert sample_data.ndim == 2
        assert sample_data.shape[1] == len(sample_vars)
        self.sample_vars = sample_vars
        self.sample_data = sample_data

    @property
    def num_vars(self):
        return len(self.sample_vars)

    @property
    def num_points(self):
        return self.sample_data.shape[0]

    @property
    def device(self):
        return self.sample_data.device

    def print_info(self):
        print("shape: {}".format(list(self.sample_data.shape)))
        print("names: {}".format(', '.join(self.sample_vars)))

    def save(self, filename: str):
        """
        Saves this point cloud to the given filename. The filename extension
        must be either csv or npz.
        """
        ext = os.path.splitext(filename)[1]
        if ext == ".csv":
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.sample_vars)
                writer.writerows(self.sample_data.numpy())
        elif ext == ".npz":
            np.savez_compressed(filename,
                                sample_vars=np.array(self.sample_vars),
                                sample_data=self.sample_data.numpy())
        else:
            raise ValueError("invalid filename extension")

    @staticmethod
    def load(filename: str) -> 'PointCloud':
        """
        Loads the data from the given csv or npz file.
        """
        ext = os.path.splitext(filename)[1]
        if ext == ".csv":
            data = np.genfromtxt(filename, delimiter=',', dtype=np.float32, names=True)
            sample_vars = list(data.dtype.names)
            sample_data = data.view(dtype=np.float32).reshape((-1, len(sample_vars)))
            return PointCloud(sample_vars, torch.tensor(sample_data))
        elif ext == ".npz":
            data = np.load(filename, allow_pickle=False)
            sample_vars = list(data["sample_vars"])
            sample_data = torch.tensor(data["sample_data"])
            return PointCloud(sample_vars, sample_data)
        else:
            raise ValueError("invalid filename extension")

    @staticmethod
    def generate(sample_vars: List[str],
                 minimums: List[float], maximums: List[float],
                 num_points: int, device='cpu') -> 'PointCloud':
        """
        Creates a new point cloud with the given number of points and
        bounding box with uniform distribution.
        """
        assert len(sample_vars) == len(minimums)
        assert len(sample_vars) == len(maximums)

        minimums = torch.tensor(minimums, device=device)
        maximums = torch.tensor(maximums, device=device)
        sample_data = torch.rand(size=(num_points, len(sample_vars)),
                                 device=device)
        sample_data = sample_data * (maximums - minimums) + minimums
        return PointCloud(sample_vars, sample_data)

    def to_device(self, device="cpu"):
        """
        Moves the sample data to the given device.
        """
        self.sample_data = self.sample_data.to(device)

    def prune_close_points(self, resolutions: List[float], keep=1) -> 'PointCloud':
        """
        Divides all variables with the given resolution and keeps at most keep
        many elements from each small rectangle in a new point cloud. If a
        resolution value is zero, then those variables do not participate in
        the decisions, so basically we project down only to those variables
        where the resolution is positive. The resolution list must be of shape
        [num_vars].
        """
        assert keep >= 1 and len(resolutions) == self.num_vars

        resolutions = torch.tensor(list(resolutions))
        multiplier = resolutions.clamp(min=0.0)
        indices = multiplier > 0.0
        multiplier[indices] = 1.0 / multiplier[indices]
        multiplier = multiplier.to(self.device)

        rounded = (self.sample_data * multiplier).round().type(torch.int64)
        multiplier = None

        # hash based filtering is not unique, but good enough
        randcoef = torch.randint(-10000000, 10000000, (self.num_vars, ),
                                 device=self.device)
        hashnums = (rounded * randcoef).sum(dim=1).cpu()
        rounded = None

        # this is slow, but good enough
        sample_data = self.sample_data.cpu()
        selected = torch.zeros(sample_data.shape[0]).bool()
        counter = Counter()
        for idx in range(sample_data.shape[0]):
            value = int(hashnums[idx])
            if counter[value] < keep:
                counter[value] += 1
                selected[idx] = True

        sample_data = sample_data[selected].to(self.device)
        return PointCloud(self.sample_vars, sample_data)

    def prune_bounding_box(self, minimums: List[float], maximums: List[float]) -> 'PointCloud':
        """
        Returns those points that lie in the specified bounding box in a new
        point cloud. The shape of the minimums and maximums lists must be of
        [vars_size]. If no bound is necessary, then use -inf or inf for that
        value.
        """
        assert len(minimums) == self.num_vars and len(maximums) == self.num_vars

        sel1 = self.sample_data >= torch.tensor(minimums, device=self.device)
        sel2 = self.sample_data <= torch.tensor(maximums, device=self.device)
        sel3 = torch.logical_and(sel1, sel2).all(dim=1)

        return PointCloud(self.sample_vars, self.sample_data[sel3])


if __name__ == '__main__':
    points = PointCloud.load('elliptic_curve.npz')
    points.print_info()
    points = points.prune_close_points([0.1, 0.0, 0.1], keep=10)
    points.print_info()
    points = points.prune_bounding_box([-1, -1, -1], [1, 2, 3])
    points.print_info()
    points = PointCloud.generate(['x', 'y', 'z'], [0, 0, 0], [1, 1, 1], 1000)
    points.print_info()
