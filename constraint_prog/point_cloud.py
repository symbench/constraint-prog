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
from numpy.lib.recfunctions import unstructured_to_structured
import torch


class PointCloud:
    def __init__(self, sample_vars: List[str], sample_data: torch.Tensor):
        """
        Creates a point table with vars_size many named coordinates.
        The shape of the sample_data must be [*,vars_size].
        """
        assert sample_data.shape[-1] == len(sample_vars)
        self.sample_vars = sample_vars
        self.sample_data = sample_data

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
