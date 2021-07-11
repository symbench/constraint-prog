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

from typing import Dict, List, Union

import numpy as np
import torch

from constraint_prog.point_cloud import PointCloud


class PointCloud2:
    def __init__(self, data: Dict[str, Union[torch.Tensor, np.ndarray]]) -> None:
        self.data = data
        assert self.data  # not empty

        self.num_points = None
        for val in data.values():
            assert isinstance(data, torch.Tensor) or type(data).__module__ == np.__name__
            num = data.shape[0]
            if self.num_points is None:
                self.num_points = num
            else:
                assert self.num_points == num

    @property
    def num_vars(self) -> int:
        return len(self.data)

    def __getitem__(self, var: str) -> Union[torch.Tensor, np.ndarray]:
        return self.data[var]

    @staticmethod
    def convert(points: PointCloud) -> 'PointCloud2':
        float_dict = dict(zip(points.float_vars, points.float_data))
        string_dict = dict(zip(points.string_vars, points.string_data))
        data_dict = float_dict
        data_dict.update(string_dict)
        return PointCloud2(data_dict=data_dict,
                           float_vars=points.float_vars)

    @staticmethod
    def load(filename: str, delimiter=',') -> 'PointCloud2':
        """
        Loads the data from the given csv or npz file.
        """
        points = PointCloud.load(filename=filename,
                                 delimiter=delimiter)
        return PointCloud2.convert(points=points)


def main():
    points = PointCloud2.load(filename='elliptic_curve.npz')


if __name__ == '__main__':
    main()
