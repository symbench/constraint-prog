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

from typing import Dict, Tuple, Union
import math
import sympy


class Shape():
    def __init__(self, name: str):
        self.name = name
        self.bounds = dict()

    def _symbol(self, name: str, spec: Union[sympy.Expr, Tuple[float, float]]) \
            -> sympy.Expr:
        if not isinstance(spec, tuple):
            return spec

        name = self.name + "_" + name
        assert name not in self.bounds and len(spec) == 2
        self.bounds[name] = (float(spec[0]), float(spec[1]))
        return sympy.Symbol(name)


class Point(Shape):
    def __init__(self, name: str):
        super(Point, self).__init__(name)

    @property
    def volume(self) -> sympy.Expr:
        return 0

    @property
    def length(self) -> sympy.Expr:
        return 0

    @property
    def diameter(self) -> sympy.Expr:
        return 0

    def enlarge(self, thickness: sympy.Expr) -> 'Point':
        return Point(self.name + "_enlarged")


class Sphere(Shape):
    def __init__(self, name: str,
                 radius: Union[sympy.Expr, Tuple[float, float]]):
        super(Sphere, self).__init__(name)
        self.radius = self._symbol("radius", radius)

    @property
    def volume(self) -> sympy.Expr:
        return (4 * sympy.pi / 3) * (self.radius ** 3)

    @property
    def length(self) -> sympy.Expr:
        return 2 * self.radius

    @property
    def diameter(self) -> sympy.Expr:
        return 2 * self.radius

    def enlarge(self, thickness: sympy.Expr) -> 'Sphere':
        return Sphere(self.name + "_enlarged",
                      self.radius + thickness)


class Cylinder(Shape):
    def __init__(self, name: str,
                 radius: Union[sympy.Expr, Tuple[float, float]],
                 length: Union[sympy.Expr, Tuple[float, float]]):
        super(Cylinder, self).__init__(name)
        self.radius = self._symbol("radius", radius)
        self.length = self._symbol("length", length)

    @property
    def volume(self) -> sympy.Expr:
        return sympy.pi * (self.radius ** 2) * self.length

    @property
    def diameter(self) -> sympy.Expr:
        return 2 * self.radius

    def enlarge(self, thickness: sympy.Expr) -> 'Cylinder':
        return Cylinder(self.name + "_enlarged",
                        self.radius + thickness,
                        self.length + thickness)


class Capsule(Shape):
    def __init__(self, name: str,
                 radius: Union[sympy.Expr, Tuple[float, float]],
                 cylinder_length: Union[sympy.Expr, Tuple[float, float]]):
        super(Capsule, self).__init__(name)
        self.radius = self._symbol("radius", radius)
        self.cylinder_length = self._symbol("cylinder_length", length)

    @property
    def volume(self) -> sympy.Expr:
        return sympy.pi * (self.radius ** 2) * (4 * self.radius / 3 + self.cylinder_length)

    @property
    def length(self) -> sympy.Expr:
        return 2 * self.radius + self.cylinder_length

    @property
    def diameter(self) -> sympy.Expr:
        return 2 * self.radius

    def enlarge(self, thickness: sympy.Expr) -> 'Capsule':
        return Capsule(self.name + "_enlarged",
                       self.radius + thickness,
                       self.cylinder_length)
