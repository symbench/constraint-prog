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

from typing import Dict, Tuple
import math
import sympy

from constraint_prog.point_cloud import PointCloud, PointFunc


class RelativePos:
    """
    This represents a relative position of one component inside another. Each
    component has its own origin, which is usually its geometric center.
    """

    def __init__(self, name: str):
        self.name = name

        self.x = sympy.Symbol(name + "_x")  # in m
        self.y = sympy.Symbol(name + "_y")  # in m
        self.z = sympy.Symbol(name + "_z")  # in m

        self.x_bounds = (-math.inf, math.inf)
        self.y_bounds = (-math.inf, math.inf)
        self.z_bounds = (-math.inf, math.inf)

    @property
    def variables(self) -> Dict[str, sympy.Expr]:
        return {
            self.name + "_x": self.x,
            self.name + "_y": self.y,
            self.name + "_z": self.z,
        }

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            self.name + "_x": self.x_bounds,
            self.name + "_y": self.y_bounds,
            self.name + "_z": self.z_bounds,
        }

    def minus(self, other: 'RelativePos') -> 'RelativePos':
        result = RelativePos(self.name + "_minus_" + other.name)
        result.x = self.x - other.x
        result.y = self.y - other.y
        result.z = self.z - other.z
        return result


class CapsuleShape:
    """
    This represents a capsule shaped object with two semispheres connected by
    a cylinder. The class maintains the two variables describing this capsule,
    its radius and length in meters. If the length is zero, then the capsule
    is a sphere. The capsule is always oriented along the x-axis.
    """

    def __init__(self, name: str):
        self.name = name

        self.radius = sympy.Symbol(name + "_radius")  # in m
        self.length = sympy.Symbol(name + "_length")  # in m

        self.radius_bounds = (0.0, math.inf)
        self.length_bounds = (0.0, math.inf)

    @property
    def volume(self) -> sympy.Expr:
        return sympy.pi * self.radius ** 2 * (4 * self.radius / 3 + self.length)

    @property
    def variables(self) -> Dict[str, sympy.Expr]:
        return {
            self.name + "_radius": self.radius,
            self.name + "_length": self.length,
        }

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            self.name + "_radius": self.radius_bounds,
            self.name + "_length": self.length_bounds,
        }

    def draw(self, turtle):
        """
        We assume that the variables are concrete numbers and we draw the
        capsule shape at the current turtle position.
        """
        pos = turtle.position()
        turtle.penup()
        turtle.setheading(270)
        turtle.forward(self.radius)
        turtle.setheading(0)
        turtle.pendown()
        turtle.forward(self.length * 0.5)
        turtle.circle(self.radius, 180)
        turtle.forward(self.length)
        turtle.circle(self.radius, 180)
        turtle.forward(self.length * 0.5)
        turtle.penup()
        turtle.goto(pos)

    def contains_point(self, relpos: 'RelativePos') -> Dict[str, sympy.Expr]:
        """
        Returns a dictionary of differentiable constraints expressing the fact
        that the given relative position is inside of this capsule.
        """
        relx = sympy.Max(0.0, sympy.Abs(relpos.x) - self.length * 0.5)
        dist = relx ** 2 + relpos.y ** 2 + relpos.z ** 2
        return {
            self.name + "_contains_" + relpos.name:
                dist <= self.radius ** 2
        }

    def excludes_point(self, relpos: 'RelativePos') -> Dict[str, sympy.Expr]:
        """
        Returns a dictionary of differentiable constraints expressing the fact
        that the given relative position is outside of this capsule.
        """
        relx = sympy.Max(0.0, sympy.Abs(relpos.x) - self.length * 0.5)
        dist = relx ** 2 + relpos.y ** 2 + relpos.z ** 2
        return {
            self.name + "_excludes_" + relpos.name:
                dist >= self.radius ** 2
        }

    def contains_capsule(self, capsule: 'CapsuleShape',
                         relpos: 'RelativePos') -> Dict[str, sympy.Expr]:
        """
        Returns a dictionary of differentiable constraints expressing the fact
        the given capsule at the relative position is fully inside this
        capsule.
        """
        relx = sympy.Max(0.0, sympy.Abs(relpos.x) + (capsule.length - self.length) * 0.5)
        dist = relx ** 2 + relpos.y ** 2 + relpos.z ** 2
        return {
            self.name + "_contains_" + capsule.name + "_at_" + relpos.name:
                dist <= (self.radius - capsule.radius) ** 2
        }

    def excludes_capsule(self, capsule: 'CapsuleShape',
                         relpos: 'RelativePos') -> Dict[str, sympy.Expr]:
        """
        Returns a dictionary of differentiable constraints expressing the fact
        the given capsule at the relative position is fully outside of this
        capsule.
        """
        relx = sympy.Max(0.0, sympy.Abs(relpos.x) - (capsule.length + self.length) * 0.5)
        dist = relx ** 2 + relpos.y ** 2 + relpos.z ** 2
        return {
            self.name + "_excludes_" + capsule.name + "_at_" + relpos.name:
                dist >= (self.radius + capsule.radius) ** 2
        }


def test_equations():
    capsule1 = CapsuleShape("capsule1")
    capsule1.radius = 50
    capsule1.length = 100
    capsule2 = CapsuleShape("capsule2")
    capsule2.radius = 10
    capsule2.length = 20
    relpos2 = RelativePos("relpos2")
    relpos2.x = 105
    relpos2.y = 40
    relpos2.z = 0
    print(capsule1.excludes_capsule(capsule2, relpos2))

    import turtle
    t = turtle.Turtle()
    t.speed(0)
    capsule1.draw(t)
    t.goto(relpos2.x, relpos2.y)
    capsule2.draw(t)
    turtle.done()


def test_newton_raphson():
    c0 = CapsuleShape("c0")
    c0.radius_bounds = (1.0, 2.0)
    c0.length_bounds = (5.0, 15.0)

    c1 = CapsuleShape("c1")
    c1.radius_bounds = (0.0, 2.0)
    c1.length_bounds = (0.0, 15.0)

    c2 = CapsuleShape("c2")
    c2.radius_bounds = (0.0, 2.0)
    c2.length_bounds = (0.0, 15.0)

    p1 = RelativePos("p1")
    p1.x_bounds = (-20.0, 20.0)
    p1.y_bounds = (-2.0, 2.0)
    p1.z_bounds = (-2.0, 2.0)

    p2 = RelativePos("p2")
    p2.x_bounds = (-20.0, 20.0)
    p2.y_bounds = (-2.0, 2.0)
    p2.z_bounds = (-2.0, 2.0)

    bounds = dict()
    bounds.update(c0.bounds)
    bounds.update(c1.bounds)
    bounds.update(c2.bounds)
    bounds.update(p1.bounds)
    bounds.update(p2.bounds)
    print(bounds)

    constraints = dict()
    constraints.update(c0.contains_capsule(c1, p1))
    constraints.update(c0.contains_capsule(c2, p2))
    constraints.update(c1.excludes_capsule(c2, p2.minus(p1)))
    print(constraints)

    num_points = 10000
    points = PointCloud.generate(bounds, num_points)
    func = PointFunc(constraints)

    for _ in range(10):
        points.add_mutations([0.5] * len(bounds), num_points)

        points = points.newton_raphson(func, bounds)

        errors = func(points)
        points = points.prune_by_tolerances(errors, 1e-5)

        points = points.prune_close_points([2.0] * len(bounds))
        print(points.num_points)


if __name__ == "__main__":
    test_newton_raphson()
