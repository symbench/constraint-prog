#!/usr/bin/env python3
# Copyright (C) 2021, Zsolt Vizi
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
from inspect import getmembers
import json
import os
import sys

import sympy

from constraint_prog import uuv_equations
from constraint_prog.sympy_func import SympyFunc


class Explorer:
    def __init__(self, args):
        with open(args.problem_json) as f:
            self.json_content = json.loads(f.read())

        self.equations = None
        self.get_equations()

    def get_equations(self):
        eqns = []
        if self.json_content["eqns"] == "uuv":
            members = getmembers(uuv_equations)
            for member in members:
                if isinstance(member[1], sympy.Eq):
                    eqns.append("uuv_equations." + member[0])
            eqns_str = eqns[0]
            for eqn_str in eqns[1:]:
                eqns_str += (', ' + eqn_str)
            self.equations = eval('[' + eqns_str + ']')

            # Alternative way: direct construction by name
            # from uuv_equations import hull_length_equation, hull_thickness_equation, mission_duration_equation
            # self.equations = [hull_length_equation, hull_thickness_equation, mission_duration_equation]

    def run(self):
        func = SympyFunc(self.equations)
        print(func.input_names)


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
    parser.add_argument('--iter', type=int,
                        default=10,
                        help='Number of iterations in the solver')
    parser.add_argument('--max-points', type=int,
                        default=1000,
                        help='Maximal number of points generated for exploration')
    args = parser.parse_args(args)

    explorer = Explorer(args=args)
    explorer.run()


if __name__ == '__main__':
    main()
