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


class Explorer:
    def __init__(self, args):
        with open(args.problem_json) as f:
            self.json_content = json.loads(f.read())

        self.components = {
            "symbols": [],
            "eqns": [],
            "constants": [],
            "derived": []
        }

        self.get_components()

    def get_components(self):
        if self.json_content["eqns"] == "uuv":
            members = getmembers(uuv_equations)
            sympy_types = getmembers(sys.modules[sympy.__name__])
            for member in members:
                member_val = member[1]
                if type(member_val) in [typ[1] for typ in sympy_types]:
                    if type(member_val) in [sympy.Symbol]:
                        self.components["symbols"].append(member)
                    elif type(member_val) not in [sympy.Eq]:
                        self.components["derived"].append(member)
                    else:
                        self.components["eqns"].append(member)
                elif type(member_val) in [int, float]:
                    self.components["constants"].append(member)

    def run(self):
        print("run")


def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('problem_json', type=str,
                        help='Path to the problem JSON file used for exploration')
    parser.add_argument('--output-dir', metavar='DIR', type=str,
                        default=os.getcwd(),
                        help='Path to output directory')
    args = parser.parse_args(args)

    explorer = Explorer(args=args)
    explorer.run()


if __name__ == '__main__':
    main()
