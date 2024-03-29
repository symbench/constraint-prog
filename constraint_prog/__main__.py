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

import argparse
import sys

from constraint_prog import explorer
from constraint_prog.point_cloud import run_pareto_front, run_plot


def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('command', help="""
    explore, pareto-front, plot
    """)
    args = parser.parse_args(sys.argv[1:2])

    # hack the program name for nested parsers
    sys.argv[0] += ' ' + args.command
    args.command = args.command.replace('_', '-')

    if args.command == 'explore':
        explorer.main(args=sys.argv[2:])
    elif args.command == 'pareto-front':
        run_pareto_front(args=sys.argv[2:])
    elif args.command == 'plot':
        run_plot(args=sys.argv[2:])
    else:
        parser.print_help()


if __name__ == '__main__':
    run()
