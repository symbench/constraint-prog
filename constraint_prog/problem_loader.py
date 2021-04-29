import importlib.util
from inspect import getmembers
import json
import os

import sympy


class ProblemLoader:
    def __init__(self, problem_json):
        self.problem_json = problem_json
        with open(self.problem_json) as f:
            self.json_content = json.loads(f.read())

        self.equations = dict()
        self.expressions = dict()
        self.get_equations_and_expressions()

        self.fixed_values = dict()
        self.constraints = self.process_constraints()

    def process_constraints(self) -> dict:
        # disregard entries that start with a dash
        constraints = {key: val
                       for (key, val) in self.json_content["variables"].items()
                       if not key.startswith('-')}

        # collect fixed values and substitute them into the equations
        self.fixed_values = {key: val["min"]
                             for (key, val) in constraints.items()
                             if val["min"] == val["max"]}
        for key, val in self.fixed_values.items():
            print("Fixing {} to {}".format(key, val))
            for val2 in self.equations.values():
                val2["expr"] = val2["expr"].subs(sympy.Symbol(key), val)
            self.expressions = {key2: val2.subs(sympy.Symbol(key), val)
                                for (key2, val2) in self.expressions.items()}
            del constraints[key]

        return constraints

    def get_equations_and_expressions(self) -> None:
        path = os.path.join(
            os.path.dirname(self.problem_json), self.json_content["source"])
        print("Loading python file:", path)
        if not os.path.exists(path):
            raise FileNotFoundError()

        spec = importlib.util.spec_from_file_location("equmod", path)
        equmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(equmod)
        members = getmembers(equmod)

        def find_member(name_):
            for member_ in members:
                if member_[0] == name_:
                    return member_[1]
            return None

        self.equations.clear()
        for (name, conf) in self.json_content["equations"].items():
            if name.startswith('-'):
                continue
            member = find_member(name)
            if member is None:
                raise ValueError("equation " + name + " not found")
            assert (isinstance(member, sympy.Eq) or
                    isinstance(member, sympy.LessThan) or
                    isinstance(member, sympy.StrictLessThan) or
                    isinstance(member, sympy.GreaterThan) or
                    isinstance(member, sympy.StrictGreaterThan))

            self.equations[name] = {
                "expr": member,
                "tolerance": conf["tolerance"]
            }

        self.expressions.clear()
        for name in self.json_content["expressions"]:
            if name.startswith('-'):
                continue
            member = find_member(name)
            if member is None:
                raise ValueError("expression " + name + " not found")
            self.expressions[name] = member
