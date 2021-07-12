from sympy import *
from sympy import Function
import torch
import numpy as np
from constraint_prog.sympy_func import SympyFunc
from battery_packing import *

if __name__=="__main__":
    x,y,z = symbols("x y z")
    print("Symbols: ")
    print(x)
    print(y)
    print(z)
    print("Symbolic call of the NN wrapper: ")
    sub_expr = x+2
    expr = battery_packing(sub_expr,y,z)
    print(expr)
    print("Evaluate of the NN wrapper in specific point: ")
    expr = battery_packing(1.0,1.0,1.0)
    print(expr)
    x = torch.tensor(1.0)
    y = torch.tensor(1.0)
    z = torch.tensor(1.0)
    print("Evaluate of the NN wrapper in specific torch tensor point: ")
    expr = battery_packing(x,y,z)
    print(expr)
    x, y, z = symbols("x y z")
    sub_expr = x + 2
    expr = battery_packing(sub_expr, y, z)
    print(expr)
    sf = SympyFunc([x,y,z])
    print(sf.input_names)
    print(sf.evaluate([expr,expr,expr],torch.tensor([1.0,1.0,1.0]),True))
