from sympy import *
from sympy import Function
import torch
import numpy as np

class battery_packing(Function):
    net = torch.load("..\\examples\\net.pt")

    @ classmethod
    def eval(cls, x,y,z):
        if x.is_Number and y.is_Number and z.is_Number:
            x = np.array(x).astype(np.float64)
            y = np.array(y).astype(np.float64)
            z = np.array(z).astype(np.float64)
            inp = torch.tensor(np.asarray([x,y,z]))
            out = cls.net(inp)
            return out.detach().numpy()[0]

    def _eval_is_real(self):
        return self.args[0].is_real

if __name__=="__main__":
    x,y,z = symbols("x y z")
    print("Symbols: ")
    print(x)
    print(y)
    print(z)
    print("Symbolic call of the NN wrapper: ")
    expr = cos(battery_packing(x,y,z))
    print(expr)
    print("Evaluate of the NN wrapper in specific point: ")
    expr = battery_packing(1.0,1.0,1.0)
    print(expr)

