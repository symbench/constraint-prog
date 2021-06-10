import torch
import sympy


class battery_packing(sympy.Function):
    net = torch.load("net.pt").float()

    @classmethod
    def eval(cls, x, y, z):
        if (isinstance(x, (sympy.Number, torch.Tensor)) and
                isinstance(y, (sympy.Number, torch.Tensor)) and
                isinstance(z, (sympy.Number, torch.Tensor))):
            return cls.net(torch.tensor([float(x), float(y), float(z)]))[0]
