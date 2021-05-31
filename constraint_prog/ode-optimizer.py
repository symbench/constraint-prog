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

from typing import Callable

import matplotlib.pyplot as plt
import torch


class ODEOptimizer:
    def __init__(self, f: Callable, fourier_order: int,
                 t_0: float, x_0: torch.Tensor, t_1: float, x_1: torch.Tensor,
                 dt: float, device: torch.device) -> None:
        self.f = f
        self.fourier_order = fourier_order
        assert t_1 >= t_0
        self.t_0, self.t_1, self.x_0, self.x_1 = t_0, t_1, x_0, x_1
        self.dt = dt
        self.device = device

        self.t = torch.linspace(start=t_0, end=t_1, steps=int((t_1 - t_0) / self.dt),
                                requires_grad=True, device=self.device).view((-1, 1))

    def run(self, coeff: torch.Tensor) -> torch.Tensor:
        coeff_ = coeff.view((4 * self.fourier_order + 2, self.x_0.shape[0]))
        u_t = self.fourier(coeff=coeff_[2 * self.fourier_order + 1:, :])
        x_t = self.fourier(coeff=coeff_[:2 * self.fourier_order + 1, :])
        dx_dt_list = []
        for idx in range(coeff_.shape[1]):
            dx_dt_list.append(torch.autograd.grad(inputs=self.t,
                                                  outputs=x_t[:, idx],
                                                  grad_outputs=torch.ones(self.t.shape[0]),
                                                  retain_graph=True
                                                  )[0]
                              )
        dx_dt = torch.stack(tensors=dx_dt_list, dim=-1).squeeze(dim=1)
        return torch.cat(
            tensors=[(x_t[0] - self.x_0).flatten(),
                     (x_t[1] - self.x_1).flatten(),
                     (dx_dt - self.f(x_t, u_t)).flatten()]
        ).unsqueeze(dim=0)

    def fourier(self, coeff: torch.Tensor) -> torch.Tensor:
        result = 0.0 * torch.zeros(size=(self.t.shape[0], coeff.shape[1]))
        result = result + torch.ones_like(input=result) * coeff[0, :]
        for j in range(1, self.fourier_order):
            result = result + torch.cos(input=j * self.t) * torch.ones(coeff.shape[1]) * coeff[2 * j - 1, :]
            result = result + torch.sin(input=j * self.t) * torch.ones(coeff.shape[1]) * coeff[2 * j, :]

        return result


def main():
    from constraint_prog.gradient_descent import gradient_descent

    def func(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            tensors=[-3 * x[:, 0] + u[:, 0],
                     2 * x[:, 1] + u[:, 1]],
            dim=-1
        )

    fourier_order = 5
    t_0, t_1 = 2.0, 4.0
    x_0 = torch.tensor(data=[0.988802, 0.761768])
    x_1 = torch.tensor(data=[-1.17613, 4.21177])
    dt = 0.01
    device = torch.device('cpu')
    ode_opt = ODEOptimizer(f=func, fourier_order=fourier_order,
                           t_0=t_0, x_0=x_0, t_1=t_1, x_1=x_1,
                           dt=dt, device=device)
    coeff = torch.randn(size=(1, (4 * fourier_order + 2) * x_0.shape[0]), device=device)

    output = gradient_descent(f=ode_opt.run, in_data=coeff, lrate=0.001, it=10000, device=device)
    coeff = output.view((4 * fourier_order + 2, x_0.shape[0]))
    x_t = ode_opt.fourier(coeff=coeff[:2 * fourier_order + 1, :]).detach().numpy()
    plt.plot(x_t[0], x_t[1])
    plt.show()


if __name__ == "__main__":
    main()
