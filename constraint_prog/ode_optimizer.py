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

from constraint_prog.func_approx import FourierFunc, PiecewiseLinearFunc


class ODEOptimizer:
    def __init__(self, f: Callable, fourier_order: int, dt: float,
                 t_0: float, x_0: torch.Tensor, t_1: float, x_1: torch.Tensor,
                 u_dim: int = 0, device: torch.device = torch.device('cpu')
                 ) -> None:
        self.f = f
        self.fourier_order = fourier_order
        assert t_1 >= t_0
        self.t_0, self.t_1, self.x_0, self.x_1 = t_0, t_1, x_0, x_1
        self.dt = dt
        self.u_dim = u_dim
        self.device = device
        # Derived members
        self.t = torch.linspace(start=t_0, end=t_1,
                                steps=int((t_1 - t_0) / self.dt),
                                device=self.device
                                ).view((-1, 1))
        self.fourier = FourierFunc(order=self.fourier_order)
        # Size values
        self.t_dim = self.t.shape[0]
        self.x_dim = self.x_0.shape[0]
        self.n_coeff_fourier = 2 * self.fourier_order + 1

    def run(self, coeff: torch.Tensor) -> torch.Tensor:
        # Reshape input coefficient tensor
        coeff_ = coeff.view((self.n_coeff_fourier, self.x_dim + self.u_dim))
        # Get Fourier polynomial for x(t) and u(t)
        x_t = self.fourier(coeff=coeff_[:, :self.x_dim], t=self.t)
        u_t = self.fourier(coeff=coeff_[:, self.x_dim:], t=self.t)
        # Get x'(t)
        dx_dt = self.fourier.d(coeff=coeff_[:, :self.x_dim], t=self.t)

        # Return with equalities in a lhs - rhs form
        return torch.cat(
            tensors=[(x_t[0] - self.x_0).flatten(),
                     (x_t[-1] - self.x_1).flatten(),
                     self.f(x=x_t, dx=dx_dt, u=u_t).flatten()]
        ).unsqueeze(dim=0)


def main():
    # Function generating the flow (dynamical system)
    def func(x: torch.Tensor, dx: torch.Tensor,  u: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            tensors=[-x[:, 0] + x[:, 1] - dx[:, 0],
                     -2 * x[:, 0] - dx[:, 1]],
            dim=-1
        )

    # Get variables needed to create an ODEOptimizer object
    device = torch.device('cpu')
    u_dim = 0
    fourier_order = 5
    t_0, t_1 = 0.0, 5.0
    x_0 = torch.tensor(data=[1.0, 1.0], device=device)
    x_1 = torch.tensor(data=[0.0877126, 0.0473586], device=device)
    dt = 0.01

    ode_opt = ODEOptimizer(f=func, fourier_order=fourier_order, u_dim=u_dim,
                           t_0=t_0, x_0=x_0, t_1=t_1, x_1=x_1,
                           dt=dt, device=device)
    coeff = torch.randn(size=(1, ode_opt.n_coeff_fourier * (ode_opt.x_dim + ode_opt.u_dim)),
                        device=device)

    # Get solution from optimization
    # from constraint_prog.gradient_descent import gradient_descent
    # output = gradient_descent(f=ode_opt.run, in_data=coeff,
    #                           lrate=0.001, it=10000, device=device)
    from constraint_prog.newton_raphson import newton_raphson
    output = newton_raphson(func=ode_opt.run, input_data=coeff)

    coeff = output.view((ode_opt.n_coeff_fourier, ode_opt.x_dim + ode_opt.u_dim))
    x_t = ode_opt.fourier(coeff=coeff[:, :ode_opt.x_dim], t=ode_opt.t).cpu().detach().numpy()
    run_output = ode_opt.run(coeff=output)
    val = (run_output.pow(2.0).sum(dim=-1)).pow(0.5)

    print("Number of points (flow):", run_output.shape[1] - 4)
    print("Squared error:", val.cpu().detach().numpy()[0])
    plt.plot(x_t[:, 0], x_t[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
    # lin = PiecewiseLinearFunc(
    #     points=torch.tensor([[[0, 1], [2, 3], [4, 5]],
    #                          [[-1, 2], [3, -4], [-5, -6]]
    #                          ])
    # )
    # print(lin(torch.tensor([0.2, 1.9, 4.1, 6.1])))
