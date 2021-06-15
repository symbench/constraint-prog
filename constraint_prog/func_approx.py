import torch


class PiecewiseLinearFunc:
    """
    Class for piecewise linear interpolation
    based on points passed at construction time
    Expected formats of points:
    1. tensor with shape (x_dim, y_dim + 1):
    [[x_i, y_{i, 1}, ..., y_{i, y_dim}]]
    2. tensor with shape (2, x_dim, y_dim)
    [[x_{i, j}], [y_{i, j}]]
    In the first case, for interpolating different functions,
    we use the same abscissas.
    In the second case, more generally, the set of abscissas can be
    different for different function
    """
    def __init__(self, points: torch.Tensor) -> None:
        self.points = points
        # Derived member variables
        if len(self.points.shape) > 2:
            # self.points has shape (2, x_dim, y_dim)
            self.x_coord = self.points[0, :, :]
            self.y_coord = self.points[1, :, :]
            self.x_dim = self.points.shape[1]
            self.y_dim = self.points.shape[2]
        else:
            # self.points has shape (x_dim, y_dim + 1)
            self.x_dim = self.points.shape[0]
            self.y_dim = self.points.shape[1] - 1
            self.x_coord = \
                self.points[:, 0].reshape((-1, 1)) * \
                torch.ones(self.y_dim)
            self.y_coord = self.points[:, 1]

        # Other member variables
        self.device = None
        self.t = None
        # Size value
        self.t_dim = None

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Function call of the class.
        For a given tensor 't' with shape (t_dim, ),
        the output has shape (t_dim, y_dim), where
        output[i, j] = interpolation of the jth function evaluated at t[i]
        If t[i] < min(x_coord[:, j]) or t[i] > max(x_coord[:, j]),
        then constant extrapolation is applied, thus return value is
        y_coord[0, j] or y_coord[-1, j]
        """
        self.device = t.device
        self.t = t
        self.t_dim = torch.flatten(input=t).shape[0]
        return self.evaluate()

    def evaluate(self) -> torch.Tensor:
        """
        Return interpolation values based on self.points.
        Output has the shape (t_dim, y_dim).
        """
        # Apply tiling to get a tensor of shape (t_dim, x_dim, y_dim)
        # then use permutation to get shape (t_dim, y_dim, x_dim)
        # x_coord_tensor[i, j, k] = x_coord[k, j]
        x_coord_tensor = torch.tile(
            input=self.x_coord,
            dims=(self.t_dim, 1, 1)
        ).permute((0, 2, 1))
        # y_coord_tensor[i, j, k] = y_coord[k, j]
        y_coord_tensor = torch.tile(
            input=self.y_coord,
            dims=(self.t_dim, 1, 1)
        ).permute((0, 2, 1))

        # Apply tiling and permutation to get a tensor of shape (t_dim, y_dim, x_dim)
        # t_tensor[i, j, k] = t[i]
        t_result = self.t.reshape((-1, 1)) * torch.ones(self.y_dim)
        t_tensor = torch.tile(
            input=t_result,
            dims=(self.x_dim, 1, 1)
        ).permute((1, 2, 0))

        # Get comparison of t_tensor and x_coord_tensor (from both direction)
        t_lt_x = t_tensor < x_coord_tensor
        x_lt_t = x_coord_tensor < t_tensor

        # Get boolean tensors for selecting interval endpoints
        # left operand of disjunction: classical case, when min(x_coord[:, j]) < t_i < max(x_coord[:, j])
        # right operand of disjunction: edge case, when t_i < min(x_coord[:, j]) or max(x_coord[:, j]) < t_i
        # Example for classical case:
        # ## Right endpoint:
        # t_lt_x = [False, False, True, True, True]
        # -> cumsum = [0, 0, 1, 2, 3]
        # -> (cumsum == 1) = [False, False, True, False, False]
        # ## Left endpoint
        # x_lt_t = [True, True, False, False, False]
        # -> x_lt_t.flip = [False, False, False, True, True]
        # -> cumsum(x_lt_t.flip) = [0, 0, 0, 1, 2]
        # -> cumsum(x_lt_t.flip).flip = [2, 1, 0, 0, 0]
        # -> (cumsum(x_lt_t.flip).flip == 1) = [False, True, False, False, False]
        idx_x = 2
        bool_tensor_r = \
            (torch.cumsum(t_lt_x, dim=idx_x) == 1).bool() | \
            ((torch.cumsum(t_lt_x, dim=idx_x) == 0).bool() &
             (torch.cumsum(x_lt_t, dim=idx_x) == self.x_dim).bool())  # for right endpoints
        bool_tensor_l = \
            (torch.cumsum(x_lt_t.flip([idx_x]), dim=idx_x).flip([idx_x]) == 1).bool() | \
            ((torch.cumsum(x_lt_t.flip([idx_x]), dim=idx_x).flip([idx_x]) == 0).bool() &
             (torch.cumsum(t_lt_x.flip([idx_x]), dim=idx_x).flip([idx_x]) == self.x_dim).bool())  # for left endpoints

        # Get x coordinates at both endpoints
        # Example:
        # >> x_coord = [[1, 2], [3, 4], [5, 6]]
        # >> t = [0.2, 1.3, 4.4, 7.1]
        # x_left_endpoints
        # [[1, 2], [1, 2], [3, 4], [5, 6]]
        # x_right_endpoints
        # [[1, 2], [3, 2], [5, 6], [5, 6]]
        x_left_endpoints = x_coord_tensor[bool_tensor_l].reshape((self.t_dim, self.y_dim))
        x_right_endpoints = x_coord_tensor[bool_tensor_r].reshape((self.t_dim, self.y_dim))

        # Get y coordinates at both endpoints
        y_left_endpoints = y_coord_tensor[bool_tensor_l].reshape((self.t_dim, self.y_dim))
        y_right_endpoints = y_coord_tensor[bool_tensor_r].reshape((self.t_dim, self.y_dim))

        # Handle situations, when left_endpoint = right_endpoint (edge cases) and
        # apply constant constant extrapolation
        bool_extrapolation = torch.abs(x_right_endpoints - x_left_endpoints) < 1e-6
        dx = x_right_endpoints - x_left_endpoints
        dx[bool_extrapolation] = 1
        dy = y_right_endpoints - y_left_endpoints

        # Calculate interpolation
        result = y_right_endpoints + dy / dx * (t_result - x_right_endpoints)

        return result


class FourierFunc:
    def __init__(self, order: int) -> None:
        self.order = order
        # Other member variables
        self.device = None
        self.t = None
        # Size value
        self.t_dim = None

    def __call__(self, coeff: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns with the Fourier approximation with shape (n_samples, t_dim, *)
        for input coefficient tensor with shape (n_samples, 2 * order + 1, *).
        We assume, that input time tensor has a shape (t_dim, ), (1, t_dim) or
        (t_dim, 1)
        """
        self.device = coeff.device
        self.t = t
        self.t_dim = torch.flatten(input=t).shape[0]
        return self.__fourier(coeff=coeff)

    def d(self, coeff: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns with the time derivative of the Fourier approximation
        with shape (n_samples, t_dim, *) for input coefficient tensor
        with shape (n_samples, 2 * order + 1, *).
        We assume, that input time tensor has a shape (t_dim, ),
        (1, t_dim) or (t_dim, 1)
        """
        self.device = coeff.device
        self.t = t
        self.t_dim = torch.flatten(input=t).shape[0]
        return self.__fourier_dt(coeff=coeff)

    def __get_t_tensor(self, v_dim: int, n_samples: int) -> torch.Tensor:
        """
        Returns with tensor of time points with shape (n_samples, order, t_dim, v_dim),
        where output[i, j, k, l] = j * self.t[k]
        """
        t_tensor = \
            torch.tile(input=self.t * torch.ones(v_dim, device=self.device),
                       dims=(n_samples, self.order, 1, 1)
                       ) * \
            torch.reshape(
                input=torch.arange(start=1, end=self.order + 1,
                                   device=self.device),
                shape=(1, self.order, 1, 1)
            )
        return t_tensor

    def __fourier(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Fourier approximation
        using input coefficients with shape (n_samples, 2 * order + 1, *)
        and returns tensor with shape (n_samples, t_dim, *)
        """
        orig_shape = coeff.shape
        if len(orig_shape) == 2:
            coeff = coeff.reshape(1, coeff.shape[0], coeff.shape[1])
        v_dim = coeff.shape[2]
        n_samples = coeff.shape[0]
        # Calculate zeroth order part
        # Broadcasting: (n_samples, self.t_dim, v_dim) * (n_samples, 1, v_dim)
        const_part = \
            torch.ones(size=(n_samples, self.t_dim, v_dim),
                       device=self.device) * \
            coeff[:, 0, :]
        # Get tensor of time points
        tt = self.__get_t_tensor(v_dim=v_dim,
                                 n_samples=n_samples)
        # Get cosine and sine part
        # Broadcasting: shape_1 * shape_2, where
        #  > shape_1 = (n_samples, self.order, self.t_dim, v_dim)
        #  > shape_2 = (n_samples, self.order, 1, v_dim)
        # Summation along dim=1: shape_1 -> (n_samples, self.t_dim, v_dim)
        cos_tensor = \
            torch.cos(input=tt) * \
            torch.reshape(input=coeff[:, 1:self.order + 1, :],
                          shape=(n_samples, self.order, 1, v_dim))
        cos_part = torch.sum(input=cos_tensor, dim=1)
        sin_tensor = \
            torch.sin(input=tt) * \
            torch.reshape(input=coeff[:, self.order + 1:, :],
                          shape=(n_samples, self.order, 1, v_dim))
        sin_part = torch.sum(input=sin_tensor, dim=1)
        # Get result as a sum of constant, cosine and sine parts
        result = const_part + cos_part + sin_part
        if len(orig_shape) == 2:
            result = result[0]
        return result

    def __fourier_dt(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        Calculates time derivative of the Fourier approximation
        using input coefficients with shape (n_samples, 2 * order + 1, *)
        and returns tensor with shape (n_samples, t_dim, *)
        """
        orig_shape = coeff.shape
        if len(orig_shape) == 2:
            coeff = coeff.reshape(1, coeff.shape[0], coeff.shape[1])
        v_dim = coeff.shape[2]
        n_samples = coeff.shape[0]
        # Get tensor of time points
        tt = self.__get_t_tensor(v_dim=v_dim,
                                 n_samples=n_samples)
        # Get cosine and sine part
        # Broadcasting: shape_1 * shape_2, where
        # shape_1 = (n_samples, self.order, self.t_dim, v_dim)
        # shape_2 = (n_samples, self.order, 1, v_dim)
        # Summation along dim=1: shape_1 -> (n_samples, self.t_dim, v_dim)
        cos_dt_tensor = \
            (-1) * torch.sin(input=tt) * \
            torch.reshape(input=coeff[:, 1:self.order + 1, :],
                          shape=(n_samples, self.order, 1, v_dim)) * \
            torch.reshape(input=torch.arange(start=1, end=self.order + 1,
                                             device=self.device),
                          shape=(1, self.order, 1, 1)
                          )
        cos_dt_part = torch.sum(input=cos_dt_tensor, dim=1)
        sin_dt_tensor = \
            torch.cos(input=tt) * \
            torch.reshape(input=coeff[:, self.order + 1:, :],
                          shape=(n_samples, self.order, 1, v_dim)) * \
            torch.reshape(input=torch.arange(start=1, end=self.order + 1,
                                             device=self.device),
                          shape=(1, self.order, 1, 1)
                          )
        sin_dt_part = torch.sum(input=sin_dt_tensor, dim=1)
        # Get result as a sum of constant, cosine and sine parts
        result = cos_dt_part + sin_dt_part
        if len(orig_shape) == 2:
            result = result[0]
        return result
