import torch
import torch.nn as nn
from warnings import warn


class FCNN(nn.Module):
    """A fully connected neural network.

    :param n_input_units: Number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_output_units: Number of units in the output layer, defaults to 1.
    :type n_output_units: int
    :param n_hidden_units: [DEPRECATED] Number of hidden units in each layer
    :type n_hidden_units: int
    :param n_hidden_layers: [DEPRECATED] Number of hidden mappsings (1 larger than the actual number of hidden layers)
    :type n_hidden_layers: int
    :param actv: The activation layer constructor after each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: class
    :param hidden_units: Number of hidden units in each hidden layer. Defaults to (32, 32).
    :type hidden_units: Tuple[int]

    .. note::
        The arguments "n_hidden_units" and "n_hidden_layers" are deprecated in favor of "hidden_units".
    """

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None,
                 actv=nn.Tanh, hidden_units=None):
        r"""Initializer method.
        """
        super(FCNN, self).__init__()

        # FORWARD COMPATIBILITY
        # If only one of {n_hidden_unit, n_hidden_layers} is specified, fill-in the other one
        if n_hidden_units is None and n_hidden_layers is not None:
            n_hidden_units = 32
        elif n_hidden_units is not None and n_hidden_layers is None:
            n_hidden_layers = 1

        # FORWARD COMPATIBILITY
        # When {n_hidden_unit, n_hidden_layers} are specified, construct an equivalent hidden_units if None is provided
        if n_hidden_units is not None or n_hidden_layers is not None:
            if hidden_units is None:
                hidden_units = tuple(n_hidden_units for _ in range(n_hidden_layers + 1))
                warn(f"`n_hidden_units` and `n_hidden_layers` are deprecated, "
                     f"pass `hidden_units={hidden_units}` instead",
                     FutureWarning)
            else:
                warn(f"Ignoring `n_hidden_units` and `n_hidden_layers` in favor of `hidden_units={hidden_units}`",
                     FutureWarning)

        # If none of {n_hidden_units, n_hidden_layers, hidden_layers} are specified, use (32, 32) by default
        if hidden_units is None:
            hidden_units = (32, 32)

        # If user passed in a list, iterator, etc., convert it to tuple
        if not isinstance(hidden_units, tuple):
            hidden_units = tuple(hidden_units)

        units = (n_input_units,) + hidden_units
        layers = []
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            layers.append(actv())
        # There's not activation in after the last layer
        layers.append(nn.Linear(units[-1], n_output_units))
        self.NN = torch.nn.Sequential(*layers)

    def forward(self, t):
        x = self.NN(t)
        return x
        
class Resnet(nn.Module):
    """A residual network with a trainable linear skip connection between input and output

    :param n_input_units: Number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_input_units: Number of units in the output layer, defaults to 1.
    :type n_input_units: int
    :param n_hidden_units: [DEPRECATED] Number of hidden units in each layer
    :type n_hidden_units: int
    :param n_hidden_layers: [DEPRECATED] Number of hidden mappsings (1 larger than the actual number of hidden layers)
    :type n_hidden_layers: int
    :param actv: the activation layer constructor after each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: class
    :param hidden_units: Number of hidden units in each hidden layer. Defaults to (32, 32).
    :param hidden_units: Tuple[int]
    """

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None, actv=nn.Tanh,
                 hidden_units=(32, 32)):
        super(Resnet, self).__init__()

        self.residual = FCNN(
            n_input_units=n_input_units,
            n_output_units=n_output_units,
            n_hidden_units=n_hidden_units,
            n_hidden_layers=n_hidden_layers,
            actv=actv,
            hidden_units=hidden_units,
        )
        self.skip_connection = nn.Linear(n_input_units, n_output_units, bias=False)

    def forward(self, t):
        x = self.skip_connection(t) + self.residual(t)
        return x