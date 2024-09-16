import torch
import torch.autograd as autograd

#The derivative of a variable with respect to another

def diff(u, t, order=1):

    # param u: the variable u in \frac{\partial u}{\partial t}
    # type u: torch.Tensor
    # param t: the variable t in \frac{\partial u}{\partial t}
    # type t: torch.Tensor
    # param order: The order of the derivative, defaults to 1.
    # type order: int
    # return type: torch.Tensor

    ones = torch.ones_like(u)
    der, = autograd.grad(u, t, create_graph=True, grad_outputs=ones, allow_unused=True)
    if der is None:
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):
        ones = torch.ones_like(der)
        der, = autograd.grad(der, t, create_graph=True, grad_outputs=ones, allow_unused=True)
        if der is None:
            return torch.zeros_like(t, requires_grad=True)
        else:
            der.requires_grad_()
    return der