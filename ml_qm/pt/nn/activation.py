'''
Created on Jun 6, 2018

@author: albertgo
'''
import torch
import torch.nn as nn
from torch.nn.functional import softplus
import math
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from torch.autograd import Function


class Gaussian(nn.Module):
    r"""Applies element-wise gaussian: exp(- x*x )
    """

    def __init__(self, beta:float = None, half_width:float = None ):
        """
            @parameter 
            beta: mutually exclusive to half_with
            half_width:  mutually exclusive to half_with,
               this computes beta as math.log(0.5) * 4./half_width/half_width
            if neither beta nor half_width is given beta is -1
        """

        super(Gaussian, self).__init__()

        if beta and half_width:
            raise Exception("beta and half_width are exclusive")

        self.beta = -1.
        if beta:       self.beta = beta
        if half_width: self.beta = math.log(0.5) * 4./half_width/half_width
        warn(f"Guassian activation created with beta={self.beta}")


    def forward(self, x):
        return torch.exp(self.beta * x * x)

class NegGaussian(nn.Module):
    r"""Applies element-wise gaussian: 1 - exp(- x*x )
    """

    def __init__(self):
        super(NegGaussian, self).__init__()

    def forward(self, x):
        return 1 - torch.exp(- x * x)


class AbsExp(nn.Module):
    r"""Applies element-wise gaussian: exp(- x*x )
    """

    def __init__(self):
        super(AbsExp, self).__init__()

    def forward(self, x):
        return torch.exp(-torch.abs(x))


class myCELU(nn.Module):
    r"""Applies element-wise gaussian: exp(- x*x )
    """

    def __init__(self, alpha):
        super(myCELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        x[x<0] = self.alpha * torch.expm1(x[x<0]/self.alpha)
        return x


class SSP(nn.Softplus):
    """ Shifted Softplus activation function """
    def __init__(self, beta=1, origin=0.5, threshold=20):
        super(SSP, self).__init__(beta, threshold)
        self.origin = origin
        self.sp0 = softplus(torch.zeros(1) + self.origin, self.beta, self.threshold).item()

    def forward(self, inpt):
        return softplus(inpt + self.origin, self.beta, self.threshold) - self.sp0


class MishSimple(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(softplus(x))


class MishFunction(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return MishFunction._forward(x)

    @staticmethod
    @torch.jit.script
    def _forward(x):
        return x * torch.tanh(softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return MishFunction._backward(grad_output, x)

    @staticmethod
    @torch.jit.script
    def _backward(grad_output, x):
        """ apparently this saves some memory"""
        tanh = softplus(x).tanh_()
        sigmoid  = x.mul_(torch.sigmoid(x))
        return grad_output * ( tanh + sigmoid * (1. - tanh * tanh))


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:

    .. math::

        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

    Plot:

    .. figure::  _static/mish.png
        :align:   center


    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def forward(self, x):
        return MishFunction.apply(x)
