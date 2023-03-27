'''
Created on Jun 6, 2018

@author: albertgo
'''

import torch
import torch.nn as nn
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611


class ExpLoss(nn.Module):
    r"""Creates a criterion that measures the exponential mean squared error between
    `n` elements in the input `x` and target `y`.

    According to eq. 6 in Roitberg 2017 paper

    Modify to normalize by the squar of batchsize to avoid infinity in exp

    """
    def __init__(self, tau):
        """ Because the loss might exceed the value range of a float this implementation
            switches from MSE the first time mse is <= start
        """
        super(ExpLoss, self).__init__()
        self.mseLoss = nn.MSELoss(size_average=False, reduce=True)
        self.tau= tau


    def forward(self, inp , target):
        assert not target.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"

        # ensure enough precision
#         inp = inp.to(dtype=torch.float64)
#         target = target.to(dtype=torch.float64)
#
        delta = torch.pow(inp - target,2).sum()/inp.shape[0]/inp.shape[0]

        # configured to just sum up losses
        eloss = self.tau * torch.exp( delta / self.tau)

        return eloss


class ANILoss(nn.Module):
    r"""Creates a criterion that measures the exponential mean squared error between
    `n` elements in the input `x` and target `y`.

    According to eq. 6 in Roitberg 2017 paper

    Modify to normalize by the squar of batchsize to avoid infinity in exp

    """
    def __init__(self):
        """ Because the loss might exceed the value range of a float this implementation
            switches from MSE the first time mse is <= start
        """
        super(ANILoss, self).__init__()
        self.mseLoss = nn.MSELoss()



    def forward(self, inp , target):
        assert not target.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"

        return torch.exp(self.mseLoss(inp,target))


class ExpLoss3(nn.Module):
    r"""Creates a criterion that measures the exponential mean squared error between
    `n` elements in the input `x` and target `y`.

    According to eq. 6 in Roitberg 2017 paper

    Modify to normalize by the squar of batchsize to avoid infinity in exp

    """
    def __init__(self):
        """ Because the loss might exceed the value range of a float this implementation
            switches from MSE the first time mse is <= start
        """
        super(ExpLoss3, self).__init__()
        self.mseLoss = nn.MSELoss()


    def forward(self, inp , target):
        assert not target.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"

        # ensure enough precision
#         inp = inp.to(dtype=torch.float64)
#         target = target.to(dtype=torch.float64)
#
        loss = self.mseLoss(inp,target)

        # configured to just sum up losses
        loss = torch.exp( loss )

        return loss


class ExpLossKhan(nn.Module):
    r"""Creates a criterion that measures the exponential mean squared error between
    `n` elements in the input `x` and target `y`.

    According to eq. 6 in Roitberg 2017 paper

    Modify to normalize by the squar of batchsize to avoid infinity in exp

    """
    def __init__(self):
        """ Because the loss might exceed the value range of a float this implementation
            switches from MSE the first time mse is <= start
        """
        super(ExpLossKhan, self).__init__()
        self.mseLoss = nn.MSELoss()


    def forward(self, inp , target):
        assert not target.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"

        # ensure enough precision
#         inp = inp.to(dtype=torch.float64)
#         target = target.to(dtype=torch.float64)
#
        loss = self.mseLoss(inp,target).sqrt()

        # configured to just sum up losses
        loss = torch.exp( loss )

        return loss


class ExpLossKhanMSE(nn.Module):
    r"""Creates a criterion that measures the exponential mean squared error between
    `n` elements in the input `x` and target `y` plus the MSE.

    """
    def __init__(self):
        """ Because the loss might exceed the value range of a float this implementation
            switches from MSE the first time mse is <= start
        """
        super(ExpLossKhanMSE, self).__init__()
        self.mseLoss = nn.MSELoss()


    def forward(self, inp , target):
        assert not target.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"

        loss = self.mseLoss(inp,target).sqrt()

        # configured to just sum up losses
        loss = torch.exp( loss ) + loss

        return loss


class ExpLoss2(nn.Module):
    r"""Creates a criterion that measures the exponential mean squared error between
    `n` elements in the input `x` and target `y`.

    According to eq. 6 in Roitberg 2017 paper
    Modified by:
    - Normalizing of batch size
    - capping with tnah
    to avoid inf in exp()

    """
    def __init__(self, tau, mmax=40. ):
        """ Because the loss might exceed the value range of a float this implementation
            switches from MSE the first time mse is <= start
        """
        super(ExpLoss2, self).__init__()
        self.mseLoss = nn.MSELoss(size_average=False, reduce=True)
        self.tau= tau
        self.mmax = mmax


    def forward(self, inp , target):
        assert not target.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"

        # ensure enough precision
#         inp = inp.to(dtype=torch.float64)
#         target = target.to(dtype=torch.float64)
        nBatch = inp.shape[0]
        delta = (inp - target)/self.mmax
        ## limit range with tanh to avoid inf in exp()
        delta = (torch.tanh(delta*delta)*self.mmax).sum()/nBatch

        # configured to just sum up losses
        eloss = self.tau * torch.exp( delta / self.tau)

        return eloss

class Power4(nn.Module):
    r"""Creates a criterion that measures the exponential mean squared error between
    `n` elements in the input `x` and target `y`.

    According to eq. 6 in Roitberg 2017 paper

    Modify to normalize by the squar of batchsize to avoid infinity in exp

    """
    def __init__(self):
        """ Because the loss might exceed the value range of a float this implementation
            switches from MSE the first time mse is <= start
        """
        super(Power4, self).__init__()


    def forward(self, inp , target):
        assert not target.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"

        loss = ((inp - target)**4).sum()/inp.shape[0]

        return loss



def createLossFunction(conf):
    lossF = globals().get(conf['loss'], None)
    if lossF is not None:
        lossF = lossF(**conf['lossParam'])
    else:
        lossF = getattr(nn, conf['loss'], None)
        if lossF is not None:
            lossF = lossF(**conf['lossParam'])
        else:
            raise TypeError("unknown loss function: " + conf['loss'])
    return lossF
