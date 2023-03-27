'''
Created on Jun 7, 2018

@author: albertgo
'''

import re
import torch
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611



class Clipper():
    """ Regularize by clipping values outside given range """

    def __init__(self, maxValue, paramNamePattern):
        self.maxValue = maxValue
        self.paramNameRe = re.compile(paramNamePattern)
        self.cnt = 0

    def regularize(self, module):
        """ look for all parameters with matching name in moduel and
            if l2 norm of parameters is > maxValeu, scale down so that
            norm is = maxValue
        """

        for name, weights in module.named_parameters():
            if self.paramNameRe.match(name):
                norm = weights.data.norm(2,-1)

                #if norm.sum()/w.shape[-1] > self.maxValue:
                #  warn("Regularizing %s: %s -> %.f" % (name, norm, self.maxValue))

                # only needed if all weights go to 0
                # the only time this happened was when the training set did not include
                # all examples with 1 and 2 heavy atoms. This is needed to make sure
                # all weights are being optimized:
                #norm = torch.clamp(norm,0.00001)

                # scale so that norm is <= maxVal
                scaling = torch.clamp(norm,0., self.maxValue)
                scaling.div_(norm + 1e-7)
#                 if scaling.mean() < 0.99:
#                     warn(scaling)

                weights.data = (weights.data * scaling.reshape(-1,1)).squeeze()
                #self.cnt += 1
                #if self.cnt % 5000 == 0: warn("Regularized %i times" % self.cnt)


def createRegularizer(conf):
    if conf['regularizer'] == "Clipper":
        regularizer = Clipper(**conf['regularizeParam'])
    elif conf['regularizer'] == "None":
        regularizer = None
    else:
        raise TypeError("unknown regularizer: " + conf['regularizer'])
    return regularizer
