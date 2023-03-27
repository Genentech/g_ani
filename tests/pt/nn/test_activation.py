import torch
import numpy as np
import pprint
from ml_qm.pt.nn.activation import MishSimple, Mish
import numpy.testing as npt

pp = pprint.PrettyPrinter(indent=2, width=2000)

torch.set_printoptions(linewidth=200)
torch.set_printoptions(precision=2, threshold=9999, linewidth=9999, sci_mode=False)

def test_Mish():
    mishS = MishSimple()
    mish  = Mish()

    xS = torch.Tensor([-100, -1.1924, 0, 10])
    x  = xS.clone()
    xS.requires_grad_(True)
    x.requires_grad_(True)

    yS = mishS(xS)
    y  = mish(x)

    npt.assert_array_almost_equal(yS.detach().cpu().numpy(), np.array([0, -0.3088, 0, 10]),3)
    npt.assert_array_almost_equal(yS.detach().cpu().numpy(), y.detach().cpu().numpy())

    yS.sum().backward()
    y.sum().backward()
    npt.assert_array_almost_equal(xS.grad.cpu().numpy(), np.array([0, 0, 0.6, 1.00]),3)
    npt.assert_array_almost_equal(xS.grad.cpu().numpy(), x.grad.cpu().numpy())
