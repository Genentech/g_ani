import torch
import numpy as np


class PrecisionUtil():
    """ Manage tensor precision defaults for models """

    def __init__(self, NNPDType, indexAddDType, lossDType, DISTStoreType = torch.float16):
        self.NNPDType = NNPDType
        self.indexAddDType = indexAddDType
        self.lossDType = lossDType
        self.npLossDType = np.float32
        self.DISTStoreType = DISTStoreType
        if lossDType == torch.float64: self.npLossDType = np.float64
        if lossDType == torch.float16: self.npLossDType = np.float16


    def indexAdd(self, resShape, dim, index, srcTnsr):
        """ torch.index_add but using precision type self.indexAddDType
            srcTnsr will be converted to indexAddDType
            returns tensor of resShape which is converted to self.defaultTDype
        """

        res = torch.zeros(resShape, dtype=self.indexAddDType, device=srcTnsr.device)
        srcTnsr = srcTnsr.to(dtype=self.indexAddDType)
        res.index_add_(dim, index, srcTnsr)
        res = res.to(self.NNPDType)
        return res


    def indexAdd_(self, destTnsr:torch.tensor, dim:int, index:torch.tensor, srcTnsr:torch.tensor):
        """ torch.index_add_ but converting srcTNst.dtype to destTNsr.dtype
        """

        srcTnsr = srcTnsr.to(dtype=destTnsr.dtype)
        destTnsr.index_add_(dim, index, srcTnsr)
        return destTnsr


    def sum(self, srcTnsr:torch.tensor, dim:int):
        """ torch.index_add_ but converting srcTNst.dtype to destTNsr.dtype
        """

        dtype = srcTnsr.dtype
        srcTnsr = srcTnsr.to(dtype=self.indexAddDType)
        return srcTnsr.sum(dim=dim).to(dtype=dtype)


def INIT_NNP_PrecisionUtil(nnpPrecision:int = 32, sumPrecision:int = 32, lossPrecision:int = 32):
    if nnpPrecision == 32:
        nnpPrecision = torch.float32
    elif nnpPrecision == 64:
        nnpPrecision = torch.float64
    elif nnpPrecision == 16:
        nnpPrecision = torch.float16
    else:
        raise RuntimeError("Unknown nnpPrecision: %s" % nnpPrecision)

    if sumPrecision == 32:
        sumPrecision = torch.float32
    elif sumPrecision == 64:
        sumPrecision = torch.float64
    elif sumPrecision == 16:
        sumPrecision = torch.float16
    else:
        raise RuntimeError("Unknown sumPrecision: %s" % sumPrecision)

    if lossPrecision == 32:
        lossPrecision = torch.float32
    elif lossPrecision == 64:
        lossPrecision = torch.float64
    elif lossPrecision == 16:
        lossPrecision = torch.float16
    else:
        raise RuntimeError("Unknown lossPrecision: %s" % lossPrecision)

    torch.set_default_dtype(nnpPrecision)
    global NNP_PRECISION
    NNP_PRECISION = PrecisionUtil(nnpPrecision, sumPrecision, lossPrecision)


# We are storing precision information in a global variable
# because this is universally useful
# defaults are here:
NNP_PRECISION = PrecisionUtil(torch.float32, torch.float32, torch.float32, torch.float16)
