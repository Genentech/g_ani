# encoding: utf-8
"""
ml_qm.optimize.optimizer -- Reads sdf files and optimizes them using NNP.

@author:     albertgo

@copyright:  2019 Genentech Inc.

"""

import logging
import os
import errno

from t_opt.unit_conversion import Units
from t_opt.NNP_computer_factory import NNPComputerFactoryInterface

log = logging.getLogger(__name__)



class NNPComputerFactory(NNPComputerFactoryInterface):
    """ Factory class for NNP_Computer """

    def __init__(self, nnpName:str):
        super().__init__(nnpName)

        # extract starting from packageRoot/nnp if not exists
        if not os.path.exists(nnpName):
            if os.path.exists(os.environ.get("NNP_PATH","") + "/" + nnpName):
                nnpName = os.environ.get("NNP_PATH","") + "/" + nnpName

        self.nnp_name = nnpName

        if os.path.isfile(nnpName):
            self.createNNP = self._createGNENNP # type: ignore

        elif os.path.isdir(nnpName):
            self.createNNP = self._createANI    # type: ignore

        else:
            raise ValueError(f"Could not find conf or determine NNP type from conf: {nnpName}")

        self._model = None


    def createNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        """ to be replaced by specific implementation"""
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.nnp_name)


    def _createGNENNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        nGPU = kwArgs.get("nGPU", 0)
        confFile = self.nnp_name

        if "dist_net" in confFile.lower():
            # import here to avoid dependency collision pytorch vs ANI
            from ml_qm.distNet.DistNetAdapter import DistNetComputer  # pylint: disable=C0415
            return DistNetComputer(nGPU, confFile,
                                   outputGrad, compute_stdev, energyOutUnits=energyOutUnits)
        else: # ANINet
            # import here to avoid dependency collision pytorch vs ANI
            from ml_qm.pt.nn import gNNP_computer
            return gNNP_computer.gNNPComputer(nGPU, confFile,
                                              outputGrad, compute_stdev, energyOutUnits=energyOutUnits)


    def _createANI(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        # pylint: disable=W0613
        # import here to avoid dependency collision pytorch vs ANI
        import t_opt.ANI_computer as ANI_computer  # pylint: disable=C0415
        return ANI_computer.ANIComputer(self.nnp_name,
                                        outputGrad, compute_stdev, energyOutUnits=energyOutUnits)
