# encoding: utf-8
"""
ml_qm.optimize.optimizer -- Reads sdf files and optimizes them using NNP.

@author:     albertgo

@copyright:  2019 Genentech Inc.

"""

import json
import logging
import os

import torch

import ml_qm.pt.nn.Precision_Util as pu
from ml_qm import AtomInfo
from ml_qm.ANIMol import ANIMol
from ml_qm.pt.nn.ani_net import ANINetInitializer
from t_opt.pytorch_computer import PytorchComputer
from t_opt.unit_conversion import Units, KCAL_MOL_to_AU, KCAL_MOL_A_to_AU

log = logging.getLogger(__name__)



class gNNPComputer(PytorchComputer):
    """
        Uses ml_qm package to calculate Neural Net Potentials for a batch of conformations.
        The conformations must have have the same atom count.
    """

    def __init__(self, nGPU:int, confFile:str, outputGrads:bool, compute_stdev:bool,
                 energyOutUnits:Units):
        """
        :param nGPU: Number of GPU to use
        :param confFile: configuration file for network json
        :param outputGrads: tif true gradients will be computed and reported
        :param compute_stdev: if true and the model is an ensample model the std_dev will be reported
        :param energyOutUnits: units for energy output
        """

        with open(confFile) as jFile:
            conf = json.load(jFile)

        device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")
        log.info(f"device={device} nGPU={nGPU} cuda.isavaialble={torch.cuda.is_available()}") # pylint: disable=W1203
        confDir = os.path.dirname(confFile)
        initAni     = ANINetInitializer(conf,device, confDir)
        _model = initAni.create_coordinate_model(compute_stdev)
        _model.eval()

        atomTypes = []
        for at in conf['atomTypes']:
            atomTypes.append(AtomInfo.NameToNum[at])
        allowed_atom_num = frozenset(atomTypes)

        atomization_energies = torch.zeros((max(atomTypes)+1,),dtype=pu.NNP_PRECISION.NNPDType)
        for at in atomTypes:
            atomization_energies[at] = ANIMol.atomEnergies[at]
        atomization_energies = atomization_energies.to(device=device)

        if energyOutUnits is Units.KCAL:
            eFactor = 1.
            fFactor = 1.
        elif energyOutUnits is Units.AU:
            eFactor = KCAL_MOL_to_AU   # if KCAL requested convert after calling NNP
            fFactor = KCAL_MOL_A_to_AU
            atomization_energies *= KCAL_MOL_to_AU
        else:
            raise TypeError("Unknown energy unit: {}".format(energyOutUnits))

        batch_by_atom_order = False  # atoms may be in random order
        mem_gb   = conf['mem_gb']
        memParam = conf['memoryParam']

        super().__init__(_model, allowed_atom_num, atomization_energies,
                         outputGrads, compute_stdev, pu.NNP_PRECISION.NNPDType,
                         mem_gb, memParam, batch_by_atom_order,
                         eFactor, fFactor, energyOutUnits)


    def maxConfsPerBatch(self, nAtom:int) -> int:
        """
        :param nAtom: given a number of atoms compute how many conformations can be processed in one batch on the GPU

        May use self.MEM_GB and self.MEM_PARAM

        Overwrite this to compute more accurate max number of conforamtion based on atom count
        """

        param = self.MEM_PARAM
        if param is None: return super().maxConfsPerBatch(nAtom)

        # mem needed =  const + nConf * ( perConf + nAt * nA_x_nC + nAt^2 * nA2_x_nC + nAt^3 * nA2_x_nC )
        atMem = nAtom * ( param['nA_x_nC'] + nAtom * (param['nA2_x_nC'] + nAtom * param['nA3_x_nC']))

        return max(1,int((self.MEM_GB * 1024 * 1024 * 1024 - param['const']) / ( param['perConf'] + atMem )))
