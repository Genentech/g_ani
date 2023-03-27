# encoding: utf-8
"""
ml_qm.optimize.optimizer -- Reads sdf files and optimizes them using NNP.

@author:     albertgo

@copyright:  2019 Genentech Inc.

"""

import json
import torch
import logging
import os
import argparse
from torch import autograd

from cdd_chem.io import get_mol_input_stream
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from cdd_chem.util import log_helper
from t_opt import atom_info
from ml_qm.pt.nn.ani_net import ANINetInitializer, SameSizeCoordsBatch
import ml_qm.pt.nn.Precision_Util as pu
from typing import Optional


log = logging.getLogger(__name__)

# noinspection PyUnboundLocalVariable
_static_grad_checker: Optional[NNPGradCheck] = None  # pylint: disable=E0601, # noqa:F821


class NNPGradCheck():
    """ To check quality of gradient """

    def __init__(self, nGPU:int, confJson:str):

        with open(confJson) as jFile:
            conf = json.load(jFile)

        self.device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")
        confDir = os.path.dirname(confJson)
        initAni     = ANINetInitializer(conf,self.device, confDir)
        self._model = initAni.create_coordinate_model(True)
        self._model.eval()

        atomTypes = []
        for at in conf['atomTypes']:
            atomTypes.append(atom_info.NameToNum[at])
        allowed_atom_num = frozenset(atomTypes)
        self.coords_batch = SameSizeCoordsBatch(allowed_atom_num, pu.NNP_PRECISION.NNPDType)
        self.nconf_ones = torch.ones((self.coords_batch), dtype=pu.NNP_PRECISION.NNPDType, device=self.device )


    def check_grad(self):
        # pylint: disable=W0603
        global _static_grad_checker

        self.coords_batch.collectConformers(True, self.device)
        _static_grad_checker = self
        coords = self.coords_batch.coords

        warn(torch.autograd.gradcheck(NNPGradCheck.compute_energy, coords, eps=1e-05, atol=1e-05, rtol=0.001, raise_exception=True))

    @staticmethod
    def compute_energy(coords:torch.tensor) -> torch.tensor:
        # pylint: disable=W0603
        global _static_grad_checker

        assert _static_grad_checker is not None

        # Overwrite coords_batch with new coordinates, must be same size
        c_batch = _static_grad_checker.coords_batch
        c_batch.coords = coords
        c_batch.zero_grad()

        energies, _ = _static_grad_checker._model.forward(c_batch) # pylint: disable=W0212

        return energies


    def compute_grad(self, energies):
        energies.backward(self.nconf_ones, retain_graph=True)
        energies.detach_()
        grad = self.coords_batch.coords.grad.data

        return grad


    def num_atoms(self):
        return self.coords_batch.n_atom_per_conf


if __name__ == '__main__':
    # Check gradients

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-in' ,  metavar='fileName',  dest='inFile', type=str, default=".sdf",
                        help='input file def=.sdf')

    parser.add_argument('-conf' ,  metavar='NNP.json',  dest='confFile', type=str, required=True,
                        help='input file *.json of ANI directory')

    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')

    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')

    args = parser.parse_args()

    log_helper.initialize_loggger(__name__, args.logFile)


    with autograd.detect_anomaly():
        with get_mol_input_stream(args.inFile)   as molS:

            n_at = 0
            for mol in molS:
                if n_at == 0:
                    nnp_grad_check = NNPGradCheck(args.nGPU, args.confFile)
                    n_at = mol.num_atoms
                elif n_at != nnp_grad_check.num_atoms():
                    nnp_grad_check.check_grad()
                    n_at = 0

                nnp_grad_check.coords_batch.addConformerMol(mol)

            if n_at != nnp_grad_check.num_atoms():
                nnp_grad_check.check_grad()
