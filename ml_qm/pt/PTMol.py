# Alberto

import numpy as np
import torch
from ase.neighborlist import neighbor_list
from ase.units import Hartree, Bohr, kcal, mol

from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611

class PTMol():
    """ Pytorch molecule container """


    def __init__(self, baseMol, requires_grad=False):
        self.baseMol = baseMol
        self.atomTensor3D = torch.tensor(baseMol.xyz, dtype=torch.float32)
        self.atomTensor3D.requires_grad_(requires_grad)


    @property
    def energy(self):
        return self.baseMol.energy

    @property
    def name(self):
        return self.baseMol.name
    @property
    def nAt(self):
        return self.baseMol.nAt

    @property
    def atNums(self):
        return self.baseMol.atNums

    @property
    def atoms(self):
        return self.baseMol.atoms

    @property
    def nHeavy(self):
        return self.baseMol.nHeavy


    @property
    def atomizationE(self):
        """ kcal/mol """
        return self.baseMol.atomizationE


    @property
    def nuclearRepulsion(self):
        """ [kcal/mol] """
        d, idx1, idx2 = self.neigborDistanceLT(cutoff=9e999)
        tAtNums = np.array(self.atNums,dtype=np.float32)

        nRepuls = (tAtNums[idx2]*tAtNums[idx1] / d.numpy()).sum()
        nRepuls *= Bohr * Hartree*mol/kcal

        return nRepuls

    @staticmethod
    def myBinCount(t):
        leng = t.max().cpu().item()+1
        #res = t.new(max.cpu().item(), dtype=torch.long64)
        res = torch.empty(leng, dtype=torch.long)
        for i in range(0,leng):
            res[i] = t[t==i].numel()

        return res

    def atNuclearRepulsion(self):

        d, idx1, idx2 = self.neigborDistance(9e5)
        #uidx, org2UniqIdx = idx1.unique(return_inverse=true)
        # bincount avaialble in pytorch 0.4.1

        #uidxCnt = torch.bincount(idx1) # only in torch 4.1
        uidxCnt = PTMol.myBinCount(idx1)

        atNums = torch.tensor(self.atNums, dtype=torch.float32)
        # atomic charge repulsion for each atom pair
        # weighted by the atomic charge of atom[idx1] to atom[idx2]
        atNRePairs = atNums[idx1] * atNums[idx2] / d * \
                     atNums[idx1] / (atNums[idx1]+atNums[idx2])


        atNRePairs = atNRePairs.cumsum(0)
        lastIdxPerAt = uidxCnt.cumsum(0)-1
        cumSum = atNRePairs.take(lastIdxPerAt)
        atNuEnergy = torch.empty_like(cumSum)
        atNuEnergy[0] = cumSum[0]
        atNuEnergy[1:] = cumSum[1:] - cumSum[:-1]

        return atNuEnergy * Bohr * Hartree*mol/kcal


    @property
    def elctronicAtomizationE(self):
        """ [kcal/mol]
            atomization energy minus nuclear repulsion E"""

        return self.atomizationE - self.nuclearRepulsion



    def neigborDistance(self, cutoff=9.):
        """
           return 3 pytorch vectors distance, AtomIndex1, AtomIndex2
                    for non-diagonal elements of distance matrix
        """
        return self._neigborDistance_NP(cutoff)


    def neigborDistanceLT(self, cutoff=9.):
        """
            return 3 pytorch vectors distance, AtomIndex1, AtomIndex2
                    for lower triangle of distance matrix
        """
        return self._neigborDistanceLT_NP(cutoff)



    def _neigborDistance_ASE(self, cutoff=9.):

        """
           For small molecules without periodic boundery conditions this is 100x slower than np 
           return 3 pytorch vectors distance, AtomIndex1, AtomIndex2
                    for non diagonal elements of distance matrix
        """

        # back propagation will be fine with ase use because just for indexes
        row_idx, col_idx = neighbor_list('ij', self.atoms, cutoff=cutoff, max_nbins=1000)
        row_idx = torch.LongTensor(row_idx)
        col_idx = torch.LongTensor(col_idx)

        d = self.atomTensor3D.expand((self.nAt,self.nAt,3))
        d = d - d.transpose(0,1)

        # keep only off diagonal diagonal elements within cutoff
        d = d[row_idx,col_idx,:]

        d = d * d       # sqrt(sum(d^2))
        d = d.sum(-1)
        d = d.sqrt()

        return d, row_idx, col_idx


    def _neigborDistance_NP(self, cutoff=9.):

        """
           return 3 pytorch vectors distance, AtomIndex1, AtomIndex2
        """

        # back propagation will be fine with np use because just as indexes
        row_idx, col_idx = np.where(~np.eye(self.atomTensor3D.shape[0],dtype=bool))
        row_idx = torch.LongTensor(row_idx)
        col_idx = torch.LongTensor(col_idx)

        d = self.atomTensor3D.expand((self.nAt,self.nAt,3))
        d = d - d.transpose(0,1)

        # keep only lower diagonal
        d = d[row_idx,col_idx,:]

        d = d * d       # sqrt(sum(d^2))
        d = d.sum(-1)   # sum xyz

        belowCutoff = d<(cutoff*cutoff)
        d = d[belowCutoff].sqrt()

        return d, row_idx[belowCutoff], col_idx[belowCutoff]

    def _neigborDistanceLT_ASE(self, cutoff=9.):

        """
            For small molecules without periodic boundery conditions this is 100x slower than np 
            return 3 pytorch vectors distance, AtomIndex1, AtomIndex2
                    for lower triangle of distance matrix
        """

        # back propagation will be fine with ase use because just for indexes
        row_idx, col_idx = neighbor_list('ij', self.atoms, cutoff=cutoff, max_nbins=1000)
        row_idxt = torch.LongTensor(row_idx[row_idx > col_idx])
        col_idxt = torch.LongTensor(col_idx[row_idx > col_idx])

        d = self.atomTensor3D.expand((self.nAt,self.nAt,3))
        d = d - d.transpose(0,1)

        # keep only lower diagonal
        d = d[row_idxt,col_idxt,:]

        d = d * d       # sqrt(sum(d^2))
        d = d.sum(-1)
        d = d.sqrt()

        return d, row_idxt, col_idxt


    def _neigborDistanceLT_NP(self, cutoff=9.):

        """
           return 3 pytorch vectors distance, AtomIndex1, AtomIndex2
        """

        # back propagation will be fine with np use because just as indexes
        row_idx, col_idx = np.tril_indices(self.atomTensor3D.shape[0],-1)
        row_idx = torch.LongTensor(row_idx)
        col_idx = torch.LongTensor(col_idx)

        d = self.atomTensor3D.expand((self.nAt,self.nAt,3))
        d = d - d.transpose(0,1)

        # keep only lower diagonal
        d = d[row_idx,col_idx,:]

        d = d * d       # sqrt(sum(d^2))
        d = d.sum(-1)   # sum xyz

        belowCutoff = d<(cutoff*cutoff)
        d = d[belowCutoff].sqrt()

        return d, row_idx[belowCutoff], col_idx[belowCutoff]
