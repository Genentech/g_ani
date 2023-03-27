## Alberto

import math
import torch
import numpy as np
from collections import OrderedDict

from itertools import combinations_with_replacement
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
import ml_qm.pt.nn.Precision_Util as pu
from builtins import staticmethod


import logging
from ml_qm.pt.ThreeDistBasis import _DistInfo

log = logging.getLogger(__name__)

class ThreeDistSqr():
    """
        This is a replacement for the "angular" descriptor as descripted by Behler and Parinello (BP).

        PB describes a triangle by the angle and the distance of the two adjacent distances.

        Here we describe the triangle by the 3 distances.
        The third distance djk is a squaed distance normalized to the range 0-1 by using the triangle equation:
            djk       = (dik^2-(dij-dik)^2)/(4*dij*dik)

        This description does not use acos() or sqrt and therefore should be differentiable
        at the extremes angle=180,0 where BP is not
    """


    def __init__(self, atomTypes, nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                       trdCenters=None, rCenters=None, cutoff=None, optimizeParam=set([]), device=None):
        # pylint: disable=W0102
        assert cutoff is not None
        assert ((nRadial is not None and rMin is not None, rMin is not None and rCenters is None)
             or (nRadial is     None and rMin is     None, rMax is     None and rCenters is not None)
                ), "either nRadial,rMin and rMin or rCenters needed"
        assert ((nThirdDist is not None and trdCenters is None)
             or (nThirdDist is     None and trdCenters is not None)
                ), "either nThirdDist or trdCenters needed"

        # needs to be sorted so that pairs are always smaller first
        atomTypes.sort()

        self.device = device

        if rCenters is not None:
            self.nRadial = len(rCenters)
            rCenters = torch.tensor(rCenters, dtype=pu.NNP_PRECISION.NNPDType, device=device)
        else:
            self.nRadial = nRadial
            rCenters = np.linspace(rMin, rMax, nRadial, dtype=np.float32)
        # avoid optimizing last radial gaussian as it might go over the cutoff
        self.rCentersOpt   = torch.tensor(rCenters[0:-1], dtype=pu.NNP_PRECISION.NNPDType, device=device)
        self.rCentersFinal = torch.tensor([rCenters[-1]], dtype=pu.NNP_PRECISION.NNPDType, device=device)


        if trdCenters is not None:
            self.nThirdDist = len(trdCenters)
            self.trdCenters = torch.tensor(trdCenters, dtype=pu.NNP_PRECISION.NNPDType, device=device)
        else:
            self.nThirdDist = nThirdDist
            self.trdCenters = torch.linspace(0, 1, nThirdDist, dtype=pu.NNP_PRECISION.NNPDType)


        self.cutoff = torch.tensor([cutoff], dtype=pu.NNP_PRECISION.NNPDType, device=device)
        self.atomTypes = atomTypes
        self.nBasisAtomTypes = len(atomTypes)
        self.nBasis = self.nThirdDist * self.nRadial

        if device is not None:
            self.rCentersOpt = self.rCentersOpt.to(device)
            self.trdCenters = self.trdCenters.to(device)

        self.atomTypePairToIdx = {}
        for idx,atTypePair in enumerate(combinations_with_replacement(atomTypes,2)):
            self.atomTypePairToIdx[atTypePair] = idx

        # mapping of a pair of atom types to a unique number 0 - nPairs = 0 - nat*(nat+1)/2
        maxAt = max(atomTypes)
        self.atomTypePairToIdxT = torch.full((maxAt+1,maxAt+1),-1,dtype=torch.long, device=device)
        for idx,atTypePair in enumerate(combinations_with_replacement(atomTypes,2)):
            self.atomTypePairToIdxT[atTypePair[0],atTypePair[1]] = idx
            self.atomTypePairToIdxT[atTypePair[1],atTypePair[0]] = idx


        self.optParameter = OrderedDict()
        if "rCenter" in optimizeParam:
            self.optParameter["rCenter"] = self.rCentersOpt
        if "trdCenter" in optimizeParam:
            self.optParameter["trdCenter"] = self.trdCenters


    def _computeThirdDistDescriptorsBatch(self, distInfo):
        """
            Overwrite!
            in parameter are as produced by  _computeDistBatch

            Compute thirdDist descriptors for all atoms in dist input
            tensor with [one row per angles][nRadial * nThirdDist] ordered as:
                # rShift  1    2    3      1    2    3      1    2    3
                # 3Shift  1    1    1      2    2    2      3    3    3

            Also return a filter (0,1) that gives the rows filters out of the input:angles due to self.cutoff limits
        """
        raise NotImplementedError


    def _computeDistBatch(self,nConf, nAtPerConf, atTypes, coords):
        ( i_triple2Coords, j_triple2Coords, k_triple2Coords,
          atTypes, distIJ, distIK, distJK ) = \
          ThreeDistSqr._compute_dist_batch_script(nConf, nAtPerConf, atTypes, coords, self.cutoff)

        return _DistInfo(nConf, nAtPerConf,
                         i_triple2Coords, j_triple2Coords, k_triple2Coords,
                         atTypes.reshape(-1), distIJ, distIK, distJK)


    @staticmethod
    @torch.jit.script
    def _compute_dist_batch_script(nConf: int, nAtPerConf:int, atTypes, coords, cutoff):
        """
            cords is numpy

            all return values are torch
            distJK [i,j,k] distance between j and k in  i,j,k normalized by max_dist = (ij+ik)
                nThirdDist is the number  of triples with distances ij and ik < cutoff
            dist[i,j,k] distances indexed by i_angle2Idx, j_angle2Idx, k_angle2Coords
            i_angle22Coords, j_angle2Coords, k_angle2Coords [nThirdDist]
        """
        nTriple= (nAtPerConf * (nAtPerConf-1) * (nAtPerConf-2))//2

        # create indexes for all possible permutations
        i_idx = torch.arange(nAtPerConf, dtype=torch.long, device=coords.device).unsqueeze(1)
        j_idx = i_idx.repeat(1,nAtPerConf,nAtPerConf).reshape(-1)
        k_idx = i_idx.repeat(nAtPerConf,nAtPerConf,1).reshape(-1)
        i_idx = i_idx.repeat(1,nAtPerConf*nAtPerConf).reshape(-1)

        #i_idx, j_idx, k_idx = np.where(np.ones((nAtPerConf,nAtPerConf,nAtPerConf))== 1)
        ltriangle = (j_idx>k_idx) & (i_idx != k_idx) & (i_idx != j_idx)
        i_idx = i_idx[ltriangle]
        j_idx = j_idx[ltriangle]
        k_idx = k_idx[ltriangle]

        distSqr = coords.unsqueeze(1)
        distSqr = distSqr.expand(-1,nAtPerConf,-1,-1)

        vectorsAll = distSqr - distSqr.transpose(1,2)

        #todo use .norm(2, -1)
        distSqr = vectorsAll**2
        distSqr = distSqr.sum(dim=-1)
        distSqr = distSqr + torch.eye(nAtPerConf,dtype=distSqr.dtype, device=distSqr.device) # avoid dist==0 for autograd
        dist = distSqr.sqrt()

        distIJ = dist[:,i_idx,j_idx].reshape(-1)
        distIK = dist[:,i_idx,k_idx].reshape(-1)
        distJK = distSqr[:,j_idx,k_idx].reshape(-1)

        # expand across all conformers
        i_idx = i_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)
        j_idx = j_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)
        k_idx = k_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)

        fltr = (distIJ > cutoff) + (distIK > cutoff) == 0
        fltr[0] = 1           # ensure tensors are not empty
        distIJ = distIJ[fltr]
        distIK = distIK[fltr]
        distJK = distJK[fltr]
        i_idx = i_idx[fltr]
        j_idx = j_idx[fltr]
        k_idx = k_idx[fltr]


        # normalize to be between 0 and 1 based on triangle equation:
        # jk distance must be between (dij-dik)^2 and (dij+dik)^2
        distJK = (distJK - (distIJ - distIK).pow(2)) / (4 * distIJ * distIK)

        # normalize by maximum distance
        nConfCounter  = torch.arange(0,nConf, dtype=torch.long, device=coords.device)
        triple2Conf = nConfCounter.unsqueeze(1).expand(-1, nTriple).reshape(-1)
        triple2Conf = triple2Conf[fltr]*int(nAtPerConf)
        i_triple2Coords = (triple2Conf + i_idx).reshape(-1)
        j_triple2Coords = (triple2Conf + j_idx).reshape(-1)
        k_triple2Coords = (triple2Conf + k_idx).reshape(-1)

        return (i_triple2Coords, j_triple2Coords, k_triple2Coords,
                atTypes.reshape(-1), distIJ, distIK, distJK)




    #@profile
    #@torch.jit.script
    def computeDescriptorBatch(self, coords, atTypes):
        """
            coords[nConf][nAt][xyz]  xyz for each atom
            atTypes [nConf][nAt]     atom type for each atom

            for now all params are np
        """

        nConf, nAt, _ = coords.shape

        nCoords = nConf * nAt
        nBasis = self.nBasis
        nBasisAtomTypes= self.nBasisAtomTypes
        nUniqueJKIdx = nBasisAtomTypes * (nBasisAtomTypes+1)//2

        # diatomic mols have mo angles
        if nAt < 3:
            return torch.zeros((nConf, nAt, nUniqueJKIdx * nBasis), device=self.device)

        if not isinstance(coords, torch.Tensor):
            if self.device is not None:
                coords = torch.from_numpy(coords).to(self.device)
                atTypes= torch.from_numpy(atTypes).to(self.device)
            else:
                coords = torch.from_numpy(coords)
                atTypes= torch.from_numpy(atTypes)

        distInfo = self._computeDistBatch(nConf, nAt, atTypes, coords)

        descriptors = self._computeThirdDistDescriptorsBatch(distInfo)
        #for d in angularDescriptors[0]: warn(d)

        descriptorIdx = distInfo.getDescriptorPositions(nUniqueJKIdx, self.atomTypePairToIdxT)

        res = pu.NNP_PRECISION.indexAdd((nCoords * nUniqueJKIdx, nBasis),
                                                    0, descriptorIdx, descriptors)

        res = res.reshape(nConf,nAt,-1)

        return res



    def nDescriptors(self):
        nAtomTypes = len(self.atomTypes)

        return nAtomTypes * (nAtomTypes+1)//2 * self.nBasis


    def getRCenters(self):
        return torch.cat((self.rCentersOpt, self.rCentersFinal))


    def to(self, device):
        if device is not None:
            if self.rCentersOpt.device != device:
                self.rCentersOpt = self.rCentersOpt.to(device)
                self.rCentersFinal = self.rCentersFinal.to(device)

            if self.trdCenters.device != device:
                self.trdCenters = self.trdCenters.to(device)

            if self.atomTypePairToIdxT.device != device:
                self.atomTypePairToIdxT = self.atomTypePairToIdxT.to(device)

        self.rCentersOpt   = self.rCentersOpt.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.rCentersFinal = self.rCentersFinal.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.trdCenters    = self.trdCenters.to(dtype=pu.NNP_PRECISION.NNPDType)

        # for backwards compatibility with old saved models pre 201904
        if isinstance(self.cutoff,float):
            self.cutoff = torch.tensor([self.cutoff], dtype=pu.NNP_PRECISION.NNPDType, device=device)
        else:
            self.cutoff        = self.cutoff.to(dtype=pu.NNP_PRECISION.NNPDType, device=device)


    def eval(self):
        """ Switch to eval mode -> no parameter optimization """
        for p in self.getOptParameter():
            p.requires_grad_(False)

    def train(self):
        """ Switch to train mode -> no parameter optimization """
        for p in self.getOptParameter():
            p.requires_grad_(True)


    def state_dict(self):
        destination = {}
        name = type(self).__name__
        destination["%s:atomTypes" % name]         = self.atomTypes
        destination["%s:nThirdDist" % name]        = self.nThirdDist
        destination["%s:nRadial" % name]           = self.nRadial
        destination["%s:rCentersOpt"% name]        = self.rCentersOpt.data
        destination["%s:rCentersFinal"% name]      = self.rCentersFinal.data
        destination["%s:trdCenters"% name]         = self.trdCenters.data
        destination["%s:cutoff"% name]             = self.cutoff
        destination["%s:nBasisAtomTypes"% name]    = self.nBasisAtomTypes
        destination["%s:nBasis"% name]             = self.nBasis
        destination["%s:atomTypePairToIdx"% name]  = self.atomTypePairToIdx
        destination["%s:atomTypePairToIdxT"% name] = self.atomTypePairToIdxT

        return destination



    def load_state_dict(self,state):
        name = type(self).__name__

        self.loadIfPresent(name, "atomTypes"        , state)
        self.loadIfPresent(name, "nThirdDist"       , state)
        self.loadIfPresent(name, "nRadial"          , state)
        self.loadIfPresent(name, "cutoff"           , state)
        self.loadIfPresent(name, "nBasisAtomTypes"  , state)
        self.loadIfPresent(name, "nBasis"           , state)
        self.loadIfPresent(name, "atomTypePairToIdx", state)

        ThreeDistSqr.updateIfPresent(self.rCentersOpt,   name, "rCentersOpt", state)
        ThreeDistSqr.updateIfPresent(self.rCentersFinal, name, "rCentersFinal", state)
        ThreeDistSqr.updateIfPresent(self.trdCenters,    name, "trdCenters", state)
        ThreeDistSqr.updateIfPresent(self.atomTypePairToIdxT, name, "atomTypePairToIdxT", state)

    def getOptParameter(self):
        return self.optParameter.values()

    def printOptParam(self):
        for n,p in self.optParameter.items():
            s= "%s" % n
            for v in p.detach().cpu().numpy():
                s += "\t%.5f" % v
            print(s)

    def loadIfPresent(self, prefix, vname, state):
        fullname = "%s:%s" % (prefix,vname)
        if state is not None and fullname in state:
            setattr(self, vname, state[fullname])

    @staticmethod
    def updateIfPresent(var, prefix, vname, state):
        vname = "%s:%s" % (prefix,vname)
        if state is not None and vname in state:
            var.copy_( state[vname]  )



class GaussianThirdDistSqrBasis(ThreeDistSqr):
    """
        Define a set of basis functions to describe the angle with  a distance^2 with nCenter gaussians
    """

    def __init__(self, atomTypes,
                       nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                       trdCenters=None, rCenters=None,
                       halfWidth=None, thirdDistHalfWidth=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        # pylint: disable=W0102
        """
        Define a set of basis functions to describe the angle with  a distance^2 with nCenter gaussians

        Either:
            nRadial: number of centers
            rMin distance of first center
            rMax distance of last center
        or
            centers a list of postions for the basis function centers
        Must be given ant the other values must be None

        halfWidth halfWidth of centers on radius
        cutoff: distance at which the descriptors will be caled to 0

        thirdDistHalfWidth: HalfWidth of gaussian in in third distance normalized to 0-1
        cutoff: dij and dik distance at which the descriptors will be 0
        """

        super().__init__(atomTypes, nThirdDist, nRadial, rMin, rMax,
                                                 trdCenters, rCenters, cutoff,
                                                 optimizeParam, device)

        assert halfWidth is not None
        assert thirdDistHalfWidth is not None

        dtype=pu.NNP_PRECISION.NNPDType

        if np.isscalar(thirdDistHalfWidth):
            thirdDistHalfWidth = torch.full((self.nThirdDist,), thirdDistHalfWidth, dtype=dtype, device=device)
        else:
            thirdDistHalfWidth = torch.tensor(thirdDistHalfWidth, dtype=dtype, device=device)

        self.nEtaThirdDist = math.log(0.5) * 4/(thirdDistHalfWidth*thirdDistHalfWidth)

        if 'nEtaThirdDist' in optimizeParam:
            self.optParameter['nEtaThirdDist'] = self.nEtaThirdDist


        #compute negative eta
        if np.isscalar(halfWidth):
            hw = torch.full((nRadial,), halfWidth, dtype=dtype, device=device)
        else:
            hw = torch.tensor(halfWidth, dtype=dtype, device=device)

        self.nEta = math.log(0.5) * 4/(hw*hw)

        if 'nEta' in optimizeParam:
            self.optParameter['nEta'] = self.nEta

        log.info("%s nBasis=%i cutoff=%.3f\n\tnThirdDist=%i trdCenter=%s\n\tangleHW=%s\n\tnRadial=%i rCenter=%s\n\tradialHW=%s\n\tnEtaThirdDist=%s\n\tnegEta=%s",
            type(self).__name__,
            self.nBasis, cutoff, self.nThirdDist, self.trdCenters, thirdDistHalfWidth,
            nRadial, self.getRCenters(), halfWidth, self.nEtaThirdDist, self.nEta)


    def to(self, device):
        super(GaussianThirdDistSqrBasis, self).to(device)
        if device is not None:
            if self.nEtaThirdDist.device != device:
                self.nEtaThirdDist = self.nEtaThirdDist.to(device)

            if self.nEta.device != device:
                self.nEta = self.nEta.to(device)

        self.nEtaThirdDist = self.nEtaThirdDist.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.nEta          = self.nEta.to(dtype=pu.NNP_PRECISION.NNPDType)

    def _computeThirdDistDescriptorsBatch(self, distInfo):
        shiftR = self.getRCenters()
        shift3 = self.trdCenters
        distJK   = distInfo.distJK
        distIJ   = distInfo.distIJ
        distIK   = distInfo.distIK

        return GaussianThirdDistSqrBasis._compute_3_dist_descriptors(
            self.nBasis, self.nEta, self.nEtaThirdDist, self.cutoff,
            shiftR, shift3, distIJ, distIK, distJK)


    @staticmethod
    @torch.jit.script
    def _compute_3_dist_descriptors(nBasis:int, nEta, nEtaThirdDist, cutoff,
                                    shiftR, shift3, distIJ, distIK, distJK):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nThirdDist] ordered as:
            [nConf][nTriples][nRadial * nThirdDist]
                # rShift  1       2        3         1        2       3        1        2       3
                # 3Shift  0       0        0        0.5      0.5     0.5      1         1       1 normilized thrid dist
        """

        rad1 = (distIJ + distIK)/2
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nThirdDist, nrShifts]

        rad1 = torch.exp(nEta * rad1 * rad1)

        # multiply with cutoff terms
        # todo: try: might be faster
        fc =      (0.5 * torch.cos(math.pi * torch.clamp(distIJ / cutoff,0,1)) + 0.5)
        fc = fc * (0.5 * torch.cos(math.pi * torch.clamp(distIK / cutoff,0,1)) + 0.5)
        rad1 = rad1 * fc.unsqueeze(-1)

        thirdDist = shift3 - distJK.unsqueeze(1)
        thirdDist = torch.exp(nEtaThirdDist * thirdDist * thirdDist)

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (thirdDist.unsqueeze(-1) * rad1).reshape(-1,nBasis)

        return d


    def state_dict(self):
        destination = super(GaussianThirdDistSqrBasis, self).state_dict()

        name = type(self).__name__
        destination["%s:nEtaThirdDist" % name] = self.nEtaThirdDist
        destination["%s:nEta" % name] = self.nEta

        return destination


    def load_state_dict(self,state):
        super(GaussianThirdDistSqrBasis, self).load_state_dict(state)

        name = type(self).__name__
        self.loadIfPresent(name, "nEtaThirdDist", state)
        ThreeDistSqr.updateIfPresent(self.nEta, name, 'nEta',  state)
