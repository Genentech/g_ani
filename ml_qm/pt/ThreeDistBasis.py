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
log = logging.getLogger(__name__)

class ThreeDist():
    """
        This is a replacement for the "angular" descriptor as descripted by Behler and Parinello (BP).

        PB describes a triangle by the angle and the distance of the two adjacent distances.

        Here we describe the triangle by the 3 distances.
        The third distance djk is normalized to the range 0-1 by using the triangle equation:
            djk       = (djk_max - djk_min) / djk_range
            djk_max   = dij + dik
            dij_min   = max(dij, dik) - min(dij, dik)
            dij_range = min(dij, dik) * 2

        This description does not use acos() and therefore should be differentiable
        at the extremes angle=180,0 where BP is not
    """


    def __init__(self, atomTypes, nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                       trdCenters=None, rCenters=None, cutoff=None, optimizeParam=set([]), device=None):
        """
            optimizeParam list of parameter types to optimize one of: "rCenter", "trdCenter"
               others might be defined by subclasses
        """

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
          ThreeDist._compute_dist_batch_script(nConf, nAtPerConf, atTypes, coords, self.cutoff)

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

        dist = coords.unsqueeze(1)
        dist = dist.expand(-1,nAtPerConf,-1,-1)

        vectorsAll = dist - dist.transpose(1,2)

        #todo use .norm(2, -1)
        dist = vectorsAll**2
        dist = dist.sum(dim=-1)
        dist = dist + torch.eye(nAtPerConf,dtype=dist.dtype, device=dist.device) # avoid dist==0 for autograd
        dist = dist.sqrt()

        distIJ = dist[:,i_idx,j_idx].reshape(-1)
        distIK = dist[:,i_idx,k_idx].reshape(-1)
        distJK = dist[:,j_idx,k_idx].reshape(-1)

        # expand across all conformers
        i_idx = i_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)
        j_idx = j_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)
        k_idx = k_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)

        filter = (distIJ > cutoff) + (distIK > cutoff) == 0
        filter[0] = 1           # ensure tensors are not empty
        distIJ = distIJ[filter]
        distIK = distIK[filter]
        distJK = distJK[filter]
        i_idx = i_idx[filter]
        j_idx = j_idx[filter]
        k_idx = k_idx[filter]

        # normalize to be between 0 and 1 based on triangle equation:
        # jk distance must be between |dij-dik| and dij+dik
        maxDist = torch.max(distIJ, distIK)
        minDist = torch.min(distIJ, distIK)
        distJK = (distJK - maxDist + minDist) / (2 * minDist)

        # normalize by maximum distance
        nConfCounter  = torch.arange(0,nConf, dtype=torch.long, device=coords.device)
        triple2Conf = nConfCounter.unsqueeze(1).expand(-1, nTriple).reshape(-1)
        triple2Conf = triple2Conf[filter]*int(nAtPerConf)
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

        ThreeDist.updateIfPresent(self.rCentersOpt,   name, "rCentersOpt", state)
        ThreeDist.updateIfPresent(self.rCentersFinal, name, "rCentersFinal", state)
        ThreeDist.updateIfPresent(self.trdCenters,    name, "trdCenters", state)
        ThreeDist.updateIfPresent(self.atomTypePairToIdxT, name, "atomTypePairToIdxT", state)

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




class _DistInfo():
    """ distance information container """

    #@profile
    #@torch.jit.script
    def __init__(self, nConf, nAtPerConf,
                 i_triple2Coords, j_triple2Coords, k_triple2Coords,
                 atTypes,
                 distIJ, distIK, distJK ):
        """ distJK[i,j,k]            distance between J and K normalized by the maximum distance
            distIJ[i,j]              distance of pair of atoms ij
            distIk[i,k]              distance of pair of atoms ik
            i_triple2Coords[nThirdDist] coordinateIdx of center atom in angle
            j_triple2Coords[nThirdDist] coordinateIdx of second atom in angle
            k_triple2Coords[nThirdDist] coordinateIdx of third atom in angle
            Ijk_atTypes[nThirdDist]     atomp type of atom in triple
        """

        self.nConf         = nConf
        self.nAt           = nAtPerConf
        self.i_triple2Coords = i_triple2Coords
#        self.j_triple2Coords = j_triple2Coords
#        self.k_triple2Coords = k_triple2Coords
        self.distJK          = distJK
        self.distIJ          = distIJ
        self.distIK          = distIK

        #self.i_AtType = atTypes[i_triple2Coords]
        self.j_AtType = atTypes[j_triple2Coords]
        self.k_AtType = atTypes[k_triple2Coords]



    #@profile
    #@torch.jit.script
    def getDescriptorPositions(self, nUniqueJKIdx, atomTypePairToIdxT):
        """ returns two vectors:
               descCoordIdx for each angle this is the coordinate of the center atom
               descJKUniqIndx: unique index of the atom types JK eg. 0 for HH, 1for HC, ....
        """

        i_triple2Coords = self.i_triple2Coords
        triple2UniqueJKIdx = atomTypePairToIdxT[self.j_AtType,self.k_AtType]

        # Each coord (indexed by i_triple2Coords has
        # nUniqueJKIdx sections for the JK atom types given by triple2UniqueJKIdx
        return i_triple2Coords * nUniqueJKIdx + triple2UniqueJKIdx



class GaussianThirdDistBasis(ThreeDist):
    def __init__(self, atomTypes,
                       nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                       trdCenters=None, rCenters=None,
                       halfWidth=None, thirdDistHalfWidth=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        """
        Define a set of basis functions to describe a distance with nCenter gaussians

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

        super(GaussianThirdDistBasis, self).__init__(atomTypes, nThirdDist, nRadial, rMin, rMax,
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

        log.info("%s nBasis=%i cutoff=%.3f\n\tnThirdDist=%i trdCenter=%s\n\tangleHW=%s\n\tnRadial=%i rCenter=%s\n\tradialHW=%s\n\tnEtaThirdDist=%s\n\tnegEta=%s" % (
            type(self).__name__,
            self.nBasis, cutoff, self.nThirdDist, self.trdCenters, thirdDistHalfWidth,
            nRadial, self.getRCenters(), halfWidth, self.nEtaThirdDist, self.nEta))


    def to(self, device):
        super(GaussianThirdDistBasis, self).to(device)
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

        return GaussianThirdDistBasis._compute_3_dist_descriptors(
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
        destination = super(GaussianThirdDistBasis, self).state_dict()

        name = type(self).__name__
        destination["%s:nEtaThirdDist" % name] = self.nEtaThirdDist
        destination["%s:nEta" % name] = self.nEta

        return destination


    def load_state_dict(self,state):
        super(GaussianThirdDistBasis, self).load_state_dict(state)

        name = type(self).__name__
        self.loadIfPresent(name, "nEtaThirdDist", state)
        ThreeDist.updateIfPresent(self.nEta, name, 'nEta',  state)



class GaussianThirdDist2Basis(GaussianThirdDistBasis):
    def __init__(self, atomTypes,
                       nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                       trdCenters=None, rCenters=None,
                       halfWidth=None, thirdDistHalfWidth=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        """
        Define a set of basis functions to describe a distance with nCenter gaussians
        Both atoms will create a peak at each atoms position in this implementation
        instead of at the average distance as in Behler PArinello

        Either:
            nRadial: number of centers
            rMin distance of first center
            rMax distance of last center
        or
            centers a list of postions for the basis function centers
        Must be given the other values umst be None

        halfWidth halfWidth of centers on radius
        cutoff: distance at which the descriptors will be caled to 0

        thirdDistHalfWidth: HalfWidth of third distance gaussian normalized to 0-1
        cutoff: dij, djk distance at which the descriptors will be 0
        """

        super(GaussianThirdDist2Basis, self).__init__(atomTypes, nThirdDist, nRadial, rMin, rMax,
                                                   trdCenters, rCenters,
                                                   halfWidth, thirdDistHalfWidth, cutoff,
                                                   optimizeParam, device)


    #@profile
    #@torch.jit.script
    def _computeThirdDistDescriptorsBatch(self, distInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nThirdDist] ordered as:
            [nConf][nTriples][nRadial * nThirdDist]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """

        distJK   = distInfo.distJK
        distIJ   = distInfo.distIJ
        distIK   = distInfo.distIK

        shiftR = self.getRCenters()
        shift3 = self.trdCenters

        rad1 = distIJ
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nThirdDist, nrShifts]
        rad1 = torch.exp(self.nEta * rad1 * rad1)
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)

        rad2 = distIK
        rad2 = shiftR - rad2.unsqueeze(1)
        rad2 = torch.exp(self.nEta * rad2 * rad2)
        rad2 = torch.transpose(rad2.unsqueeze(-1), dim0=-2, dim1=-1)

        thirdDist = shift3 - distJK.unsqueeze(1)
        thirdDist = torch.exp(self.nEtaThirdDist * thirdDist * thirdDist)

        # add each component for each radial shift with component for each angular shift
        d1 = (thirdDist.unsqueeze(-1) * rad1).reshape(-1,self.nBasis)
        d2 = (thirdDist.unsqueeze(-1) * rad2).reshape(-1,self.nBasis)

        d = (d1 + d2) / 2

        # multiply with cutoff terms
        d = d * (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5).unsqueeze(-1)
        d = d * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5).unsqueeze(-1)

        return d


class GaussianThirdDistMinBasis(GaussianThirdDistBasis):
    def __init__(self, atomTypes,
                       nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                       trdCenters=None, rCenters=None,
                       halfWidth=None, thirdDistHalfWidth=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        """
        Define a set of basis functions to describe a distance with nCenter gaussians
        Same as GaussianThirdDistBasis using min(Dij,Dik)
        """

        super(GaussianThirdDistMinBasis, self).__init__(atomTypes, nThirdDist, nRadial, rMin, rMax,
                                                   trdCenters, rCenters,
                                                   halfWidth, thirdDistHalfWidth, cutoff,
                                                   optimizeParam, device)


    #@profile
    #@torch.jit.script
    def _computeThirdDistDescriptorsBatch(self, distInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nThirdDist] ordered as:
            [nConf][nTriples][nRadial * nThirdDist]
                # rShift  1       2        3         1        2       3        1        2       3
                # 3Shift  0       0        0        0.5      0.5     0.5      1         1       1 normilized thrid dist
        """

        distJK   = distInfo.distJK * distInfo.distJK
        distIJ   = distInfo.distIJ
        distIK   = distInfo.distIK

        shiftR = self.getRCenters()
        shift3 = self.trdCenters

        rad1 = torch.min(distIJ, distIK)
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nThirdDist, nrShifts]

        rad1 = torch.exp(self.nEta * rad1 * rad1)

        # multiply with cutoff terms
        # todo: try: might be faster
        fc =      (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5)
        fc = fc * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5)
        rad1 = rad1 * fc.unsqueeze(-1)

#         rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5).unsqueeze(-1)
#         rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5).unsqueeze(-1)

        thirdDist = shift3 - distJK.unsqueeze(1)
        thirdDist = torch.exp(self.nEtaThirdDist * thirdDist * thirdDist)

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (thirdDist.unsqueeze(-1) * rad1).reshape(-1,self.nBasis)

        return d



class GaussianThirdDistCombBasis(ThreeDist):
    def __init__(self, atomTypes,
                       nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                       trdCenters=None, rCenters=None,
                       halfWidth=None, thirdDistFactor=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        """
        Define a set of basis functions to describe a distance with nCenter gaussians
        The radial and angular components are combined in one by Pytagoras before computing the gaussain.

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

        super(GaussianThirdDistCombBasis, self).__init__(atomTypes, nThirdDist, nRadial, rMin, rMax,
                                                 trdCenters, rCenters, cutoff,
                                                 optimizeParam, device)

        assert halfWidth is not None
        assert thirdDistFactor is not None

        dtype=pu.NNP_PRECISION.NNPDType

        if np.isscalar(thirdDistFactor):
            self.thirdDistFactor = torch.full((self.nThirdDist,), thirdDistFactor, dtype=dtype, device=device)
        else:
            self.thirdDistFactor = torch.tensor(thirdDistFactor, dtype=dtype, device=device)

        if 'thirdDistFactor' in optimizeParam:
            self.optParameter['thirdDistFactor'] = self.thirdDistFactor


        #compute negative eta
        if np.isscalar(halfWidth):
            hw = torch.full((nRadial,), halfWidth, dtype=dtype, device=device)
        else:
            hw = torch.tensor(halfWidth, dtype=dtype, device=device)

        self.nEta = math.log(0.5) * 4/(hw*hw)

        if 'nEta' in optimizeParam:
            self.optParameter['nEta'] = self.nEta

        log.info("%s nBasis=%i cutoff=%.3f\n\tnThirdDist=%i trdCenter=%s\n\tangleF=%s\n\tnRadial=%i rCenter=%s\n\tradialHW=%s\n\tthirdDistFactor=%s\n\tnegEta=%s" % (
            type(self).__name__,
            self.nBasis, cutoff, self.nThirdDist, self.trdCenters, thirdDistFactor,
            nRadial, self.getRCenters(), halfWidth, self.thirdDistFactor, self.nEta))


    def to(self, device):
        super(GaussianThirdDistCombBasis, self).to(device)
        if device is not None:
            if self.thirdDistFactor.device != device:
                self.thirdDistFactor = self.thirdDistFactor.to(device)

            if self.nEta.device != device:
                self.nEta = self.nEta.to(device)

        self.thirdDistFactor = self.thirdDistFactor.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.nEta            = self.nEta.to(dtype=pu.NNP_PRECISION.NNPDType)


    #@torch.jit.script
    def _computeThirdDistDescriptorsBatch(self, distInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nThirdDist] ordered as:
            [nConf][nTriples][nRadial * nThirdDist]
                # rShift  1       2        3         1        2       3        1        2       3
                # 3Shift  0       0        0        0.5      0.5     0.5      1         1       1 normilized thrid dist
        """

        distJK   = distInfo.distJK
        distIJ   = distInfo.distIJ
        distIK   = distInfo.distIK

        shiftR = self.getRCenters()
        shift3 = self.trdCenters

        rad1 = (distIJ + distIK)/2
        rad1 = shiftR - rad1.unsqueeze(1)
        rad1 = rad1 * rad1

        thirdDist = shift3 - distJK.unsqueeze(1)
        thirdDist = thirdDist * thirdDist * self.thirdDistFactor

        combDist = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        combDist = (thirdDist.unsqueeze(-1) + combDist).reshape(-1,self.nBasis)

        # combDist is [nConf,nBasis] with the nBasis being nRad*nTrd with the nRad index running first
        # therefore nEta needs to be expanded to get the nBasis size
        nEta = self.nEta.expand(self.thirdDistFactor.shape[0],-1).flatten()
        combDist = torch.exp(combDist.sqrt() * nEta)

        fc = (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5)
        fc = fc * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5)
        combDist = combDist * fc.unsqueeze(1)

        return combDist


    def state_dict(self):
        destination = super(GaussianThirdDistCombBasis, self).state_dict()

        name = type(self).__name__
        destination["%s:thirdDistFactor" % name] = self.thirdDistFactor
        destination["%s:nEta" % name] = self.nEta

        return destination


    def load_state_dict(self,state):
        super(GaussianThirdDistCombBasis, self).load_state_dict(state)

        name = type(self).__name__
        self.loadIfPresent(name, "thirdDistFactor", state)
        ThreeDist.updateIfPresent(self.nEta, name, 'nEta',  state)



class Bump2ThirdDistanceBasis(ThreeDist):
    def __init__(self, atomTypes,
                       nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                       trdCenters=None, rCenters=None,
                       halfWidth=None, maxWidthMultiplier=None,
                       thirdDistHalfWidth=None, max3rdDistWidthMultiplier=None,
                       optimizeParam=set(), device=None ):

        """

        #### Note this does not show correct behavior
        ### need to rethink addition vs multiplication of radial terms
        ### need to rethink addition vs multiplication of angular term
        ### currently large values are only computed if delatR1 ~= shiftR and deltaR2 ~= same shiftR

        Define a set of basis functions to describe a distance with nCenter gaussians

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth HalfWidth of centers on radius
        thirdDistHalfWidth: HalfWidth of 3rdDist gaussian normalized to 0-1
        maxWidthMultiplier: halfWidth*maxRadiusMultiplier = radius at which bump function is 0

        The formula used is exp(rC1/(1-deltaR^2/rC2) - rC1 + aC1/(1-deltaA^2/aC2) - AC1)
           c1 and c2 are adjusted by the halfWidth and the maxRadiusMultiplier
        """
        assert ((nRadial is not None and rMin is not None, rMax is not None and rCenters is None)
             or (nRadial is     None and rMin is     None, rMax is     None and rCenters is not None)
                ), "either nRadial,rMin and rMax or rCenters needed"
        assert ((nThirdDist is not None and trdCenters is None)
             or (nThirdDist is     None and trdCenters is not None)
                ), "either nThirdDist or trdCenters needed"

        cutoff = self._computeCutoff(nRadial, rMax, rCenters, halfWidth, maxWidthMultiplier)
        dtype=pu.NNP_PRECISION.NNPDType

        super(Bump2ThirdDistanceBasis, self).__init__(atomTypes, nThirdDist, nRadial, rMin, rMax,
                                                trdCenters, rCenters, cutoff,
                                                optimizeParam, device)

        if np.isscalar(halfWidth):
            assert np.isscalar(maxWidthMultiplier), "halfWidth and maxWidthMultiplier must be of same length"

            maxWidth = halfWidth * maxWidthMultiplier
            c2Base = maxWidth * maxWidth / 4.
            c1Base = math.log(0.5) * maxWidth * maxWidth / halfWidth / halfWidth - math.log(0.5)

            # only the first len-1 c2 will be optimized to avoid the last going into cutoff
            self.rC2Opt = torch.full((nRadial-1,), c2Base, dtype=dtype, device=device)
            self.rC2Final = torch.tensor([c2Base], dtype=dtype, device=device)
            self.rC1 = torch.full((nRadial,), c1Base, dtype=dtype, device=device)

            del maxWidth, c1Base, c2Base
        else:
            assert len(maxWidthMultiplier) == len(halfWidth), "halfWidth and maxWidthMultiplier must be of same length"

            halfWidth          = np.array(halfWidth, dtype=np.float32)
            maxWidthMultiplier = np.array(maxWidthMultiplier, dtype=np.float32)
            maxWidth           = halfWidth * maxWidthMultiplier

            # read center parameters from lists arguments
            c2 = maxWidth * maxWidth / 4.
            c1 = math.log(0.5) * maxWidth * maxWidth / halfWidth / halfWidth - math.log(0.5)

            self.rC2Opt   = torch.tensor(c2[0:-1], dtype=dtype, device=device)
            self.rC2Final = torch.tensor([c2[-1]], dtype=dtype, device=device)
            self.rC1 = torch.tensor(c1, dtype=dtype, device=device)

            del maxWidth, c1, c2

        if np.isscalar(thirdDistHalfWidth):
            assert np.isscalar(max3rdDistWidthMultiplier), "halfWidth and max3rdDistWidthMultiplier must be of same length"

            maxWidth = thirdDistHalfWidth * max3rdDistWidthMultiplier
            c2Base = maxWidth * maxWidth / 4. * math.pi * math.pi
            c1Base = math.log(0.5) * maxWidth * maxWidth / thirdDistHalfWidth / thirdDistHalfWidth - math.log(0.5)

            # only the first len-1 c2 will be optimized to avoid the last going into cutoff
            self.aC2 = torch.full((self.nThirdDist,), c2Base, dtype=dtype, device=device)
            self.aC1 = torch.full((self.nThirdDist,), c1Base, dtype=dtype, device=device)
        else:
            assert len(max3rdDistWidthMultiplier) == len(thirdDistHalfWidth), "halfWidth and maxWidthMultiplier must be of same length"

            thirdDistHalfWidth = np.array(thirdDistHalfWidth, dtype=np.float32)
            max3rdDistWidthMultiplier = np.array(max3rdDistWidthMultiplier, dtype=np.float32)
            maxWidth =  thirdDistHalfWidth * max3rdDistWidthMultiplier

            # read center parameters from lists arguments
            c2 = maxWidth * maxWidth / 4. * math.pi * math.pi
            c1 = math.log(0.5) * maxWidth * maxWidth / thirdDistHalfWidth / thirdDistHalfWidth - math.log(0.5)

            self.aC2 = torch.tensor(c2, dtype=pu.NNP_PRECISION.NNPDType, device=device)
            self.aC1 = torch.tensor(c1, dtype=pu.NNP_PRECISION.NNPDType, device=device)


        log.info("%s with cutoff=%.4f, nBasis=%i, nThirdDist=%i nRadial=%i\n\t3rdDistHW=%s\n\tradialHW=%s\n\trC1=%s\n\trC2=%s\n\taC1=%s\n\taC2=%s" % (
            type(self).__name__, self.cutoff, self.nBasis, nThirdDist, nRadial, thirdDistHalfWidth, halfWidth,
            self.rC1, self.getRC2(), self.aC1, self.aC2))

        if "rC1" in optimizeParam:
            self.optParameter['rC1'] = self.rC1
        if "rC2" in optimizeParam:
            self.optParameter['rC2Opt'] = self.rC2Opt

        if "aC1" in optimizeParam:
            self.optParameter['aC1'] = self.aC1
        if "aC2" in optimizeParam:
            self.optParameter['aC2'] = self.aC2



    #@torch.jit.script
    def _computeCutoff(self, nRadial, rMax, rCenters, halfWidth, maxWidthMultiplier):
        mwm = maxWidthMultiplier if np.isscalar(maxWidthMultiplier) else maxWidthMultiplier[-1]
        hw = halfWidth if np.isscalar(halfWidth) else halfWidth[-1]
        if nRadial is None:
            cutoff = rCenters[-1] + hw / 2. * mwm
        else:
            cutoff = rMax + hw / 2. * mwm
        return cutoff



    def getRC2(self):
        return torch.cat((self.rC2Opt, self.rC2Final))


    def to(self, device):
        super(Bump2ThirdDistanceBasis, self).to(device)
        if device is not None:
            if self.rC1.device != device:
                self.rC1 = self.rC1.to(device)

            if self.rC2Opt.device != device:
                self.rC2Opt   = self.rC2Opt.to(device)
                self.rC2Final = self.rC2Final.to(device)

            if self.aC1.device != device:
                self.aC1 = self.aC1.to(device)

            if self.aC2.device != device:
                self.aC2 = self.aC2.to(device)


    def state_dict(self):
        destination = super(Bump2ThirdDistanceBasis, self).state_dict()

        name = type(self).__name__
        destination["%s:rC1"      % name] = self.rC1
        destination["%s:rC2Opt"   % name] = self.rC2Opt
        destination["%s:rC2Final" % name] = self.rC2Final
        destination["%s:aC1"      % name] = self.aC1
        destination["%s:aC2"      % name] = self.aC2

        return destination


    def load_state_dict(self,state):
        super(Bump2ThirdDistanceBasis, self).load_state_dict(state)

        name = type(self).__name__
        self.updateIfPresent(self.rC1,      name, "rC1", state)
        self.updateIfPresent(self.rC2Opt,   name, "rC2Opt", state)
        self.updateIfPresent(self.rC2Final, name, "rC2Final", state)
        self.updateIfPresent(self.aC1,      name, "aC1", state)
        self.updateIfPresent(self.aC2,      name, "aC2", state)



    #@profile
    #@torch.jit.script
    def _computeThirdDistDescriptorsBatch(self, distInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nThirdDist] ordered as:
            [nConf][nTriples][nRadial * nThirdDist]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """


        distJK  = distInfo.distJK
        rad1    = distInfo.distIJ
        rad2    = distInfo.distIK

        shiftR = self.getRCenters()
        rC2 = self.getRC2()
        shift3 = self.trdCenters

        # expand to have one row per triple i,j,k
        rad1 = shiftR - rad1.unsqueeze(1)
        rad2 = shiftR - rad2.unsqueeze(1)

        # rad1 is [nThirdDist][nrShifts]
        thirdDist = shift3 - distJK.unsqueeze(1)
        thirdDist = self.aC1 / (1. - torch.clamp(thirdDist * thirdDist / self.aC2, 0.,0.9999)) - self.aC1
        # thirdDist is [nThirdDist][nAShifts]

        rad1  =        self.rC1 / (1. - torch.clamp(rad1 * rad1 / rC2, 0.,0.9999)) - self.rC1
        rad1  = rad1 + self.rC1 / (1. - torch.clamp(rad2 * rad2 / rC2 ,0.,0.9999)) - self.rC1
        rad1 = rad1/2.

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (thirdDist.unsqueeze(-1) + rad1).reshape(-1, self.nBasis)

        d = torch.exp( d )

        return d


class BumpThirdDistanceBasis(Bump2ThirdDistanceBasis):
    def __init__(self, atomTypes,
                      nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                      trdCenters=None, rCenters=None,
                      halfWidth=None, maxWidthMultiplier=None,
                      thirdDistHalfWidth=None, max3rdDistWidthMultiplier=None,
                      optimizeParam=set(), device=None ):
        """
        Note: We need to reevaluate the cutoff computation since the average of dij and dik is used.

        Define a set of basis functions to describe a distance with nCenter gaussians

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth HalfWidth of centers on radius
        thirdDistHalfWidth: HalfWidth of 3rdDist gaussian normalized to 0-1
        maxWidthMultiplier: halfWidth*maxRadiusMultiplier = radius at which bump function is 0

        The formula used is exp(rC1/(1-deltaR^2/rC2) - rC1 + aC1/(1-deltaA^2/aC2) - AC1)
           c1 and c2 are adjusted by the halfWidth and the maxRadiusMultiplier
        """

        super(BumpThirdDistanceBasis, self).__init__(atomTypes, nThirdDist, nRadial,  rMin, rMax,
                                               trdCenters, rCenters,
                                               halfWidth, maxWidthMultiplier,
                                               thirdDistHalfWidth, max3rdDistWidthMultiplier, optimizeParam, device)


    #@torch.jit.script
    def _computeCutoff(self, nRadial, rMax, rCenters, halfWidth, maxWidthMultiplier):

        # Need to multiply by 2 because BumpAngularBasis is using average distance

        if nRadial is None:
            mwm = maxWidthMultiplier if np.isscalar(maxWidthMultiplier) else maxWidthMultiplier[-1]
            hw = halfWidth if np.isscalar(halfWidth) else halfWidth[-1]
            cutoff = rCenters[-1] + hw / 2. * mwm
        else:
            cutoff = rMax + halfWidth / 2. * maxWidthMultiplier
        return cutoff * 2.


    #@profile
    #@torch.jit.script
    def _computeThirdDistDescriptorsBatch(self, distInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nThirdDist] ordered as:
            [nConf][nTriples][nRadial * nThirdDist]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """

        distJK   = distInfo.distJK
        distIJ   = distInfo.distIJ
        distIK   = distInfo.distIK

        shiftR = self.getRCenters()
        rC2 = self.getRC2()
        shift3 = self.trdCenters

        rad1 = (distIJ + distIK)/2

        # verticalize and expand to have one row per triple i,j,k
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nThirdDist][nrShifts]
        thirdDist = shift3 - distJK.unsqueeze(1)
        thirdDist = self.aC1 / (1. - torch.clamp(thirdDist * thirdDist / self.aC2, 0.,0.9999)) - self.aC1
        # thirdDist is [nThirdDist][nAShifts]

        rad1 = self.rC1 / (1. - torch.clamp(rad1 * rad1 /      rC2, 0.,0.9999)) - self.rC1

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (thirdDist.unsqueeze(-1) + rad1).reshape(-1, self.nBasis)

        d = torch.exp( d )

        return d



class Bump3ThirdDistanceBasis(Bump2ThirdDistanceBasis):
    def __init__(self, atomTypes,
                      nThirdDist=None, nRadial=None, rMin=None, rMax=None,
                      trdCenters=None, rCenters=None,
                      halfWidth=None, maxWidthMultiplier=None,
                      thirdDistHalfWidth=None, max3rdDistWidthMultiplier=None,
                      cutoff=None, optimizeParam=set(), device=None ):
        """
        Define a set of basis functions to describe a distance with nCenter gaussians

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth HalfWidth of centers on radius
        thirdDistHalfWidth: HalfWidth of 3rdDist gaussian normalized to 0-1
        maxWidthMultiplier: halfWidth*maxRadiusMultiplier = radius at which bump function is 0

        The formula used is exp(rC1/(1-deltaR^2/rC2) - rC1 + aC1/(1-deltaA^2/aC2) - AC1)
           c1 and c2 are adjusted by the halfWidth and the maxRadiusMultiplier
        """

        super(Bump3ThirdDistanceBasis, self).__init__(atomTypes, nThirdDist, nRadial,  rMin, rMax,
                                               trdCenters, rCenters,
                                               halfWidth, maxWidthMultiplier,
                                               thirdDistHalfWidth, max3rdDistWidthMultiplier, optimizeParam, device)


    #@profile
    #@torch.jit.script
    def _computeThirdDistDescriptorsBatch(self, distInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nThirdDist] ordered as:
            [nConf][nTriples][nRadial * nThirdDist]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """


        distJK = distInfo.distJK
        distIJ = distInfo.distIJ
        distIK = distInfo.distIK

        shiftR = self.getRCenters()
        rC2 = self.getRC2()
        shift3 = self.trdCenters

        # expand to have one row per triple i,j,k
        rad1 = shiftR - distIJ.unsqueeze(1)
        rad2 = shiftR - distIK.unsqueeze(1)

        # rad1 is [nThirdDist][nrShifts]
        thirdDist  = shift3 - distJK.unsqueeze(1)
        # thirdDist is [nThirdDist][nAShifts]

        rad1  = self.rC1 / (1. - torch.clamp(rad1 * rad1 / rC2, 0.,0.9999)) - self.rC1
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)

        rad2  = self.rC1 / (1. - torch.clamp(rad2 * rad2 / rC2 ,0.,0.9999)) - self.rC1
        rad2 = torch.transpose(rad2.unsqueeze(-1), dim0=-2, dim1=-1)

        thirdDist  = self.aC1 / (1. - torch.clamp(thirdDist  * thirdDist  / self.aC2, 0.,0.9999)) - self.aC1

        # add each component for each radial shift with component for each angular shift
        d1 = (thirdDist.unsqueeze(-1) + rad1).reshape(-1, self.nBasis)
        d2 = (thirdDist.unsqueeze(-1) + rad2).reshape(-1, self.nBasis)

        d = (torch.exp( d1 ) + torch.exp( d2 )) / 2

        # needed or this would have instability when one neighbor moves out of cutoff
        d = d * (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5).unsqueeze(-1)
        d = d * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5).unsqueeze(-1)

        return d
