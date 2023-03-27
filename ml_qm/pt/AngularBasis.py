## Alberto
# pylint: disable=C0302

import ml_qm.pt.nn.Precision_Util as pu
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from itertools import combinations_with_replacement
import math
import torch
import numpy as np
from collections import OrderedDict

import logging
log = logging.getLogger(__name__)


class AngularBasis():
    """ Basis for Atomic environment descriptors that describes triple of atoms and their angle """

    def __init__(self, atomTypes, nAngles=None, nRadial=None, rMin=None, rMax=None,
                       aCenters=None, rCenters=None, cutoff=None, optimizeParam=set([]), device=None):
        # pylint: disable=W0102
        """
            optimizeParam list of parameter types to optimize one of: "rCenter", "aCenter"
               others might be defined by subclasses
        """

        assert cutoff is not None
        assert ((nRadial is not None and rMin is not None, rMin is not None and rCenters is None)
             or (nRadial is     None and rMin is     None, rMax is     None and rCenters is not None)
                ), "either nRadial,rMin and rMin or rCenters needed"
        assert ((nAngles is not None and aCenters is None)
             or (nAngles is     None and aCenters is not None)
                ), "either nAngles or aCenters needed"

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


        if aCenters is not None:
            self.nAngles = len(aCenters)
            self.aCenters = torch.tensor(aCenters, dtype=pu.NNP_PRECISION.NNPDType, device=device)
        else:
            self.nAngles = nAngles
            self.aCenters = torch.linspace(0, math.pi, nAngles, dtype=pu.NNP_PRECISION.NNPDType)


        self.cutoff = cutoff
        self.atomTypes = atomTypes
        self.nBasisAtomTypes = len(atomTypes)
        self.nBasis = self.nAngles * self.nRadial

        if device is not None:
            self.rCentersOpt = self.rCentersOpt.to(device)
            self.aCenters = self.aCenters.to(device)

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
        if "aCenter" in optimizeParam:
            self.optParameter["aCenter"] = self.aCenters


    def _computeAngularDescriptors(self, mol):
        """
            Overwrite!
            Compute angular descriptors for all atoms of mol
            as in formula 4 in ANI--1 paper
            Return list of atom indices for centr, neighbor1 and neighbor2 as well as descriptor array

        """
        raise NotImplementedError


    def _computeAngularDescriptorsBatch(self, angInfo):
        """
            Overwrite!
            in parameter are as produced by  _computeAnglesBatch

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nAngles] ordered as:
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles

            Also return a filter (0,1) that gives the rows filteres out of the input:angles due to self.cutoff limits
        """
        raise NotImplementedError


    #@torch.jit.script
    def _computeAnglesBatch(self,nConf, nAtPerConf, atTypes, coords):
        """
            cords is numpy

            all return values are torch
            ang [i,j,k] with angles indexed by i_angle2Coords, j_angle2Coords, k_angle2Coords
                i_angle2Idx[nAngles] being the center of the angle
                nAngles is the number  of angles with distances < cutoff
            dist[i,j,k] distances indexed by i_angle2Idx, j_angle2Idx, k_angle2Coords
            i_angle22Coords, j_angle2Coords, k_angle2Coords [nAngles]
        """
        nTriple       = (nAtPerConf * (nAtPerConf-1) * (nAtPerConf-2))//2
        cutoff = self.cutoff

        # create indexes for all possible permutations
        i_idx = torch.arange(nAtPerConf, dtype=torch.long, device=self.device).unsqueeze(1)
        k_idx = i_idx.repeat(nAtPerConf,nAtPerConf,1).reshape(-1)
        j_idx = i_idx.repeat(1,nAtPerConf,nAtPerConf).reshape(-1)
        i_idx = i_idx.repeat(1,nAtPerConf*nAtPerConf).reshape(-1)

        #i_idx, j_idx, k_idx = np.where(np.ones((nAtPerConf,nAtPerConf,nAtPerConf))== 1)
        ltriangle = (j_idx>k_idx) & (i_idx != k_idx) & (i_idx != j_idx)
        i_idx = i_idx[ltriangle]
        j_idx = j_idx[ltriangle]
        k_idx = k_idx[ltriangle]

        dist = coords.unsqueeze(1)
        dist = dist.expand(-1,nAtPerConf,-1,-1)

        vectorsAll = dist - dist.transpose(1,2)
        # can we work with: filter = dist<self.cutoff
        # smaller filter but then we have to expand to cover all tripples
        # could we use:
        #   filter = dist<cutoff
        #   distIJ = scatter(???)
        # might be difficult: we need to have the indexes for scatter ahead and then filter

        vectorsIJ = vectorsAll[:,i_idx,j_idx].reshape(-1,3)
        vectorsIK = vectorsAll[:,i_idx,k_idx].reshape(-1,3)

        #todo use .norm(2, -1)
        dist = vectorsAll**2
        dist = dist.sum(dim=-1)
        dist = dist + torch.eye(nAtPerConf,dtype=dist.dtype, device=dist.device) # avoid dist==0 for autograd
        dist = dist.sqrt()

        distIJ = dist[:,i_idx,j_idx].reshape(-1)
        distIK = dist[:,i_idx,k_idx].reshape(-1)

        # expand across all conformers
        i_idx = i_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)
        j_idx = j_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)
        k_idx = k_idx.unsqueeze(0).expand(nConf,-1).reshape(-1)

        fltr = (distIJ > cutoff) + (distIK > cutoff) == 0
        fltr[0] = 1           # ensure tensors are not empty
        distIJ = distIJ[fltr]
        distIK = distIK[fltr]
        vectorsIJ = vectorsIJ[fltr]
        vectorsIK = vectorsIK[fltr]
#         distIJ = vectorsIJ.norm(2, dim=-1)
#         distIK = vectorsIK.norm(2, dim=-1)
        i_idx = i_idx[fltr]
        j_idx = j_idx[fltr]
        k_idx = k_idx[fltr]

        nConfCounter  = torch.arange(0,nConf, dtype=torch.long, device=coords.device)
        triple2Conf = nConfCounter.unsqueeze(1).expand(-1, nTriple).reshape(-1)
        triple2Conf = triple2Conf[fltr]*int(nAtPerConf)
        i_triple2Coords = (triple2Conf + i_idx).reshape(-1)
        j_triple2Coords = (triple2Conf + j_idx).reshape(-1)
        k_triple2Coords = (triple2Conf + k_idx).reshape(-1)


        # Vector multiply each vector in vectors(from atom ix to others within cutoff)
        # to all other vectors from ix
        # and compute angle
        #print("j j1 %s %s" %(j,j1))
        #todo use torch distance calc of cosine sim
        cosAng = torch.nn.functional.cosine_similarity(vectorsIJ,vectorsIK)
#         ang = (vectorsIJ * vectorsIK).sum(-1)
#         ang /= distIJ * distIK
#         ang = torch.acos(ang * 0.99999)

        return _BatchAngularInfo(nConf, nAtPerConf,
                                 i_triple2Coords, j_triple2Coords, k_triple2Coords,
                                 atTypes.reshape(-1), distIJ, distIK, cosAng)



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

        # diatomic mols have no angles
        if nAt < 3:
            return torch.zeros((nConf, nAt, nUniqueJKIdx * nBasis), device=self.device)

        if not isinstance(coords, torch.Tensor):
            if self.device is not None:
                coords = torch.from_numpy(coords).to(self.device)
                atTypes= torch.from_numpy(atTypes).to(self.device)
            else:
                coords = torch.from_numpy(coords)
                atTypes= torch.from_numpy(atTypes)

        angInfo = self._computeAnglesBatch(nConf, nAt, atTypes, coords)

        angularDescriptors = self._computeAngularDescriptorsBatch(angInfo)
        #for d in angularDescriptors[0]: warn(d)

        descriptorIdx = angInfo.getDescriptorPositions(nUniqueJKIdx, self.atomTypePairToIdxT)

        # TODO add nUniqueJKIdx dimension to allow more parallelization
        # switch to double
        res = pu.NNP_PRECISION.indexAdd((nCoords * nUniqueJKIdx, nBasis),
                                                     0, descriptorIdx, angularDescriptors)
        res = res.reshape(nConf,nAt,-1)

        return res




    def computeDescriptors(self, mol):
        """
            mol -- conformation as eg. PTMol

            returns -- map from central atom type to array of descriptors for each atom
               # { 1 : [ [ descAt1], [descAt2], ... ],
               #   6 : [[ ... ]] }

        """
        nAt = mol.nAt
        if self.device is None:
            atNums = torch.from_numpy(mol.atoms.numbers)
        else:
            atNums = torch.from_numpy(mol.atoms.numbers).to(self.device)
        nAtomTypes = len(self.atomTypes)

        desc, idx1o, idx2o, idx3o = self._computeAngularDescriptors(mol)
        if self.device is not None:
            desc  = desc.to(self.device)
            idx1o = idx1o.to(self.device)
            idx2o = idx2o.to(self.device)
            idx3o = idx3o.to(self.device)

        descriptTensor = torch.zeros((nAt,
                                      nAtomTypes * (nAtomTypes+1)//2,
                                      self.nBasis),
                                     dtype=pu.NNP_PRECISION.NNPDType,
                                     device=self.device)

        # sum for all pairs of identical atom types on centerAt atom
        for centerAt, at1, at2, d in zip(idx1o, idx2o, idx3o, desc):
            #atNumC = atNums[centerAt]
            atNum1 = atNums[at1]
            atNum2 = atNums[at2]

            if atNum1 > atNum2:
                atNum1, atNum2 = atNum2, atNum1

            try:
                pairIdx = self.atomTypePairToIdx[(atNum1.item(), atNum2.item())]
            except KeyError as err:
                raise TypeError("AtomType not known %i or %i" % (atNum1.item(), atNum2.item())) from err

            descriptTensor[centerAt, pairIdx ] += d

        res = {}
        for centerType in self.atomTypes:
            res[centerType] = descriptTensor[atNums == centerType]

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

            if self.aCenters.device != device:
                self.aCenters = self.aCenters.to(device)

            if self.atomTypePairToIdxT.device != device:
                self.atomTypePairToIdxT = self.atomTypePairToIdxT.to(device)

        self.rCentersOpt = self.rCentersOpt.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.rCentersFinal = self.rCentersFinal.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.aCenters = self.aCenters.to(dtype=pu.NNP_PRECISION.NNPDType)

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
        destination["%s:nAngles" % name]           = self.nAngles
        destination["%s:nRadial" % name]           = self.nRadial
        destination["%s:rCentersOpt"% name]        = self.rCentersOpt.data
        destination["%s:rCentersFinal"% name]      = self.rCentersFinal.data
        destination["%s:aCenters"% name]           = self.aCenters.data
        destination["%s:cutoff"% name]             = self.cutoff
        destination["%s:nBasisAtomTypes"% name]    = self.nBasisAtomTypes
        destination["%s:nBasis"% name]             = self.nBasis
        destination["%s:atomTypePairToIdx"% name]  = self.atomTypePairToIdx
        destination["%s:atomTypePairToIdxT"% name] = self.atomTypePairToIdxT

        return destination



    def load_state_dict(self,state):
        name = type(self).__name__

        self.loadIfPresent(name, "atomTypes"        , state)
        self.loadIfPresent(name, "nAngles"          , state)
        self.loadIfPresent(name, "nRadial"          , state)
        self.loadIfPresent(name, "cutoff"           , state)
        self.loadIfPresent(name, "nBasisAtomTypes"  , state)
        self.loadIfPresent(name, "nBasis"           , state)
        self.loadIfPresent(name, "atomTypePairToIdx", state)

        AngularBasis.updateIfPresent(self.rCentersOpt,   name, "rCentersOpt", state)
        AngularBasis.updateIfPresent(self.rCentersFinal, name, "rCentersFinal", state)
        AngularBasis.updateIfPresent(self.aCenters,      name, "aCenters", state)
        AngularBasis.updateIfPresent(self.atomTypePairToIdxT, name, "atomTypePairToIdxT", state)

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




class _BatchAngularInfo():
    """ infor on angluar relationschip for batch of compounds """

    #@profile
    #@torch.jit.script
    def __init__(self, nConf, nAtPerConf,
                 i_triple2Coords, j_triple2Coords, k_triple2Coords,
                 atTypes,
                 distIJ, distIK, cosAngles ):
        """ cosAngles[i,j,k]         cosine of angle between i(J)K
            distIJ[i,j]              distance of pair of atoms ij
            distIk[i,k]              distance of pair of atoms ik
            i_triple2Coords[nAngles] coordinateIdx of center atom in angle
            j_triple2Coords[nAngles] coordinateIdx of second atom in angle
            k_triple2Coords[nAngles] coordinateIdx of third atom in angle
            Ijk_atTypes[nAngles]     atomp type of atom in triple
        """

        self.nConf         = nConf
        self.nAt           = nAtPerConf
        self.i_triple2Coords = i_triple2Coords
#        self.j_triple2Coords = j_triple2Coords
#        self.k_triple2Coords = k_triple2Coords
        self.cosAngles          = cosAngles
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



class GaussianAngularBasis(AngularBasis):
    """ Default ANI like gaussians as basis """
    def __init__(self, atomTypes,
                       nAngles=None, nRadial=None, rMin=None, rMax=None,
                       aCenters=None, rCenters=None,
                       halfWidth=None, angleHalfWidth=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        # pylint: disable=W1201,W0102
        """
        Define a set of basis functions to describe a distance with nCenter gaussians

        Either:
            nRadial: number of centers
            rMin distance of first center
            rMax distance of last center
        or
            centers a list of postions for the basis function centers
        Must be given the other values umst be None

        halfWidth halfWidth of centers on radius
        cutoff: distance at which the descriptors will be caled to 0

        halfWidth HalfWidth of centers on radius
        angleHalfWidth: HalfWidth of angle gaussian in PI, eg. if 0.5 then the halfWidth of the angluar term will be PI/2
        cutoff: distance at which the descriptors will be called to 0
        """

        super().__init__(atomTypes, nAngles, nRadial, rMin, rMax,
                                                   aCenters, rCenters, cutoff,
                                                   optimizeParam, device)

        assert halfWidth is not None
        assert angleHalfWidth is not None

        dtype=pu.NNP_PRECISION.NNPDType

        if np.isscalar(angleHalfWidth):
            angleHalfWidth = torch.full((self.nAngles,), angleHalfWidth, dtype=dtype, device=device)
        else:
            angleHalfWidth = torch.tensor(angleHalfWidth, dtype=dtype, device=device)

        self.zeta   = -1./torch.log2((1+torch.cos(angleHalfWidth/2.*math.pi))/2.)
        # inverse: =ACOS((2^(-1/B2)*2)-1)/PI()*2

        if 'zeta' in optimizeParam:
            self.optParameter['zeta'] = self.zeta


        #compute negative eta
        if np.isscalar(halfWidth):
            hw = torch.full((nRadial,), halfWidth, dtype=dtype, device=device)
        else:
            hw = torch.tensor(halfWidth, dtype=dtype, device=device)

        self.nEta = math.log(0.5) * 4/(hw*hw)

        if 'nEta' in optimizeParam:
            self.optParameter['nEta'] = self.nEta

        log.info("%s nBasis=%i cutoff=%.3f\n\tnAngles=%i aCenter=%s\n\tangleHW=%s\n\tnRadial=%i rCenter=%s\n\tradialHW=%s\n\tzeta=%s\n\tnegEta=%s" % (
            type(self).__name__,
            self.nBasis, cutoff, self.nAngles, self.aCenters, angleHalfWidth,
            nRadial, self.getRCenters(), halfWidth, self.zeta, self.nEta))


    def to(self, device):
        super().to(device)
        if device is not None:
            if self.zeta.device != device:
                self.zeta = self.zeta.to(device)

            if self.nEta.device != device:
                self.nEta = self.nEta.to(device)

        self.zeta = self.zeta.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.nEta = self.nEta.to(dtype=pu.NNP_PRECISION.NNPDType)


    #@torch.jit.script
    def _computeAngularDescriptors(self, mol):
        """
            Compute angular descriptors for all atoms of mol
            as in fromula 4 in ANI--1 paper
            Return list of atom indices for centr, neighbor1 and neighbor2 as well as descriptor array

        """

        xyz = mol.atomTensor3D

        atNums = mol.atoms.numbers
        shiftR = self.getRCenters()
        shiftA = self.aCenters

        # dist is vector with distances, idx1 and idx2 are indexes into pairs of atoms for each dist
        dist, idx1, idx2 = mol.neigborDistance(self.cutoff)
        vectors = xyz[idx2] - xyz[idx1]

        # output tensors
        # idx1o: index of center atom
        # idx2o: index of second atom
        # idx3o: index of third  atom
        # descriptor according to formula 4 in ANI-1 paper without sum
        # computed from angle between 1-2 and 1-3
        idx1o = torch.tensor([], dtype=torch.long, device=self.device)
        idx2o = torch.tensor([], dtype=torch.long, device=self.device)
        idx3o = torch.tensor([], dtype=torch.long, device=self.device)
        desc  = torch.tensor([], device=self.device)

        for ix,_ in enumerate(atNums):
            j         = idx2[idx1==ix]
            distanceI = dist[idx1==ix]
            vectorsI  = vectors[idx1==ix]
            #print("j,dI %s %s" %(j, distanceI))
            for j1,distance1,vector1 in zip(j,distanceI,vectorsI):
                # distance1,vector1 are from center atom (ix) to neighbor1 j1
                # j, distanceI,vectorsI are from center to other atoms

                # rows should always represent atoms and columns shiftR and shiftA variations
                # we are only doing the lower triangle (j<j1)
                distOther = distanceI[j<j1].view(-1,1)
                nOther = len(distOther)
                if nOther > 0:

                    # Vector multiply each vector in vectors(from atom ix to others within cutoff)
                    # to all other vectors from ix
                    # and compute angle
                    #print("j j1 %s %s" %(j,j1))
                    ang = torch.acos(
                        torch.matmul(
                            vector1,
                            vectorsI[j<j1].transpose(0,1)/(distance1*distanceI[j<j1]))
                        .clamp(-1.,1.))
                    ang = ang.view(-1,1)
                    #print("ang dist %s %s" % (ang, distanceI[j<j1]))

                    # rows should always represent atoms and columns shiftR and shiftA variations
                    distOther = distanceI[j<j1].view(-1,1)

                    rad1 = shiftR - (distance1+distOther)/2
                    rad1 = torch.exp(self.nEta * rad1 * rad1)
                    rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distance1 / self.cutoff,0,1)) + 0.5)
                    #rad1 is len(distOther) * len(shiftR)

                    # atoms are in rows shift parameters in columns
                    # multiply each radial shift with each atoms cutoff function
                    rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distOther / self.cutoff,0,1)) + 0.5)

                    d = torch.pow(1. + torch.cos(ang - shiftA), self.zeta)
                    d = d * torch.pow(2.,1.-self.zeta)

                    # add one dimension so we get all elements of row in rad1
                    # multiplied with all elements in same row in d
                    # the elements in each row should be (rhift1-ashift1, rshift1-aschift2,... rshift2-aschift1, ...)

                    d = torch.transpose(d.unsqueeze(-1),0,1)
                    d = torch.transpose(rad1 * d,0,1).reshape(distOther.shape[0],-1)
                    #print("rad1=%s\nd=%s" % (rad1,d))

                    idx1o = torch.cat((idx1o,torch.tensor([ix] * nOther, dtype=torch.long)))  ## ix is numpy
                    idx2o = torch.cat((idx2o,j1.expand(nOther)))
                    idx3o = torch.cat((idx3o,j[j<j1]))
                    desc = torch.cat((desc, d))


        # print("idx=%s\njdx=%s\nkx=%s\nang=%s\ndesc=%s" % (idxo,jdxo,kdxo,desc))

        return desc, idx1o, idx2o, idx3o


    #@profile
    #@torch.jit.script
    def _computeAngularDescriptorsBatch(self, angInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nAngles] ordered as:
            [nConf][nTriples][nRadial * nAngles]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """

        # 0.99999 needed because numeric issues might yield |values| > 1.
        angles   = torch.acos(angInfo.cosAngles * 0.99999)
        distIJ   = angInfo.distIJ
        distIK   = angInfo.distIK

        shiftR = self.getRCenters()
        shiftA = self.aCenters

        rad1 = (distIJ + distIK)/2
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nAngles, nrShifts]

        rad1 = torch.exp(self.nEta * rad1 * rad1)

        # multiply with cutoff terms
        rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5).unsqueeze(-1)
        rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5).unsqueeze(-1)


        ang  = shiftA - angles.unsqueeze(1)
        # ang is angle differences [nAngles, nAShifts]

        # torchani devides 1+cos by 2
        # torchani does not include "*  2.**(1.-self.zeta)"
        #log.info(f"ang.shape {ang.shape}")
        ang = torch.pow(1. + torch.cos(ang), self.zeta) *  2.**(1.-self.zeta)

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (ang.unsqueeze(-1) * rad1).reshape(-1,self.nBasis)

        return d


    def state_dict(self):
        destination = super().state_dict()

        name = type(self).__name__
        destination["%s:zeta" % name] = self.zeta
        destination["%s:nEta" % name] = self.nEta

        return destination


    def load_state_dict(self,state):
        super().load_state_dict(state)

        name = type(self).__name__
        self.loadIfPresent(name, "zeta", state)
        AngularBasis.updateIfPresent(self.nEta, name, 'nEta',  state)



class Gaussian2AngularBasis(GaussianAngularBasis):
    """ Basis """
    def __init__(self, atomTypes,
                       nAngles=None, nRadial=None, rMin=None, rMax=None,
                       aCenters=None, rCenters=None,
                       halfWidth=None, angleHalfWidth=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        # pylint: disable=W0102,W0235
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

        halfWidth HalfWidth of centers on radius
        angleHalfWidth: HalfWidth of angle gaussian in PI, eg. if 0.5 then the halfWidth of the angluar term will be PI/2
        cutoff: distance at which the descriptors will be called to 0
        """

        super().__init__(atomTypes, nAngles, nRadial, rMin, rMax,
                         aCenters, rCenters,
                         halfWidth, angleHalfWidth, cutoff,
                         optimizeParam, device)

    #@profile
    #@torch.jit.script
    def _computeAngularDescriptorsBatch(self, angInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nAngles] ordered as:
            [nConf][nTriples][nRadial * nAngles]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """

        # 0.99999 needed because numeric issues might yield |values| > 1.
        angles   = torch.acos(angInfo.cosAngles * 0.99999)
        distIJ   = angInfo.distIJ
        distIK   = angInfo.distIK

        shiftR = self.getRCenters()
        shiftA = self.aCenters

        rad1 = distIJ
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nAngles, nrShifts]
        rad1 = torch.exp(self.nEta * rad1 * rad1)
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)

        rad2 = distIK
        rad2 = shiftR - rad2.unsqueeze(1)
        rad2 = torch.exp(self.nEta * rad2 * rad2)
        rad2 = torch.transpose(rad2.unsqueeze(-1), dim0=-2, dim1=-1)

        ang  = shiftA - angles.unsqueeze(1)
        # ang is angle differences [nAngles, nAShifts]

        ang = torch.pow(1. + torch.cos(ang), self.zeta) *  2.**(1.-self.zeta)

        # add each component for each radial shift with component for each angular shift
        d1 = (ang.unsqueeze(-1) * rad1).reshape(-1,self.nBasis)
        d2 = (ang.unsqueeze(-1) * rad2).reshape(-1,self.nBasis)

        d = (d1 + d2) / 2

        # multiply with cutoff terms
        d = d * (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5).unsqueeze(-1)
        d = d * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5).unsqueeze(-1)

        return d



class GaussianCosineBasis(AngularBasis):
    """ use cosine of angle instead of angle itself """
    def __init__(self, atomTypes,
                       nAngles=None, nRadial=None, rMin=None, rMax=None,
                       aCenters=None, rCenters=None,
                       halfWidth=None, angleHalfWidth=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        # pylint: disable=W1201,W0102
        """
        Define a set of basis functions to describes the angle between three atoms i(j)k
        with nCenter gaussians basis along the cosine of the angle.
        Similar to the angular term of the Behler APrinello descriptors used in the
        original ANI-1 paper but using the cosine of the angle instead of the radians.

        rCenters:
        Either:
            nRadial: number of centers
            rMin distance of first center
            rMax distance of last center
        or
            rCenters a list of postions for the basis function centers
        Must be given the other values umst be None

        either aCenters or nAngles:
        aCenters can be 0 to PI

        halfWidth halfWidth of centers on radius
        angleHalfWidth: HalfWidth of angle gaussian in PI, eg. if 0.5 then the halfWidth of the angluar term will be PI/2
        cutoff: distance at which the descriptors will be called to 0
        """

        super().__init__(atomTypes, nAngles, nRadial, rMin, rMax,
                                                   aCenters, rCenters, cutoff,
                                                   optimizeParam, device)

        assert halfWidth is not None
        assert angleHalfWidth is not None

        dtype=pu.NNP_PRECISION.NNPDType

        # convert center positions into cosine scale
        torch.cos_(self.aCenters)

        #compute angular negative eta
        if np.isscalar(angleHalfWidth):
            angleHalfWidth = torch.full((self.nAngles,), angleHalfWidth, dtype=dtype, device=device)
        else:
            angleHalfWidth = torch.tensor(angleHalfWidth, dtype=dtype, device=device)

        self.aNEta = math.log(0.5) * 4/(angleHalfWidth*angleHalfWidth)

        if 'aNEta' in optimizeParam:
            self.optParameter['aNEta'] = self.aNEta



        #compute radial negative eta
        if np.isscalar(halfWidth):
            hw = torch.full((nRadial,), halfWidth, dtype=dtype, device=device)
        else:
            hw = torch.tensor(halfWidth, dtype=dtype, device=device)

        self.nEta = math.log(0.5) * 4/(hw*hw)

        if 'nEta' in optimizeParam:
            self.optParameter['nEta'] = self.nEta

        log.info("%s nBasis=%i cutoff=%.3f\n\tnAngles=%i aCenter=%s\n\tangleHW=%s\n\tnRadial=%i rCenter=%s\n\tradialHW=%s\n\taNegEta=%s\n\tnegEta=%s" % (
            type(self).__name__,
            self.nBasis, cutoff, self.nAngles, self.aCenters, angleHalfWidth,
            nRadial, self.getRCenters(), halfWidth, self.aNEta, self.nEta))


    def to(self, device):
        super().to(device)
        if device is not None:
            if self.aNEta.device != device:
                self.aNEta = self.aNEta.to(device)

            if self.nEta.device != device:
                self.nEta = self.nEta.to(device)

        self.aNEta = self.aNEta.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.nEta = self.nEta.to(dtype=pu.NNP_PRECISION.NNPDType)


    #@profile
    #@torch.jit.script
    def _computeAngularDescriptorsBatch(self, angInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nAngles] ordered as:
            [nConf][nTriples][nRadial * nAngles]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  1       1        1         0        0       0       -1       -1      -1     cos Angles
        """

        cosAng   = angInfo.cosAngles
        distIJ   = angInfo.distIJ
        distIK   = angInfo.distIK

        shiftR = self.getRCenters()
        shiftA = self.aCenters

        rad1 = (distIJ + distIK)/2
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nAngles, nrShifts]

        rad1 = torch.exp(self.nEta * rad1 * rad1)

        # multiply with cutoff terms
        rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5).unsqueeze(-1)
        rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5).unsqueeze(-1)

        ang  = shiftA - cosAng.unsqueeze(1)
        # ang is cosine angle differences [cosAng, nAShifts]
        ang = torch.exp(self.aNEta * ang * ang)

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (ang.unsqueeze(-1) * rad1).reshape(-1,self.nBasis)

        return d

    def _computeAngularDescriptors(self, mol):
        raise NotImplementedError("This is a slower method and not needed")



class GaussianTANIAngularBasis(GaussianAngularBasis):
    """ Basis """
    def __init__(self, atomTypes,
                       nAngles=None, nRadial=None, rMin=None, rMax=None,
                       aCenters=None, rCenters=None,
                       halfWidth=None, angleHalfWidth=None,
                       cutoff=None, optimizeParam=set(), device=None ):
        # pylint: disable=W0102,W0235
        """
        Everythong same as torchANi



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

        halfWidth HalfWidth of centers on radius
        angleHalfWidth: HalfWidth of angle gaussian in PI, eg. if 0.5 then the halfWidth of the angluar term will be PI/2
        cutoff: distance at which the descriptors will be called to 0
        """

        super().__init__(atomTypes, nAngles, nRadial, rMin, rMax,
                         aCenters, rCenters,
                         halfWidth, angleHalfWidth, cutoff,
                         optimizeParam, device)


    #@profile
    #@torch.jit.script
    def _computeAngularDescriptorsBatch(self, angInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nAngles] ordered as:
            [nConf][nTriples][nRadial * nAngles]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """

        # 0.99999 needed because numeric issues might yield |values| > 1.
        angles   = torch.acos(angInfo.cosAngles * 0.999999)
        distIJ   = angInfo.distIJ
        distIK   = angInfo.distIK

        shiftR = self.getRCenters()
        shiftA = self.aCenters

        rad1 = (distIJ + distIK)/2
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nAngles, nrShifts]

        rad1 = torch.exp(self.nEta * rad1 * rad1)

        # multiply with cutoff terms
        rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5).unsqueeze(-1)
        rad1 = rad1 * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5).unsqueeze(-1)


        ang  = shiftA - angles.unsqueeze(1)
        # ang is angle differences [nAngles, nAShifts]

        ang = ((1. + torch.cos(ang))/2) ** self.zeta

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (ang.unsqueeze(-1) * rad1).reshape(-1,self.nBasis) * 2

        return d




class Bump2AngularBasis(AngularBasis):
    """ use cheaper bump function rather than cosine """
    def __init__(self, atomTypes,
                       nAngles=None, nRadial=None, rMin=None, rMax=None,
                       aCenters=None, rCenters=None,
                       halfWidth=None, maxWidthMultiplier=None,
                       angleHalfWidth=None, maxAngleWidthMultiplier=None,
                       optimizeParam=set(), device=None ):
        # pylint: disable=W1201,W0102
        """
        # pylint: disable=W0102
        #### Note this does not show correct behavior
        ### need to rethink addition vs multiplication of radial terms
        ### need to rethink addition vs multiplication of angular term
        ### currently large values are only computed if delatR1 ~= shiftR and deltaR2 ~= same shiftR

        Define a set of basis functions to describe a distance with nCenter gaussians

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth HalfWidth of centers on radius
        angleHalfWidth: HalfWidth of angle gaussian in PI, eg. if 0.5 then the halfWidth of the angluar term will be PI/2
        maxWidthMultiplier: halfWidth*maxRadiusMultiplier = radius at which bump function is 0

        The formula used is exp(rC1/(1-deltaR^2/rC2) - rC1 + aC1/(1-deltaA^2/aC2) - AC1)
           c1 and c2 are adjusted by the halfWidth and the maxRadiusMultiplier
        """
        assert ((nRadial is not None and rMin is not None, rMax is not None and rCenters is None)
             or (nRadial is     None and rMin is     None, rMax is     None and rCenters is not None)
                ), "either nRadial,rMin and rMax or rCenters needed"
        assert ((nAngles is not None and aCenters is None)
             or (nAngles is     None and aCenters is not None)
                ), "either nAngles or rCenters aCenters"

        dtype=pu.NNP_PRECISION.NNPDType
        cutoff = self._computeCutoff(nRadial, rMax, rCenters, halfWidth, maxWidthMultiplier)

        super().__init__(atomTypes, nAngles, nRadial, rMin, rMax,
                          aCenters, rCenters, cutoff,
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


        if np.isscalar(angleHalfWidth):
            assert np.isscalar(maxAngleWidthMultiplier), "halfWidth and maxAngleWidthMultiplier must be of same length"

            maxWidth = angleHalfWidth * maxAngleWidthMultiplier
            c2Base = maxWidth * maxWidth / 4. * math.pi * math.pi
            c1Base = math.log(0.5) * maxWidth * maxWidth / angleHalfWidth / angleHalfWidth - math.log(0.5)

            # only the first len-1 c2 will be optimized to avoid the last going into cutoff
            self.aC2 = torch.full((self.nAngles,), c2Base, dtype=dtype, device=device)
            self.aC1 = torch.full((self.nAngles,), c1Base, dtype=dtype, device=device)
        else:
            assert len(maxAngleWidthMultiplier) == len(angleHalfWidth), "halfWidth and maxWidthMultiplier must be of same length"

            angleHalfWidth = np.array(angleHalfWidth, dtype=np.float32)
            maxAngleWidthMultiplier = np.array(maxAngleWidthMultiplier, dtype=np.float32)
            maxWidth =  angleHalfWidth * maxAngleWidthMultiplier

            # read center parameters from lists arguments
            c2 = maxWidth * maxWidth / 4. * math.pi * math.pi
            c1 = math.log(0.5) * maxWidth * maxWidth / angleHalfWidth / angleHalfWidth - math.log(0.5)

            self.aC2 = torch.tensor(c2, dtype=dtype, device=device)
            self.aC1 = torch.tensor(c1, dtype=dtype, device=device)


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
        # pylint: disable=R0201
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
        super().to(device)
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

        self.rC1 = self.rC1.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.rC2Opt   = self.rC2Opt.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.rC2Final = self.rC2Final.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.aC1 = self.aC1.to(dtype=pu.NNP_PRECISION.NNPDType)
        self.aC2 = self.aC2.to(dtype=pu.NNP_PRECISION.NNPDType)


    def state_dict(self):
        destination = super().state_dict()

        name = type(self).__name__
        destination["%s:rC1"      % name] = self.rC1
        destination["%s:rC2Opt"   % name] = self.rC2Opt
        destination["%s:rC2Final" % name] = self.rC2Final
        destination["%s:aC1"      % name] = self.aC1
        destination["%s:aC2"      % name] = self.aC2

        return destination


    def load_state_dict(self,state):
        super().load_state_dict(state)

        name = type(self).__name__
        self.updateIfPresent(self.rC1,      name, "rC1", state)
        self.updateIfPresent(self.rC2Opt,   name, "rC2Opt", state)
        self.updateIfPresent(self.rC2Final, name, "rC2Final", state)
        self.updateIfPresent(self.aC1,      name, "aC1", state)
        self.updateIfPresent(self.aC2,      name, "aC2", state)


    #@profile
    #@torch.jit.script
    def _computeAngularDescriptors(self, mol):
        """
            Compute angular descriptors for all atoms of mol
            as in fromula 4 in ANI--1 paper
            Return list of atom indices for centr, neighbor1 and neighbor2 as well as descriptor array

        """

        xyz = mol.atomTensor3D

        atNums = mol.atoms.numbers
        shiftR = self.getRCenters()
        rC2    = self.getRC2()
        shiftA = self.aCenters

        # dist is vector with distances, idx1 and idx2 are indexes into pairs of atoms for each dist
        dist, idx1, idx2 = mol.neigborDistance(self.cutoff)
        vectors = xyz[idx2] - xyz[idx1]

        # output tensors
        # idx1o: index of center atom
        # idx2o: index of second atom
        # idx3o: index of third  atom
        # descriptor according to formula 4 in ANI-1 paper without sum and prefactor (2^(1-zeta))
        # computed from angle between 1-2 and 1-3
        idx1o = torch.tensor([], dtype=torch.long, device=self.device)
        idx2o = torch.tensor([], dtype=torch.long, device=self.device)
        idx3o = torch.tensor([], dtype=torch.long, device=self.device)
        desc  = torch.tensor([], device=self.device)

        for ix,_ in enumerate(atNums):
            j         = idx2[idx1==ix]
            distanceI = dist[idx1==ix]
            vectorsI  = vectors[idx1==ix]
            #print("j,dI %s %s" %(j, distanceI))
            for j1,distance1,vector1 in zip(j,distanceI,vectorsI):
                # distance1,vector1 are from center atom (ix) to neighbor1 j1
                # j, distanceI,vectorsI are from center to other atoms

                # rows should always represent atoms and columns shiftR and shiftA variations
                # we are only doing the lower triangle (j<j1)
                distOther = distanceI[j<j1].view(-1,1)
                nOther = len(distOther)

                if nOther > 0:

                    # Vector multiply each vector in vectors(from atom ix to others within cutoff)
                    # to all other vectors from ix
                    # and compute angle
                    #print("j j1 %s %s" %(j,j1))
                    ang = torch.acos(
                        torch.matmul(
                            vector1,
                            vectorsI[j<j1].transpose(0,1)/(distance1*distanceI[j<j1]))
                        .clamp(-1.,1.))
                    ang = ang.view(-1,1)
                    #print("ang dist %s %s" % (ang, distanceI[j<j1]))

                    # rows should always represent atoms and columns shiftR and shiftA variations
                    distOther = distanceI[j<j1].view(-1,1)

                    rad1 = shiftR - distance1
                    rad1 = self.rC1 / (1. - torch.clamp(rad1 * rad1 / rC2 ,0.,0.9999)) - self.rC1

                    rad2 = shiftR - distOther
                    rad1 = rad1 + self.rC1 / (1. - torch.clamp(rad2 * rad2 / rC2 ,0.,0.9999)) - self.rC1
                    rad1 = rad1/2.

                    ang = ang - shiftA
                    ang  = self.aC1 / (1. - torch.clamp(ang  * ang  / self.aC2 ,0.,0.9999)) - self.aC1
                    ang = torch.transpose(ang.unsqueeze(-1),dim0=0, dim1=1)

                    # add each component for each radial shift with component for each angular shift
                    d = torch.transpose(rad1 + ang, dim0=0,dim1=1).reshape(distOther.shape[0],-1)
                    d = torch.exp( d )

                    idx1o = torch.cat((idx1o,torch.tensor([ix] * nOther, dtype=torch.long)))  ## ix is numpy
                    idx2o = torch.cat((idx2o,j1.expand(nOther)))
                    idx3o = torch.cat((idx3o,j[j<j1]))
                    desc = torch.cat((desc, d))


        # print("idx=%s\njdx=%s\nkx=%s\nang=%s\ndesc=%s" % (idxo,jdxo,kdxo,desc))

        # possible TODO: compute this product after sum computeDescriptors to speed up

        return desc, idx1o, idx2o, idx3o

    #@profile
    #@torch.jit.script
    def _computeAngularDescriptorsBatch(self, angInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nAngles] ordered as:
            [nConf][nTriples][nRadial * nAngles]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """

        # 0.99999 needed because numeric issues might yield |values| > 1.
        angles   = torch.acos(angInfo.cosAngles * 0.99999)
        rad1    = angInfo.distIJ
        rad2    = angInfo.distIK

        shiftR = self.getRCenters()
        rC2 = self.getRC2()
        shiftA = self.aCenters

        # expand to have one row per triple i,j,k
        rad1 = shiftR - rad1.unsqueeze(1)
        rad2 = shiftR - rad2.unsqueeze(1)

        # rad1 is [nAngles][nrShifts]
        ang  = shiftA - angles.unsqueeze(1)
        # ang is [nAngles][nAShifts]

        rad1  =        self.rC1 / (1. - torch.clamp(rad1 * rad1 / rC2, 0.,0.9999)) - self.rC1
        rad1  = rad1 + self.rC1 / (1. - torch.clamp(rad2 * rad2 / rC2 ,0.,0.9999)) - self.rC1
        rad1 = rad1/2.

        ang  = self.aC1 / (1. - torch.clamp(ang  * ang  / self.aC2, 0.,0.9999)) - self.aC1

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (ang.unsqueeze(-1) + rad1).reshape(-1, self.nBasis)

        d = torch.exp( d )

        return d


class BumpAngularBasis(Bump2AngularBasis):
    """ Use faster bumpfunction for cutoff rather than cosine """
    def __init__(self, atomTypes,
                      nAngles=None, nRadial=None, rMin=None, rMax=None,
                      aCenters=None, rCenters=None,
                      halfWidth=None, maxWidthMultiplier=None,
                      angleHalfWidth=None, maxAngleWidthMultiplier=None,
                      cutoff=None, optimizeParam=set(), device=None ):
        # pylint: disable=W1201
        """
        Define a set of basis functions to describe a distance with nCenter gaussians

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth HalfWidth of centers on radius
        angleHalfWidth: HalfWidth of angle gaussian in PI, eg. if 0.5 then the halfWidth of the angluar term will be PI/2
        maxWidthMultiplier: halfWidth*maxRadiusMultiplier = radius at which bump function is 0

        The formula used is exp(rC1/(1-deltaR^2/rC2) - rC1 + aC1/(1-deltaA^2/aC2) - AC1)
           c1 and c2 are adjusted by the halfWidth and the maxRadiusMultiplier
        """

        super().__init__(atomTypes, nAngles, nRadial,  rMin, rMax,
                         aCenters, rCenters,
                         halfWidth, maxWidthMultiplier,
                         angleHalfWidth, maxAngleWidthMultiplier, optimizeParam, device)


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
    def _computeAngularDescriptors(self, mol):
        """
            Compute angular descriptors for all atoms of mol
            as in formula 4 in ANI--1 paper
            Return list of atom indices for centr, neighbor1 and neighbor2 as well as descriptor array

        """

        xyz = mol.atomTensor3D


        atNums = mol.atoms.numbers
        shiftR = self.getRCenters()
        rC2 = self.getRC2()
        shiftA = self.aCenters

        # dist is vector with distances, idx1 and idx2 are indexes into pairs of atoms for each dist
        dist, idx1, idx2 = mol.neigborDistance(self.cutoff*2)
        if self.device is not None:
            xyz = xyz.to(self.device)
            dist = dist.to(self.device)
            idx1 = idx1.to(self.device)
            idx2 = idx2.to(self.device)

        vectors = xyz[idx2] - xyz[idx1]

        # output tensors
        # idx1o: index of center atom
        # idx2o: index of second atom
        # idx3o: index of third  atom
        # descriptor according to formula 4 in ANI-1 paper without sum and pre-factor (2^(1-zeta))
        # computed from angle between 1-2 and 1-3
        idx1o = torch.tensor([], dtype=torch.long, device=self.device)
        idx2o = torch.tensor([], dtype=torch.long, device=self.device)
        idx3o = torch.tensor([], dtype=torch.long, device=self.device)
        desc  = torch.tensor([], device=self.device)

        for ix,_ in enumerate(atNums):
            j         = idx2[idx1==ix]
            distanceI = dist[idx1==ix]
            vectorsI  = vectors[idx1==ix]
            #print("j,dI %s %s" %(j, distanceI))
            for j1,distance1,vector1 in zip(j,distanceI,vectorsI):
                # distance1,vector1 are from center atom (ix) to neighbor1 j1
                # j, distanceI,vectorsI are from center to other atoms

                # rows should always represent atoms and columns shiftR and shiftA variations
                # we are only doing the lower triangle (j<j1)
                distOther = distanceI[j<j1].view(-1,1)
                nOther = len(distOther)

                if nOther > 0:

                    # Vector multiply each vector in vectors(from atom ix to others within cutoff)
                    # to all other vectors from ix
                    # and compute angle
                    #print("j j1 %s %s" %(j,j1))
                    ang = torch.acos(
                        torch.matmul(
                            vector1,
                            vectorsI[j<j1].transpose(0,1)/(distance1*distanceI[j<j1]))
                        .clamp(-1.,1.))
                    ang = ang.view(-1,1)
                    #print("ang dist %s %s" % (ang, distanceI[j<j1]))

                    # rows should always represent atoms and columns shiftR and shiftA variations
                    #distOther = distanceI[j<j1].view(-1,1)

                    rad1 = shiftR - (distance1+distOther)/2
                    ang = ang - shiftA
                    rad1 = self.rC1 / (1. - torch.clamp(rad1 * rad1 / rC2,     0.,0.9999)) - self.rC1
                    ang  = self.aC1 / (1. - torch.clamp(ang  * ang  / self.aC2,0.,0.9999)) - self.aC1
                    ang = torch.transpose(ang.unsqueeze(-1),dim0=0, dim1=1)

                    # add each component for each radial shift with component for each angular shift
                    d = torch.transpose(rad1 + ang, dim0=0,dim1=1).reshape(distOther.shape[0],-1)
                    d = torch.exp( d )

                    idx1o = torch.cat((idx1o,torch.tensor([ix] * nOther, dtype=torch.long, device=self.device)))  ## ix is numpy
                    idx2o = torch.cat((idx2o,j1.expand(nOther)))
                    idx3o = torch.cat((idx3o,j[j<j1]))
                    desc = torch.cat((desc, d))


        # print("idx=%s\njdx=%s\nkx=%s\nang=%s\ndesc=%s" % (idxo,jdxo,kdxo,desc))

        # possible TODO: compute this product after sum computeDescriptors to speed up

        return desc, idx1o, idx2o, idx3o


    #@profile
    #@torch.jit.script
    def _computeAngularDescriptorsBatch(self, angInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nAngles] ordered as:
            [nConf][nTriples][nRadial * nAngles]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """

        # 0.99999 needed because numeric issues might yield |values| > 1.
        angles   = torch.acos(angInfo.cosAngles * 0.99999)
        distIJ   = angInfo.distIJ
        distIK   = angInfo.distIK

        shiftR = self.getRCenters()
        rC2 = self.getRC2()
        shiftA = self.aCenters

        rad1 = (distIJ + distIK)/2

        # verticalize and expand to have one row per triple i,j,k
        rad1 = shiftR - rad1.unsqueeze(1)
        # rad1 is [nAngles][nrShifts]
        ang  = shiftA - angles.unsqueeze(1)
        # ang is [nAngles][nAShifts]

        rad1 = self.rC1 / (1. - torch.clamp(rad1 * rad1 /      rC2, 0.,0.9999)) - self.rC1
        ang  = self.aC1 / (1. - torch.clamp(ang  * ang  / self.aC2, 0.,0.9999)) - self.aC1

        # add each component for each radial shift with component for each angular shift
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)
        d = (ang.unsqueeze(-1) + rad1).reshape(-1, self.nBasis)

        d = torch.exp( d )

        return d



class Bump3AngularBasis(Bump2AngularBasis):
    """ Use faster bumpfunction for cutoff rather than cosine """
    def __init__(self, atomTypes,
                      nAngles=None, nRadial=None, rMin=None, rMax=None,
                      aCenters=None, rCenters=None,
                      halfWidth=None, maxWidthMultiplier=None,
                      angleHalfWidth=None, maxAngleWidthMultiplier=None,
                      cutoff=None, optimizeParam=set(), device=None ):
        # pylint: disable=W1201,W0613
        """
        Define a set of basis functions to describe a distance with nCenter gaussians

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth HalfWidth of centers on radius
        angleHalfWidth: HalfWidth of angle gaussian in PI, eg. if 0.5 then the halfWidth of the angluar term will be PI/2
        maxWidthMultiplier: halfWidth*maxRadiusMultiplier = radius at which bump function is 0

        The formula used is exp(rC1/(1-deltaR^2/rC2) - rC1 + aC1/(1-deltaA^2/aC2) - AC1)
           c1 and c2 are adjusted by the halfWidth and the maxRadiusMultiplier
        """

        super().__init__(atomTypes, nAngles, nRadial,  rMin, rMax,
                         aCenters, rCenters,
                         halfWidth, maxWidthMultiplier,
                         angleHalfWidth, maxAngleWidthMultiplier, optimizeParam, device)


    #@profile
    #@torch.jit.script
    def _computeAngularDescriptorsBatch(self, angInfo):
        """
            in parameter are as produced by  _computeAnglesBatch()

            Compute angular descriptors for all atoms in angles input
            tensor with [one row per angles][nRadial * nAngles] ordered as:
            [nConf][nTriples][nRadial * nAngles]
                # rShift  1       2        3         1        2       3        1        2       3
                # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
        """


        # 0.99999 needed because numeric issues might yield |values| > 1.
        angles    = torch.acos(angInfo.cosAngles * 0.99999)
        distIJ    = angInfo.distIJ
        distIK    = angInfo.distIK

        shiftR = self.getRCenters()
        rC2 = self.getRC2()
        shiftA = self.aCenters

        # expand to have one row per triple i,j,k
        rad1 = shiftR - distIJ.unsqueeze(1)
        rad2 = shiftR - distIK.unsqueeze(1)

        # rad1 is [nAngles][nrShifts]
        ang  = shiftA - angles.unsqueeze(1)
        # ang is [nAngles][nAShifts]

        rad1  = self.rC1 / (1. - torch.clamp(rad1 * rad1 / rC2, 0.,0.9999)) - self.rC1
        rad1 = torch.transpose(rad1.unsqueeze(-1), dim0=-2, dim1=-1)

        rad2  = self.rC1 / (1. - torch.clamp(rad2 * rad2 / rC2 ,0.,0.9999)) - self.rC1
        rad2 = torch.transpose(rad2.unsqueeze(-1), dim0=-2, dim1=-1)

        ang  = self.aC1 / (1. - torch.clamp(ang  * ang  / self.aC2, 0.,0.9999)) - self.aC1

        # add each component for each radial shift with component for each angular shift
        d1 = (ang.unsqueeze(-1) + rad1).reshape(-1, self.nBasis)
        d2 = (ang.unsqueeze(-1) + rad2).reshape(-1, self.nBasis)

        d = (torch.exp( d1 ) + torch.exp( d2 )) / 2

        # needed or this would have instability when one neighbor moves out of cutoff
        d = d * (0.5 * torch.cos(math.pi * torch.clamp(distIJ / self.cutoff,0,1)) + 0.5).unsqueeze(-1)
        d = d * (0.5 * torch.cos(math.pi * torch.clamp(distIK / self.cutoff,0,1)) + 0.5).unsqueeze(-1)

        return d
