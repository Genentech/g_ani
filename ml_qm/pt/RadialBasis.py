## Alberto

import ml_qm.pt.nn.Precision_Util as pu
from ml_qm import AtomInfo
from collections import OrderedDict
import math
import torch
import numpy as np
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611


import logging
log = logging.getLogger(__name__)



class RadialBasis():
    """ Basis to compute radial dependicies of pairs of atoms """

    def __init__(self, atomTypes,
                 nCenters=None, centerMin=None, centerMax=None,
                 centers=None,
                 cutoff=None, optimizeParam=False, optimizePositions=False, device=None):

        """
           Either:
                nCenter: number of centers
                centerMin distance of first center
                centerMax distance of last center
            or
                centers a list of postions for the basis function centers
            Must be given the other values ust be None
        """

        assert cutoff is not None
        assert ((nCenters is not None and centerMin is not None, centerMax is not None and centers is None)
             or (nCenters is     None and centerMin is     None, centerMax is     None and centers is not None)
                ), "either nCenter,min and max or centers needed"

        self.device = device
        self.optimizeParam = optimizeParam
        self.optimizePositions = optimizePositions

        if centers is not None:
            self.nCenters = len(centers)
            centers = np.array(centers)
        else:
            self.nCenters = nCenters
            centers = np.linspace(centerMin, centerMax, nCenters, dtype=np.float32)

        # avoid optimizing last gaussian as it might go over the cutoff
        self.centersOpt = torch.tensor(centers[0:-1], dtype=pu.NNP_PRECISION.NNPDType, device=device)
        self.centerFinal= torch.tensor([centers[-1]], dtype=pu.NNP_PRECISION.NNPDType, device=device)

        self.cutoff = cutoff
        self.atomTypes = atomTypes
        self.nBasisCounter = torch.arange(0,self.nCenters,dtype=torch.long, device=self.device)

        # map from atomType to unique index
        self.atomTypeToIdx = {}
        for idx,atType in enumerate(atomTypes):
            self.atomTypeToIdx[atType] = idx

        self.atType2IdxInDesc = torch.full((max(atomTypes)+1,), -1, dtype=torch.long, device=device)
        for i, att in enumerate(atomTypes):
            self.atType2IdxInDesc[att] = i

        self.optParameter = OrderedDict()
        if optimizePositions:
            self.optParameter["centersOpt"] = self.centersOpt



    def _computeRadialDescriptors(self,distInfo):
        """
            Overwrite!
            Compute radial descriptors for all r in vector r
            return matrix with r rows and nCenter columns

        """
        raise NotImplementedError


    def _getDistancesBatch(self, nConf:int, nAtPerConf:int, atTypes:torch.tensor, coords:torch.tensor):
        """
        returns _BatchDistInfo
           atTypes [nConfs][nAt] atomTypes of atoms in confroamtions
           dist is tensor: [nConfs][nAtom * (nAtom-1)] distance between atom1 and atom2
           rowidx, colIdx: [nConfs][natoms * nAtoms-1)]
                for each distance give the atom indexes of the two pairs within this conformer
                Diagonal elements are excluded
        """

        i_pair2Coords, j_pair2Coords, distIJ = \
           RadialBasis._get_distances_batch_script(nConf, nAtPerConf, coords, self.cutoff)
        return _BatchDistInfo(nConf, nAtPerConf, atTypes.reshape(-1), i_pair2Coords, j_pair2Coords, distIJ )

    #@profile
    @staticmethod
    @torch.jit.script
    def _get_distances_batch_script(nConf:int , nAtPerConf:int, coords, cutoff:float):
        """
        static method for use in jit.
        return
        ------
        i_pair2Coords, j_pair2Coords, distIJ
        """
        nPairs = nAtPerConf * (nAtPerConf-1)

        # create indexes for all possible permutationscoords
        i_idx = torch.arange(nAtPerConf, dtype=torch.long, device=coords.device).unsqueeze(1)
        j_idx = i_idx.repeat(nAtPerConf,1).reshape(-1)
        i_idx = i_idx.repeat(1,nAtPerConf).reshape(-1)

        ltriangle = i_idx != j_idx
        i_idx = i_idx[ltriangle]
        j_idx = j_idx[ltriangle]

        #add aditional dimenstion so we can have one rrow per pair
        # [nconfs][nAtoms][xyz] -> [nconfs][nAtoms][nAtoms][xyz]
        distIJ = coords.unsqueeze(1)  # coords is nconf, nat, 3
        distIJ = distIJ.expand(-1,nAtPerConf,-1,-1)

        distIJ = distIJ - distIJ.transpose(1,2)
        # keep only non-diagonals (maybe using torch.cdist on the whole matrix would be faster? and require less memory)
        distIJ = distIJ[:,i_idx,j_idx]

        # TODO use .norm(2, -1)
        distIJ = distIJ**2
        distIJ = distIJ.sum(dim=-1)
        distIJ = distIJ.sqrt()

        i_idx = i_idx.expand(nConf,-1)
        j_idx = j_idx.expand(nConf,-1)


        fltr = distIJ < cutoff
        fltr[0] = 1           # ensure tensors are not empty

        distIJ = distIJ[fltr]
        i_idx = i_idx[fltr]
        j_idx = j_idx[fltr]

        pair2Conf = torch.arange(0,nConf,   dtype=torch.long, device=coords.device)
        pair2Conf = pair2Conf.reshape(-1,1).expand(-1,nPairs)
        pair2Conf = pair2Conf[fltr]*int(nAtPerConf)

        i_pair2Coords = pair2Conf + i_idx
        j_pair2Coords = pair2Conf + j_idx

        return i_pair2Coords, j_pair2Coords, distIJ


    #@profile
    def computeDescriptorBatch(self, coords, atTypes):
        """
            coords[nConf][nAt][xyz]  xyz for each atom
            atTypes [nConf][nAt]     atom type for each atom
        """

        nConf, nAt, _ = coords.shape
        nCoords       = nConf * nAt

        # diatomic mols have mo angles
        if nAt < 2:
            return torch.zeros((nCoords, nAt, self.nCenters), device=self.device)

        nAtomTypes= len(self.atomTypes)
        nBasis = self.nCenters

        if not isinstance(coords, torch.Tensor):
            if self.device is None:
                coords = torch.from_numpy(coords)
                atTypes= torch.from_numpy(atTypes)
            else:
                coords = torch.from_numpy(coords).to(self.device)
                atTypes= torch.from_numpy(atTypes).to(self.device)

        distInfo  = self._getDistancesBatch(nConf, nAt, atTypes, coords)

        # result is one radial descriptor per atom pair
        # index from pair to atomIdx in conf are in i_idx, j_idx
        radialDescriptors = self._computeRadialDescriptors(distInfo)


        descriptorIdx = distInfo.getDescriptorPositions(nAtomTypes, self.atType2IdxInDesc)

        res = pu.NNP_PRECISION.indexAdd((nCoords * nAtomTypes, nBasis),
                                                         0, descriptorIdx, radialDescriptors)

        res = res.reshape(nConf,nAt,-1)

        return res




    def computeDescriptors(self, mol):
        """
            mol -- conformation as eg. GDBMolPT

            returns -- map from central atom type to array of descriptors for each atom
               # { 1 : [ [ descAt1], [descAt2], ... ],
               #   6 : [[ ... ]] }

        """

        nAt = mol.nAt
        atNums = torch.tensor(mol.atNums, dtype=torch.long, device=self.device)
        uatNums = np.unique(mol.atNums).tolist()
        nBasis = self.nCenters

        descriptTensor = torch.zeros((nAt, len(self.atomTypes), nBasis), dtype=pu.NNP_PRECISION.NNPDType, device=self.device)

        # dist is vector with distances, idx1 and idx2 are indexes into pairs of atoms for each dist
        # The list only holds the upper triangle elements
        (dist, idx1, idx2) = mol.neigborDistanceLT(cutoff=self.cutoff)
        if self.device is not None:
            dist  = dist.to(self.device)
            idx1  = idx1.to(self.device)
            idx2  = idx2.to(self.device)

        distInfo = SimpleDistInfo(dist)
        radialDescriptors = self._computeRadialDescriptors(distInfo).reshape(-1,nBasis)
        # add elements for lower triangle

        # sum over upper pairs
        for (at1,at2,d) in zip(idx1,idx2, radialDescriptors):
            atNum2 = atNums[at2]

            try:
                descTypeIdx = self.atomTypeToIdx[atNum2.item()]
            except KeyError as err:
                raise TypeError("AtomType not known %i" % atNum2.item()) from err

            descriptTensor[at1, descTypeIdx ] += d

        # sum over lower triangle
        for (at1,at2,d) in zip(idx2,idx1,radialDescriptors):
            atNum2 = atNums[at2]
            descriptTensor[at1, self.atomTypeToIdx[atNum2.item()]] += d

        res = {}
        #for centerType in self.atomTypes:
        for centerType in uatNums:
            res[centerType] = descriptTensor[atNums == centerType]

        return res


    def nDescriptors(self):
        return self.nCenters * len(self.atomTypes)

    def getCenters(self):
        return torch.cat((self.centersOpt, self.centerFinal))


    def to(self, device):
        if device is not None:
            if self.atType2IdxInDesc.device != device:
                self.atType2IdxInDesc = self.atType2IdxInDesc.to(device)

            if  self.centersOpt.device != device:
                self.centerFinal = self.centerFinal.to(device)
                self.centersOpt  = self.centersOpt.to(device)


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
        destination["%s:atomTypes" % name]        = self.atomTypes
        destination["%s:nCenters" % name]         = self.nCenters
        destination["%s:centersOpt"% name]        = self.centersOpt.data
        destination["%s:centerFinal"% name]       = self.centerFinal.data
        destination["%s:cutoff"% name]            = self.cutoff
        destination["%s:atomTypeToIdx"% name]     = self.atomTypeToIdx
        destination["%s:atType2IdxInDesc"% name]  = self.atType2IdxInDesc.data

        return destination


    def load_state_dict(self,state):
        name = type(self).__name__
        self.loadIfPresent(name, "atomTypes"        , state)
        self.loadIfPresent(name, "nCenters"         , state)
        self.loadIfPresent(name, "cutoff"           , state)
        self.loadIfPresent(name, "atomTypeToIdx"    , state)
        self.loadIfPresent(name, "optimizeParam"    , state)
        self.loadIfPresent(name, "optimizePositions", state)

        self.updateIfPresent(self.atType2IdxInDesc, name, "atType2IdxInDesc", state)

        self.updateIfPresent(self.centersOpt, name, "centersOpt", state)
        self.updateIfPresent(self.centerFinal, name, "centerFinal", state)


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


class SimpleDistInfo():
    """ simple version for use with single distance execution """
    def __init__(self, dist):
        self.distIJ = dist



class _BatchDistInfo():
    """ version for use with batch distance execution """

    def __init__(self, nConf, nAtPerConf, atTypes, i_pair2Coords, j_pair2Coords, distIJ):
        """ distIJ[I,J]           distance of pair of atoms
            i_pair2Coords[nPairs] index of coordinate of first atom in pair
            j_pair2Coords[nPairs] index of coordinate of second atom in pair
            atTypes[nPairs]       atomTypes by coordinateIndex

            i_pair2Coords[nPairs] mapping from first atom pair to conformation number
            j_pair2Coords[nPairs] mapping from second atom pair to conformation number
        """

        self.nConf          = nConf
        self.nAt            = nAtPerConf
        self.atTypes        = atTypes
        self.distIJ         = distIJ
        self.i_pair2Coords  = i_pair2Coords
        #self.j_pair2Coords  = j_pair2Coords

        # get atomtype i and j for each descriptor
        self.j_atType = self.atTypes[j_pair2Coords]


    @property
    def i_atType(self):
        """ only compute if requested """
        return self.atTypes[self.i_pair2Coords]

    #@profile
    def getDescriptorPositions(self, nBasisAtomTypes, atType2IdxInDesc):
        """ returns two vectors:
               descCoordIdx for each distance descriptor this is the coordinate of the center atom
               neighborIndx: unique index of the neighbor atom type e.g 0 for H, 1 for C, ....
        """

        i_pair2Coords = self.i_pair2Coords
        neighborIndx = atType2IdxInDesc[self.j_atType]

        # Each coord (indexed by i_triple2Coords has
        # nUniqueJKIdx sections for the JK atom types given by triple2UniqueJKIdx
        return i_pair2Coords * nBasisAtomTypes + neighborIndx





class GaussianRadialBasis(RadialBasis):
    """ AEV Basis using Gaussian do softbin distance ANI type AEV """
    def __init__(self, atomTypes, nCenters=None, centerMin=None, centerMax=None,
                 centers=None, halfWidth=None, cutoff=None,
                 optimizeParam=False, optimizePositions=False, device=None):
        # pylint: disable=W1201
        """
        Define a set of basis functions to describe a distance with nCenter gaussians

        Either:
            nCenter: number of centers
            centerMin distance of first center
            centerMax distance of last center
        or
            centers a list of postions for the basis function centers
        Must be given the other values umst be None

        halfWidth halfWidth of centers on radius
        cutoff: distance at which the descriptors will be caled to 0
        """

        super().__init__(atomTypes, nCenters, centerMin, centerMax, centers, cutoff,
                         optimizeParam, optimizePositions, device)

        assert halfWidth is not None

        if nCenters is None: nCenters = len(centers)

        #compute negative eta
        if np.isscalar(halfWidth):
            self.hw = torch.full((nCenters,), halfWidth, dtype=pu.NNP_PRECISION.NNPDType, device=device)
        else:
            self.hw = torch.tensor(halfWidth, dtype=pu.NNP_PRECISION.NNPDType, device=device)

        self.nEta = self._computeNEta(self.hw)

        if optimizeParam:
            self.optParameter['nEta'] = self.nEta

        log.info("%s cutoff=%.3f\n\tcenters=%s\n\thalfWidth=%s\n\tnEta=%s" % (
              type(self).__name__, cutoff, self.getCenters(), self.hw, self.nEta ))



    def _computeNEta(self, halfWidth):                  # pylint: disable=R0201
        return math.log(0.5) * 4./halfWidth/halfWidth


    def to(self, device):
        super(GaussianRadialBasis, self).to(device)
        if device is not None:
            if self.hw.device != device:
                self.hw = self.hw.to(device)

            if self.nEta.device != device:
                self.nEta = self.nEta.to(device)

    #@profile
    def _computeRadialDescriptors(self, distInfo):
        """
            Compute radial descriptors for all r in vector r or a matrix of vectors of r
            (one per atom)
            return matrix with r rows and nCenter columns

        """
        return GaussianRadialBasis._compute_radial_descriptors_script(distInfo.distIJ, self.getCenters(), self.nEta, self.cutoff)


    @staticmethod
    @torch.jit.script
    def _compute_radial_descriptors_script(r, centers, nEta, cutoff:float):

        desc = centers - r.unsqueeze(-1)
        desc = desc * desc
        # torchani has a multiplier with 0.25 here too (aev.py:98)
        desc = torch.exp( desc * nEta)
        desc = desc * (0.5 * torch.cos(math.pi * torch.clamp(r.unsqueeze(-1) /cutoff,0,1)) + 0.5)
        return desc


    def state_dict(self):
        destination = super(GaussianRadialBasis, self).state_dict()

        name = type(self).__name__
        destination["%s:hw" % name]  = self.hw.data
        destination["%s:nEta"% name] = self.nEta.data

        return destination


    def load_state_dict(self,state):
        super(GaussianRadialBasis, self).load_state_dict(state)

        name = type(self).__name__

        RadialBasis.updateIfPresent(self.hw,   name, 'hw',  state)
        RadialBasis.updateIfPresent(self.nEta, name, 'nEta', state)




class SlaterRadialBasis(GaussianRadialBasis):
    """ Like a slator orbital """

    def __init__(self, atomTypes, nCenters=None, centerMin=None, centerMax=None,
                 centers=None, halfWidth=None, cutoff=None,
                 optimizeParam=False, optimizePositions=False, device=None):
        """
        Define a set of basis functions to describe a distance with nCenter slater function
        This has a discontinuity at deltaR = 0 but we assume that will not affect optimizations

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth halfWidth of centers on radius
        cutoff: distance at which the descriptors will be caled to 0
        """

        super().__init__(atomTypes, nCenters, centerMin, centerMax,
                         centers, halfWidth, cutoff,
                         optimizeParam, optimizePositions, device)



    def _computeNEta(self, halfWidth):       # pylint: disable=R0201
        return 2. * math.log(0.5) / halfWidth



    #@profile
    def _computeRadialDescriptors(self, distInfo):
        """
            Compute radial descriptors for all r in vector r or a matrix of vectors of r
            (one per atom)
            return matrix with r rows and nCenter columns

        """
        r = distInfo.distIJ
        centers = self.getCenters()

        desc = centers - r.unsqueeze(-1)
        # torchani has a multiplier with 0.25 here too (aev.py:98)
        desc = torch.exp( desc.abs() * self.nEta)
        desc = desc * (0.5 * torch.cos(math.pi * torch.clamp(r.unsqueeze(-1) /self.cutoff,0,1)) + 0.5)
        return desc






class GausVDWRadialBasis(GaussianRadialBasis):

    def __init__(self, atomTypes, nCenters=None, centerMin=None, centerMax=None,
                 centers=None, halfWidth=None, cutoff=None,
                 optimizeParam=False, optimizePositions=False, device=None):
        """
        Define a set of basis functions to describe a distance with nCenter gaussians relative
        to the average vdW radius of the two atoms

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth halfWidth of centers on radius
        cutoff: distance at which the descriptors will be caled to 0
        """

        super().__init__(atomTypes, nCenters, centerMin, centerMax,
                         centers, halfWidth, cutoff,
                         optimizeParam, optimizePositions, device)

        dtype=pu.NNP_PRECISION.NNPDType

        #compute negative eta
        self.vdwByAtomType = torch.full((max(atomTypes)+1,), -1, dtype=dtype, device=device)
        for i in range(1,max(atomTypes)+1):
            self.vdwByAtomType[i] = AtomInfo.NumToVDWRadius[i]

        if optimizeParam:
            self.optParameter['vdwRadius'] = self.vdwByAtomType


    def _computeRadialDescriptors(self, distInfo):
        """
            Compute radial descriptors for all r in vector r or a matrix of vectors of r
            (one per atom)
            return matrix with r rows and nCenter columns

        """
        dist = distInfo.distIJ
        i_atType = distInfo.i_atType
        j_atType = distInfo.j_atType
        distScaled = dist * 2. \
                    / (self.vdwByAtomType[i_atType] + self.vdwByAtomType[j_atType])

        centers = self.getCenters()

        desc = centers - distScaled.unsqueeze(-1)
        desc = desc * desc
        desc = torch.exp( desc * self.nEta)
        desc = desc * (0.5 * torch.cos(math.pi * torch.clamp(dist.unsqueeze(-1) /self.cutoff,0,1)) + 0.5)
        return desc



class Bump3RadialBasis(RadialBasis):
    """ same as BumpRadialBasis but with cos() cutoff from GaussianBasis """

    def __init__(self, atomTypes, nCenters=None, centerMin=None, centerMax=None,
                 centers=None,
                 halfWidth=None, maxWidthMultiplier=2,
                 optimizeParam=False, optimizePositions=False, device=None):
        # pylint: disable=W1201
        """
        Define a set of basis functions to describe a distance with nCenter gaussians

        nCenter: number of centers
        centerMin distance of first center
        centerMax distance of last center
        halfWidth halfWidth of centers on radius
        maxWidthMultiplier: halfWidth*maxRadiusMultiplier = radius at which bump function is 0

        The formula used is exp(c1/(1-deltaR^2/c2) - c1)
           c1 and c2 are adjusted by the halfWidth and the maxRadiusMultiplier
        """

        assert ((nCenters is not None and centerMin is not None, centerMax is not None and centers is None)
             or (nCenters is     None and centerMin is     None, centerMax is     None and centers is not None)
                ), "either nCenter,min and max or centers needed"
        assert halfWidth is not None

        dtype=pu.NNP_PRECISION.NNPDType


        if nCenters is None:
            mwm = maxWidthMultiplier if np.isscalar(maxWidthMultiplier) else maxWidthMultiplier[-1]

            #TODO test this should be:
            #hw  = halfWidth if np.isscalar(halfWidth) else halfWidth[-1]
            #cutoff = rCenters[-1] + hw / 2. * mwm
            cutoff = max(centers) + mwm / 2.
        else:
            cutoff = centerMax + halfWidth/2. * maxWidthMultiplier

        super().__init__(atomTypes, nCenters, centerMin, centerMax, centers,
                         cutoff, optimizeParam, optimizePositions, device)

        if np.isscalar(halfWidth):
            assert np.isscalar(maxWidthMultiplier), "halfWidth and maxWidthMultiplier must be of same length"

            maxWidth = halfWidth * maxWidthMultiplier
            c2Base = maxWidth * maxWidth / 4.
            c1Base = math.log(0.5) * maxWidth * maxWidth / halfWidth / halfWidth - math.log(0.5)

            # only the first len-1 c2 will be optimized to avoid the last going into cutoff
            self.c2Opt = torch.full((nCenters-1,), c2Base, dtype=dtype, device=device)
            self.c2Final = torch.tensor([c2Base], dtype=dtype, device=device)
            self.c1 = torch.full((nCenters,), c1Base, dtype=dtype, device=device)

            log.info("%s nCenter=%i min=%.3f max=%.3f halfWidth=%.3f maxWidthMult=%.3f cutoff=%.3f c1=%.4f c2=%.4f" % (
                  type(self).__name__, nCenters, centerMin, centerMax, halfWidth, maxWidthMultiplier, cutoff, c1Base, c2Base ))
        else:
            assert len(maxWidthMultiplier) == len(halfWidth), "halfWidth and maxWidthMultiplier must be of same length"

            maxWidth = np.array(halfWidth, dtype=np.float32) * np.array(maxWidthMultiplier, dtype=np.float32)
            # read center parameters from lists arguments
            c2 = maxWidth * maxWidth / 4.
            c1 = math.log(0.5) * maxWidth * maxWidth / halfWidth / halfWidth - math.log(0.5)

            self.c2Opt   = torch.tensor(c2[0:-1], dtype=dtype, device=device)
            self.c2Final = torch.tensor([c2[-1]], dtype=dtype, device=device)
            self.c1 = torch.tensor(c1, dtype=dtype, device=device)

            log.info("%s cutoff=%.3f\ncenters=%s\nhalfWidth=%s\nmaxWidthMult=%s\nc1=%s\nc2=%s\n" % (
              type(self).__name__, cutoff, self.centers, halfWidth, maxWidthMultiplier, self.c1, self.c2))



        if optimizeParam:
            self.optParameter['c1'] = self.c1
            self.optParameter['c2Opt'] = self.c2Opt



    def to(self, device):
        super(Bump3RadialBasis, self).to(device)
        if device is not None:
            if self.c1.device != device:
                self.c1 = self.c1.to(device)

            if self.c2Opt.device != device:
                self.c2Opt   = self.c2Opt.to(device)
                self.c2Final = self.c2Final.to(device)


    def state_dict(self):
        destination = super(Bump3RadialBasis, self).state_dict()

        name = type(self).__name__
        destination["%s:c1" % name] = self.c1.data
        destination["%s:c2Opt" % name] = self.c2Opt.data
        destination["%s:c2Final" % name] = self.c2Final.data

        return destination


    def load_state_dict(self,state):
        super(Bump3RadialBasis, self).load_state_dict(state)

        name = type(self).__name__

        RadialBasis.updateIfPresent(self.c1,      name, "c1", state)
        RadialBasis.updateIfPresent(self.c2Opt,   name, "c2Opt",   state)
        RadialBasis.updateIfPresent(self.c2Final, name, "c2Final", state)


    def _computeRadialDescriptors(self,distInfo):
        """
            Compute radial descriptors for all r in vector r
            return matrix with r rows and nCenter columns

        """
        r = distInfo.distIJ
        centers = self.getCenters()
        c2 = self.getC2()

        desc = centers - r.unsqueeze(-1)
        desc = 1. - torch.clamp(desc * desc / c2, 0,.999999)
        desc = torch.exp( self.c1 / desc - self.c1 )
        desc = desc * (0.5 * torch.cos(math.pi * torch.clamp(r.unsqueeze(-1) /self.cutoff,0,1)) + 0.5)
        return desc


    def getC2(self):
        return torch.cat((self.c2Opt, self.c2Final))




class BumpRadialBasis(Bump3RadialBasis):
    def __init__(self, atomTypes, nCenters=None, centerMin=None, centerMax=None,
                 centers=None,
                 halfWidth=None, maxWidthMultiplier=2,
                 optimizeParam=False, optimizePositions=False, device=None):
        super().__init__(atomTypes,
                         nCenters, centerMin, centerMax, centers,
                         halfWidth, maxWidthMultiplier,
                         optimizeParam, optimizePositions, device)



    def _computeRadialDescriptors(self,distInfo):
        """
            Compute radial descriptors for all r in vector r
            return matrix with r rows and nCenter columns

        """

        r = distInfo.distIJ
        centers = self.getCenters()
        c2 = self.getC2()

        desc = centers - r.unsqueeze(-1)
        desc = 1. - torch.clamp(desc * desc / c2, 0,.999999)
        desc = torch.exp( self.c1 / desc - self.c1 )
        return desc





class LinearRadialBasis(RadialBasis):
    def __init__(self, atomTypes, nCenters=None, centerMin=None, centerMax=None,
                 centers=None, cutoff=None,
                 optimizeParam=False, optimizePositions=False, device=None):
        # pylint: disable=W1201
        """
        Define a set of basis functions to describe a distance with nCenter as a linear function

        Either:
            nCenter: number of centers
            centerMin distance of first center
            centerMax distance of last center
        or
            centers a list of postions for the basis function centers
        Must be given the other values umst be None

        halfWidth halfWidth of centers on radius
        cutoff: distance at which the descriptors will be caled to 0
        """

        super().__init__(atomTypes, nCenters, centerMin, centerMax, centers, cutoff,
                         optimizeParam, optimizePositions, device)

        if nCenters is None: nCenters = len(centers)

        log.info("%s cutoff=%.3f\n\tcenters=%s\n" % (
              type(self).__name__, cutoff, self.getCenters()))



    #@profile
    def _computeRadialDescriptors(self, distInfo):
        """
            Compute radial descriptors for all r in vector r or a matrix of vectors of r
            (one per atom)
            return matrix with r rows and nCenter columns

        """
        # pylint: disable=W1201
        r = distInfo.distIJ
        centers = self.getCenters()

        desc = r.unsqueeze(-1) - centers
        desc = desc * (0.5 * torch.cos(math.pi * torch.clamp(r.unsqueeze(-1) /self.cutoff,0,1)) + 0.5)
        return desc



class SigmoidRadialBasis(RadialBasis):
    def __init__(self, atomTypes, nCenters=None, centerMin=None, centerMax=None,
                 centers=None, halfWidth=None, cutoff=None,
                 optimizeParam=False, optimizePositions=False, device=None):
        # pylint: disable=W1201
        """
        Define a set of basis functions to describe a distance with nCenter sigmoid functions

        Either:
            nCenter: number of centers
            centerMin distance of first center
            centerMax distance of last center
        or
            centers a list of postions for the basis function centers
        Must be given the other values umst be None

        halfWidth halfWidth of centers on radius
        cutoff: distance at which the descriptors will 0
        """

        super().__init__(atomTypes, nCenters, centerMin, centerMax, centers, cutoff,
                         optimizeParam, optimizePositions, device)

        assert halfWidth is not None

        dtype=pu.NNP_PRECISION.NNPDType

        if nCenters is None: nCenters = len(centers)

        #compute negative eta
        if np.isscalar(halfWidth):
            self.hw = torch.full((nCenters,), halfWidth, dtype=dtype, device=device)
        else:
            self.hw = torch.tensor(halfWidth, dtype=dtype, device=device)

        # 2*LN(1/3)/B2
        self.lmbda = 2 * math.log(1/3.) / self.hw

        if optimizeParam:
            self.optParameter['lmbda'] = self.lmbda

        log.info("%s cutoff=%.3f\n\tcenters=%s\n\thalfWidth=%s\n\tlmbda=%s" % (
              type(self).__name__, cutoff, self.getCenters(), self.hw, self.lmbda ))


    def to(self, device):
        super(SigmoidRadialBasis, self).to(device)
        if device is not None:
            if self.hw.device != device:
                self.hw = self.hw.to(device)

            if self.lmbda.device != device:
                self.lmbda = self.lmbda.to(device)

    #@profile
    def _computeRadialDescriptors(self, distInfo):
        """
            Compute radial descriptors for all r in vector r or a matrix of vectors of r
            (one per atom)
            return matrix with r rows and nCenter columns

        """
        r = distInfo.distIJ
        centers = self.getCenters()

        desc = r.unsqueeze(-1) - centers
        # 1/(1+EXP($E$1*A2))*2 - 1
        desc = 1./ (1. + torch.exp(self.lmbda * desc)) * 2 - 1.
        desc = desc * (0.5 * torch.cos(math.pi * torch.clamp(r.unsqueeze(-1) /self.cutoff,0,1)) + 0.5)
        return desc


    def state_dict(self):
        destination = super(SigmoidRadialBasis, self).state_dict()

        name = type(self).__name__
        destination["%s:hw" % name]  = self.hw.data
        destination["%s:lmbda"% name] = self.lmbda.data

        return destination


    def load_state_dict(self,state):
        super(SigmoidRadialBasis, self).load_state_dict(state)

        name = type(self).__name__

        RadialBasis.updateIfPresent(self.hw,   name, 'hw',  state)
        RadialBasis.updateIfPresent(self.lmbda, name, 'lmbda', state)
