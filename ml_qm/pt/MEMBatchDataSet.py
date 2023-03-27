#!/bin/env python
"""
Created on May 14, 2018

@author: albertgo
"""

from builtins import RuntimeError
import ml_qm.pt.nn.Precision_Util as pu
from ml_qm.ANIMol import ANIMol
import numpy as np
from ml_qm.pt.PTMol import PTMol
from ml_qm.pt.MEMDataSet import TypedDataSet, ANIMEMSplitDataSet
from ml_qm.pt import torch_util as tu
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
import random

import torch
from torch.utils.data.sampler import RandomSampler, BatchSampler,\
    SequentialSampler

import logging
log = logging.getLogger(__name__)


torch.set_printoptions(linewidth=200)


class MEMBatchDataSet():
    """
        MEMBatchDataSet provides access to batches of conformations in TorchConfBatchDescription objects
        via an iterator interface (__getitem__())
        The batches of conformations are taken from CoordsBatch's in the memDataSet.
        CoordsBatch are grouped by atom count for GPU computation of descriptors.
        The _coordsBatch2ConfBatchDescription() method takes such batches and:
           - computes the Symmetry functions
           - reassemples subsets now by atom type in preparation fot passing these
             along to atom specific Neural NEtworks.

        Iteration is thread save as long as shuffle is not called.
    """


    def __init__(self, memDataSet:ANIMEMSplitDataSet,  batchSize, descriptorComputer, setType='train',
                 requiresXYZGrad=False, dropLast=False, fuzz=None, device=None, debug=False):

        self.descriptorComputer = descriptorComputer
        self.device = device
        self.debug  = debug
        self.fuzzMax= None
        self.setType= setType
        self.requiresXYZGrad = requiresXYZGrad
        if fuzz is not None:
            self.fuzzMax   = fuzz['fuzzMax']
            self.fuzzMin   = fuzz['fuzzMin']
            self.fuzzEpochs= fuzz['fuzzEpochs']
            self.fuzzSlope  = (self.fuzzMax/self.fuzzMin-1)/self.fuzzEpochs

        self.batchSize = batchSize
        self.epoch  = 0

        pinMem = False
        if device is not None and device.type != 'cpu': pinMem = True

        self.coordBatches = []
        typedDataSet = TypedDataSet(memDataSet, setType)
        if setType == 'train':
            sampler = RandomSampler(typedDataSet)
        else:
            sampler = SequentialSampler(typedDataSet)
        batchsampler = BatchSampler(sampler, batchSize, dropLast)

        for idxs in batchsampler:
            self.coordBatches.append(typedDataSet.getCoordsBatch(idxs, pinMem))

        del typedDataSet

        log.info("Read %s batches", len(self.coordBatches) )


    def setEpoch(self, epoch):
        self.epoch = epoch


    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        pass


    def __len__(self):
        return len(self.coordBatches) > 0

    def getSubBatchSize(self, batchSize):
        """ return the number of subBatches if this is split into batches of batchSize """

        assert self.batchSize % batchSize == 0
        assert len(self.coordBatches), "No data in this MEMBatchDataSet"

        # last batch might have less than self.batchSize
        subLen = (len(self)-1) * (self.batchSize // batchSize) \
             + (self.coordBatches[-1].nConfs-1) // batchSize + 1

        log.info("Subsetting from batchsize %i to %i, from %i batches to %i, last orgBatch was %i",
                 self.batchSize, batchSize, len(self), subLen, self.coordBatches[-1].nConfs)

        return subLen


    def shuffle(self):
        random.shuffle(self.coordBatches)


    def __getitem__(self, index):
        """ next TorchConfBatchDescription """
        coordsBatch = self.coordBatches[index]
        try:
            return self._coordsBatch2ConfBatchDescription(coordsBatch)
        except IndexError as e:
            raise RuntimeError() from e


    #@profile
    def _coordsBatch2ConfBatchDescription(self, coordsBatch):
        # pylint: disable=R0915
        """ Compute the Atomic Environment Desccriptors and return a TorchConfBatchDescription
            which is pre-grouped by atom type.

            Arguments
            ---------
            coordsBatch : batch of coordinates already split by atoms per molecule
        """
        batchEnergies     = coordsBatch.batchEnergies
        batchAt2ConfNum   = coordsBatch.batchAt2ConfNum
        at2AtType         = coordsBatch.at2AtType
        uAtTypes          = coordsBatch.uAtTypes
        atTypeCount       = coordsBatch.atTypeCount
        nConfs            = coordsBatch.nConfs

        if self.device is not None:
            if self.device.type == 'cuda':
                at2AtType       = at2AtType.to(self.device, non_blocking=True)
                batchAt2ConfNum = batchAt2ConfNum.to(self.device, non_blocking=True)
                if batchEnergies is not None:
                    batchEnergies   = batchEnergies.to(dtype=pu.NNP_PRECISION.lossDType, device=self.device, non_blocking=True)
            else:
                at2AtType       = at2AtType.to(self.device)
                batchAt2ConfNum = batchAt2ConfNum.to(self.device)
                if batchEnergies is not None:
                    batchEnergies   = batchEnergies.to(dtype=pu.NNP_PRECISION.lossDType, device=self.device)


        nDescriptor = self.descriptorComputer.nDescriptors()
        batchDescList = []
        coordsList    = [] if self.requiresXYZGrad else None
        # loop over groups of conformers by atom type and compute descriptors
        for coords,atTypes in coordsBatch.nextCoordsByAtomCount():
            if self.device is not None:
                if self.device.type == 'cuda':
                    coords  = coords.to(dtype=pu.NNP_PRECISION.NNPDType, device=self.device, non_blocking=True)
                    atTypes = atTypes.to(self.device, non_blocking=True)
                else:
                    coords  = coords.to(dtype=pu.NNP_PRECISION.NNPDType, device=self.device)
                    atTypes = atTypes.to(self.device)

            if self.setType == 'train' and self.fuzzMax is not None and not self.debug:
                # after self.fuzzEpochs epochs fuzMin is reached
                #fuzz = max(1e-3/(self.epoch+1), 1e-5)
                #fuzz = max(self.fuzzMax - (self.fuzzMax - self.fuzzMin) * self.epoch/self.fuzzEpochs, self.fuzzMin)
                fuzz = max(self.fuzzMax/(self.fuzzSlope * self.epoch +1), self.fuzzMin)
                coords = coords + torch.rand_like(coords) * fuzz * 2 - fuzz

            if self.requiresXYZGrad:
                coords.requires_grad_()
                coordsList.append(coords)

            #######################################################################
            desc = self.descriptorComputer.computeDescriptorBatch(coords, atTypes)
            #######################################################################
            desc = desc.reshape(-1, nDescriptor)
            batchDescList.append( desc )

            del atTypes, coords

        batchDesc = torch.cat(batchDescList)
        del batchDescList

        #sort by atType
        at2AtType, sortByTypeUIdx = at2AtType.sort()

        batchDesc       = batchDesc[sortByTypeUIdx]
        batchAt2ConfNum = batchAt2ConfNum[sortByTypeUIdx]

        #split by atom type
        descByAtTypeList    = batchDesc.split(atTypeCount)       # per atom type 1,6,7,8 descriptors per atom
        del batchDesc

        batchAt2ConfNumList = batchAt2ConfNum.split(atTypeCount) # per atom type 1,6,7,8 reference from atom position to conformer number
        del batchAt2ConfNum

        # atoms in coordsList are still sorted by original atom order in conf
        cbd = TorchConfBatchDescription(nConfs, uAtTypes, descByAtTypeList, batchAt2ConfNumList,
                                        batchEnergies, coordsList)

        if self.debug:
            #warn(batchEnergies.shape, batchAt2ConfNum.shape, at2AtType.shape, uAtTypes, atTypeCount)
            self.checkConf(coordsBatch, cbd, random.randrange(coordsBatch.batchEnergies.shape[0]))

        return cbd


    def checkConf(self, coordsBatch, confBatchDesc, confNum):
        # pylint: disable=R0915
        torch.set_printoptions(linewidth=200)

        batchEnergies    = coordsBatch.batchEnergies.to(confBatchDesc.results.device)
        batchAt2ConfNum  = coordsBatch.batchAt2ConfNum.to(confBatchDesc.results.device)
        at2AtType        = coordsBatch.at2AtType.to(confBatchDesc.results.device)

        startAt = 0
        startConfNum = 0
        for coords,atTypes in coordsBatch.nextCoordsByAtomCount(): # pylint: disable=W0631
            nConf, nAt = atTypes.shape

            if startConfNum + nConf > confNum:
                break

            startConfNum += nConf
            startAt += nConf * nAt

        coords = coords.to(confBatchDesc.results.device)
        atTypes= atTypes.to(confBatchDesc.results.device)
        energy = batchEnergies[confNum]


        startAt = startAt + (confNum - startConfNum)*nAt
        coord = coords[confNum - startConfNum]
        atType = atTypes[confNum - startConfNum]

        for atT1,atT2 in zip(atType, at2AtType[startAt:startAt+nAt]):
            assert atT1 == atT2

        assert batchAt2ConfNum[startAt] == confNum


        #################################################################

        assert energy == confBatchDesc.results[confNum]

        atType, sortIdx = atType.sort()
        coord = coord[sortIdx]
        uAtType, cntUAt = tu.uniqueCount(atType)
        uAtTypeIdx = 0

        mol = PTMol(ANIMol(str(confNum), energy, atType, coord))
        individualDescs = self.descriptorComputer.computeDescriptors(mol)
        #individualAtoms = mol.atNums

        hadDiff = False
        for i, atT in enumerate(confBatchDesc.uAtomTypes):
            if uAtTypeIdx >= len(uAtType) or atT != uAtType[uAtTypeIdx]:
                # warn("Conf %d has no %s" % (confNum, atT))
                continue

            uAtTypeIdx += 1

            confNums  = confBatchDesc.at2confNumList[i]
            atDescs   = confBatchDesc.atomDiscriptorList[i]

            assert confNums.shape[0] == atDescs.shape[0]
            atTypeDesc =  atDescs[confNums == confNum]
            assert atTypeDesc.shape[0] == cntUAt[uAtTypeIdx-1]

            individualDesc = individualDescs[0,torch.tensor(atType) == int(atT)]
            maxDiff = -9999
            for atDesc in atTypeDesc:
                minDiff = 99999
                for iAtDesc in individualDesc:
                    diff = (atDesc - iAtDesc).abs().sum()
                    if diff < minDiff:
                        minPair = (atDesc, iAtDesc)
                        minDiff = diff
                if minDiff > maxDiff: maxDiff = minDiff
                if minDiff > 0.01:
                    if not hadDiff:
                        log.debug("energy: %f atomTypes: %s",  energy, atType )
                        warn(coord)
                    hadDiff = True
                    log.debug("diff  %s". minPair[0] - minPair[1])
                    log.debug("batch %s", minPair[0])
                    log.debug("indi  %s", minPair[1])

            if maxDiff > 0.001:
                log.debug("atType=%s, largest Diff = %s", atT,maxDiff)



class MultiBatchComputerDataSet():

    """ This dataset uses an underlying MEMBatchDataSet that computes descriptors 
        for a dataset with larger batchsize. But splits the results into smaller batch sizes.

        The goal is to improve performance by large batch descriptor calculation while having flexibility to
        Use smaller batch sizes while learning.
    """

    def __init__(self, memBatchDataSet:MEMBatchDataSet, batchSize:int, dropLast:bool):
        """ memBatchDataSet: parent dataset with large batch size
            batchSize: new smaller batch size, must be multiple of parents size
            dropLast: if last batch is smaller than batchSize do not return it
        """
        self.memBatchDataSet = memBatchDataSet
        self.batchSize = batchSize
        self.length = self.memBatchDataSet.getSubBatchSize(batchSize)
        self.drop_last = dropLast

        assert self.memBatchDataSet.batchSize % batchSize == 0, \
            "Parents batch size must be multiple of this batch size"

        self.batchSizeRatio =  self.memBatchDataSet.batchSize // batchSize

        # do not shuffle if unless this is training run
        if memBatchDataSet.setType != 'train': self.next = self.nextSlice # type: ignore


    def setEpoch(self, epoch):
        self.memBatchDataSet.setEpoch(epoch)


    def __iter__(self):
        return self.next()

    def __len__(self):
        return self.length

    def shuffle(self):
        self.memBatchDataSet.shuffle()


    def next(self):
        return self.nextShuffle()

    def nextSlice(self):
        """ next TorchConfBatchDescription """
        for torchBatchDesc in self.memBatchDataSet:
            startPos = 0
            while ((not self.drop_last) and startPos < torchBatchDesc.nConfs) \
                  or startPos + self.batchSize <= torchBatchDesc.nConfs:
                subTorchBatchDesc = torchBatchDesc.subSet(startPos, self.batchSize)
                startPos = startPos + self.batchSize
                yield subTorchBatchDesc
                del subTorchBatchDesc

        return


    def nextShuffle(self):
        """ next TorchConfBatchDescription """

        for torchBatchDesc in self.memBatchDataSet:
            device = torchBatchDesc.results.device

            suffeledMolConfs = torch.randperm(torchBatchDesc.nConfs, device=device)

            startPos = 0
            while (not self.drop_last and startPos < torchBatchDesc.nConfs) \
                  or startPos + self.batchSize <= torchBatchDesc.nConfs:

                endPos = startPos+self.batchSize
                shuffled = suffeledMolConfs[startPos:endPos]

                subTorchBatchDesc = torchBatchDesc.getByIndex(shuffled, torchBatchDesc.nConfs)

                startPos = endPos
                yield subTorchBatchDesc
                del subTorchBatchDesc

        return


class TorchConfBatchDescription():
    """
            Batch of Conformational descriptors, already separated by unique atom type
            ready to go into atom_nets.
    """
    def __init__(self, nConfs, uAtomTypes, atomDescritorsList, at2confNumList, results = None, coordsList = None):
        """
            Batch of Conformational descriptors, already separated by unique atom type
            ready to go into atom_nets.

            uAtomTypes array of unique atom types e.g. 1,6,7,8
            atomDescritorsList list of atomicDescriptors one element per uAtomType
               each element tensor with one entry per atom with given atom type, and nBasis columns
            at2confNumList list of conformer numbers for each atom

            e.g. if the batch contains conf for each NH3 and OH2 you will have:

               resultList        =  [ -40.2342, -36,666 ]  energies

                                       3 + 1 Hydrogen           1 N              1 O
               uAtomTypes =            1,                       7,               8
               atomDiscriptorList = [ tensor[5,nBases] , tensor[1,nBasis, tensor[1,nbasis] ]
               at2confNumList     = [ [0,0,0,1,1]              [0]               [1] ]

            results:  tensor[nConf] of labels (energies), may be None if not none in prediction

            coordsList: coordinates of atoms in atomDiscriptorList. not-None only if gradients are required
                        this is split into segments by atomsPerConf
                        because tensors do not allow for varying number of atoms.

        """


        self.nConfs             = nConfs
        self.uAtomTypes         = uAtomTypes
        self.atomDiscriptorList = atomDescritorsList
        self.at2confNumList     = at2confNumList
        self.results            = results
        self.coordsList         = coordsList


    def subSet(self, startPos, batchSize):
        """ create a subset of this TorchConfBatchDescription with batchSize conformations """
        results = None
        if self.results is not None:
            results = self.results[ startPos: startPos+batchSize ]

        #print("subset1", startPos,batchSize, results.shape[0])

        batchDescList       = []
        batchAt2ConfNumList = []
        batchCoordsList     = [] if self.coordsList is not None else None
        uAtTypes = np.array([], np.int64)
        nConfs = min(batchSize, self.nConfs - startPos )
        for i in range(len(self.uAtomTypes)):
            atType      = self.uAtomTypes[i]
            atDesc      = self.atomDiscriptorList[i]
            at2ConfNums = self.at2confNumList[i]

            #print( "   %s %s %s %s %s %s" % (atType, at2ConfNums.min(), at2ConfNums.max(), atDesc.shape, self.results.sum(), self.uAtomTypes))
            fltr = (at2ConfNums >= startPos) * (at2ConfNums < (startPos+batchSize))
            at2ConfNums = at2ConfNums[fltr]
            if at2ConfNums.shape[0] > 0:
                at2ConfNums = at2ConfNums - startPos
                uAtTypes = np.append(uAtTypes, atType)
                batchAt2ConfNumList.append( at2ConfNums )
                batchDescList.append( atDesc[fltr] )
                if self.coordsList is not None:
                    batchCoordsList.append(self.coordsList[i][fltr])
                #print( "   %s %s %s %s %s %s" % (atType, at2ConfNums.min(), at2ConfNums.max(), atDesc[fltr].shape, uAtTypes, results.sum()))

        return TorchConfBatchDescription(nConfs, uAtTypes, batchDescList, batchAt2ConfNumList, results, batchCoordsList)


    def getByIndex(self, indxs, maxSize ):
        """ great a subset of this TorchConfBatchDescription by selecting specific indices.
           maxsize is the max of indxs + 1 """

        results = None
        if self.results is not None: results = self.results[ indxs ]
        #print("subset1", startPos,batchSize, results.shape[0])

        nConfs = indxs.shape[0]
        counter = torch.arange(nConfs, device=indxs.device)
        oldConfNumToIndxMap = torch.full((maxSize,), -1, dtype=torch.int64, device=indxs.device)
        oldConfNumToIndxMap[indxs] = counter
        del counter

        batchDescList       = []
        batchAt2ConfNumList = []
        batchCoordsList     = [] if self.coordsList is not None else None
        uAtTypes = np.array([], np.int64)
        for i in range(len(self.uAtomTypes)):
            atType      = self.uAtomTypes[i]
            atDesc      = self.atomDiscriptorList[i]
            at2ConfNums = self.at2confNumList[i]

            #print( "   %s %s %s %s %s %s" % (atType, at2ConfNums.min(), at2ConfNums.max(), atDesc.shape, self.results.sum(), self.uAtomTypes))
            fltr = oldConfNumToIndxMap[at2ConfNums] >= 0
            at2ConfNums = at2ConfNums[fltr]

            if at2ConfNums.shape[0] > 0:
                at2ConfNums = oldConfNumToIndxMap[at2ConfNums]

                uAtTypes = np.append(uAtTypes, atType)
                batchAt2ConfNumList.append( at2ConfNums )
                batchDescList.append( atDesc[fltr] )
                if self.coordsList is not None:
                    batchCoordsList.append(self.coordsList[i][fltr])
                #print( "   %s %s %s %s %s %s" % (atType, at2ConfNums.min(), at2ConfNums.max(), atDesc[fltr].shape, uAtTypes, results.sum()))

        return TorchConfBatchDescription(nConfs, uAtTypes, batchDescList, batchAt2ConfNumList, results, batchCoordsList)


    def printInfo(self):
        #warn("nConf=nresults = %i uAtomTypes=%s" % (self.nConfs, self.uAtomTypes))
        for i,_ in enumerate(self.uAtomTypes):
            log.info("shapes of atomDiscriptor: %s at2confNumList: %s countsPerConfList: %s" %
                 (self.atomDiscriptorList[i].shape,
                  self.at2confNumList[i].shape,
                  self.countsPerConfList[i].shape ))
