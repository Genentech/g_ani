"""
Created on Aug 16, 2018

@author: albertgo
"""

import torch
from typing import Sequence

from cdd_chem.util.io import warn

import logging
import gc
log = logging.getLogger(__name__)

class TensorBuffer:
    """ Tensor that can expand """

    def __init__(self, init_size:Sequence[int], chunk_size:int = 1000, **kwargs):
        """
            A buffer supported by a tensor that support append().
            The buffer must be required_grad=False
            @param
               init_size: size of initial tensor
               append_chunk: allocate additional space in this size
               kwargs: pass to initial buffer creation

        """
        self.buffer = torch.empty(init_size, **kwargs)
        assert not self.buffer.requires_grad
        self.chunk_size = chunk_size
        self.nRows = 0

    def append(self, data:torch.tensor):
        """ append rows to this buffer """
        new_rows = data.shape[0]
        cur_rows = self.buffer.shape[0]
        new_chunks = 0
        while cur_rows + new_chunks * self.chunk_size < self.nRows + new_rows:
            new_chunks += 1

        newshape = list(self.buffer.shape)
        newshape[0] = cur_rows+new_chunks*self.chunk_size
        self.buffer.resize_(newshape)

        self.buffer[self.nRows:self.nRows+new_rows] = data
        self.nRows += new_rows

        return self

    def finalize(self, log_name=None, dtype=None, device='cpu'):
        """
            returns tenor resized to fit data exactly.
            :param log_name name for the tnesor for output to log
            :param dtype of returned tensor
            :param device device into which the returned tensor is placed
        """

        newshape=list(self.buffer.shape)
        newshape[0] = self.nRows
        self.buffer.resize_(newshape)
        self.buffer = self.buffer.to(dtype=dtype, device=device)

        if log_name:
            log.info(f'finalizing TB {log_name} nRows = {self.nRows}')

        return self.buffer

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def device(self):
        return self.buffer.device


    def __len__(self) -> int:
        return self.nRows


#@profile
def sumById(tnsor, groupIds, dim=0):
    """
       tensor is a vector with values that need to be summed by groups
       groupIds is a vector of ids, sums are to be computed for each group of consecutive ids
       ids must be positive

       return: sums, posLastInGroup
           vector with sums,
           boolean vector flagging positions of last indexes in each group

    """

    groupIds = torch.transpose(groupIds,0,dim)       # transpose to make working dim the first
    grpsShiftedUp = torch.empty_like(groupIds)
    grpsShiftedUp[:-1] = groupIds[1:]
    grpsShiftedUp[-1] = -1                           # shifted
    isLast = grpsShiftedUp != groupIds               # isLast(!=)

    isLast = torch.transpose(isLast,0,dim)           # undo tranpose

    tnsor = tnsor.cumsum(dim)                          # cumsum in the dimension specified
    tnsor = tnsor[isLast]                              # keep only last of each group
                                                     # this also removes dim

    isLastOfDim = torch.transpose(torch.zeros_like(isLast),0,dim)
    isLastOfDim[-1] = 1                              # flag last element in each group
    isLastOfDim = torch.transpose(isLastOfDim,0,dim) # find last group element for each group
    isLastOfDim = isLastOfDim[isLast]                # in same order as in tnsor , set value to 0

    res = torch.empty_like(tnsor)

    res[0]  = tnsor[0]
    res[1:] = tnsor[1:] - tnsor[:-1]                           # undo cumulative effect by subtracting prvious group
                                                             # unfortunately this creates an artifact on the first element of each group
    isLastOfDim = isLastOfDim[:-1]
    res[1:][isLastOfDim] += tnsor[:-1][isLastOfDim]           # undo effect of group shift to first sum of each group

    return res, isLast



def sumById2(tnsor, groupIds, nGroups=None):
    """
       This is about 8x faster than sumById for 1M x 768 tensor

       tnsor is a vector with values that need to be summed by groups
       groupIds is a vector of ids, sums are to be computed for each group of consecutive ids
       ids must be positive

       if nGroups is None it will be computed

       return: sums, posfirstInGroup
           vector with sums,
           boolean vector flagging positions of first indexes in each group

    """

    grpsShiftedDown = torch.empty_like(groupIds)
    grpsShiftedDown[1:] = groupIds[:-1]
    grpsShiftedDown[0]  = -1                         # shifted
    isFirst = grpsShiftedDown != groupIds            # isLast(!=)
    isFirst[0] = 0
    reduceToIdx = isFirst.cumsum(0)                  # compute the index into which each entry is to be added
                                                     # = the group number = cumsum()
                                                     # elements of first group 0, of second second 1, ....

    if nGroups is None: nGroups = reduceToIdx[-1].cpu().item() + 1

    size = list(tnsor.shape)
    size[0] = nGroups
    res = torch.zeros(size, dtype=tnsor.dtype, device=tnsor.device)
    res = res.index_add(0, reduceToIdx, tnsor)
    isFirst[0] = 1

    return res, isFirst


#@profile
def sumByIdFilter(tnsor, groupIds, fltr, dim=0):
    """
       tnsor is a vector with values that need to be summed by groups
       groupIds is a vector of ids, sums are to be computed for each group of consecutive ids
       ids must be positive
       fltr is a filter that can be applied to speed things up because it is
           known that the values in tnsor are 0
       return: sums, modifiedFilter, posLastInGroup
           vector with sums,
           modifiedFilter is a modfied version iof the input filter that might contain a few more elements,
                          it is to be used with posLastInGroup like otherProp[modifiedFilter][posLastInGroup]
           posLastInGroup: boolean vector flagging positions of last indexes in each group

    """

    isLastOfDim = torch.zeros_like(groupIds, dtype=torch.uint8).bool()
    isLastOfDim = torch.transpose(isLastOfDim,0,dim)
    isLastOfDim[-1] = True                           # flag last element in each group
    isLastOfDim = torch.transpose(isLastOfDim,0,dim)

    fltr = fltr | isLastOfDim                    # ensure last element of each group is kept
    groupIds = groupIds[fltr]
    grpsShiftedUp = torch.empty_like(groupIds)
    grpsShiftedUp[:-1] = groupIds[1:]
    grpsShiftedUp[-1] = -1                           # shifted
    isLast = grpsShiftedUp != groupIds               # isLast(!=)

    tnsor = tnsor.cumsum(dim)                          # cumsum in the dimnsion speecified
    tnsor = tnsor[fltr][isLast]                      # keep only last of each group
                                                     # this also removes dim

                                                     # find last group element for each group
    isLastOfDim = isLastOfDim[fltr][isLast]        # in same order as in tnsor , set value to 0

    res = torch.empty_like(tnsor)

    res[0]  = tnsor[0]
    res[1:] = tnsor[1:] - tnsor[:-1]                           # undo cumulative effect by subtracting prvious group
                                                             # unfortunately this creates an artifact on the first element of each group
    isLastOfDim = isLastOfDim[:-1]
    res[1:][isLastOfDim] += tnsor[:-1][isLastOfDim]           # undo effect of group shift to first sum of each group

    return res, isLast



def sumByGroupCounts(tnsor, grpCounts):
    """ Here is the trick in summing up the values for all atoms for each group:
    Input:
        tnsor a vector of groups of values that need to be sumed up per group
        grpCounts the counts of elements in each group

    return sums, indexToLastElementofGroup
           both are of size len(grpCounts)

    Algorithm:
     1 tnsor = Compute cumulative sum of tnsor
       -> The grpCounts[0]-1 element of tnsor
          will contain the sum for the first group (0)
          The grpCounts[0]+grpCounts[1]-1 element of tnsor
          will contain the sum of atomic energies for groups 0 and 1
    2 resultIdx = grpCounts.cumsum(0)-1
       -> compute the index of the last element for each group
    3 outTensor = tnsor.take(resultIdx)
       -> extract the cumulative sums for each group
    4 outTensor[1:] = outTensor[1:] - outTensor[:-1]
       -> subtract the energy of the i-1 group to revert the cumulative nature of the sum
    """

    tnsor = tnsor.cumsum(0)
    resultIdx = grpCounts.cumsum(0)-1
    tnsor = tnsor.take(resultIdx)
    outTensor = torch.empty_like(tnsor)
    outTensor[0]  = tnsor[0]
    outTensor[1:] = tnsor[1:] - tnsor[:-1]

    return outTensor, resultIdx



def uniqueCount(tnsor):
    """
        tnsor: vector with consecutive groups of elements
        return uniqValues, countOfEach: the unique elements in tnsor and their counts
    """

    # here is the trick:
    # tnsor is values that are in groups with consecutive duplicates
    #
    #    - create copy of tnsor shifted one up: tnsorShiftedUp
    #      if elements differ from the original we have a last element of a group
    #    - by using != we can find positions of the last item of each group    # isLast(!=)
    #    - filter tnsor so that we keep only one value per group                # unique(filtered)
    #    - inverse is number of replicates, use cumsum to sum the whole vector # count(1)
    #    - shift the isFirst vector down by one -> you get the isLast vector   # isLast
    #    - get the cumsum of the last element of each group                    # count(2)
    #    - substract the previous element to undo the cumsum                   # count(3)
    #    - add one to include count for the first element                      # count(4)
    #
    #  at2MolIdx shiftD  isFirst(!=) unique(filtered) count(1) isLast count(2) count(3) count(4)
    #        0    -1       1                0          0  0      1      0        0        1
    #        2     0       1                1          0  0      0
    #        2     1       0                           1->1  ->  1      1        1        2
    #        5     1       1                2          0  1      1      1        0        1
    #        4     2       1                3          0  1      1      1        0        1
    #
    #  at2MolIdx shiftU  isFirst(!=) unique(filtered) count(1) isLast count(2) count(3) count(4)
    #        0    -1       1                0          0  0      1      0        0        1
    #        2     0       1                1          0  0      0
    #        2     1       0                           1->1  ->  1      1        1        2
    #        5     1       1                2          0  1      1      1        0        1
    #        4     2       1                3          0  1      0
    #        4     3       0                           1  2      1      2        1        2

    tnsorShiftedUp = tnsor[1:].resize_as_(tnsor)
    tnsorShiftedUp[-1] = -1                                    # shifted

    isLast = tnsorShiftedUp != tnsor                            # isLast(!=)
    tnsor    = tnsor[ isLast ]                                  # unique(filtered)

    cumSum = (1-isLast.long()).cumsum(0)                             # counts(1)
    cumSum = cumSum[isLast]                                   # count(2)
    countUniq = torch.empty_like(cumSum)
    countUniq[0]  = cumSum[0]
    countUniq[1:] = cumSum[1:] - cumSum[:-1]                  # count(3)  undo cumsum
    countUniq = countUniq.add_(1)                             # count(4)

    return tnsor, countUniq


def print_all_tensors():
    for obj in gc.get_objects():
        # noinspection PyBroadException
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except: pass     # noqa: E722; # pylint: disable=W0702



if __name__ == '__main__':
    from cdd_chem.util.debug.time_it import MyTimer

    device = torch.device("cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")

    tnsr = torch.randn((1000000,768,), device=device)
    grpIds = torch.randint(1,500,(1000000,), dtype=torch.long, device=device)
    grpIds, _ = grpIds.sort(0)

    sm = sumById(tnsr, grpIds, dim=0)
    torch.cuda.synchronize()
    with MyTimer("sumById"):
        # noinspection PyRedeclaration
        sm = sumById(tnsr, grpIds, dim=0)
        torch.cuda.synchronize()

    sm2 = sumById2(tnsr, grpIds)
    torch.cuda.synchronize()
    with MyTimer("sumById"):
        # noinspection PyRedeclaration
        sm2 = sumById2(tnsr, grpIds)
        torch.cuda.synchronize()

    warn("Max deviation %f, sumDeviation = %f"%( (sm[0]-sm2[0]).abs().max(), (sm[0]-sm2[0]).abs().sum() ))
    pass
