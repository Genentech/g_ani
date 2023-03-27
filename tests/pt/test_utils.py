
import numpy.testing as npt
import torch

from ml_qm.pt import torch_util as tu
import pprint
pp = pprint.PrettyPrinter(indent=2, width=2000)

torch.set_printoptions(linewidth=200)

def test_tensor_buffer():
    tb = tu.TensorBuffer((1,2),1)
    t1 = torch.ones((1,2))
    t2 = torch.full((2,2),2)

    assert tb.buffer.shape[0] == 1
    assert tb.nRows           == 0

    tb.append(t1)
    assert tb.buffer.shape[0] == 1
    assert tb.nRows           == 1
    assert tb.buffer.sum()    == 2

    tb.append(t2)
    assert tb.buffer.shape[0] == 3
    assert tb.nRows           == 3
    assert tb.buffer.sum()    == 10
    assert tb.buffer[0].sum() == 2


    tb = tu.TensorBuffer((1,2),10)
    assert tb.buffer.shape[0] == 1
    assert tb.nRows           == 0

    tb.append(t1)
    assert tb.buffer.shape[0] == 1
    assert tb.nRows           == 1
    assert tb.buffer.sum()    == 2

    tb.append(t2)
    assert tb.buffer.shape[0]   == 11
    assert tb.nRows             == 3
    assert tb.buffer[0:3].sum() == 10
    assert tb.buffer[0].sum()   == 2

    tb.append(torch.full((10,2),-1))
    assert tb.buffer.shape[0]    == 21
    assert tb.nRows              == 13
    assert tb.buffer[0:13].sum() == -10
    assert tb.buffer[0:3].sum()  == 10

    tb.append(torch.full((8,2),0))
    assert tb.buffer.shape[0]   == 21
    assert tb.nRows             == 21
    assert tb.buffer.sum()      == -10
    assert tb.buffer[0:13].sum()== -10

    tb.append(torch.full((10,2),1))
    assert tb.buffer.shape[0]   == 31
    assert tb.nRows             == 31
    assert tb.buffer.sum()      == 10
    assert tb.buffer[0:21].sum()== -10

    tb.append(torch.full((11,2),2))
    assert tb.buffer.shape[0]   == 51
    assert tb.nRows             == 42
    assert tb.buffer[0:42].sum()== 54
    assert tb.buffer[0:31].sum()== 10


def test_gather():
    t = torch.tensor([[1,3,2],
                      [4,2,3],
                      [7,9,8],
                      [-1,-3,-2]])
    t2 = torch.tensor([[[1.1,1.2],[3.1,3.2],[2.1,2.2]],
                       [[4.1,4.2],[2.1,2.2],[3.1,3.2]],
                       [[1.7,2.7],[1.9,2.9],[1.8,2.8]],
                       [[1.2,2.2],[1.3,2.3],[1.1,2.1]]])

    t2sorted = torch.tensor([[[1.1,1.2],[2.1,2.2],[3.1,3.2]],
                             [[2.1,2.2],[3.1,3.2],[4.1,4.2]],
                             [[1.7,2.7],[1.8,2.8],[1.9,2.9]],
                             [[1.3,2.3],[1.1,2.1],[1.2,2.2]]])

    s,idx = t.sort(-1)

    nLastDim = t2.shape[-1]
    nLast2Dim = t2.shape[-2]
    nLast3Dim = t2.shape[-3]
    lastDimCounter = torch.arange(0,nLastDim,dtype=torch.long)
    last3DimCounter = torch.arange(0,nLast3Dim,dtype=torch.long)
    t2 = t2.reshape(-1)[(idx*nLastDim+(last3DimCounter*nLastDim*nLast2Dim).unsqueeze(-1)).unsqueeze(-1).expand(-1,-1,nLastDim) + lastDimCounter]

    npt.assert_equal(t2sorted,t2.numpy())




    # if t2 has same shape as t it gather will work
    t2 = torch.tensor([[2,4,3],
                       [5,3,4],
                       [8,10,9],
                       [0,-2,-1]])
    s,idx = t.sort(-1)
    npt.assert_equal(s,(t2.gather(1,idx)-1).numpy())



def test_uniqCount():

    uniq, counts= tu.uniqueCount(torch.LongTensor([1,1,2,2,3,5,6,6,6,6]))
    npt.assert_equal(uniq.numpy(), [ 1,2,3,5,6])
    npt.assert_equal(counts.numpy(), [2,2,1,1,4])

    uniq, counts= tu.uniqueCount(torch.LongTensor([5]))
    npt.assert_equal(uniq.numpy(),   [5])
    npt.assert_equal(counts.numpy(), [1])


    uniq, counts= tu.uniqueCount(torch.LongTensor([5,2,2,1]))
    npt.assert_equal(uniq.numpy(),   [5,2,1])
    npt.assert_equal(counts.numpy(), [1,2,1])


def test_sumById2():

    sums, poss = tu.sumById2(torch.tensor([1.1,1.2, 2.1,2.2, 3.1, 5.5, 6.1,6.2,6.3,6.4]),
                            torch.tensor([1,1,     2,2,     3,   5,   6,6,6,6]))
    npt.assert_allclose(sums.numpy(),    [2.3,     4.3,     3.1, 5.5, 25])
    npt.assert_equal(poss.numpy(),       [1,  0,   1,  0,   1,   1,   1,  0,  0,  0])

    sums, poss = tu.sumById2(torch.tensor([5.5]),
                            torch.tensor([1]))
    npt.assert_allclose(sums.numpy(),    [5.5])
    npt.assert_equal(poss.numpy(),       [1])

    sums, poss = tu.sumById2(torch.tensor([1.1, 2.1,2.2, 3.1, 5.5, 6.1]),
                            torch.tensor([1,   2,2,     3,   5,   6]))
    npt.assert_allclose(sums.numpy(),    [1.1, 4.3,     3.1, 5.5, 6.1])
    npt.assert_equal(poss.numpy(),       [1,   1,  0,   1,   1,   1])



def test_sumById():

    sums, poss = tu.sumById(torch.tensor([1.1,1.2, 2.1,2.2, 3.1, 5.5, 6.1,6.2,6.3,6.4]),
                            torch.tensor([1,1,     2,2,     3,   5,   6,6,6,6]), 0)
    npt.assert_allclose(sums.numpy(),    [2.3,     4.3,     3.1, 5.5, 25])
    npt.assert_equal(poss.numpy(),       [0,  1,   0,  1,   1,   1,   0,  0,  0,  1])

    sums, poss = tu.sumById(torch.tensor([5.5]),
                            torch.tensor([1]), 0)
    npt.assert_allclose(sums.numpy(),    [5.5])
    npt.assert_equal(poss.numpy(),       [1])

    sums, poss = tu.sumById(torch.tensor([1.1, 2.1,2.2, 3.1, 5.5, 6.1]),
                            torch.tensor([1,   2,2,     3,   5,   6]), 0)
    npt.assert_allclose(sums.numpy(),    [1.1, 4.3,     3.1, 5.5, 6.1])
    npt.assert_equal(poss.numpy(),       [1,   0,  1,   1,   1,   1])


    sums, poss = tu.sumById(torch.tensor([ [1.1, 2.1,2.2, 3.1, 5.5, 6.1],
                                           [1.1,1.2, 2.1, 3.1, 6.1,6.2]]),

                            torch.tensor([ [1,   2,2,     3,   5,   6],
                                           [1,   1,  22,  33,  66, 66]]), 1)

    npt.assert_allclose(sums.numpy(),      [1.1, 4.3,     3.1, 5.5, 6.1,
                                            2.3,     2.1, 3.1, 12.3   ], 1e-4)
    npt.assert_equal(poss.numpy(),       [ [1,   0,  1,   1,   1,   1],
                                           [0,   1,  1,   1,   0,   1]])


def test_sumByIdFilter():

    sums, poss = tu.sumByIdFilter(torch.tensor([0. ,0,1.2, 0,2.1,2.2, 3.1, 5.5, 6.1,6.2,6.3,6.4,0]),
                                  torch.tensor([1,1,1,     2,  2,  2,   3,   5, 6,6,6,6,6]),
                              torch.ByteTensor([0,  0, 1,  0, 1, 1,   1,    1,  1,  1,  1,  1,  0]).bool(), 0)
    npt.assert_allclose(sums.numpy(),          [1.2,       4.3,       3.1, 5.5, 25],1e-4)
    npt.assert_equal(poss.numpy(),             [      1,   0,  1,   1,   1,   0,  0,  0,  0,  1])

    sums, poss = tu.sumByIdFilter(torch.tensor([5.5]),
                                  torch.tensor([1]),
                              torch.ByteTensor([1]).bool(), 0)
    npt.assert_allclose(sums.numpy(),    [5.5])
    npt.assert_equal(poss.numpy(),       [1])

#     # this will fail due to []
#     sums, poss = tu.sumByIdFilter(torch.tensor([0.]),
#                                   torch.tensor([1]),
#                               torch.ByteTensor([0]).bool(), 0)
#     npt.assert_allclose(sums.numpy(),    [])
#     npt.assert_equal(poss.numpy(),       [])

    sums, poss = tu.sumById(torch.tensor([1.1, 2.1,2.2, 3.1, 5.5, 6.1]),
                            torch.tensor([1,   2,2,     3,   5,   6]), 0)
    npt.assert_allclose(sums.numpy(),    [1.1, 4.3,     3.1, 5.5, 6.1])
    npt.assert_equal(poss.numpy(),       [1,   0,  1,   1,   1,   1])


    sums, poss = tu.sumByIdFilter(torch.tensor([ [1.1,0,0, 2.1,2.2, 3.1, 5.5, 0,6.1],
                                                 [1.1,1.2, 0,0,0, 2.1, 3.1, 6.1,6.2]]),

                                  torch.tensor([ [1,1,1,   2,2,     3,   5,   6,6],
                                                 [1,   1,  3,3,3, 22,  33,  66, 66]]),
                              torch.ByteTensor([ [1,0,0,   1,1,     1,   1,   0,1],
                                                 [1,1,     0,0,0,  1,   1,   1,1]]).bool(),   1)

    npt.assert_allclose(sums.numpy(), [1.1, 4.3,     3.1, 5.5, 6.1,
                                       2.3,     2.1, 3.1, 12.3   ], 1e-4)
    npt.assert_equal(poss.numpy(),    [ 1,   0,  1,   1,   1,   1,
                                        0,   1,  1,   1,   0,   1])



def test_sumByGroupCounts():

    sums, poss = tu.sumByGroupCounts(torch.tensor([1.1,1.2, 2.1,2.2, 3.1, 5.5, 6.1,6.2,6.3,6.4]),
                                     torch.tensor([2,       2,       1,   1,   4]))
    npt.assert_allclose(sums.numpy(),    [2.3,     4.3,     3.1, 5.5, 25])
    npt.assert_equal(poss.numpy(),       [1,       3,       4,   5,   9])

    sums, poss = tu.sumByGroupCounts(torch.tensor([5.5]),
                                     torch.tensor([1]))
    npt.assert_allclose(sums.numpy(),    [5.5])
    npt.assert_equal(poss.numpy(),       [0])

    sums, poss = tu.sumByGroupCounts(torch.tensor([1.1, 2.1,2.2, 3.1, 5.5, 6.1]),
                                     torch.tensor([1,   2,       1,   1,   1]))
    npt.assert_allclose(sums.numpy(),    [1.1, 4.3,     3.1, 5.5, 6.1])
    npt.assert_equal(poss.numpy(),       [0,   2,       3,   4,   5])
