import torch
import numpy as np
import numpy.testing as npt
import pprint

from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from ml_qm.GDBMol import GDBMol
from ml_qm.pt.PTMol import PTMol
from ml_qm.GDBMol import demoMols
from ml_qm.ANIMol import ANIMol
from ml_qm.pt.RadialBasis import SlaterRadialBasis, SigmoidRadialBasis,\
    LinearRadialBasis, GaussianRadialBasis, SimpleDistInfo, BumpRadialBasis
pp = pprint.PrettyPrinter(indent=2, width=2000)

torch.set_printoptions(linewidth=200)
np.set_printoptions(4,linewidth=140,suppress=True)


def test_Slater():
    atomTypes= torch.LongTensor([ 1,6,7,8])
    xyz = torch.tensor(
        [[0.,0,0],
         [1,0,0],
         [0,2,0],
         [-3,0,0]])

    mol = PTMol(ANIMol("HCNO", 0.0, atomTypes, xyz))

    rbasis = SlaterRadialBasis([1,6,7,8], 16,0.5,4.34375,None, 0.416277306,4.6)

    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = rbasis.computeDescriptorBatch(coords, atTypes)

    # this is 4x the torvchani results due to difference in formula
    expectedH0 = \
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.168 , 0.3943, 0.8517, 0.3628, 0.1545, 0.0658, 0.028 , 0.0119, 0.0051, 0.0022, 0.0009, 0.0004, 0.0002, 0.0001, 0.    , 0.    ,
         0.0041, 0.0096, 0.0224, 0.0527, 0.1237, 0.2904, 0.5311, 0.2262, 0.0964, 0.0411, 0.0175, 0.0074, 0.0032, 0.0014, 0.0006, 0.0002,
         0.0001, 0.0002, 0.0004, 0.0008, 0.002 , 0.0047, 0.0109, 0.0257, 0.0603, 0.1416, 0.2192, 0.0934, 0.0398, 0.0169, 0.0072, 0.0031]
    expectedC1 = \
        [0.168 , 0.3943, 0.8517, 0.3628, 0.1545, 0.0658, 0.028 , 0.0119, 0.0051, 0.0022, 0.0009, 0.0004, 0.0002, 0.0001, 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.0016, 0.0038, 0.0089, 0.0208, 0.0489, 0.1147, 0.2694, 0.4306, 0.1834, 0.0781, 0.0333, 0.0142, 0.006 , 0.0026, 0.0011, 0.0005,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0001, 0.0001, 0.0003, 0.0008, 0.0018, 0.0043, 0.0101, 0.0236, 0.0309, 0.0132]
    expectedN2 = \
        [0.0041, 0.0096, 0.0224, 0.0527, 0.1237, 0.2904, 0.5311, 0.2262, 0.0964, 0.0411, 0.0175, 0.0074, 0.0032, 0.0014, 0.0006, 0.0002,
         0.0016, 0.0038, 0.0089, 0.0208, 0.0489, 0.1147, 0.2694, 0.4306, 0.1834, 0.0781, 0.0333, 0.0142, 0.006 , 0.0026, 0.0011, 0.0005,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.0001, 0.0003, 0.0006, 0.0014, 0.0033, 0.0077, 0.0182, 0.0427, 0.1002, 0.0523, 0.0223, 0.0095]
    expectedO3 = \
        [0.0001, 0.0002, 0.0004, 0.0008, 0.002 , 0.0047, 0.0109, 0.0257, 0.0603, 0.1416, 0.2192, 0.0934, 0.0398, 0.0169, 0.0072, 0.0031,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0001, 0.0001, 0.0003, 0.0008, 0.0018, 0.0043, 0.0101, 0.0236, 0.0309, 0.0132,
         0.    , 0.    , 0.    , 0.    , 0.0001, 0.0003, 0.0006, 0.0014, 0.0033, 0.0077, 0.0182, 0.0427, 0.1002, 0.0523, 0.0223, 0.0095,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]
    npt.assert_allclose(res[0,0].numpy(), expectedH0, atol=1e-4)
    npt.assert_allclose(res[0,1].numpy(), expectedC1, atol=1e-4)
    npt.assert_allclose(res[0,2].numpy(), expectedN2, atol=1e-4)
    npt.assert_allclose(res[0,3].numpy(), expectedO3, atol=1e-4)




def test_sigmoid():
    atomTypes= torch.LongTensor([ 1,6,7,8])
    xyz = torch.tensor(
        [[0.,0,0],
         [1,0,0],
         [0,2,0],
         [-3,0,0]])

    mol = PTMol(ANIMol("HCNO", 0.0, atomTypes, xyz))

    rbasis = SigmoidRadialBasis([1,6,7,8], 16,0.5,4.34375,None, 0.416277306,4.6)

    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = rbasis.computeDescriptorBatch(coords, atTypes)

    # this is 4x the torvchani results due to difference in formula
    expected = \
      [[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
         0.7695,  0.5035, -0.0293, -0.5418, -0.7833, -0.8596, -0.8805, -0.8859, -0.8874, -0.8877, -0.8878, -0.8878, -0.8879, -0.8879, -0.8879, -0.8879,
         0.6013,  0.6   ,  0.5952,  0.5769,  0.511 ,  0.3133, -0.0594, -0.3911, -0.5391, -0.5849, -0.5973, -0.6006, -0.6014, -0.6017, -0.6017, -0.6017,
         0.27  ,  0.27  ,  0.27  ,  0.2699,  0.2697,  0.2691,  0.2666,  0.2573,  0.224 ,  0.1271, -0.0441, -0.1853, -0.2452, -0.2633, -0.2682, -0.2695],
       [ 0.7695,  0.5035, -0.0293, -0.5418, -0.7833, -0.8596, -0.8805, -0.8859, -0.8874, -0.8877, -0.8878, -0.8878, -0.8879, -0.8879, -0.8879, -0.8879,
         0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
         0.5217,  0.5214,  0.5202,  0.5155,  0.4979,  0.4351,  0.2509, -0.0788, -0.3547, -0.4728, -0.5087, -0.5184, -0.5209, -0.5216, -0.5218, -0.5218,
         0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0412,  0.0408,  0.0392,  0.0335,  0.0173, -0.0094, -0.0298],
       [ 0.6013,  0.6   ,  0.5952,  0.5769,  0.511 ,  0.3133, -0.0594, -0.3911, -0.5391, -0.5849, -0.5973, -0.6006, -0.6014, -0.6017, -0.6017, -0.6017,
         0.5217,  0.5214,  0.5202,  0.5155,  0.4979,  0.4351,  0.2509, -0.0788, -0.3547, -0.4728, -0.5087, -0.5184, -0.5209, -0.5216, -0.5218, -0.5218,
         0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
         0.111 ,  0.111 ,  0.111 ,  0.1109,  0.1109,  0.1109,  0.1109,  0.1107,  0.1101,  0.1077,  0.099 ,  0.0709,  0.0089, -0.0592, -0.0948, -0.1065],
       [ 0.27  ,  0.27  ,  0.27  ,  0.2699,  0.2697,  0.2691,  0.2666,  0.2573,  0.224 ,  0.1271, -0.0441, -0.1853, -0.2452, -0.2633, -0.2682, -0.2695,
         0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0414,  0.0412,  0.0408,  0.0392,  0.0335,  0.0173, -0.0094, -0.0298,
         0.111 ,  0.111 ,  0.111 ,  0.1109,  0.1109,  0.1109,  0.1109,  0.1107,  0.1101,  0.1077,  0.099 ,  0.0709,  0.0089, -0.0592, -0.0948, -0.1065,
         0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ]]

    npt.assert_allclose(res[0].numpy(), expected, atol=1e-4)


def test_Linear():
    atomTypes= torch.LongTensor([ 1,6,7,8])
    xyz = torch.tensor(
        [[0.,0,0],
         [1,0,0],
         [0,2,0],
         [-3,0,0]])

    mol = PTMol(ANIMol("HCNO", 0.0, atomTypes, xyz))

    rbasis = LinearRadialBasis([1,6,7,8], 16,0.5,4.34375,None, 4.6)

    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = rbasis.computeDescriptorBatch(coords, atTypes)

    # this is 4x the torvchani results due to difference in formula
    expected = \
        [[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
         0.4439,  0.2164, -0.0111, -0.2386, -0.4661, -0.6936, -0.9212, -1.1487, -1.3762, -1.6037, -1.8312, -2.0587, -2.2862, -2.5137, -2.7413, -2.9688,
         0.9026,  0.7484,  0.5942,  0.44  ,  0.2858,  0.1316, -0.0226, -0.1768, -0.331 , -0.4851, -0.6393, -0.7935, -0.9477, -1.1019, -1.2561, -1.4103,
         0.6749,  0.6057,  0.5366,  0.4674,  0.3982,  0.329 ,  0.2598,  0.1907,  0.1215,  0.0523, -0.0169, -0.0861, -0.1552, -0.2244, -0.2936, -0.3628],
       [ 0.4439,  0.2164, -0.0111, -0.2386, -0.4661, -0.6936, -0.9212, -1.1487, -1.3762, -1.6037, -1.8312, -2.0587, -2.2862, -2.5137, -2.7413, -2.9688,
         0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
         0.9059,  0.7722,  0.6385,  0.5048,  0.3711,  0.2373,  0.1036, -0.0301, -0.1638, -0.2975, -0.4313, -0.565 , -0.6987, -0.8324, -0.9661, -1.0998,
         0.1449,  0.1343,  0.1237,  0.1131,  0.1025,  0.0918,  0.0812,  0.0706,  0.06  ,  0.0494,  0.0388,  0.0282,  0.0176,  0.007 , -0.0036, -0.0142],
       [ 0.9026,  0.7484,  0.5942,  0.44  ,  0.2858,  0.1316, -0.0226, -0.1768, -0.331 , -0.4851, -0.6393, -0.7935, -0.9477, -1.1019, -1.2561, -1.4103,
         0.9059,  0.7722,  0.6385,  0.5048,  0.3711,  0.2373,  0.1036, -0.0301, -0.1638, -0.2975, -0.4313, -0.565 , -0.6987, -0.8324, -0.9661, -1.0998,
         0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
         0.3446,  0.3161,  0.2877,  0.2593,  0.2308,  0.2024,  0.174 ,  0.1455,  0.1171,  0.0887,  0.0603,  0.0318,  0.0034, -0.025 , -0.0535, -0.0819],
       [ 0.6749,  0.6057,  0.5366,  0.4674,  0.3982,  0.329 ,  0.2598,  0.1907,  0.1215,  0.0523, -0.0169, -0.0861, -0.1552, -0.2244, -0.2936, -0.3628,
         0.1449,  0.1343,  0.1237,  0.1131,  0.1025,  0.0918,  0.0812,  0.0706,  0.06  ,  0.0494,  0.0388,  0.0282,  0.0176,  0.007 , -0.0036, -0.0142,
         0.3446,  0.3161,  0.2877,  0.2593,  0.2308,  0.2024,  0.174 ,  0.1455,  0.1171,  0.0887,  0.0603,  0.0318,  0.0034, -0.025 , -0.0535, -0.0819,
         0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ]]

    npt.assert_allclose(res[0].numpy(), expected, atol=1e-4)


def test_KHAN():
    atomTypes= torch.LongTensor([ 1,6,7,8])
    xyz = torch.tensor(
        [[0.,0,0],
         [1,0,0],
         [0,2,0],
         [-3,0,0]])

    mol = PTMol(ANIMol("HCNO", 0.0, atomTypes, xyz))

    rbasis = GaussianRadialBasis([1,6,7,8], 16,0.5,4.34375,None, 0.416277306,4.6)

    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = rbasis.computeDescriptorBatch(coords, atTypes)

    # this is 4x the torvchani results due to difference in formula
    expectedH0 = \
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.0163, 0.3432, 0.8856, 0.2796, 0.0108, 0.0001, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.0001, 0.0163, 0.2798, 0.5883, 0.1513, 0.0048, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0001, 0.0106, 0.1481, 0.2536, 0.0531, 0.0014, 0.    , 0.    , 0.    ]
    expectedC1 = \
        [0.0163, 0.3432, 0.8856, 0.2796, 0.0108, 0.0001, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.0002, 0.0191, 0.2777, 0.4948, 0.1078, 0.0029, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0023, 0.0262, 0.0366, 0.0062]
    expectedN2 = \
        [0.    , 0.    , 0.    , 0.0001, 0.0163, 0.2798, 0.5883, 0.1513, 0.0048, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.0002, 0.0191, 0.2777, 0.4948, 0.1078, 0.0029, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.001 , 0.0298, 0.1093, 0.0491, 0.0027, 0.    ]
    expectedO3 = \
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0001, 0.0106, 0.1481, 0.2536, 0.0531, 0.0014, 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0023, 0.0262, 0.0366, 0.0062,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.001 , 0.0298, 0.1093, 0.0491, 0.0027, 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]
    npt.assert_allclose(res[0,0].numpy(), expectedH0, atol=1e-4)
    npt.assert_allclose(res[0,1].numpy(), expectedC1, atol=1e-4)
    npt.assert_allclose(res[0,2].numpy(), expectedN2, atol=1e-4)
    npt.assert_allclose(res[0,3].numpy(), expectedO3, atol=1e-4)


def test_TorchAni():
    atomTypes= torch.LongTensor([ 1,6,7,8])
    xyz = torch.tensor(
        [[0.,0,0],
         [1,0,0],
         [0,2,0],
         [-3,0,0]])

    mol = PTMol(ANIMol("HCNO", 0.0, atomTypes, xyz))

    rbasis = GaussianRadialBasis([1,6,7,8], 16,0.9,4.9313,None, 0.416277306,5.2)

    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = rbasis.computeDescriptorBatch(coords, atTypes)

    # this is 4x the torvchani results due to difference in formula
    expectedC = \
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.7767, 0.5779, 0.0426, 0.0003, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.0043, 0.1703, 0.6706, 0.2617, 0.0101, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0085, 0.1769, 0.3654, 0.0748, 0.0015, 0.    , 0.    , 0.    , 0.    , 0.    ]
    expectedH1 = \
        [0.7767, 0.5779, 0.0426, 0.0003, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.0068, 0.2047, 0.6085, 0.1793, 0.0052, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
         0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0001, 0.0083, 0.0904, 0.0979, 0.0105, 0.0001, 0.    ]

    npt.assert_allclose(res[0,0].numpy(), expectedC, atol=1e-4)
    npt.assert_allclose(res[0,1].numpy(), expectedH1, atol=1e-4)

    atomTypes= torch.LongTensor([ 6,1,1,1,1])
    xyz = torch.tensor(
        [[-0.0035,  0.0102,  0.0194],
         [-0.7955,  0.5767, -0.5472],
         [-0.3938, -0.9799,  0.2723],
         [ 0.6345,  0.4474,  0.9357],
         [ 0.5958, -0.1652, -0.8916]])

    mol = PTMol(ANIMol("CH4", 0.0, atomTypes, xyz))
    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)

    res = rbasis.computeDescriptorBatch(coords, atTypes)
    r,_=res[0].sort(descending=True)
    r = r[:,0:12].numpy()
    np.set_printoptions(4,linewidth=268,suppress=True, threshold=10000)

    expected = \
      [[3.379 , 1.5474, 0.828 , 0.0233, 0.0001, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
       [1.3926, 1.1363, 0.8637, 0.5605, 0.428 , 0.3908, 0.1892, 0.0339, 0.0267, 0.0041, 0.0002, 0.0002],
       [1.8079, 1.3646, 0.818 , 0.4903, 0.3042, 0.1353, 0.1302, 0.0061, 0.0022, 0.0015, 0.    , 0.    ],
       [1.8686, 0.8617, 0.8416, 0.6199, 0.3521, 0.209 , 0.0471, 0.0305, 0.0143, 0.0003, 0.0002, 0.0001],
       [1.7424, 1.0602, 0.8356, 0.6778, 0.4573, 0.1536, 0.1514, 0.0376, 0.003 , 0.0027, 0.0002, 0.    ]]
    npt.assert_allclose(r, expected, atol=1e-4)


def test_Batch():
    mol = PTMol(GDBMol(demoMols['C']))
    rbasis = GaussianRadialBasis([1,6], 6,0.5,3.5,None, 1,4)
    resSingle = rbasis.computeDescriptors(mol)

    # add two copies of mol to coords
    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)
    res = rbasis.computeDescriptorBatch(coords, atTypes)

    cSingle = resSingle[6].reshape(-1)
    hSingle = resSingle[1][0].reshape(-1)
    npt.assert_allclose(cSingle, res[0][0], 1e-4)
    npt.assert_allclose(cSingle, res[1][0], 1e-4)
    npt.assert_allclose(hSingle, res[0][1], 1e-4)
    npt.assert_allclose(hSingle, res[1][1], 1e-4)

    #cutoff
    rbasis = GaussianRadialBasis([1,6], 6,0.5,3.5,None, 1,1.7831)
    resSingle = rbasis.computeDescriptors(mol)
    res       = rbasis.computeDescriptorBatch(coords, atTypes)

    cSingle = resSingle[6].reshape(-1)
    hSingle = resSingle[1][0].reshape(-1)
    npt.assert_allclose(cSingle, res[0][0], 1e-4)
    npt.assert_allclose(cSingle, res[1][0], 1e-4)
    npt.assert_allclose(hSingle, res[0][1], 1e-4)
    npt.assert_allclose(hSingle, res[1][1], 1e-4)


def test_HNO():
    energy = 43.477840
    atomTypes= torch.LongTensor([ 1,  7,  8])
    xyz = torch.tensor(
        [[-0.9863,  0.8474,  0.0000],
         [ 0.0707,  0.5792,  0.0000],
         [ 0.0594, -0.6163,  0.0000]])

    mol = PTMol(ANIMol("C3H7N", energy, atomTypes, xyz))
    #                                  nCenters, centerMin, centerMax, centers, halfWidth(.416=>Eta=16), cutoff",
    rb = GaussianRadialBasis([1,6,7,8], 32,        0.5,       4.34,    None,      0.416,                 4.6)
    bDesc = rb.computeDescriptorBatch(np.expand_dims(xyz.numpy(),0), atomTypes.numpy())
    sDesc = rb.computeDescriptors(mol)

    countDiffs = 0
    for atT, bd in zip(atomTypes, bDesc[0]):
        sdesc = sDesc[atT.item()]
        minDiff = 999999
        for sd in sdesc:
            sd = sd.reshape(-1)
            diff = (sd - bd).abs().sum()
            if diff < minDiff:
                minPair = (sd,bd)
                minDiff = diff
        if minDiff > 0.01:
            warn("AtomType = %s" % atT)
            warn("diff %s\nsingle %s\nbatch  %s\n" % (minPair[0] - minPair[1],  minPair[0],minPair[1]))
            countDiffs += 1

    assert countDiffs == 0



def test_GaussianRadialBasis():

    #       atomTypes, nCenters, centerMin, centerMax, centers, halfWidth, cutoff
    rb = GaussianRadialBasis([1,6], 7, 0.5, 3.5, None, 1, 4)
    r = torch.tensor([0, 0.5, 1])
    distInfo = SimpleDistInfo(r)
    res= rb._computeRadialDescriptors(distInfo)   # pylint: disable=W0212

    expect = np.array(
        [[5.00000000e-01, 6.25000000e-02, 1.95312500e-03, 1.52587891e-05, 2.98023224e-08, 1.45519152e-11, 1.77635684e-15],
     [9.61939766e-01, 4.80969883e-01, 6.01212354e-02, 1.87878861e-03, 1.46780360e-05, 2.86680390e-08, 1.39980659e-11],
            [4.26776695e-01, 8.53553391e-01, 4.26776695e-01, 5.33470869e-02, 1.66709647e-03, 1.30241911e-05, 2.54378733e-08]])

    npt.assert_allclose(res.numpy(), expect, 1e-5)


def test_CH4():
    mol = PTMol(GDBMol(demoMols['C']))
    rbasis = GaussianRadialBasis([1,6], 6,0.5,3.5,None,1,4)
    res = rbasis.computeDescriptors(mol)
    #pp.pprint(ad.computeDescriptors(mol))

    # indexed by atom then by surrounding atom type, e.g. CH4
    #  [C0: [[CH], [CC]],
    #   H1: [[HH], [HC]],
    #   H2: [[HH], [HC]], ....
    expectC = \
        [[[1.2522770e+00, 3.3078914e+00, 1.1869587e+00, 5.7856642e-02, 3.8309369e-04, 3.4458031e-07],
          [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]]
    expectH = \
        [[[1.8262722e-02, 4.8100501e-01, 1.7209464e+00, 8.3640903e-01, 5.5220921e-02, 4.9524743e-04],
          [3.1306532e-01, 8.2697195e-01, 2.9674280e-01, 1.4464494e-02, 9.5776704e-05, 8.6148987e-08]],

         [[1.8262632e-02, 4.8100367e-01, 1.7209451e+00, 8.3641016e-01, 5.5221111e-02, 4.9525016e-04],
          [3.1306696e-01, 8.2697231e-01, 2.9674152e-01, 1.4464362e-02, 9.5775373e-05, 8.6147374e-08]],

         [[1.8261345e-02, 4.8098427e-01, 1.7209276e+00, 8.3642691e-01, 5.5223886e-02, 4.9528998e-04],
          [3.1307295e-01, 8.2697368e-01, 2.9673684e-01, 1.4463881e-02, 9.5770520e-05, 8.6141505e-08]],

         [[1.8261347e-02, 4.8098433e-01, 1.7209277e+00, 8.3642685e-01, 5.5223875e-02, 4.9528986e-04],
          [3.1307161e-01, 8.2697338e-01, 2.9673788e-01, 1.4463988e-02, 9.5771597e-05, 8.6142805e-08]]]

    npt.assert_allclose(res[6].numpy(), expectC, 1e-5)
    npt.assert_allclose(res[1].numpy(), expectH, 1e-5)




def test_OH3():
    # pylint: disable=W0105
    mol = PTMol(GDBMol(demoMols['angleTest']))

    rbasis = GaussianRadialBasis([8,1], 3,1,3,None,2,9)

    r = torch.tensor([1,2,3.])
    distInfo = SimpleDistInfo(r)
    res= rbasis._computeRadialDescriptors(distInfo)   # pylint: disable=W0212

    # column sum should be the same as below for OH
    expect = \
        [[ 0.96985,  0.48492,  0.06062],
         [ 0.44151,  0.88302,  0.44151],
         [ 0.04688,  0.37500,  0.75000]]
    npt.assert_allclose(res.numpy(), expect, 2e-4)

    """ Now try:
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal istances are 1-2: 2.3,    2-3: 3.6

                  H:2
                  |
                  |
    H:-3 ---------- O:0 ----- H:1
    """
    res = rbasis.computeDescriptors(mol)
    #pp.pprint(ad.computeDescriptors(mol))


    # indexed by atom then by surrounding atom type, e.g. CH4
    #  [O0: [[OO], [OH]],
    #   H1: [[OH], [HH]],
    #   H2: [[OH], [HH]], ....
    expectO = \
        [[[ 0.0000,  0.0000,  0.0000],
          [ 1.4582,  1.7429,  1.2521]]]
    expectH = \
        [[[ 0.96985,  0.48492,  0.06062],
          [ 0.29775,  0.85954,  0.86413]],

         [[ 0.44151,  0.88302,  0.44151],
          [ 0.30251,  0.93234,  1.07762]],

         [[ 0.04699,  0.37500,  0.75000],
          [ 0.00706,  0.14622,  0.80031]]]

    npt.assert_array_almost_equal(res[8].numpy(), expectO, 3)
    npt.assert_array_almost_equal(res[1].numpy(), expectH, 3)


def test_GaussianRadialBasisSimple():

    #       atomTypes, nCenters, centerMin, centerMax, halfWidth, cutoff
    rb = GaussianRadialBasis([1,6], 1, 1, 1, None, 1, 9)
    r = torch.tensor([-1., 0., 0.2, 0.5, 1, 1.5, 1.8, 2.0, 2.1])
    distInfo = SimpleDistInfo(r)
    res= rb._computeRadialDescriptors(distInfo)   # pylint: disable=W0212

    expect = [0., 0.0625, 0.1694, 0.4962, 0.9698, 0.4665, 0.1534, 0.0552, 0.0304]

    npt.assert_array_almost_equal(expect, res.numpy().flatten(), 3)


def test_BumpRadialBasis():
    #       atomTypes, nCenters, centerMin, centerMax, centers, halfWidth, maxWidthMultiplier
    rb = BumpRadialBasis([1,6], 1, 1, 1, None, 1, 2)
    r = torch.tensor([-1., 0, 0.2, 0.5, 1, 1.5, 1.8, 2.0, 2.1])
    distInfo = SimpleDistInfo(r)
    res = rb._computeRadialDescriptors(distInfo)   # pylint: disable=W0212

    expect = [0.,   0.,   0.0248,   0.5,   1.,     0.5,   0.0248,  0.,    0.]

    npt.assert_allclose(expect, res.numpy().flatten(), 1e-3)


def test_BatchBump():
    mol = PTMol(GDBMol(demoMols['C']))
    rbasis = BumpRadialBasis([1,6], 6,0.5,3.5,None,1,3)
    resSingle = rbasis.computeDescriptors(mol)

    # add two copies of mol to coords
    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)
    res = rbasis.computeDescriptorBatch(coords, atTypes)

    cSingle = resSingle[6].reshape(-1)
    hSingle = resSingle[1][0].reshape(-1)
    npt.assert_allclose(cSingle, res[0][0], atol=1e-4)
    npt.assert_allclose(cSingle, res[1][0], atol=1e-4)
    npt.assert_allclose(hSingle, res[0][1], atol=1e-4)
    npt.assert_allclose(hSingle, res[1][1], atol=1e-4)


# def test_Batch2():
#     rbasis = GaussianRadialBasis([1,6,7,8], 6,0.5,3.5,None,1,4)
#
#     coords = np.asarray([[ [ 8.1452723e-19, -4.5979507e-03,  1.3252889e-01],
#                            [-1.8325644e-17,  6.3922459e-01, -6.1352789e-01],
#                            [ 5.3985029e-18, -5.6625175e-01, -5.5660444e-01]]])
#     aTypeList = np.asarray([[8, 1, 1]])
#
#     res = rbasis.computeDescriptorBatch(coords, aTypeList)
#
