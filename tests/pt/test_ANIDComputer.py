import numpy.testing as npt
import pprint
from ml_qm.GDBMol import GDBMol
from ml_qm.pt.PTMol import PTMol
from ml_qm.GDBMol import demoMols
from ml_qm.pt.RadialBasis import GaussianRadialBasis
from ml_qm.pt.AngularBasis import GaussianAngularBasis
from ml_qm.pt import ANIDComputer as AComput
pp = pprint.PrettyPrinter(indent=2, width=2000)


def testRadial():
    mol = PTMol(GDBMol(demoMols['C']))
    rbasis = GaussianRadialBasis([1,6],6,0.5,3.5,None,1,4)
    ad = AComput.ANIDComputer([1,6], rbasis)
    res = ad.computeDescriptors(mol)
    #pp.pprint(ad.computeDescriptors(mol))

    # indexed by atom then by surrounding atom type, e.g. CH4
    #
    #  {C: C0 [CH..., CC...]],
    #   H: H1 [HH..., HC...]],
    #      H2 [HH..., HC...]], ....
    #  }
    expectC = \
        [1.2522770e+00, 3.3078914e+00, 1.1869587e+00, 5.7856642e-02, 3.8309369e-04, 3.4458031e-07,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]
    expectH = \
        [[1.8262722e-02, 4.8100501e-01, 1.7209464e+00, 8.3640903e-01, 5.5220921e-02, 4.9524743e-04,
          3.1306532e-01, 8.2697195e-01, 2.9674280e-01, 1.4464494e-02, 9.5776704e-05, 8.6148987e-08],

         [1.8262632e-02, 4.8100367e-01, 1.7209451e+00, 8.3641016e-01, 5.5221111e-02, 4.9525016e-04,
          3.1306696e-01, 8.2697231e-01, 2.9674152e-01, 1.4464362e-02, 9.5775373e-05, 8.6147374e-08],

         [1.8261345e-02, 4.8098427e-01, 1.7209276e+00, 8.3642691e-01, 5.5223886e-02, 4.9528998e-04,
          3.1307295e-01, 8.2697368e-01, 2.9673684e-01, 1.4463881e-02, 9.5770520e-05, 8.6141505e-08],

         [1.8261347e-02, 4.8098433e-01, 1.7209277e+00, 8.3642685e-01, 5.5223875e-02, 4.9528986e-04,
          3.1307161e-01, 8.2697338e-01, 2.9673788e-01, 1.4463988e-02, 9.5771597e-05, 8.6142805e-08]]

    npt.assert_allclose(res[0,0].numpy(), expectC, 1e-5)
    npt.assert_allclose(res[0,1:5].numpy(), expectH, 1e-5)


def testRadialAngular():
    gMol = PTMol(GDBMol(demoMols['C'], quiet=True) )
    rbasis = GaussianRadialBasis([1,6],3,1,3,None,1,4)
    #atomTypes, nAngles, nRadial, rMin, rMax, rCntr, aCntr, halfWidth, angleHalfWidth, cutoff
    abasis = GaussianAngularBasis([1,6], 2, 2, 1, 2, None,None, 1, 0.4286, 4)
    ad = AComput.ANIDComputer([1,6], rbasis, abasis)
    res = ad.computeDescriptors(gMol)

    # indexed by atom then by surrounding atom type, e.g. CH4
    # number per atom: 2*3 = 6 radial and 2 * (2 *(2+1)/2) = 12 angular => total 18
    #         | radial 2 * 3 | ------------------- angular  Center(-H)-H ------------------------| angular Center(-H)-C, .... |
    #  {C: C0 [CH..., CC...  , rshift1:ashift1, rshift1:ashift2, rshift2:ashift1, rshift2:ashift2                              ]],
    #   H: H1 [HH..., HC...  , rshift1:ashift1, rshift1:ashift2, rshift2:ashift1, rshift2:ashift2
    #      H2 [HH..., HC...  , rshift1:ashift1, rshift1:ashift2, rshift2:ashift1, rshift2:ashift2  ....                        ]]
    #  }
    expectC = \
        [  3.2318,  0.3363, 0.0001,  0.0000,  0.0000,  0.0000, # radial
            0.0110, 0.0011, 0.7040, 0.0733,                    # C(-H)-H
            0.0000, 0.0000, 0.0000, 0.0000,                    # C(-C)-H
            0.0000, 0.0000, 0.0000, 0.0000]                    # C(-C)-C

    expectH = \
        [[ 0.3203, 1.5398, 0.0289, 0.8080, 0.0841, 0.,                              #radial
           0.0667, 0.3205, 0.0001, 0.0004, 0.9580, 0.6776, 0., 0., 0., 0., 0., 0. ],#angular
         [ 0.3203, 1.5398, 0.0289, 0.8080, 0.0841, 0.,
           0.0667, 0.3205, 0.0001, 0.0004, 0.9581, 0.6776, 0., 0., 0., 0., 0., 0. ],
         [ 0.3203, 1.5398, 0.0289, 0.8080, 0.0841, 0.,
           0.0667, 0.3205, 0.0001, 0.0004, 0.9581, 0.6776, 0., 0., 0., 0., 0., 0. ],
         [ 0.3203, 1.5398, 0.0289, 0.8080, 0.0841, 0.,
           0.0667, 0.3205, 0.0001, 0.0004, 0.9581, 0.6776, 0., 0., 0., 0., 0., 0. ]]

    npt.assert_array_almost_equal(res[0,0].numpy(), expectC, decimal=3)
    npt.assert_array_almost_equal(res[0,1:5].numpy(), expectH, decimal=3)
