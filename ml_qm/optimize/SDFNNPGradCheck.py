#!/usr/bin/env python
"""
# Alberto

Can be used to check numeric gradients vs analytical.
e.g. running with -conf nnp/ani22/bb/bb.3_dd.json -in ml_qm_data/4667.opt.bb.3.sdf -displacement d gives:

displacement
------------ ------------------------------------------
0.01        maxDelta2 113.0676, RMSD 62.87708709947841
0.0001      maxDelta2 0.011066, RMSD 0.6276956318294786
0.00001     maxDelta2 0.000120, RMSD 0.06353211096961109
0.000001    maxDelta2 0.000049, RMSD 0.027992
0.0000001   maxDelta2 0.027568, RMSD 0.522784
"""
import argparse
import numpy as np
import logging
import sys
import torch
import copy



from cdd_chem.io import get_mol_input_stream, get_mol_output_stream
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from cdd_chem.util import log_helper
from cdd_chem.mol import BaseMol
from t_opt.abstract_NNP_computer import AbstractNNPComputer
from t_opt.unit_conversion import Units
from t_opt.opt_util import OPT_ENERGY_TAG, OPT_STD_TAG
from ml_qm.optimize.NNP_computer_factory import NNPComputerFactory



log = logging.getLogger(__name__)

TESTRUN = 0
PROFILE = 0

torch.set_num_threads(4)

class GradChecker():
    """ help check quality of gradient """

    def __init__(self, nnpComputer:AbstractNNPComputer, displacement:float):

        self._nnpComputer = nnpComputer
        self.displacement = displacement


    def check(self, mol1:BaseMol ):
        mol_list = []
        mol_list.append(copy.deepcopy(mol1))

        self._nnpComputer.outputGrads = True
        efIterator = self._nnpComputer.computeBatch(mol_list)
        mol1, e0, std, grad0 = efIterator.__next__()
        grad0 = grad0.reshape(-1)

        mol_list = []
        natoms = mol1.num_atoms
        coords = mol1.coordinates.reshape(-1)

        for i in range(natoms*3):
            new_coords = copy.deepcopy(coords)
            new_coords[i] += self.displacement
            new_mol = copy.deepcopy(mol1)
            new_mol.coordinates = new_coords.reshape(-1,3)
            mol_list.append(new_mol)

        self._nnpComputer.outputGrads = False
        numeric_grad = np.zeros(grad0.shape)
        efIterator = self._nnpComputer.computeBatch(mol_list)
        for i, (mol, e, std, _) in enumerate(efIterator):
            numeric_grad[i] = (e - e0) / self.displacement
            mol[OPT_ENERGY_TAG] = e
            mol[OPT_STD_TAG] = std

        delta2 = (grad0 - numeric_grad)**2
        mol = mol_list[delta2.argmax()]
        mol['maxDelta2'] = delta2.max()
        mol['grad RMSD'] = np.sqrt(delta2.sum())

        warn(f'maxDelta2 {delta2.max():f}, RMSD {np.sqrt(delta2.sum()):f} '
             f'numericMaxAbsGrad {np.fabs(numeric_grad).max():f} analMaxAbsGrad {np.fabs(grad0).max():f} '
             f' e0 {e0:f} eMaxGRadDev {float(mol[OPT_ENERGY_TAG]):f}')

        return mol




def main(argv=None):
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-in' ,  metavar='fileName',  dest='inFile', type=str, default=".sdf",
                        help='input file def=.sdf')

    parser.add_argument('-conf' ,  metavar='NNP.json',  dest='confFile', type=str, required=True,
                        help='input file *.json of ANI directory')

    parser.add_argument('-out' ,  metavar='fileName',  dest='outFile', type=str, default=".sdf",
                        help='will contain conformation with biggest gradient deviation')

    parser.add_argument('-displacement', type=float, default=0.000001,
                        help='Displacement to add to coordinates to compute numerical derivative')

    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')

    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')

    args = parser.parse_args()

    log_helper.initialize_loggger(__name__, args.logFile)

    if not args.inFile: args.inFile = ".sdf"
    if not args.outFile: args.outFile = ".sdf"

    nnpFactory = NNPComputerFactory(args.confFile)
    nnpCmptr = nnpFactory.createNNP(True, True, energyOutUnits=Units.KCAL, **vars(args))
    gradCheck = GradChecker(nnpCmptr, args.displacement)

    with get_mol_output_stream(args.outFile) as out,  \
         get_mol_input_stream(args.inFile) as molS:
        for mol in molS:
            mol = gradCheck.check(mol)
            out.write_mol(mol)



if __name__ == "__main__":
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE == 1:
        import cProfile
        import pstats
        profile_filename = 'ml_qm.optimize.optimizer_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = sys.stderr
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    elif PROFILE == 2:
        import line_profiler # pylint: disable=E0401
        import atexit
        from t_opt.batch_lbfgs import BatchLBFGS # pylint: disable=C0412

        prof = line_profiler.LineProfiler()
        atexit.register(prof.print_stats)

        prof.add_function(BatchLBFGS.optimize)
#         prof.add_function(optmz.brent_wiki)
#         prof.add_function(optmz.Froot.evaluate)
#         prof.add_function(optmz.trust_step)
#         prof.add_function(optmz.getCartesianNorm)
#         prof.add_function(optmz.get_delta_prime)
#         prof.add_function(internal.InternalCoordinates.newCartesian)

        prof_wrapper = prof(main)
        prof_wrapper()
        sys.exit(0)

    elif PROFILE == 3:
        from pyinstrument import Profiler # pylint: disable=E0401
        profiler = Profiler()
        profiler.start()

        main()

        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        sys.exit(0)

    elif PROFILE == 4:
        # this needs lots of memory
        import torch.autograd  # pylint: disable=C0412
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            main()
        # use chrome://tracing  to open file
        prof.export_chrome_trace("profile_Chrome.json")
        sys.exit(0)

    sys.exit(main())
