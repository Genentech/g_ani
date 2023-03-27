#!/usr/bin/env python

# Alberto

import argparse
import numpy as np
import logging
import sys
import torch
import re
from typing import Iterable, Type, List, Tuple, Optional, Iterator

from cdd_chem.io import get_mol_input_stream, get_mol_output_stream
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from cdd_chem.util import log_helper
from cdd_chem.util import iterate
from cdd_chem.mol import BaseMol
from t_opt.unit_conversion import Units
from t_opt.opt_util import OPT_ENERGY_TAG, OPT_FORCE_TAG, OPT_STD_TAG,\
    OPT_MAX_FORCE_TAG, OPT_STD_PER_ATOM_TAG
from t_opt.abstract_NNP_computer import AbstractNNPComputer
from ml_qm.optimize.NNP_computer_factory import NNPComputerFactory


log = logging.getLogger(__name__)

TESTRUN = 0
PROFILE = 0

torch.set_num_threads(4)

class SDFComputer:
    """ compute NNP energy for one sdf record """

    def __init__(self, molInStream:Iterable[Type[BaseMol]], nnpComputer:AbstractNNPComputer, tag_prefix:str = "NNP::"):

        self.molIn:Iterator[BaseMol] = iterate.PushbackIterator(molInStream)
        self._nnpComputer = nnpComputer
        self._currentMol = 0
        self._molBatch: List[BaseMol]   = []
        self.countMol = 0
        self.efIterator = None
        self.batch_natoms = 0
        self.batch_atom_order: Optional[Tuple] = None

        self.is_same_batch = self.is_same_batch_by_at_count
        if nnpComputer.batch_by_atom_order:
            self.is_same_batch = self.is_same_batch_by_at_order

        self.energy_tag    = re.sub("NNP_", tag_prefix, OPT_ENERGY_TAG)
        self.std_tag       = re.sub("NNP_", tag_prefix, OPT_STD_TAG)
        self.std_per_atom_tag= re.sub("NNP_", tag_prefix, OPT_STD_PER_ATOM_TAG)
        self.force_tag     = re.sub("NNP_", tag_prefix, OPT_FORCE_TAG)
        self.max_force_tag = re.sub("NNP_", tag_prefix, OPT_MAX_FORCE_TAG)


    def is_same_batch_by_at_count(self, mol:BaseMol):
        return mol.num_atoms == self.batch_natoms


    def is_same_batch_by_at_order(self,mol:BaseMol):
        mol_ats = tuple(at.atomic_num for at in mol.atoms)
        if mol_ats == self.batch_atom_order: return True
        if not self.batch_atom_order:  # this is a new batch
            self.batch_atom_order = mol_ats
            return True
        return False


    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __exit__(self, *args):
        pass


    def __next__(self):
        # pylint: disable=W1203
        if self._currentMol >= len(self._molBatch):
            self._molBatch.clear()
            nMols = 0
            self.batch_natoms = 0
            batch_size = 1    # initial value will be adjusted in first record
            while nMols < batch_size and self.molIn.has_next():
                mol = self.molIn.__next__()

                for at in mol.atoms:
                    if at.atomic_num not in self._nnpComputer.allowed_atom_num:
                        log.warning(f"Unknown atom type {at.atomic_num} in {mol.title}")
                        break
                else:
                    nAt = mol.num_atoms
                    if self.batch_natoms == 0:
                        self.batch_natoms = nAt

                        # assume memory requirement goes with nAtom^3
                        batch_size = self._nnpComputer.maxConfsPerBatch(self.batch_natoms)
                        log.debug(f"nAt: {self.batch_natoms}, batchSize: {batch_size}")

                    if not self.is_same_batch(mol):
                        # batch must either have ame atom count gANI or same atom sequece neuroChem
                        self.batch_atom_order = None
                        self.molIn.pushback(mol)
                        break

                    self._molBatch.append(mol)
                    nMols += 1
                    self.countMol += 1
                    if log.isEnabledFor(logging.WARN):
                        if self.countMol % 50 == 0:   print('.', file=sys.stderr, flush=True, end="")
                        if self.countMol % 2000 == 0: warn(f' sdfNNP: {self.countMol}')

            if nMols == 0:
                if log.isEnabledFor(logging.WARN):
                    warn(f' sdfNNP: {self.countMol}')
                raise StopIteration

            self.efIterator = self._nnpComputer.computeBatch(self._molBatch)
            self._currentMol = 0

        mol, e, std, grad = self.efIterator.__next__()

        mol[self.energy_tag] = f'{e:.1f}'

        if std:
            mol[self.std_tag]   = f'{std:.1f}'
            mol[self.std_per_atom_tag]   = f'{std/self.batch_natoms:.2f}'

        if self._nnpComputer.outputGrads:
            max_force = np.sqrt(np.power(grad,2).sum(1).max())
            mol[self.max_force_tag] = f"{max_force:.1f}"

            frc = np.array_str(-1 * grad,-1, 1, True)
            frc = re.sub("\\s+"," ",frc)
            frc = re.sub("([0-9.\\]]) ([-+0-9\\[])","\\1,\\2",frc)
            mol[self.force_tag] = frc

        self._currentMol += 1
        return mol

def main(argv=None):
    if argv is not None:
        sys.argv.extend(argv)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-in' ,  metavar='fileName',  dest='inFile', type=str, default=".sdf",
                        help='input file def=.sdf')

    parser.add_argument('-out' ,  metavar='fileName',  dest='outFile', type=str, default=".sdf",
                        help='input file def=.sdf')

    parser.add_argument('-conf' ,  metavar='NNP.json',  dest='confFile', type=str, required=True,
                        help='input file *.json of ANI directory')

    parser.add_argument('-computeForce', default=False, action='store_true',
                        help='Compute forces on atoms by using autograd on energy')

    parser.add_argument('-computeSTDev', default=False, action='store_true',
                        help='Compute standard deviation from ensamble models')

    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')

    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')

    parser.add_argument('-prefix',  metavar='str', type=str, default = "NNP",
                        help='Prefix for all sdf tag names')

    parser.add_argument('-gradAnomalyDetect', default=False, action='store_true',
                        help='For debugging force computation')

    args = parser.parse_args()

    log_helper.initialize_loggger(__name__, args.logFile)

    if args.gradAnomalyDetect:
        torch.autograd.set_detect_anomaly(True)

    if not args.inFile: args.inFile = ".sdf"
    if not args.outFile: args.outFile = ".sdf"

    nnpFactory = NNPComputerFactory(args.confFile)
    nnpCmptr = nnpFactory.createNNP(args.computeForce, args.computeSTDev, energyOutUnits=Units.KCAL, **vars(args))

    prefix = args.prefix
    if prefix and prefix[-1] not in ("_", ":"): prefix = prefix + "::"

    with get_mol_output_stream(args.outFile) as out,  \
         get_mol_input_stream(args.inFile)   as molS, \
         SDFComputer(molS, nnpCmptr, prefix) as cmptr:
        for mol in cmptr:
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
        import line_profiler    # pylint: disable=E0401
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
        import torch.autograd    # pylint: disable=C0412
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            main()
        # use chrome://tracing  to open file
        prof.export_chrome_trace("profile_Chrome.json")
        sys.exit(0)

    sys.exit(main())
