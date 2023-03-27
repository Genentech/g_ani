#!/usr/bin/env python
# encoding: utf-8
'''
ml_qm.optimize.optimizer -- Reads sdf files and optimizes them using NNP.

@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

import sys

import argparse
from argparse import RawDescriptionHelpFormatter
import logging

from cdd_chem.rdkit.io import MolInputStream, MolOutputStream
from cdd_chem.util import log_helper
from t_opt.unit_conversion import Units
from ml_qm.optimize.geometric.geomeTRIC_optimizer import GeomeTRICOptimizer
from ml_qm.optimize.NNP_computer_factory import NNPComputerFactory

log = logging.getLogger(__name__)

TESTRUN = 0
PROFILE = 0


def main(argv=None): # IGNORE:C0111

    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Optimize all molecules in sdf", formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-in",  dest="inFile", type=str, default=".sdf",
                        help="input file (def=.sdf)")
    parser.add_argument("-out", dest="outFile",  type=str, default=".sdf",
                        help="output file (def=.sdf)")
    parser.add_argument('-conf' ,  metavar='fileName', type=str, required=True,
                        help='nnp configuration file *.json or ANI directory name')
    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')
    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')
    parser.add_argument('-harm_constr' ,  metavar='k[kcal/mol/a]' ,  type=float, default=0,
                        help='Force Constant of harmonic constraint that pulls input molecules back to their input coordinates')
    parser.add_argument('-constraint' ,  metavar='type|file' ,  type=str,
                        help='heavyAtoms|geomeTRIC constraint-file')

    parser.add_argument('--epsilon', type=float, default=1e-5, help='Small eigenvalue threshold.')
    parser.add_argument('--check',   type=int, default=0, help='Check coordinates every N steps to see whether it has changed.')
    parser.add_argument('--verbose', action='store_true', help='Write out the displacements.')
    parser.add_argument('--reset',   action='store_true', help='Reset Hessian when eigenvalues are under epsilon.')
    parser.add_argument('--rfo',     action='store_true', help='Use rational function optimization (default is trust-radius Newton Raphson).')
    parser.add_argument('--trust',   type=float, default=0.1, help='Starting trust radius.')
    parser.add_argument('--tmax',    type=float, default=0.3, help='Maximum trust radius.')
    parser.add_argument('--maxiter', type=int, default=300, help='Maximum number of optimization steps.')
    parser.add_argument('--radii',   type=str, nargs="+", default=["Na","0.0"], help='List of atomic radii for coordinate system.')
    parser.add_argument('--coordsys',type=str, choices=['cart','prim','dlc','hdlc','tric'], default='tric', help='Coordinate system to use for optimization')


    # Process arguments
    args = parser.parse_args()

    inFile  = args.inFile
    outFile = args.outFile
    constraint = args.constraint
    harm_constr= args.harm_constr
    del args.constraint, args.harm_constr

    log_helper.initialize_loggger(__name__, args.logFile)

    nnpFactory = NNPComputerFactory(args.conf)
    nnpCmptr = nnpFactory.createNNP(True, False, energyOutUnits=Units.AU, **vars(args))

    with MolOutputStream(outFile) as out, \
         MolInputStream(inFile) as molS,  \
         GeomeTRICOptimizer(molS, nnpCmptr, constraint, harm_constr, **vars(args)) as sdfOptizer:
        for mol in sdfOptizer:
            out.write_mol(mol)

    return 0


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
        import line_profiler                   # pylint: disable=E0401
        import atexit                          # pylint: disable=E0401
        import geometric.optimize as optmz     # pylint: disable=E0401
        import geometric.internal as internal  # pylint: disable=E0401

        prof = line_profiler.LineProfiler()
        atexit.register(prof.print_stats)

        prof.add_function(optmz.Optimizer.step)
        prof.add_function(optmz.brent_wiki)
        prof.add_function(optmz.Froot.evaluate)
        prof.add_function(optmz.trust_step)
        prof.add_function(optmz.getCartesianNorm)
        prof.add_function(optmz.get_delta_prime)
        prof.add_function(internal.InternalCoordinates.newCartesian)

        prof_wrapper = prof(main)
        prof_wrapper()
        sys.exit(0)

    sys.exit(main())
