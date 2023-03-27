#!/usr/bin/env python
# pylint: skip-file
# this code is not supported at teh moment it will not run with newer versions of geometric

'''
ml_qm.optimize.optimizer -- Reads sdf files and optimizes them using NNP.

@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

import geometric.optimize as gto
from geometric.internal import CartesianCoordinates,\
    PrimitiveInternalCoordinates, DelocalizedInternalCoordinates
from geometric.engine import Engine as GTEngine
from geometric.molecule import Molecule as GTMolecule
import tempfile
import numpy as np
from numpy.linalg.linalg import LinAlgError

import logging
from t_opt.opt_util import OPT_ENERGY_TAG, OPT_STATUS_TAG, OPT_STEPS
import math
from t_opt.unit_conversion import A_to_AU, KCAL_MOL_A2_to_AU, bohr2ang,\
    KCAL_MOL_to_AU
log = logging.getLogger(__name__)




class NNPEngine(GTEngine):
    """
       Wrapper around geometric.engine to hold energy and grad computed by the NNP.
    """
    def __init__(self, molecule, molIdx, harm_constraint_AU=0.):
        """
        Arguments
        ---------
        molecule: Molecule to be optimized
        molIdx: counter to id molecule
        harm_contraint: harmonic constraint pulling back to initial coordinates in [AU]
        """

        super().__init__(molecule)
        self.molIdx = molIdx
        self.energy_AU = None
        self.grad_AU   = None
        self.harm_constraint_AU = harm_constraint_AU
        self.in_coords = molecule.xyzs[0] * A_to_AU
        self.harm_constraint_AU = harm_constraint_AU
        self.lastComputedEnergyNC_AU = None # excludes harmonic constrain potential

    def calc(self, coords_AU, dirname):
        """
            This simply returns the energy and forces stored in this container by
            the NNPComputer.


        """

        # copied from engine.Psi4.calc_new. why do we need to do this?
        # I think the xyzs[0] is just the input conf???
        self.M.xyzs[0] = coords_AU.reshape(-1, 3) / A_to_AU

        #log.debug("[AU]: e=%.5f bl=%.5f,%.5f g=%.4f; e[kcal]=%.1f" % (
        #            self.energy_AU, coords_AU[0],coords_AU[3], self.grad_AU[0,0],
        #            self.energy_AU / KCAL_MOL_to_AU))

        self.lastComputedEnergyNC_AU = self.energy_AU

        if self.harm_constraint_AU == 0.:
            return self.energy_AU, self.grad_AU.flatten()

        # add harmonic constraint
        # E = k ( delta(atom1)^2 + delta(atom2)^2 + ... )
        # F = 2 * k * deta(a_i)
        delta = self.in_coords - coords_AU.reshape(-1, 3)
        delta2 = np.power(delta,2)

        harmE = self.harm_constraint_AU * delta2.sum()
        harmGrad = -2. * self.harm_constraint_AU * delta

        return self.energy_AU + harmE, (self.grad_AU+harmGrad).flatten()


class GeomeTRICOptimizer():
    """ Iterator that takes a iterator of Mol's and returns an iterator of Mol's
        The output Mol will have the updated geometry the following output fields:
            - NNP_Energy_kcal_mol
    """
    def __init__(self, molInStream, nnpComputer, constraint=None, harm_constr=0., **kwargs):
        self.molIn = molInStream

        self.constraint = constraint
        self.harm_constraint_AU = harm_constr * KCAL_MOL_A2_to_AU

        self._nnpComputer = nnpComputer
        self._currentMol = 0
        self._molBatch   = []
        self.tmpDir = tempfile.mkdtemp(".tmp", "sdfOpt.")

        optParams = gto.OptParams(**kwargs)
#         optParams.Convergence_energy = kwargs.get('convergence_energy', 8e-5)
#         optParams.Convergence_drms = kwargs.get('convergence_drms', 5.e-3)
#         optParams.Convergence_dmax = kwargs.get('convergence_dmax', 1.2e-2)
#         optParams.Convergence_grms = kwargs.get('convergence_grms', 2.8e-3)
#         optParams.Convergence_gmax = kwargs.get('convergence_gmax', 1.2e-2)
#         optParams.trust = kwargs.get('trust', 0.4)
#         optParams.tmax = kwargs.get('tmax', 0.6)

        self.trust = optParams.trust
        self._optParams = optParams

        #=========================================#
        #| Set up the internal coordinate system |#
        #=========================================#
        # First item in tuple: The class to be initialized
        # Second item in tuple: Whether to connect nonbonded fragments
        # Third item in tuple: Whether to throw in all Cartesians (no effect if second item is True)
        CoordSysDict = {'cart':(CartesianCoordinates, False, False),
                        'prim':(PrimitiveInternalCoordinates, True, False),
                        'dlc':(DelocalizedInternalCoordinates, True, False),
                        'hdlc':(DelocalizedInternalCoordinates, False, True),
                        'tric':(DelocalizedInternalCoordinates, False, False)}

        coordsys = kwargs.get('coordsys', 'tric').lower()
        self._CoordClass, self._connect, self._addcart = CoordSysDict[coordsys]



    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __exit__(self, *args):
        pass


    def __next__(self):
        if self._currentMol >= len(self._molBatch):
            self._molBatch = []
            nMols = 0
            batch_natoms = 0
            batch_size = 1    # initial value will be adjusted in first record

            while nMols < batch_size and self.molIn.has_next():
                mol = self.molIn.__next__()

                nAt = mol.num_atoms
                if batch_natoms == 0:
                    batch_natoms = nAt

                    # assume memory requirement goes with nAtom^3
                    batch_size = int(self._nnpComputer.mol_mem_GB * math.pow(1024/batch_natoms,3)
                                     / self._nnpComputer.BYTES_PER_MOL)
                    log.debug("nAt: {batch_natoms}, batchSize: {batch_size}")

                self._molBatch.append(mol)
                nMols += 1

            if nMols == 0: raise StopIteration

            self.optimizeMols(self._molBatch)
            self._currentMol = 0

        res= self._molBatch[self._currentMol]
        self._currentMol += 1
        return res



    def prepareOptimizers(self, mol_batch):
        """ # initialize objects for Geometric!!!!!

        Parameter:
        ----------
        mol_batch: list of Mol's

        """

        countMol = 0
        gt_optmzrs = []

        constraintStr = None
        if self.constraint not in [None, "heavyAtoms"]:
            # Read in the constraints
            constraintStr = open(self.constraint).read()

        efIterator = self._nnpComputer.computeBatch(mol_batch)
        for molIdx, (mol, e, std, grad) in enumerate(efIterator): # pylint: disable=W0612
            gtMol = GTMolecule()
            gtMol.elem = mol.atom_symbols
            gtMol.xyzs = [mol.coordinates] # both are in A
            gtEngine = NNPEngine(gtMol, molIdx, self.harm_constraint_AU)
            gtEngine.energy_AU = e
            gtEngine.grad_AU = grad
            coords = gtMol.xyzs[0].flatten() * A_to_AU

            if self.constraint == "heavyAtoms":
                constraintStr = "$freeze\nxyz "
                for i, ele in enumerate(gtMol.elem):
                    if ele != "H" : constraintStr += "{:1d},".format(i+1)
                if constraintStr[-1] != ",":
                    constraintStr = None
                else:
                    constraintStr = constraintStr[0:-1]

            if constraintStr is not None:
                Cons, CVals = gto.ParseConstraints(gtMol, constraintStr)
            else:
                Cons = None
                CVals = None

            # gtMol needs to be in Angstrom
            gtIC = self._CoordClass(gtMol, build=True, connect=self._connect, addcart=self._addcart,
                                    constraints=Cons, cvals=CVals[0] if CVals is not None else None)
            #tmpDir = tempfile.mkdtemp(".tmp", str(countMol) + ".", self.tmpDir)
            tmpDir = self.tmpDir

            optmzr = gto.Optimizer(coords, gtMol, gtIC, gtEngine, tmpDir, self._optParams)
            optmzr.calcEnergyForce()  # pull from NNPEngine
            optmzr.prepareFirstStep()
            gt_optmzrs.append(optmzr)

            countMol += 1

        return gt_optmzrs


    def optimizeMols(self, mol_batch):
        """
            Optimize a batch of Mol instances using geomeTRIC.

            Parameter
            ---------
               mol_batch (potentially large) batch of cdd_chem.rdkit.Mol objects 
        """
        gt_optmzrs = self.prepareOptimizers(mol_batch)

        # Optimization Loop, while not all have completed optimization
        while len(gt_optmzrs) > 0:
            next_optmzr = []
            nextMolsToBeOptimized = []

            # take one step, energy and gradient must have been stored in optObj
            for mol, optmzr in zip(mol_batch, gt_optmzrs):
                try:
                    optmzr.step()
                    coordsA = optmzr.X.reshape(-1,3) * bohr2ang
                    mol.coordinates = coordsA
                except LinAlgError as ex:
                    optmzr.state = gto.OPT_STATE.FAILED
                    log.critical("Optimization Failed %s" % ex)

            # compute energies and forces for new coords of whole batch
            efIterator = self._nnpComputer.computeBatch(mol_batch)

            # transfer e and grad to optOpj and evaluate last step
            for optmzr, (mol,e, std, grad) in zip(gt_optmzrs, efIterator):

                if optmzr.state is gto.OPT_STATE.NEEDS_EVALUATION:
                    # update energy and forces
                    optmzr.engine.energy_AU = e
                    optmzr.engine.grad_AU = grad
                    optmzr.calcEnergyForce()
#                     log.debug("e=%f bl=%f,%f g=%f" % (
#                         optObj.engine.energy_AU,
#                         optObj.X.reshape(-1,3)[0,0],optObj.X.reshape(-1,3)[1,0],
#                         optObj.engine.grad_AU[0,0]))

                    # evaluate step
                    optmzr.evaluateStep()

                if optmzr.state in [gto.OPT_STATE.CONVERGED, gto.OPT_STATE.FAILED]:
                    enrgy = optmzr.engine.lastComputedEnergyNC_AU / KCAL_MOL_to_AU
                    mol[OPT_ENERGY_TAG] = f'{enrgy:.1f}'
                    mol[OPT_STATUS_TAG] = str(optmzr.state)
                    if log.getEffectiveLevel() <= logging.INFO:
                        mol[OPT_STEPS] = optmzr.Iteration
                    continue

                # add to list for next cycle
                nextMolsToBeOptimized.append(mol)
                next_optmzr.append(optmzr)

            if len(next_optmzr) == 0: break  ######## All Done

            # step and evaluation completed, next step for remaining conformations
            mol_batch = nextMolsToBeOptimized
            gt_optmzrs = next_optmzr
