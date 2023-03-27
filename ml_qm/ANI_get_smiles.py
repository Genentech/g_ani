#!/bin/env python

'''
Created on Jun 25, 2018

@author: albertgo
'''
import glob
import sys
import argparse

import ANI.lib.pyanitools as pya # noqa: N813

from ase.units import Hartree, kcal
from ase.units import mol as mol_unit
from cdd_chem.util.io import warn

import re



class ANIGetInfo():
    """ info from ani file """

    def __init__(self, outFile):

        self.totalCnt = 0
        self.outFile = outFile
        print("Smiles\tcount\tdelatE", file=outFile)

    def parseFile(self, inFile):
        # Construct the data loader class
        warn("Processing: %s" % inFile)

        adl = pya.anidataloader(inFile)


        molNum = 0
        # Print the species of the data set one by one
        for data in adl:

            # Extract the data
            name = data['path']
            m = re.search('/([^/]+)', name)
            name = m.group(1)

            coords = data['coordinates'] # noqa: F841; # pylint: disable=W0612
            e = data['energies']
            smiles = "".join(data['smiles'])
#             atomTypes = list(AtomInfo.NameToNum[a] for a in data['species'])

            minE = min(e)
            print("%s\t%i\t%f" %
                 (smiles, len(e), (max(e)-minE) * Hartree*mol_unit/kcal),
                 file=self.outFile)

            #get first conformer for checking
#             mol = ANIMol(name, 0., atomTypes,coords[0])

            molNum += 1

#             for i, (e, xyz) in enumerate(zip(e, coords)):
#
#                 if self.totalCnt % 120 == 119:
#                     sys.stderr.write('.')
#                     sys.stderr.flush()
#                 if self.totalCnt % 5000 == 4999:
#                     warn(' %i read %i written' % (self.totalCnt+1, self.writeCnt))
#                 self.totalCnt += 1
#
#
#                 mol = PTMol(ANIMol(myname, e * Hartree*mol_unit/kcal, atomTypes,xyz))
#

        adl.cleanup()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-inPattern' ,  metavar='fileNamePattern',  type=str, required=True,
                        help='input file pattern eg. /tmp/ani_gdb_*.h5')
    parser.add_argument('-out' ,  metavar='out-file',  type=str,
                        help='out tab file')


    args = parser.parse_args()

    out = sys.stdout
    if args.out is not None:
        out = open(args.out,"wt")

    aGI = ANIGetInfo(out)
    for f in glob.glob(args.inPattern):
        aGI.parseFile(f)

    out.close()
