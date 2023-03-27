#!/bin/env python

'''
Created on Jun 25, 2018

@author: albertgo
'''
import sys
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import ANI.lib.pyanitools as pya

from ase.units import Hartree, kcal
from ase.units import mol as mol_unit
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611

import re
import h5py
import math
from t_opt.atom_info import NameToNum
from typing import Optional, Any



class ANIGetInfoAni1():
    """ get info from ani files """

    def __init__(self, in_file_name:str, outFile):

        self.totalCnt = 0
        self.inFile = in_file_name
        self.outFile = outFile
        print("Smiles\tcount\tdelatE", file=outFile)

    def parseFile(self):
        # Construct the data loader class
        warn("Processing: %s" % self.inFile)

        adl = pya.anidataloader(self.inFile)


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

class ANIGetInfo():
    """ get info from ani files """

    def __init__(self, in_file_name:str, outFile:Optional[str],
                 at_num_field:Optional[str], at_sym_field:Optional[str],
                 energy_field:str, coord_field:str, force_field:str):

        warn("Processing: %s" % in_file_name)
        self.inFile = h5py.File(in_file_name, "r")

        out = sys.stdout
        if outFile is not None:
            out = open(args.out,"wt")

        self.outFile = out
        self.at_nums = np.array([1,6,7,8,9,16,17])
        self.at_num_field = at_num_field
        self.at_sym_field = at_sym_field
        self.energy_field = energy_field
        self.coord_field  = coord_field
        self.force_field  = force_field

    def parseFile(self):
        # pylint: disable=R0915
        # Construct the data loader class

        cnt_conf = 0
        cnt_group = 0
        sumE = 0    # noqa: F841; # pylint: disable=W0612
        sumAt = {}  # noqa: F841; # pylint: disable=W0612
        countData  = np.zeros((1000000,len(self.at_nums)), dtype=int)
        energyData = np.zeros((1000000,), dtype=np.float)
        at_num_to_pos = np.full((self.at_nums.max()+1,), -1, dtype=int)
        for i,at_num in enumerate(self.at_nums):
            at_num_to_pos[at_num] = i

        for name,item in self.inFile.items():
            if self.at_num_field:
                at_nums = item[self.at_num_field][()] # np.array(uint8)
            else:
                at_nums = [ NameToNum[s.decode('ascii')] for s in item[self.at_sym_field][()] ] # np.array(uint8)
            conf_energies = item[self.energy_field][()] # np.array((nMol,nAt])
            at_coord = item[self.coord_field][()] # np.array((nMol,nAt,3]) # noqa: F841; # pylint: disable=W0612
            at_force = item[self.force_field][()] # np.array((nMol,nAt,3]) # noqa: F841; # pylint: disable=W0612
            nconf = conf_energies.shape[0]
            #warn(f"{cnt_group}: at_nums: {at_nums.shape[0]} num_conf: {conf_energies.shape[0]}, coord: {at_coord.shape} frc: {at_force.shape}")

            while len(energyData) < cnt_conf+nconf:
                energyData = np.append(energyData, np.zeros((1000000)), axis=0)
                countData  = np.append(countData,  np.zeros((1000000, len(self.at_nums)), dtype=int), axis=0)

            energyData[cnt_conf:cnt_conf+nconf] = conf_energies
            uat_nums, uat_counts = np.unique(at_nums, return_counts=True)
            countData[cnt_conf:cnt_conf+nconf,at_num_to_pos[uat_nums]] = uat_counts

            cnt_conf += nconf
            cnt_group += 1
            if cnt_conf // 100000 != (cnt_conf - nconf) // 100000:
                warn(f"conf {cnt_conf}")

        warn(f"read {cnt_conf} conformations in {cnt_group} groups")

        assert countData.shape[0] == energyData.shape[0]
        countData = countData[0:cnt_conf]
        energyData = energyData[0:cnt_conf]

        model = LinearRegression(fit_intercept=False)
        model.fit(countData, energyData)
        pred = model.predict(countData)
        rmse = math.sqrt(mean_squared_error(pred, energyData))
        deviation = (pred-energyData)

        warn("Atomic energies fitted:")
        warn(f"   r^2 = {model.score(countData, energyData)} rmse={rmse}")
        warn(f"   min deviation={deviation.min()} max deviation={deviation.max()}")
        warn("AtomNum\tAE[au]")
        for atnum,ae in zip(self.at_nums, model.coef_):
            warn(f"{atnum}\t{ae}")


        cnt_conf = 0

        warn("\nWriting energies")
        print("ID\tgID\tTITLE\tcount\tnatoms\tE\taE\tdelta\tdelta/n\tdelta/rootN", file=self.outFile)

        for key in self.inFile.keys():
            name = str(key)
            item = self.inFile[key]
            if self.at_num_field:
                at_nums = item[self.at_num_field][()] # np.array(uint8)
            else:
                at_nums = [ NameToNum[s.decode('ascii')] for s in item[self.at_sym_field][()] ] # np.array(uint8)
            conf_energies = item[self.energy_field][()] # np.array((nMol,nAt])
            nconf = conf_energies.shape[0]

            for i,e in enumerate(conf_energies):
                predAE = pred[cnt_conf]
                delta = e - predAE
                nAt = len(at_nums)
                print(f'{cnt_conf}\t{i}\t{name}\t{nconf}\t{nAt}\t{e}\t{predAE:.6f}\t{delta:.6f}\t{delta/nAt:.5f}\t{delta/math.sqrt(nAt):.5f}',
                      file= self.outFile)
                cnt_conf += 1

            if cnt_conf // 100000 != (cnt_conf - nconf) // 100000:
                warn(f"conf {cnt_conf}")



    def close(self):
        self.outFile.close()
        self.inFile.close()

class ANIGetInfo20191001(ANIGetInfo):
    """ 20191001 file format """
    def __init__(self,in_file_name:str, outFile:Optional[str] = None):
        super().__init__(in_file_name, outFile, "atomic_numbers", None, "wb97x_dz.energy", "coordinates", "wb97x_dz.forces")

class ANIGetInfo202001(ANIGetInfo):
    """ 202001 file format """
    def __init__(self,in_file_name:str, outFile:Optional[str] = None):
        super().__init__(in_file_name, outFile, None, "species", "energies", "coordinates", "forces")


if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-in' ,  dest='inf', metavar='fileName',  type=str, required=True,
                        help='input file *.h5')
    parser.add_argument('-out' ,  metavar='out-file',  type=str, required=False,
                        help='out tab file')
    parser.add_argument('-type' ,  type=str, required=False, default="202001",
                        help='ANI1|20191001|202001')


    args = parser.parse_args()

    aGI:Any

    if args.type == '202001':
        aGI = ANIGetInfo202001(args.inf, args.out)
    elif args.type == '20191001':
        aGI = ANIGetInfo20191001(args.inf, args.out)
    else:
        aGI = ANIGetInfoAni1(args.inf, args.out)

    aGI.parseFile()
    aGI.close()


# if __name__ == '__main__':
#
#
#     parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
#                                      description="")
#
#     parser.add_argument('-inPattern' ,  metavar='fileNamePattern',  type=str, required=True,
#                         help='input file pattern eg. /tmp/ani_gdb_*.h5')
#     parser.add_argument('-out' ,  metavar='out-file',  type=str,
#                         help='out tab file')
#
#
#     args = parser.parse_args()
#
#     out = sys.stdout
#     if args.out is not None:
#         out = open(args.out,"wt")
#
#     aGI = ANIGetInfo(out)
#     for f in glob.glob(args.inPattern):
#         aGI.parseFile(f)
#
#     out.close()
