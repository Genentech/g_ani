import torch
import numpy as np
import numpy.testing as npt
import yaml
import os
from ml_qm.distNet.data_set import DataSet
from typing import Dict, Optional

def setup_2NH3_Examples(rep:int = 1, yml_name:str = "scripts/testDistNet.yml",
                        ang_padding_map:Optional[Dict[int,int]] = None):
    ymlFile = os.path.dirname(__file__)
    ymlFile = f'{ymlFile}/../../{yml_name}'

    with open(ymlFile) as yFile:
        conf = yaml.safe_load(yFile)

    conf['angleNet']['angularCutoff'] = 3.2  # just for testing
    conf['radialNet']['radialCutoff'] = 3.1   # just for testing
    conf['trainData']['skipFactor'] = 1
    if ang_padding_map:
        conf['padding'] = conf.get('padding',{})
        conf['padding']['ang_neighbors'] = ang_padding_map

    nGPU=1
    device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")

    atom_types = [7,1,1,1]
    #         H2        H6
    #         |         |_ H7
    #         |         | /|
    #    H3---N-H1      N-H5
    conformers = np.array([[[0,0,0],[1.,0.,0],[0,2,0],[-3,0,0]],
                           [[0,0,0],[1.,0.,0],[0,2,0],[1,1,0]]])
    e = np.array([-35456.,-35456.2])

    n_confs = conformers.shape[0]
    conformers = np.stack([conformers for _ in range(rep)], axis=0)
    e = np.tile(e,rep)
    conformers = conformers.reshape(n_confs*rep,-1,3)

    data_set = DataSet(conf)
    data_set.add_conformers(atom_types, conformers, e)

    data_set.finalize()

    return conf, device, data_set, conformers


def setup_CH4_2NH3_Examples():
    ymlFile = os.path.dirname(__file__)
    ymlFile = f'{ymlFile}/../../scripts/testDistNet.yml'
    with open(ymlFile) as yFile:
        conf = yaml.safe_load(yFile)

    conf['angleNet']['angularCutoff'] = 3.2  # just for testing
    conf['radialNet']['radialCutoff'] = 3.1  # just for testing
    conf['trainData']['skipFactor'] = 1

    nGPU=1
    device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")

    data_set = DataSet(conf)

    atom_types = [6,1,1,1,1]
    #         H2
    #         | H4
    #         |/
    #    H3---C-H1
    conformers = np.array([[[0,0,0],[1.,0.,0],[0,2,0],[-3,0,0],[0,0,1]]])
    e = np.array([-25401])

    data_set.add_conformers(atom_types, conformers, e)

    atom_types = [7,1,1,1]
    #         H2        H6
    #         |         | H7
    #         |         |/
    #    H3---N-H1      N-H5
    conformers = np.array([[[0,0,0],[1.,0.,0],[0,2,0],[-3,0,0]],
                           [[0,0,0],[1.,0.,0],[0,2,0],[1,1,0]]])
    e = np.array([-35456.,-35456.2])

    data_set.add_conformers(atom_types, conformers, e)

    data_set.finalize()

    return conf, device, data_set, conformers


def setup_C3H12_N4H12_Examples(rep:int = 1, yml_name:str = "scripts/testDistNet.yml"):
    ymlFile = os.path.dirname(__file__)
    ymlFile = f'{ymlFile}/../../{yml_name}'
    with open(ymlFile) as yFile:
        conf = yaml.safe_load(yFile)

    conf['angleNet']['angularCutoff'] = 3.2  # just for testing
    conf['trainData']['skipFactor'] = 1

    nGPU=1
    device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")

    data_set = DataSet(conf)

    atom_types = [6,1,1,1,1,6,1,1,1,1,6,1,1,1,1]
    #         H2
    #         | H4
    #         |/
    #    H3---C-H1
    # plus copies shifted to +3x, +6x
    conformers = np.array([[[0,0,0],[1.,0.,0],[0,2,0],[-3,0,0],[0,0,1],
                            [3,0,0],[4.,0.,0],[3,2,0],[ 0,0,0],[3,0,1],
                            [6,0,0],[7.,0.,0],[6,2,0],[ 3,0,0],[6,0,1]]])
    e = np.array([-76204])

    n_confs = conformers.shape[0]
    conformers = np.stack([conformers for _ in range(rep)], axis=0)
    e = np.tile(e,rep)
    conformers = conformers.reshape(n_confs*rep,-1,3)

    data_set.add_conformers(atom_types, conformers, e)

    atom_types = [7,1,1,1,7,1,1,1,7,1,1,1,7,1,1,1]
    #         H2        H6
    #         |         | H7
    #         |         |/
    #    H3---N-H1      N-H5
    # plus copies shifted to +3x, +6x, +9x
    conformers = np.array([[[0,0,0],[1.,0.,0],[0,2,0],[-3,0,0],
                            [3,0,0],[4.,0.,0],[3,2,0],[ 0,0,0],
                            [6,0,0],[7.,0.,0],[6,2,0],[ 3,0,0],
                            [9,0,0],[10,0.,0],[9,2,0],[ 6,0,0]],
                           [[0,0,0],[1.,0.,0],[0,2,0],[ 1,1,0],
                            [3,0,0],[4.,0.,0],[3,2,0],[ 4,1,0],
                            [6,0,0],[7.,0.,0],[6,2,0],[ 7,1,0],
                            [9,0,0],[10,0.,0],[9,2,0],[10,1,0]
                            ]])
    e = np.array([-141824.,-141825])

    n_confs = conformers.shape[0]
    conformers = np.stack([conformers for _ in range(rep)], axis=0)
    e = np.tile(e,rep)
    conformers = conformers.reshape(n_confs*rep,-1,3)

    data_set.add_conformers(atom_types, conformers, e)

    data_set.finalize()

    return conf, device, data_set, conformers


class TestDataSet():
    def setup(self):
        self.conf, self.device, self.data_set, self.conformers \
            = setup_2NH3_Examples()

    def test_init(self):
        ds = self.data_set
        assert ds.n_confs == 2
        assert ds.n_atoms == 8
        conf_exp = np.array([[0.2425, 4.0000e+00],[0.0425, 4.0000e+00]])
        npt.assert_allclose(ds.conformations.buffer,conf_exp,rtol=0.001)

        # radial
        assert len(ds.rad_dist_map) == 2
        aij = ds.rad_dist_map['atom_ij_idx'].buffer
        dij = ds.rad_dist_map['dist_ij'].buffer
        assert aij.shape[0] == dij.shape[0]
        npt.assert_equal(aij.t(), np.array(
            [[0,     0,      0,      1,      1,      2,      2,      3,      4,      4,      4, 5, 5, 5, 6, 6, 6, 7, 7, 7],
             [1,     2,      3,      0,      2,      0,      1,      0,      5,      6,      7, 4, 6, 7, 4, 5, 7, 4, 5, 6]]))
        npt.assert_array_almost_equal(dij,np.array(
            [1.0000, 2.0000, 3.0000, 1.0000, 2.2363, 2.0000, 2.2363, 3.0000, 1.0000, 2.0000,
             1.4141, 1.0000, 2.2363, 1.0000, 2.0000, 2.2363, 1.4141, 1.4141, 1.0000, 1.4141]),4)

        # angular
        at_idx_exp = np.array([[0, 7],[0, 1],[0, 1],[0, 1],
                               [1, 7],[1, 1],[1, 1],[1, 1]])

        npt.assert_equal(ds.atoms_long.buffer, at_idx_exp)
        npt.assert_array_almost_equal(ds.atoms_xyz.buffer,self.conformers.reshape(-1,3),4)
        assert len(ds.ang_neighbor_map) == 2

        # 2 neighbors, 2 center atoms (1,2) with each 2 neighbors (0,2) (0,1)
        idx_exp  = np.array([[[1, 0],[1, 2]],
                             [[2, 0],[2, 1]]])
        dist_exp = np.array([[1.0000, 2.2363],
                             [2.0000, 2.2363]])
        i_buffer, j_buffer, d_buffer = ds.ang_neighbor_map[2]
        ij = torch.broadcast_tensors(i_buffer.buffer,j_buffer.buffer)

        npt.assert_equal(torch.stack(ij,dim=2), idx_exp)
        npt.assert_array_almost_equal(d_buffer.buffer,dist_exp,4)

        # 3 neighbors
        idx_exp  = np.array([[[0, 1],[0, 2],[0, 3]],
                             [[4, 5],[4, 6],[4, 7]],
                             [[5, 4],[5, 6],[5, 7]],
                             [[6, 4],[6, 5],[6, 7]],
                             [[7, 4],[7, 5],[7, 6]]])

        dist_exp = np.array([[1.0000, 2.0000, 3.0000],
                             [1.0000, 2.0000, 1.4141],
                             [1.0000, 2.2363, 1.0000],
                             [2.0000, 2.2363, 1.4141],
                             [1.4141, 1.0000, 1.4141]])

        i_buffer, j_buffer, d_buffer = ds.ang_neighbor_map[3]
        ij = torch.broadcast_tensors(i_buffer.buffer,j_buffer.buffer)

        npt.assert_equal(torch.stack(ij,dim=2), idx_exp)
        npt.assert_array_almost_equal(d_buffer.buffer,dist_exp,4)
