import torch
import numpy as np
import numpy.testing as npt
from ml_qm.distNet.dist_net import RadialNet
from ml_qm.distNet.data_loader import DataLoader
from tests.distNet.test_data_set import setup_2NH3_Examples,\
    setup_CH4_2NH3_Examples
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611

from cdd_chem.util import log_helper
log_helper.initialize_loggger(__name__, None)

class TestRadialNet():
    def setup_method(self):
        self.conf, self.device, data_set, _ = setup_2NH3_Examples()
        self.conf['radialNet']['radialCutoff'] = 3.2  # just for testing
        confIds = torch.tensor([0,1])
        self.dloader = DataLoader(data_set, confIds, 2)
        self.module = RadialNet(self.conf)
        self.module = self.module.to(self.device)
        self.device_data_set = data_set.to(self.device)


    def test_Radial_Net(self):
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_rad_center_atom_idx']
            desc  = inp['batch_rad_desc']

            assert i_idx.shape[0] == 8 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([0, 1, 2, 3, 4, 5, 6, 7]))


class TestRadialNet2():
    def setup_method(self):
        self.conf, self.device, self.data_set, _ = setup_CH4_2NH3_Examples()
        self.module = RadialNet(self.conf)
        self.module = self.module.to(self.device)
        self.device_data_set = self.data_set.to(self.device)

    def test_Radial_Net100(self):
        confIds = torch.tensor([0], device=self.device)
        dloader = DataLoader(self.data_set, confIds, 2)
        dloader.setEpoch(0)
        for batch in dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_rad_center_atom_idx']
            desc  = inp['batch_rad_desc']

            assert i_idx.shape[0] == 5 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([0, 1, 2, 3, 4]))

    def test_Radial_Net010(self):
        confIds = torch.tensor([1],device=self.device)
        dloader = DataLoader(self.data_set, confIds, 2)
        dloader.setEpoch(0)
        for batch in dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_rad_center_atom_idx']
            desc  = inp['batch_rad_desc']

            assert i_idx.shape[0] == 4 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([5, 6, 7, 8]))

    def test_Radial_Net110(self):
        confIds = torch.tensor([0,1],device=self.device)
        dloader = DataLoader(self.data_set, confIds, 2)
        dloader.setEpoch(0)
        for batch in dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_rad_center_atom_idx']
            desc  = inp['batch_rad_desc']

            assert i_idx.shape[0] == 9 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))

    def test_Radial_Net111(self):
        confIds = torch.tensor([0,1,2],device=self.device)
        dloader = DataLoader(self.data_set, confIds, 5)
        dloader.setEpoch(0)
        for batch in dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_rad_center_atom_idx']
            desc  = inp['batch_rad_desc']

            assert i_idx.shape[0] == 13 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]))

    def test_Radial_Net001(self):
        confIds = torch.tensor([2],device=self.device)
        dloader = DataLoader(self.data_set, confIds, 2)
        dloader.setEpoch(0)
        for batch in dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_rad_center_atom_idx']
            desc  = inp['batch_rad_desc']

            assert i_idx.shape[0] == 4 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([9,  10, 11, 12]))
