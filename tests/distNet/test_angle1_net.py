
import torch
import numpy as np
import numpy.testing as npt
from ml_qm.distNet.dist_net import AngleNet1
from ml_qm.distNet.data_loader import DataLoader
from tests.distNet.test_data_set import setup_2NH3_Examples,\
    setup_CH4_2NH3_Examples
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611

from cdd_chem.util import log_helper
log_helper.initialize_loggger(__name__, None)

class TestAngleNet1():
    def setup(self):
        self.conf, device, self.data_set, _ = setup_2NH3_Examples()
        confIds = torch.tensor([0,1])
        self.dloader = DataLoader(self.data_set, confIds, 2)
        self.data_set = self.data_set.to(device)
        self.module = AngleNet1(self.conf)


    def test_Angle1_Net(self):
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            inp = self.module(self.data_set, batch)
            i_idx = inp['batch_ang_center_atom_idx']
            desc  = inp['batch_ang_desc']

            assert i_idx.shape[0] == 7 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([0, 4, 5, 6, 7, 1, 2]))


    def test_Angle1_Net_Padding(self):
        conf_pad, device, data_set_pad, _ = setup_2NH3_Examples(ang_padding_map={2:3})
        confIds = torch.tensor([0,1])
        dloader_pad = DataLoader(data_set_pad, confIds, 2)
        data_set_pad = data_set_pad.to(device)

        self.dloader.setEpoch(0)
        dloader_pad.setEpoch(0)
        for batch, batch_pad in zip(self.dloader, dloader_pad):
            assert len(batch['batch_ang_neighbor_map'].keys()) > len(batch_pad['batch_ang_neighbor_map'].keys())

            inp = self.module(self.data_set, batch)
            inp_pad = self.module(data_set_pad, batch_pad)
            i_idx     = inp['batch_ang_center_atom_idx']
            i_idx_pad = inp_pad['batch_ang_center_atom_idx']
            npt.assert_equal(i_idx.cpu().numpy(), i_idx_pad.cpu().numpy())

            desc      = inp['batch_ang_desc']
            desc_pad  = inp_pad['batch_ang_desc']
            npt.assert_array_almost_equal(desc.detach_().cpu(), desc_pad.detach_().cpu(),3)


class TestAngleNet12():
    def setup(self):
        self.conf, self.device, self.data_set, _ = setup_CH4_2NH3_Examples()
        self.module = AngleNet1(self.conf)
        self.module = self.module.to(self.device)
        self.device_data_set = self.data_set.to(self.device)


    def test_Angle1_Net110(self):
        confIds = torch.tensor([0,1])
        self.dloader = DataLoader(self.data_set, confIds, 2)
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_ang_center_atom_idx']
            desc  = inp['batch_ang_desc']

            assert i_idx.shape[0] == 8 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([0, 4, 1, 2, 5, 3, 6, 7]))

    def test_Angle1_Net100(self):
        confIds = torch.tensor([0])
        self.dloader = DataLoader(self.data_set, confIds, 2)
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_ang_center_atom_idx']
            desc  = inp['batch_ang_desc']

            assert i_idx.shape[0] == 5 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([0, 4, 1, 2, 3]))

    def test_Angle1_Net111(self):
        confIds = torch.tensor([0,1,2])
        self.dloader = DataLoader(self.data_set, confIds, 3)
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_ang_center_atom_idx']
            desc  = inp['batch_ang_desc']

            assert i_idx.shape[0] == 12 and desc.shape[0] == i_idx.shape[0]
            npt.assert_equal(i_idx, np.array([0,  4,  1,  2,  5,  9, 10, 11, 12,  3,  6,  7]))
