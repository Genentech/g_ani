import torch
import numpy as np
import numpy.testing as npt
from ml_qm.distNet.dist_net import ComputeDescriptors
from ml_qm.distNet.data_loader import DataLoader
from tests.distNet.test_data_set import setup_2NH3_Examples, setup_CH4_2NH3_Examples # noqa: F401

from cdd_chem.util import log_helper
log_helper.initialize_loggger(__name__, None)

class TestComputeDescriptors():
    def setup_method(self):
        self.conf, self.device, data_set, _ = setup_2NH3_Examples()
        self.conf['radialNet']['radialCutoff'] = 3.2  # just for testing
        confIds = torch.tensor([0,1])
        self.dloader = DataLoader(data_set, confIds, 2)
        self.module = ComputeDescriptors(self.conf)
        self.module = self.module.to(self.device)
        self.device_data_set = data_set.to(self.device)


    def test_compute_descriptors(self):
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            inp = self.module(self.device_data_set, batch)
            i_idx = inp['batch_center_atom_idx']
            desc  = inp['batch_desc']

            assert i_idx.shape[0] == 8 and desc.shape[0] == i_idx.shape[0]
            assert i_idx.unique().shape[0] == i_idx.shape[0]
            assert desc.shape[1] == self.module.n_output
            npt.assert_equal(i_idx, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
