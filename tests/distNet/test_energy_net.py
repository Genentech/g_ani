import torch
import numpy as np
import numpy.testing as npt
from ml_qm.distNet.dist_net import EnergyNet
from ml_qm.distNet.data_loader import DataLoader
from tests.distNet.test_data_set import setup_2NH3_Examples, setup_CH4_2NH3_Examples  # noqa: F401; # pylint: disable=W0611
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611

from cdd_chem.util import log_helper
log_helper.initialize_loggger(__name__, None)

class TestEnergyNet():
    def setup(self):
        self.conf, self.device, data_set, _ = setup_CH4_2NH3_Examples()
        self.conf['radialNet']['radialCutoff'] = 3.2  # just for testing
        confIds = torch.tensor([0,1,2])
        self.dloader = DataLoader(data_set, confIds, 4)
        self.module = EnergyNet(self.conf)
        self.module = self.module.to(self.device)
        self.module.train()
        self.device_data_set = data_set.to(self.device)


    def test_Energy(self):
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            inp = self.module(self.device_data_set, batch)
            conf_idx = inp['batch_output_conf_idx']
            e        = inp['batch_output']

            assert conf_idx.shape[0] == 3 and e.shape[0] == conf_idx.shape[0]
            assert conf_idx.unique().shape[0] == conf_idx.shape[0]
            npt.assert_equal(conf_idx, np.array([0, 1, 2]))

            e.backward(torch.ones((3,),device=self.device))
