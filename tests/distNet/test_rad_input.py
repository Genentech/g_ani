import torch
import numpy as np
import numpy.testing as npt
from ml_qm.distNet.dist_net import ComputeRadialInput
from ml_qm.distNet.data_loader import DataLoader
from tests.distNet.test_data_set import setup_2NH3_Examples

class TestToInput():
    def setup(self):
        duplicates = 2
        self.conf, self.device, data_set, _ = setup_2NH3_Examples(duplicates)
        self.conf['radialNet']['radialCutoff'] = 3.2  # just for testing

        confIds = torch.arange(2 * duplicates, device=self.device)
        self.dloader = DataLoader(data_set, confIds, 2, nCPU=1, set_type='val')
        self.module = ComputeRadialInput(self.conf)
        self.module = self.module.to(self.device)
        self.device_data_set = data_set.to(self.device)

    def test_Radial(self):

        self.dloader.setEpoch(0)
        for i,batch in enumerate(self.dloader):

            # prepare input
            device = self.device_data_set.ZERO.device
            batch_dist_map = batch.pop('batch_dist_map')
            atom_ij_idx = batch_dist_map['batch_atom_ij_idx'].to(device, non_blocking=True)
            dist_ij = batch_dist_map['batch_dist_ij'].to(device, dtype=torch.float32, non_blocking=True)
            batch['batch_dist_ij']     = dist_ij
            batch['batch_atom_ij_idx'] = atom_ij_idx

            batch = self.module(self.device_data_set, batch)
            batch_rad_input = batch['batch_rad_input']
            batch_rad_center= batch['batch_rad_center_atom_idx']


            if(i == 0 ):
                npt.assert_equal(batch_rad_center,
                    np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]) )
            npt.assert_array_almost_equal(batch_rad_input[0],
                    np.array([ 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0455, 0.0746, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.9091, -1.0746]),3)
            npt.assert_array_almost_equal(batch_rad_input[1],
                    np.array([ 2.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.045,  0.075,  0.5  ,  0.   ,  0.   ,  0.5  ,  0.   ,  0.   ,  0.   ,  0.   ,  -0.955,  -0.537]),3)
            npt.assert_array_almost_equal(batch_rad_input[6],
                    np.array([ 2.236,  1.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.   , -1.909, -1.075,  0.447,  0.   ,  0.   ,  0.447,  0.   ,  0.   ,  0.   ,  0.   , -0.854, -0.481]),3)
            npt.assert_array_almost_equal(batch_rad_input[19],
                    np.array([ 1.414,  1.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.   , -1.909, -1.075,  0.707,  0.   ,  0.   ,  0.707,  0.   , 0.   ,  0.   ,  0.   , -1.35 , -0.76]),3)
