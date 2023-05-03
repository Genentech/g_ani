import torch
import numpy.testing as npt
from ml_qm.distNet.dist_net import Compute1AngleInput
from ml_qm.distNet.data_loader import DataLoader
from tests.distNet.test_data_set import setup_2NH3_Examples

class TestToInput():
    def setup_method(self):
        self.conf, self.device, data_set, _ = setup_2NH3_Examples()
        confIds = torch.tensor([0,1])
        self.dloader = DataLoader(data_set, confIds, 2)
        self.data_set = data_set.to(self.device)
        self.module = Compute1AngleInput(data_set.atom_types.n_feature)
        self.module = self.module.to(self.device)


    def test_triples(self):
        exp_dist = { 2:
                ( #triple_dist_ij, triple_dist_ik, triple_dist_jk
                  # i_idx.flat = [1, 2]   j_idx = [[0, 2], [0, 1]]
                  # 10    10      12    12        20    20      21    21        ij is repeated
                 [[[1.00, 1.00], [2.24, 2.24]], [[2.00, 2.00], [2.24, 2.24]]],
                  # 10    12      10    12        20    21      20    21        ij1,ij2 is repeated
                 [[[1.00, 2.24], [1.00, 2.24]], [[2.00, 2.24], [2.00, 2.24]]],
                  # 100   102     120   122      200    201     210   211
                    [[[0.00, 0.38], [0.38, 0.00]], [[0.00, 0.19], [0.19, 0.00]]]),
            3: \
                (  # i_idx.flat [0, 4, 5, 6, 7],
                   # j_idx [[1, 2, 3], [5, 6, 7], [4, 6, 7], [4, 5, 7], [4, 5, 6]]
                   #triple_dist_ij
                     # 01    01    01     02                   03
                   [[[1.00, 1.00, 1.00], [2.00, 2.00, 2.00], [3.00, 3.00, 3.00]],
                     # 45    45    45     46                  47
                    [[1.00, 1.00, 1.00], [2.00, 2.00, 2.00], [1.41, 1.41, 1.41]],
                     # 54                 56                  57
                    [[1.00, 1.00, 1.00], [2.23, 2.23, 2.23], [1.00, 1.00, 1.00]],
                    [[2.00, 2.00, 2.00], [2.23, 2.23, 2.23], [1.41, 1.41, 1.41]],
                    [[1.41, 1.41, 1.41], [1.00, 1.00, 1.00], [1.41, 1.41, 1.41]]],
                   #triple_dist_ik
                     # 01   02    03       01   02    03       01   02    03
                   [[[1.00, 2.00, 3.00], [1.00, 2.00, 3.00], [1.00, 2.00, 3.00]],
                    #  45   46    47       45   46    47       45   46    47
                    [[1.00, 2.00, 1.41], [1.00, 2.00, 1.41], [1.00, 2.00, 1.41]],
                    [[1.00, 2.24, 1.00], [1.00, 2.24, 1.00], [1.00, 2.24, 1.00]],
                    [[2.00, 2.24, 1.41], [2.00, 2.24, 1.41], [2.00, 2.24, 1.41]],
                    [[1.41, 1.00, 1.41], [1.41, 1.00, 1.41], [1.41, 1.00, 1.41]]],
                   # triple_dist_jk
                     # 011  012   013     021   022   023     031   032   033
                   [[[0.00, 0.62, 1.00], [0.62, 0.00, 0.65], [1.00, 0.65, 0.00]],
                     # 455  456   457     465   466   467     475   476   477
                    [[0.00, 0.62, 0.29], [0.62, 0.00, 0.29], [0.29, 0.29, 0.00]],
                    [[0.00, 0.38, 0.71], [0.38, 0.00, 0.09], [0.71, 0.09, 0.00]],
                    [[0.00, 0.19, 0.29], [0.19, 0.00, 0.06], [0.29, 0.06, 0.00]],
                    [[0.00, 0.29, 0.71], [0.29, 0.00, 0.91], [0.71, 0.91, 0.00]]])
        }

        ds = self.data_set
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            batch_ang_neighbor_map = batch['batch_ang_neighbor_map']
            for nNeigh, (i_idx, j_idx, dist_ij) in batch_ang_neighbor_map.items():
                if nNeigh == 1: continue

                triple_dist_ij, triple_dist_ik, triple_dist_jk \
                   = self.module._computeTriples(nNeigh, i_idx, j_idx, dist_ij, ds.atoms_xyz)
                exp_ij, exp_ik, exp_jk = exp_dist[nNeigh]
                npt.assert_array_almost_equal(triple_dist_ij,exp_ij,2)
                npt.assert_array_almost_equal(triple_dist_ik,exp_ik,2)
                npt.assert_array_almost_equal(triple_dist_jk,exp_jk,2)


    def test_Ang(self):
        ds = self.data_set
        at_types = ds.atom_types
        n_feature = at_types.n_feature

        # positions of embeddings of H and N in feature vector
        # start at 3 because first 3 elements are dist_ijk
        grp1Pos = [ 3 + at_types.at_prop_names_2_col['g1'] + i * n_feature for i in range(0,3)]
        grp5Pos = [ 3 + at_types.at_prop_names_2_col['g5'] + i * n_feature for i in range(0,3)]
        row1Pos = [ 3 + at_types.at_prop_names_2_col['r1'] + i * n_feature for i in range(0,3)]
        row2Pos = [ 3 + at_types.at_prop_names_2_col['r2'] + i * n_feature for i in range(0,3)]

        self.dloader.setEpoch(0)
        for batch in self.dloader:
            batch_ang_neighbor_map = batch['batch_ang_neighbor_map']
            for nNeigh, (i_idx, j_idx, dist_ij) in batch_ang_neighbor_map.items():
                if nNeigh == 1: continue

                center_atom_idx, desc = self.module(nNeigh, i_idx, j_idx, dist_ij,
                                             ds.atoms_xyz, ds.atoms_long, ds.atom_types.atom_embedding)

                if nNeigh == 3:
                    npt.assert_array_almost_equal(desc[4,5],
                        torch.tensor([ 1.414, 1. , 0.911, 1. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , -1.909, -1.075, 0.707, 0. , 0. , 0.707, 0. , 0. , 0. , 0. , -1.35 , -0.76 , 1. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , -1.909, -1.075]),3)

                at_num_i = ds.atoms_long[center_atom_idx,1].numpy()

                for iatom in range(desc.shape[0]):
                    for trpl in range(nNeigh*(nNeigh-1)):
                        feat = desc[iatom,trpl].numpy()

                        assert at_num_i[iatom] == 7 or at_num_i[iatom] == 1
                        if at_num_i[iatom] == 7:
                            assert feat[grp5Pos[0]] == 1 and feat[row2Pos[0]] == 1, "N atom missmatch"
                            assert feat[grp1Pos[0]] == 0 and feat[row1Pos[0]] == 0, "N atom missmatch"
                        else:
                            assert feat[grp1Pos[0]] == 1 and feat[row1Pos[0]] == 1, "H atom missmatch"
                            assert feat[grp5Pos[0]] == 0 and feat[row2Pos[0]] == 0, "H atom missmatch"

                        for ijk in range(0,3):
                            assert feat[grp1Pos[ijk]] > 0. or feat[grp5Pos[ijk]] > 0., "atom does not seem to be H or N"
                            assert feat[row1Pos[ijk]] > 0. or feat[row2Pos[ijk]] > 0., "atom does not seem to be H or N"


    def test_Ang_pad(self):
        conf, device, data_set, _ = setup_2NH3_Examples(ang_padding_map={2:3})
        confIds = torch.tensor([0,1])
        dloader = DataLoader(data_set, confIds, 2)
        module = Compute1AngleInput(data_set.atom_types.n_feature)
        module = module.to(device)

        ds = dloader.data_set.to(device)
        at_types = ds.atom_types
        n_feature = at_types.n_feature

        # positions of embeddings of H and N in feature vector
        # start at 3 because first 3 elements are dist_ijk
        grp1Pos = [ 3 + at_types.at_prop_names_2_col['g1'] + i * n_feature for i in range(0,3)]
        grp5Pos = [ 3 + at_types.at_prop_names_2_col['g5'] + i * n_feature for i in range(0,3)]
        row1Pos = [ 3 + at_types.at_prop_names_2_col['r1'] + i * n_feature for i in range(0,3)]
        row2Pos = [ 3 + at_types.at_prop_names_2_col['r2'] + i * n_feature for i in range(0,3)]

        dloader.setEpoch(0)
        for batch in dloader:
            batch_ang_neighbor_map = batch['batch_ang_neighbor_map']
            for nNeigh, (i_idx, j_idx, dist_ij) in batch_ang_neighbor_map.items():
                if nNeigh == 1: continue
                center_atom_idx, desc = module(nNeigh, i_idx, j_idx, dist_ij,
                                             ds.atoms_xyz, ds.atoms_long, ds.atom_types.atom_embedding)

                if nNeigh == 3:
                    npt.assert_array_almost_equal(desc[4,5],
                        torch.tensor([ 1.414, 1. , 0.911, 1. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , -1.909, -1.075, 0.707, 0. , 0. , 0.707, 0. , 0. , 0. , 0. , -1.35 , -0.76 , 1. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , -1.909,-1.075]),3)

                    npt.assert_array_almost_equal(desc[-1,-1],  # a padded example
                        torch.tensor([ conf['angleNet']['angularCutoff'],
                                       2.24,  0.28,  1.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00, -1.91, -1.07,  0.31,  0.00,  0.00,  0.31,  0.00,  0.00,  0.00,  0.00, -0.60, -0.34,  0.45,  0.00,  0.00,  0.45,  0.00,  0.00,  0.00,  0.00, -0.85, -0.48]),2)

                # dimensions are:
                #   nExamples of triple by center atom i
                #   nNeigh * (nNeigh -1): number or triples for this nNeigh
                #   3: ijk
                at_num_i = ds.atoms_long[center_atom_idx,1].numpy()

                for iatom in range(desc.shape[0]):
                    for trpl in range(nNeigh*(nNeigh-1)):
                        feat = desc[iatom,trpl].numpy()

                        assert at_num_i[iatom] == 7 or at_num_i[iatom] == 1
                        if at_num_i[iatom] == 7:
                            assert feat[grp5Pos[0]] == 1 and feat[row2Pos[0]] == 1, "N atom missmatch"
                            assert feat[grp1Pos[0]] == 0 and feat[row1Pos[0]] == 0, "N atom missmatch"
                        else:
                            assert feat[grp1Pos[0]] == 1 and feat[row1Pos[0]] == 1, "H atom missmatch"
                            assert feat[grp5Pos[0]] == 0 and feat[row2Pos[0]] == 0, "H atom missmatch"

                        for ijk in range(0,3):
                            assert feat[grp1Pos[ijk]] > 0. or feat[grp5Pos[ijk]] > 0., "atom does not seem to be H or N"
                            assert feat[row1Pos[ijk]] > 0. or feat[row2Pos[ijk]] > 0., "atom does not seem to be H or N"
