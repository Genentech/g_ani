import torch
import numpy.testing as npt
from ml_qm.distNet.dist_net import ComputeRealAngle2Input
from ml_qm.distNet.data_loader import DataLoader
from tests.distNet.test_data_set import setup_2NH3_Examples


class TestToInput():
    def setup(self):
        self.conf, self.device, data_set, _ = setup_2NH3_Examples()
        cutoff = self.conf['angleNet']['angularCutoff']
        confIds = torch.tensor([0,1])
        self.dloader = DataLoader(data_set, confIds, 2)
        self.data_set = data_set.to(self.device)
        self.module = ComputeRealAngle2Input(data_set.atom_types.n_feature,cutoff)
        self.module = self.module.to(self.device)


    def test_triples(self):
        exp_dist = { 2:
                ( #triple_dist_ij, triple_dist_ik, triple_angle_jk
                  # i_idx.flat = [1, 2]   j_idx = [[0, 2], [0, 1]]
                  # 10    10         12    12           20    20        21    21        ij is repeated
                 [[[0.778, 0.778], [0.093, 0.093]], [[0.154, 0.154], [0.093, 0.093]]],
                  # 10    12         10    12           20    21        20    21        ij1,ij2 is repeated
                 [[[0.778, 0.093], [0.778, 0.093]], [[0.154, 0.093], [0.154, 0.093]]],
                  # 100   102     120   122      200    201     210   211
                 [[[0.01, 1.11], [1.11, 0.01]], [[0.01, 0.46], [0.46, 0.01]]]),
            3: \
                (  # i_idx.flat [0, 4, 5, 6, 7],
                   # j_idx [[1, 2, 3], [5, 6, 7], [4, 6, 7], [4, 5, 7], [4, 5, 6]]
                   #triple_dist_ij
                     # 01    01    01        02                     03
                   [[[0.778, 0.778, 0.778], [0.154, 0.154, 0.154], [0.003, 0.003, 0.003]],
                     # 45    45    45        46                     47
                    [[0.778, 0.778, 0.778], [0.154, 0.154, 0.154], [0.418, 0.418, 0.418]],
                     # 54                    56                     57
                    [[0.778, 0.778, 0.778], [0.093, 0.093, 0.093], [0.778, 0.778, 0.778]],
                    [[0.154, 0.154, 0.154], [0.093, 0.093, 0.093], [0.418, 0.418, 0.418]],
                    [[0.418, 0.418, 0.418], [0.778, 0.778, 0.778], [0.418, 0.418, 0.418]]],
                   #triple_dist_ik
                     # 01   02    03          01   02    03          01   02    03
                    [[[0.778, 0.154, 0.003], [0.778, 0.154, 0.003], [0.778, 0.154, 0.003]],
                    #  45   46    47          45   46    47          45   46    47
                     [[0.778, 0.154, 0.418], [0.778, 0.154, 0.418], [0.778, 0.154, 0.418]], 
                     [[0.778, 0.093, 0.778], [0.778, 0.093, 0.778], [0.778, 0.093, 0.778]], 
                     [[0.154, 0.093, 0.418], [0.154, 0.093, 0.418], [0.154, 0.093, 0.418]], 
                     [[0.418, 0.778, 0.418], [0.418, 0.778, 0.418], [0.418, 0.778, 0.418]]],
                   # triple_angle_jk
                     # 011  012   013     021   022   023     031   032   033
                   [[[0.01, 1.57, 3.13], [1.57, 0.01, 1.57], [3.13, 1.57, 0.01]],
                    [[0.01, 1.57, 0.79], [1.57, 0.01, 0.79], [0.79, 0.79, 0.01]],
                    [[0.01, 1.11, 1.57], [1.11, 0.01, 0.46], [1.57, 0.46, 0.01]],
                    [[0.01, 0.46, 0.79], [0.46, 0.01, 0.32], [0.79, 0.32, 0.01]],
                    [[0.01, 0.79, 1.57], [0.79, 0.01, 2.36], [1.57, 2.36, 0.01]]])
        }

        ds = self.data_set
        self.dloader.setEpoch(0)
        for batch in self.dloader:
            batch_ang_neighbor_map = batch['batch_ang_neighbor_map']
            for nNeigh, (i_idx, j_idx, dist_ij) in batch_ang_neighbor_map.items():
                if nNeigh == 1: continue

                triple_dist_ij, triple_dist_ik, triple_angle_jk \
                   = self.module._computeTriples(nNeigh, i_idx, j_idx, dist_ij, ds.atoms_xyz)
                exp_ij, exp_ik, exp_jk = exp_dist[nNeigh]
                npt.assert_array_almost_equal(triple_dist_ij,exp_ij,3)
                npt.assert_array_almost_equal(triple_dist_ik,exp_ik,3)
                npt.assert_array_almost_equal(triple_angle_jk,exp_jk,2)


    def test_Ang(self):
        ds = self.data_set
        at_types = ds.atom_types
        n_feature = at_types.n_feature

        # positions of embeddings of H and N in feature vector
        # start at 3 because first 3 elements are dist_ijk
        grp1Pos = [ 1 + at_types.at_prop_names_2_col['g1'] + i * n_feature for i in range(0,3)]
        grp5Pos = [ 1 + at_types.at_prop_names_2_col['g5'] + i * n_feature for i in range(0,3)]
        row1Pos = [ 1 + at_types.at_prop_names_2_col['r1'] + i * n_feature for i in range(0,3)]
        row2Pos = [ 1 + at_types.at_prop_names_2_col['r2'] + i * n_feature for i in range(0,3)]

        self.dloader.setEpoch(0)
        for batch in self.dloader:
            batch_ang_neighbor_map = batch['batch_ang_neighbor_map']
            for nNeigh, (i_idx, j_idx, dist_ij) in batch_ang_neighbor_map.items():
                if nNeigh == 1: continue
                center_atom_idx, desc = self.module(nNeigh, i_idx, j_idx, dist_ij,
                                             ds.atoms_xyz, ds.atoms_long, ds.atom_types.atom_embedding)

                if nNeigh == 3:
                    npt.assert_array_almost_equal(desc[4,5],
                        torch.tensor([ 2.356,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1.909, -1.075,  0.418,  0.,  0.,  0.418,  0.,  0.,  0.,  0., -0.798, -0.449,  0.778,  0.,  0.,  0.778,  0.,  0.,  0.,  0., -1.485, -0.836]),3)

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
        cutoff = self.conf['angleNet']['angularCutoff']
        confIds = torch.tensor([0,1])
        dloader = DataLoader(data_set, confIds, 2)
        module = ComputeRealAngle2Input(data_set.atom_types.n_feature,cutoff)
        module = module.to(device)

        ds = dloader.data_set.to(device)
        at_types = ds.atom_types
        n_feature = at_types.n_feature

        # positions of embeddings of H and N in feature vector
        # start at 3 because first 3 elements are dist_ijk
        grp1Pos = [ 1 + at_types.at_prop_names_2_col['g1'] + i * n_feature for i in range(0,3)]
        grp5Pos = [ 1 + at_types.at_prop_names_2_col['g5'] + i * n_feature for i in range(0,3)]
        row1Pos = [ 1 + at_types.at_prop_names_2_col['r1'] + i * n_feature for i in range(0,3)]
        row2Pos = [ 1 + at_types.at_prop_names_2_col['r2'] + i * n_feature for i in range(0,3)]

        dloader.setEpoch(0)
        for batch in dloader:
            batch_ang_neighbor_map = batch['batch_ang_neighbor_map']
            for nNeigh, (i_idx, j_idx, dist_ij) in batch_ang_neighbor_map.items():
                if nNeigh == 1: continue
                center_atom_idx, desc = module(nNeigh, i_idx, j_idx, dist_ij,
                                             ds.atoms_xyz, ds.atoms_long, ds.atom_types.atom_embedding)

                if nNeigh == 3:
                    npt.assert_array_almost_equal(desc[4,5],
                        torch.tensor([ 2.356,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1.909, -1.075,  0.418,  0.,  0.,  0.418,  0.,  0.,  0.,  0., -0.798, -0.449,  0.778,  0.,  0.,  0.778,  0.,  0.,  0.,  0., -1.485, -0.836]),3)

                    npt.assert_array_almost_equal(desc[-1,-1],  # a padded example
                        torch.tensor([ 1.571, 1., 0., 0., 1., 0., 0., 0., 0., -1.909, -1.075, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.093, 0., 0., 0.093, 0., 0., 0., 0., -0.177, -0.100]),2)

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
