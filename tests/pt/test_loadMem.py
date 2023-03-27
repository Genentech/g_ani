import torch
import json

from ml_qm.pt.nn.ani_net import ANINetInitializer
import copy

torch.set_printoptions(linewidth=200)


class TestMemLoad():
    def setup(self):
        jsonFile = 'scripts/trainHCON_Bump_48.json'
        with open(jsonFile) as jFile:
            self.conf = json.load(jFile)

        nGPU=1
        self.device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")


        initAni    = ANINetInitializer(self.conf,self.device)
        self.descComput = initAni.create_descriptor_computer()
        self.model      = initAni.create_model(None, self.descComput, False)


    def test_freeze(self):
        pstat = [ p.requires_grad for p in self.model.parameters()]
        assert sum( pstat ) == 32
        allActiveState = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(allActiveState)
        pstat = [ p.requires_grad for p in self.model.parameters()]
        assert sum( pstat ) == 32

        conf = copy.deepcopy(self.conf)
        conf['layers'][1]['requires_grad'] = False
        initAni    = ANINetInitializer(conf,self.device)
        model      = initAni.create_model(None, self.descComput, False)

        pstat = [ p.requires_grad for p in model.parameters()]
        assert sum( pstat ) == 24

        # verify that load_state_dict does not affect requires_grad
        model.load_state_dict(allActiveState)
        pstat = [ p.requires_grad for p in model.parameters()]
        assert sum( pstat ) == 24
