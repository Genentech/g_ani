import logging

import re

from torch import nn, optim
from torch.nn.modules.module import Module
import torch
from typing import Dict, Any

from ml_qm import AtomInfo
import ml_qm.pt.nn.activation
import math
from ml_qm.pt.ranger import Ranger
import copy

log = logging.getLogger(__name__)

def initWeigths(layer, nInputs, nOutputs, initWeight = None):
    if not initWeight: initWeight = {"method": "xavier_normal", "gain": 1 }

    meth = initWeight['method']
    gain = initWeight['gain']
    if meth == 'uniform':
        stdv = gain / math.sqrt(nInputs)
        nn.init.uniform_(layer.weight, -stdv, stdv)
    elif meth == 'normal':
        stdv = gain / math.sqrt(nInputs)
        nn.init.normal_(layer.weight, 0., stdv)
    elif meth == 'khan':
        stdv = (gain / (nInputs + nOutputs)) ** 0.5
        nn.init.normal_(layer.weight, 0., stdv)
    elif meth == 'xavier_uniform':
        nn.init.xavier_uniform_(layer.weight, gain=gain)
    elif meth == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight, gain=gain)
    else:
        raise TypeError("Unknown weight initialization '%s'" % meth)


def initBias(layer, initBias = None):
    if not initBias: return

    meth = initBias['method']
    param = initBias.get('param', None)
    if meth == 'uniform':
        if type(param) is list or type(param) is tuple:
            layer.bias.data.uniform_(*param)
        else:
            layer.bias.data.uniform_(**param)
    else:
        raise TypeError(f"unknown bias initialization: {initBias}")


def create_activation(act_func:str, act_param={}) -> Module:
    if act_func is None or act_func == 'None': return None        # for bohb opt to signal no activation
    actFun = getattr(ml_qm.pt.nn.activation, act_func, None)
    if actFun is not None:
        actFun = actFun(**act_param)
    else:
        actFun = getattr(nn, act_func, None)
        if actFun is not None:
            actFun = actFun(**act_param)
        else:
            raise TypeError("unknown activation function: " + act_func)
    return actFun


def createOptimizer(model, conf):
    op = conf['optParam']
    if isinstance(op, Dict):
        optParam = { 'params' : model.parameters(),  **conf['optParam'] }
    else:   # allow for separate opt params for given parameter name (regexp)
        optParamList = []
        knownParamSet = set()

        def getParams(regexp, model):
            pList = []
            for nParam in model.named_parameters():
                pName = nParam[0]
                if pName not in knownParamSet and re.search(regexp, pName):
                    pList.append(nParam[1])
                    knownParamSet.add(pName)
            return pList

        for opp in op:
            optParamList.append(
                { 'params': getParams(opp['regexp'], model), **opp['param'] })
        optParam = { 'params' : optParamList }

        def getParamNames(model):
            pList = []
            for nParam in model.named_parameters():
                pName = nParam[0]
                if pName not in knownParamSet:
                    pList.append(pName)
            return pList

        remainingParam = getParamNames(model)
        if len(remainingParam) > 0:
            for p in remainingParam:
                log.warning(f'Parameter is not being optimized: {p}')

    oType = conf['optType']
    if oType == 'Ranger':
        optimizer = Ranger
    else:
        optimizer = getattr(optim, oType, None)
    if optimizer is not None:
        optimizer = optimizer(**optParam)
    else:
        raise TypeError("unknown optimizer: " + oType)

    return optimizer




class LinearLayer(nn.Module):
    """ Linear layer configured via Dict can have:
        bias, activation,batchNorm,dropout, initWeiight, initBias instructions
    """
    def __init__(self, layer_num:int, nInputs:int, layer_conf:Dict[str,Any], layer_name_prefix:str):
        super(LinearLayer, self).__init__()

        self.mod_dict = nn.ModuleDict()

        nOut       = layer_conf['nodes']
        bias       = layer_conf['bias']
        activation = layer_conf.get('activation', None)
        batchNorm  = layer_conf.get('batchNorm', None)
        dropOutPct = layer_conf.get('dropOutPct', 0)
        requires_grad = layer_conf.get('requires_grad', True)

        layer = nn.Linear(nInputs, nOut, bias=bias)
        initWeigths(layer, nInputs, nOut, layer_conf.get('initWeight', None))
        if bias: initBias(layer, layer_conf.get('initBias', None))
        name = f"{layer_name_prefix}"
        self.mod_dict.add_module(name,layer)

        # activation
        if activation is not None:
            activation = create_activation(
                            activation['type'], activation.get('param',{}))
            name = f"{layer_name_prefix} act"
            self.mod_dict.add_module(name, activation)

        if  batchNorm is not None:
            name = f"{layer_name_prefix} batchNorm"
            bNorm = nn.BatchNorm1d(nOut, *batchNorm.get('param',[]))
            self.mod_dict.add_module(name, bNorm)

        if dropOutPct > 0:
            name = f"{layer_name_prefix} dropOut"
            self.mod_dict.add_module(name, nn.Dropout(dropOutPct / 100., False))

        layer.requires_grad_(requires_grad)
        self.n_output = nOut


    def forward(self, inp:torch.tensor) -> torch.tensor:
        for layr in self.mod_dict.values():
            if isinstance(layr,nn.BatchNorm1d):
                s=inp.shape
                inp = layr(inp.reshape(-1,self.n_output)).reshape(s)
            else:
                inp = layr(inp)

        return inp



class AtomTypes:

    """ Embeddign of atomtypes """

    atom_embedding:nn.Embedding
    atom_vdw:torch.tensor
    n_atom_type:int
    n_feature:int
    at_prop_names_2_col:Dict[str,int]

    def __init__(self, atom_types:Dict[str,Dict[str,float]], atom_embedding_conf):
        self.n_atom_type = len(atom_types)
        self.n_feature = len(list(atom_types.values())[0])
        self.atom_vdw = torch.tensor(AtomInfo.NumToVDWRadius)
        max_atom_num = max([AtomInfo.NameToNum[k] for k in atom_types.keys()])

        at_props = torch.full((max_atom_num+1, self.n_feature),-1, dtype=torch.float32)
        at_prop_names = list(list(atom_types.values())[0].keys())
        self.at_prop_names_2_col = { nam:i for (i,nam) in enumerate(at_prop_names)}

        for at_sym, at_prop in atom_types.items():
            at_num = AtomInfo.NameToNum[at_sym]
            at_props[at_num] = torch.tensor(list(at_prop.values()), dtype=torch.float32)

        # normalize by avg and stdev if given

        if atom_embedding_conf and atom_embedding_conf.get('normalization', None):
            for nam, stat in atom_embedding_conf['normalization'].items():
                prop_col = self.at_prop_names_2_col[nam]
                at_props[:, prop_col] = (
                    at_props[:, prop_col] - stat['avg']) / stat['sigma']

        self.atom_embedding = nn.Embedding.from_pretrained(at_props, freeze=True)

    def to_(self, device):
        """ move to device """
        self.atom_embedding = self.atom_embedding.to(device)
        self.atom_vdw       = self.atom_vdw.to(device)


    def to(self, device):
        """ clone to device """
        newAt = copy.deepcopy(self)  # deepcopy needed because to() does not create new object
        newAt.atom_embedding = newAt.atom_embedding.to(device)
        newAt.atom_vdw       = self.atom_vdw.to(device)

        return newAt
