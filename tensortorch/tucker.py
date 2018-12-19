import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class TuckerLinear(nn.Module):
    def __init__(self, in_modes, out_modes, ranks, bias=True, cache=False):
        """
        cache: if cache is True, pre calculated W_tsr until user reset the variable
        """
        super().__init__()
        assert len(in_modes) == len(out_modes) == len(ranks)
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.ranks = ranks
        self.cache = cache
        self._W_linear = None

        self.core, self.factors = self._create_tucker_params(in_modes, out_modes, ranks)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(int(np.prod(out_modes))))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        CONST = (0.05 / np.prod(self.ranks + self.ranks)**0.5)**(1.0 / (len(self.in_modes) + len(self.out_modes) + 1))
        nn.init.normal_(self.core, 0, CONST)
        for ii in range(len(self.factors)):
            nn.init.normal_(self.factors[ii], 0, CONST)
            pass
        if self.bias is not None:
            self.bias.data.zero_()

    def reset(self):
        self._W_linear = None

    @property
    def W_linear(self):
        if not self.cache:
            return _tensor_to_matrix(self.in_modes, self.out_modes,
                                     _tucker_cores_to_tensor(self.core, list(self.factors)))
        if self._W_linear is None:
            self._W_linear = _tensor_to_matrix(self.in_modes, self.out_modes,
                                               _tucker_cores_to_tensor(self.core, list(self.factors)))
        else:
            pass
        return self._W_linear

    def forward(self, input):
        return F.linear(input, self.W_linear.t(), self.bias)

    def _create_tucker_params(self, in_modes, out_modes, ranks) :
        assert len(in_modes) == len(out_modes) == len(ranks)
        modes = in_modes + out_modes # extend list
        core = nn.Parameter(torch.Tensor(*list(ranks+ranks)).normal_())
        factors = []
        for mm, rr in zip(modes, ranks+ranks) :
            factors.append(nn.Parameter(torch.Tensor(mm, rr).normal_()))
        factors = nn.ParameterList(factors)
        return core, factors


def _tensor_to_matrix(in_modes, out_modes, tensor) :
    return tensor.contiguous().view(int(np.prod(in_modes)), int(np.prod(out_modes)))

def _n_mode_product(core, factor, mode) :
    assert factor.dim() == 2
    # core = [i_1,..,i_j,..,i_D]
    core_shape = list(core.shape) # j = mode
    core_tmp = core.transpose(mode, -1) # [i_1,..,i_D,i_j]
    new_core_shape = list(core_tmp.shape)
    core_tmp = core_tmp.contiguous().view(-1, core_shape[mode]) # [prod([i_1,..,i_D]), i_j ]
    core_tmp = core_tmp.mm(factor.t()) # [prod([i_1,..,i_D]), m_j]
    core_tmp = core_tmp.view(*new_core_shape[0:-1], factor.shape[0]) # [i_1,..,i_D,m_j]
    core_tmp = core_tmp.transpose(mode, -1) # [i_1,..,m_j,..,i_D]
    return core_tmp


def _tucker_cores_to_tensor(core, list_factors) :
    n_dim = len(list_factors)
    assert n_dim == core.dim()
    tensor_out = core.contiguous()
    for ii in range(n_dim) :
        tensor_out = _n_mode_product(tensor_out, list_factors[ii], ii)
    return tensor_out

