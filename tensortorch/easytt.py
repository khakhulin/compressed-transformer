import numpy as np

import torch
import torch.nn as nn


def tt_dot(in_modes, out_modes, ranks, inputs, weight, bias=None):
    assert len(in_modes) == len(out_modes) == len(ranks) - 1
    assert inputs.shape[2] == np.prod(in_modes)
    b, seq, _ = inputs.size()
    res = inputs
    res = res.view(-1, int(np.prod(in_modes)))
    res = res.transpose(1, 0)
    res = res.contiguous()
    dim = len(in_modes)
    for ii in range(dim):
        res = res.view(ranks[ii] * in_modes[ii], -1)
        res = torch.matmul(weight[ii], res)
        res = res.view(out_modes[ii], -1)
        res = res.transpose(1, 0)
        res = res.contiguous()
    # res = res.view(-1, int(np.prod(out_modes)))
    res = res.view(b,seq, int(np.prod(out_modes)))
    if bias is not None:
        res += bias
    return res

def seq_tt_dot(in_modes, out_modes, ranks, inputs, weight, bias=None):
    assert len(in_modes) == len(out_modes) == len(ranks) - 1
    assert inputs.shape[2] == np.prod(in_modes)
    b, seq, _ = inputs.size()
    res = inputs
    res = res.view(-1, int(np.prod(in_modes)))
    res = res.transpose(1, 0)
    res = res.contiguous()
    dim = len(in_modes)
    for ii in range(dim):
        res = res.view(ranks[ii] * in_modes[ii], -1)
        res = torch.matmul(weight[ii], res)
        res = res.view(out_modes[ii], -1)
        # res = res.transpose(1, 0)
        # res = res.contiguous()
    # res = res.view(-1, int(np.prod(out_modes)))
    res = res.view(b,seq, int(np.prod(out_modes)))
    if bias is not None:
        res += bias
    return res


class TTLayer(nn.Module):
    def __init__(self, in_modes, out_modes, ranks, bias=True):
        super().__init__()
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.ranks = ranks
        dim = len(self.in_modes)

        assert len(self.in_modes) == len(self.out_modes) == len(self.ranks) - 1

        self._create_tt_cores(self.in_modes, self.out_modes, self.ranks)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(np.prod(out_modes)))
        else:
            self.register_parameter('bias', None)
        print("2")
        self.reset_parameters()

    def reset_normal(self):
        normal_z = ((((0.05 ** 2) / np.prod(self.ranks))) ** (1 / (len(self.ranks) - 1))) ** 0.5
        for i in range(len(self.weight)):
            nn.init.normal_(self.weight[i], 0, normal_z)

    def reset_parameters(self):
        self.reset_normal()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return seq_tt_dot(self.in_modes, self.out_modes, self.ranks, input, self.weight, self.bias)

    def _create_tt_cores(self, in_modes, out_modes, ranks):
        """
        in_modes: shape of initial tensor
        out_modes: shape of out tensor
        Total tensor shape is element_wise_product(in_modes,out_modes)
        ranks: desirable ranks of tt len(ranks) + 1 == len(in_modes)
        return: weights
        """
        dim = len(in_modes)
        _tt_cores_list = []

        for i in range(dim):
            _tt_cores_list.append(nn.Parameter(torch.Tensor(out_modes[i] * ranks[i + 1], in_modes[i] * ranks[i])))

        self.weight = nn.ParameterList(_tt_cores_list)
