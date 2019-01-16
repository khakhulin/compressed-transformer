import numpy as np
import torch
from torch import nn
import tntorch as tn


def matrix_to_tt_cores(matrix, shapes, ranks):
  shapes = np.asarray(shapes)
  matrix = matrix.reshape(list(shapes.flatten()))
  d = len(shapes[0])
  transpose_idx = list(np.arange(2 * d).reshape(2, d).T.flatten())
  matrix = matrix.permute(*transpose_idx)
  newshape = np.prod(shapes, 0)
  matrix = matrix.reshape(list(newshape))
  tt = tn.Tensor(matrix, ranks_tt=ranks)

  newcores = []
  for core, s1, s2, r1, r2 in zip(tt.cores,
                                  shapes[0], shapes[1],
                                  tt.ranks_tt, tt.ranks_tt[1:]):
    newcores.append(core.reshape((r1, s1, s2, r2)))
  return newcores


def ttmatmul(cores, t, shapes, ranks):
  ranks = [1] + ranks + [1]
  tshape = t.shape

  t = t.transpose(1, 0)
  t = t.reshape((-1, shapes[1][-1], 1))
  ndims = len(cores)
  for i in reversed(range(ndims)):
    t = torch.einsum('aijb,rjb->ira', (cores[i], t))
    if i:
      t = t.reshape((-1, shapes[1][i - 1], ranks[i]))
  t = t.reshape((int(np.prod(shapes[0])), tshape[1]))
  return t


def transpose(cores):
  result = []
  for c in cores:
    result.append(c.permute((0, 2, 1, 3)))
  return result


def matmultt(t, cores, shapes, ranks):
  t = t.transpose(1, 0)
  cores = transpose(cores)
  shapes = [shapes[1], shapes[0]]
  return ttmatmul(cores, t, shapes, ranks).transpose(1, 0)


class TTLayer(nn.Module):
  def __init__(self, layer, shapes, ranks):
    super(TTLayer, self).__init__()
    self.shapes = shapes
    self.ranks = ranks
    with torch.no_grad():
      weight = layer.weight.transpose(1, 0)
      self.cores = nn.ParameterList(
          map(nn.Parameter, matrix_to_tt_cores(weight, shapes, ranks)))
    self.bias = layer.bias

  def forward(self, inputs):
    out = matmultt(inputs, self.cores, self.shapes, self.ranks)
    out = out + self.bias
    return out
