"""
utility functions
"""
import os

import scipy
from scipy.stats import sem
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from torch_geometric.nn.conv.gcn_conv import gcn_norm
#plus______________________________________________
import torch.nn.functional as F
from torch_geometric.utils import softmax
#_______________________________________________________

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class MaxNFEException(Exception): pass


def rms_norm(tensor):
  return tensor.pow(2).mean().sqrt()


def make_norm(state):
  if isinstance(state, tuple):
    state = state[0]
  state_size = state.numel()

  def norm(aug_state):
    y = aug_state[1:1 + state_size]
    adj_y = aug_state[1 + state_size:1 + 2 * state_size]
    return max(rms_norm(y), rms_norm(adj_y))

  return norm


def print_model_params(model):
  total_num_params = 0
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      total_num_params += param.numel()
  print("Model has a total of {} params".format(total_num_params))


def adjust_learning_rate(optimizer, lr, epoch, burnin=50):
  if epoch <= burnin:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr * epoch / burnin


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not int(fill_value) == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-0.5)
  deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
  return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def coo2tensor(coo, device=None):
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  values = coo.data
  v = torch.FloatTensor(values)
  shape = coo.shape
  print('adjacency matrix generated with shape {}'.format(shape))
  # test
  return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_sym_adj(data, opt, improved=False):
  edge_index, edge_weight = gcn_norm(  # yapf: disable
    data.edge_index, data.edge_attr, data.num_nodes,
    improved, opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  return coo2tensor(coo)


def get_rw_adj_old(data, opt):
  if opt['self_loop_weight'] > 0:
    edge_index, edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                       fill_value=opt['self_loop_weight'])
  else:
    edge_index, edge_weight = data.edge_index, data.edge_attr
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  normed_csc = normalize(coo, norm='l1', axis=0)
  return coo2tensor(normed_csc.tocoo())


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]

  return edge_index, edge_weight
#plus
def compute_transP( cd, edge_index):
        """

        :param cd: class distribution [N, D]
        :param edge_index: [2, E]
        :return: transition probability (transP) [E, 1]
        """

        # edge_index: [2, E]
        row, col = edge_index  # row, col: [1, E] each

        # Indexing: [N, D] -> [E, D]
        p_i = cd[row]
        p_j = cd[col]

        # Transition Probability
        pipj = (p_i * p_j).sum(dim=-1)  # [E, 1]
        
        #plus,添加一层神经网络去学习结点间的关系
        # m2=torch.ones(pipj.shape[0],requires_grad=True)
        # m2.to(device='cuda')
        # # m2=torch.nn.Linear(pipj.shape[1], pipj.shape[1])
        # m2=torch.nn.Parameter(m2)
        # m2= torch.nn.Parameter(torch.ones(pipj.shape[0]).cuda(), requires_grad= True)
        
        pipj=pipj
        # pipj=self.m2(pipj)
        # transP = torch.nn.functional.normalize(pipj,dim=0)
        transP = F.softmax(pipj)
        

        # with torch.no_grad():
        #     sum_pipj = scatter_add(pipj, row)

        return transP
#plus
def get_rw_adj_cad(x,edge_index):
  cd = F.softmax(x, dim=-1)
  EPS = 1e-15
  entropy = -(cd * torch.log(cd + EPS)).sum(dim=-1)

  #  Compute a transition matrix: transP
  transP =  compute_transP(cd, edge_index)
  return edge_index, 1+transP


#plus
def get_edge_weight(data,norm_dim=1,fill_value=0.,dtype=None,tau=[5,]):
  # waveplus________________________
  edge_weight = torch.ones((data.edge_index.size(1),), dtype=dtype,
                           device=data.edge_index.device)
  from collections import defaultdict
  chev_order = 3
  thre = 0.0001
  # tau = [5, ]
  num_sample = 10
  num_nodes = data.x.shape[0]
  num_feats = data.x.shape[1]
  import pickle as pkl
  import scipy.sparse as sparse
  from pygsp import graphs, filters, plotting, utils
  adj_lists = defaultdict(set)
  wave_lists = defaultdict(set)  # wavelet 基过滤后的列表
  us=data.edge_index[0]
  vs=data.edge_index[1]
  for i in range(len(us)):
    adj_lists[us[i]].add(vs[i])
    adj_lists[vs[i]].add(us[i])

  adj_mat = sparse.lil_matrix((num_nodes, num_nodes))
  for p1 in adj_lists:
    for p2 in adj_lists[p1]:
      adj_mat[p1, p2] = 1
  G = graphs.Graph(adj_mat)
  G.estimate_lmax()
  f = filters.Heat(G, tau)  # 此处的参数可变
  chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
  s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
  for i in range(s.shape[0]):
    ls = []
    neis = []
    for j in range(s.shape[1]):
      if s[i][j] > thre:
        ls.append((j, s[i][j]))
    if len(ls) < num_sample:
      for k in range(len(ls)):
        neis.append(ls[k][0])
    else:
      ls = sorted(ls, key=lambda x: x[1], reverse=True)
      for k in range(num_sample):
        neis.append(ls[k][0])
    wave_lists[i] = set(neis)

  adj = np.zeros((data.x.shape[0], data.x.shape[0]), dtype='float32')
  wave = np.zeros((data.x.shape[0], data.x.shape[0]), dtype='float32')
  wave_new = np.zeros((data.x.shape[0], data.x.shape[0]), dtype='float32')

  value_of_s_inwavelist = []
  for paper1, nodes_list in wave_lists.items():
    for paper2 in nodes_list:
      if paper1 != paper2:
        value_of_s_inwavelist.append(s[paper1, paper2] - thre)
  max_of_s = np.max(value_of_s_inwavelist)
  min_of_s = np.min(value_of_s_inwavelist)
  mean_of_s = np.mean(value_of_s_inwavelist)  # np.mean(value_of_s_inwavelist)#np.sum(s[:,:])/(len(wave_lists))
  std_of_s = np.std(value_of_s_inwavelist)
  median_of_s = np.median(value_of_s_inwavelist)
  for paper1, nodes_list in wave_lists.items():
    for paper2 in nodes_list:
      wave[paper1, paper2] = 1.0
      wave[paper2, paper1] = 1.0

  for paper1, nodes_list in adj_lists.items():
    for paper2 in nodes_list:
      adj[paper1, paper2] = 1.0
      adj[paper2, paper1] = 1.0

  for paper1, nodes_list in wave_lists.items():
    for paper2 in nodes_list:
      if paper1 == paper2:
        wave_new[paper1, paper2] = 1.0
      else:
        wave_new[paper1, paper2] = 1.0 + (s[paper1, paper2] - thre - min_of_s) / (
                max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）
        wave_new[paper2, paper1] = (1.0 + (s[paper2, paper1] - thre - min_of_s) / (
                max_of_s - min_of_s) ) # 实验一下老师新说的，看会不会有效果（原来为1）
  wave_new=torch.from_numpy(wave_new)
  wave = torch.from_numpy(wave)
  adj = torch.from_numpy(adj)
  times=0

  for i in range(data.edge_index.size(1)):
    edge_weight[i]= wave[us[i], vs[i]]
    # if edge_weight[i] == 0:
    #   edge_weight[i] = 1

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(data.edge_index, edge_weight, 1 + (max_of_s-min_of_s-thre)/(max_of_s - min_of_s) , num_nodes)#fill_value
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight
  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]

  def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)
  i=0
  edge_index_0=edge_index[0]
  edge_index_1=edge_index[1]
  while i<len(edge_index_0):
    if edge_weight[i]==0:
      edge_weight = del_tensor_ele(edge_weight, i)
      edge_index_0 = del_tensor_ele(edge_index_0, i)
      edge_index_1 = del_tensor_ele(edge_index_1, i)
      times+=1
    else:
      i+=1
  edge_index=torch.stack((edge_index_0, edge_index_1), dim=0)
  # #other method----------------------------------------
  # edge_weight=[]
  # edge_index_0=[]
  # edge_index_1=[]
  # for i, nodes_list in wave_lists.items():
  #   for j in nodes_list:
  #     if wave_new[i,j]!=0:
  #       edge_index_0.append(i)
  #       edge_index_1.append(j)
  #       edge_weight.append(wave[i,j])
  # edge_weight=torch.tensor(edge_weight,dtype=dtype,device=edge_index.device)
  # edge_index_0 = torch.tensor(edge_index_0, dtype=torch.int64, device=edge_index.device)
  # edge_index_1 = torch.tensor(edge_index_1, dtype=torch.int64, device=edge_index.device)
  # edge_index = torch.stack((edge_index_0, edge_index_1), dim=0)
  # row, col = edge_index_0, edge_index_1
  # indices = row if norm_dim == 0 else col
  # deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
  # deg_inv_sqrt = deg.pow_(-1)
  # edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
  # # other method----------------------------------------

  return edge_index, edge_weight

def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  return torch.sparse.FloatTensor(i, v * d, s.size())


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval


def get_full_adjacency(num_nodes):
  # what is the format of the edge index?
  edge_index = torch.zeros((2, num_nodes ** 2),dtype=torch.long)
  for idx in range(num_nodes):
    edge_index[0][idx * num_nodes: (idx + 1) * num_nodes] = idx
    edge_index[1][idx * num_nodes: (idx + 1) * num_nodes] = torch.arange(0, num_nodes,dtype=torch.long)
  return edge_index



from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr


# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
  r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
  out = src - src.max()
  # out = out.exp()
  out = (out + torch.sqrt(out ** 2 + 4)) / 2

  if ptr is not None:
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
  elif index is not None:
    N = maybe_num_nodes(index, num_nodes)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
  else:
    raise NotImplementedError

  return out / (out_sum + 1e-16)


# Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val


class DummyDataset(object):
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class DummyData(object):
  def __init__(self, edge_index=None, edge_Attr=None, num_nodes=None):
    self.edge_index = edge_index
    self.edge_attr = edge_Attr
    self.num_nodes = num_nodes
