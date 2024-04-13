import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException

# plus CAD--------------------------------------------------------------------------------------------------------------------------------------
import argparse
import torch
import torch.nn.functional as F
import time
from torch_geometric.utils import add_self_loops

from AdaCAD_cora_copy import AdaCAD
from torch_geometric.utils import softmax, sort_edge_index, degree
from torch_scatter import scatter_add



parser = argparse.ArgumentParser()
parser.add_argument('--is_debug', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_drop', type=bool, default=True)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--add_selfloops', type=bool, default=True)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--entropy_regularization', type=float, default=0.5)
args = parser.parse_args()
# plus CAD--------------------------------------------------------------------------------------------------------------------------------------


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    # plus CAD------------------------------------------------

    # self.adgs = AdaCAD(K=args.K,
    #                        beta=args.beta,
    #                        dropout=args.dropout,
    #                        edge_index=data.edge_index,
    #                        inchannel= 7
    #                        )
    # # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # # self.adgs.to(device)
    # # self.data=data
    # self.K=1
    self.transP=None

    # plus CAD------------------------------------------------

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax
  # def cad(self, x):
  #   # plus,添加一层神经网络去预测节点的类别，对于下面的聚合更有意义
  #   # x=self.m1(x)

  #   # Step 1: Class Distribution & Entropy Regularization
  #   cd = F.softmax(x, dim=-1)
  #   EPS = 1e-15
  #   entropy = -(cd * torch.log(cd + EPS)).sum(dim=-1)

  #   # Step 2: Compute a transition matrix: transP
  #   if self.transP==None:
  #     self.transP, self.sum_pipj = self.compute_transP(cd, self.edge_index)

  #   # Step 3: gamma
  #   with torch.no_grad():
  #       deg = degree(self.edge_index[0])
  #       deg[deg==0] = 1
  #       cont_i = self.sum_pipj / deg

  #       gamma =  cont_i
  #   # gamma=F.softmax(gamma)

  #   # Step 4: Aggregate features
  #   x = F.dropout(x, p=0.5, training=True)
  #   H = x
  #   # x_list=[H]


  #   for k in range(self.K):
  #       x =torch_sparse.spmm(self.edge_index,self.transP, x.shape[0], x.shape[0], x)#self.propagate(self.edge_index, x=x, transP=transP)


  #   x = (1 - gamma.unsqueeze(dim=-1)) * H + gamma.unsqueeze(dim=-1) * x#[-1]
  #   return x, entropy

  def compute_transP(self, cd, edge_index):
      
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
        

    # with torch.no_grad():
    #     sum_pipj = scatter_add(pipj, row)
    transP = softmax(pipj, row, num_nodes=cd.size(0))
        

        

    return transP

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1


    ax = self.sparse_multiply(x)
    # if self.transP==None:
    # self.transP= self.compute_transP(x,self.edge_index)
    # ax =torch_sparse.spmm(self.edge_index, self.transP, x.shape[0], x.shape[0], x)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    f = alpha * (ax - x)
    # # #plus 使用cad重新计算f
    # z, ent  = self.cad(x)
    # f = alpha * (z - x)
    # # print('test')

    if self.opt['add_source']:
      f = f + self.beta_train * self.x0


    return f
