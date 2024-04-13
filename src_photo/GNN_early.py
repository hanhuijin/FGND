"""
A GNN used at test time that supports early stopping during the integrator
"""

import torch
import torch.nn.functional as F
import argparse
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import time
from data import get_dataset
# from run_GNN import get_optimizer, train, test
from early_stop_solver import EarlyStopInt
from base_classes import BaseGNN
from model_configurations import set_block, set_function


# plus CAD--------------------------------------------------------------------------------------------------------------------------------------
import argparse
import torch
import torch.nn.functional as F
import time
from torch_geometric.utils import add_self_loops

# from AdaCAD_cora import AdaCAD
# import sys
# sys.path.append(r"/home/hhj/file/graph-neural-pde-main/src")
from AdaCAD_cora_for_grand_cad import AdaCAD


#cora 
# parser = argparse.ArgumentParser()
# parser.add_argument('--is_debug', type=bool, default=False)
# parser.add_argument('--runs', type=int, default=100)
# parser.add_argument('--dataset', type=str, default='Cora')
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--lr_drop', type=bool, default=True)
# parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--early_stopping', type=int, default=10)
# parser.add_argument('--add_selfloops', type=bool, default=True)
# parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--dropout', type=float, default=0.5)
# parser.add_argument('--normalize_features', type=bool, default=True)
# parser.add_argument('--K', type=int, default=1)#13
# parser.add_argument('--beta', type=float, default=0.1)
# parser.add_argument('--entropy_regularization', type=float, default=0.5)
# args = parser.parse_args()
# citeseer
# parser = argparse.ArgumentParser()
# parser.add_argument('--is_debug', type=bool, default=False)
# parser.add_argument('--runs', type=int, default=100)
# parser.add_argument('--dataset', type=str, default='Citeseer')
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--lr_drop', type=bool, default=True)
# parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--early_stopping', type=int, default=10)
# parser.add_argument('--add_selfloops', type=bool, default=True)
# parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--dropout', type=float, default=0.5)
# parser.add_argument('--normalize_features', type=bool, default=True)
# parser.add_argument('--K', type=int, default=1)#13
# parser.add_argument('--beta', type=float, default=0.15)
# parser.add_argument('--entropy_regularization', type=float, default=0.5)
# args = parser.parse_args()
#Pubmed
# parser = argparse.ArgumentParser()
# parser.add_argument('--is_debug', type=bool, default=False)
# parser.add_argument('--runs', type=int, default=100)
# parser.add_argument('--dataset', type=str, default='Pubmed')
# parser.add_argument('--epochs', type=int, default=300)
# parser.add_argument('--lr', type=float, default=0.03)
# parser.add_argument('--lr_drop', type=bool, default=True)
# parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--early_stopping', type=int, default=30)
# parser.add_argument('--add_selfloops', type=bool, default=False)
# parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--dropout', type=float, default=0.3)
# parser.add_argument('--normalize_features', type=bool, default=True)
# parser.add_argument('--K', type=int, default=8)#8
# parser.add_argument('--beta', type=float, default=0.85)#0.85
# parser.add_argument('--entropy_regularization', type=float, default=0.5)
# args = parser.parse_args()
#Computers
# parser = argparse.ArgumentParser()
# parser.add_argument('--is_debug', type=bool, default=False)
# parser.add_argument('--runs', type=int, default=100)
# parser.add_argument('--dataset', type=str, default='Computers')
# parser.add_argument('--epochs', type=int, default=300)
# parser.add_argument('--lr', type=float, default=0.03)
# parser.add_argument('--lr_drop', type=bool, default=True)
# parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--early_stopping', type=int, default=30)
# parser.add_argument('--add_selfloops', type=bool, default=False)
# parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--dropout', type=float, default=0.3)
# parser.add_argument('--normalize_features', type=bool, default=True)
# parser.add_argument('--K', type=int, default=1)#8
# parser.add_argument('--beta', type=float, default=0.75)#0.85
# parser.add_argument('--entropy_regularization', type=float, default=0.5)
# args = parser.parse_args()

#Photo
parser = argparse.ArgumentParser()
parser.add_argument('--is_debug', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--dataset', type=str, default='Photo')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.03)
parser.add_argument('--lr_drop', type=bool, default=True)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=30)
parser.add_argument('--add_selfloops', type=bool, default=False)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=2)#8
parser.add_argument('--beta', type=float, default=0.8)#0.85
parser.add_argument('--entropy_regularization', type=float, default=0.5)
args = parser.parse_args()
#CoauthorCS
# parser = argparse.ArgumentParser()
# parser.add_argument('--is_debug', type=bool, default=False)
# parser.add_argument('--runs', type=int, default=100)
# parser.add_argument('--dataset', type=str, default='CoauthorCS')
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--lr_drop', type=bool, default=True)
# parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--early_stopping', type=int, default=10)
# parser.add_argument('--add_selfloops', type=bool, default=True)
# parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--dropout', type=float, default=0.5)
# parser.add_argument('--normalize_features', type=bool, default=True)
# parser.add_argument('--K', type=int, default=6)
# parser.add_argument('--beta', type=float, default=0.65)#0.65
# parser.add_argument('--entropy_regularization', type=float, default=0.5)
# args = parser.parse_args()

#texas
# parser = argparse.ArgumentParser()
# parser.add_argument('--is_debug', type=bool, default=False)
# parser.add_argument('--runs', type=int, default=100)
# parser.add_argument('--dataset', type=str, default='CoauthorCS')
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--lr_drop', type=bool, default=True)
# parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--early_stopping', type=int, default=10)
# parser.add_argument('--add_selfloops', type=bool, default=True)
# parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--dropout', type=float, default=0.5)
# parser.add_argument('--normalize_features', type=bool, default=True)
# parser.add_argument('--K', type=int, default=2)
# parser.add_argument('--beta', type=float, default=0.96)#0.65
# parser.add_argument('--entropy_regularization', type=float, default=0.5)
# args = parser.parse_args()
# plus CAD--------------------------------------------------------------------------------------------------------------------------------------

class GNNEarly(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNNEarly, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    self.device = device
    time_tensor = torch.tensor([0, self.T]).to(device)
    # self.regularization_fns = ()
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
    # overwrite the test integrator with this custom one
    with torch.no_grad():
      self.odeblock.test_integrator = EarlyStopInt(self.T, self.opt, self.device)
      self.set_solver_data(dataset.data)


    # plus CAD------------------------------------------------
    self.plus = torch.nn.Parameter(torch.tensor(0.5))
    self.adgs = AdaCAD(K=args.K,
                           beta=args.beta,
                           dropout=args.dropout,
                           edge_index=self.odeblock.odefunc.edge_index,
                           inchannel= dataset.num_classes
                           )
    self.adgs1 = AdaCAD(K=args.K,
                           beta=args.beta,
                           dropout=args.dropout,
                           edge_index=self.odeblock.odefunc.edge_index,
                           inchannel= dataset.num_classes
                           )
    self.adgs2 = AdaCAD(K=3,
                           beta=args.beta,
                           dropout=args.dropout,
                           edge_index=self.odeblock.odefunc.edge_index,
                           inchannel= dataset.num_classes
                           )


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # self.adgs.to(device)
    self.data=dataset.data
    
    # plus CAD------------------------------------------------
  def reset_parameters(self):#plus
        
    self.m1.reset_parameters()
    self.m2.reset_parameters()
    self.adgs.reset_parameters()
    self.adgs1.reset_parameters()
    self.adgs2.reset_parameters()

  def set_solver_m2(self):
    self.odeblock.test_integrator.m2_weight = self.m2.weight.data.detach().clone().to(self.device)
    self.odeblock.test_integrator.m2_bias = self.m2.bias.data.detach().clone().to(self.device)
    self.odeblock.test_integrator.adgs = self.adgs
    self.odeblock.test_integrator.adgs1 = self.adgs1
    self.odeblock.test_integrator.adgs2 = self.adgs2

  def set_solver_data(self, data):
    self.odeblock.test_integrator.data = data


  def cleanup(self):
    del self.odeblock.test_integrator.m2
    torch.cuda.empty_cache()

  def calculate_same_class_ratio(self,x):

    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)
    self.odeblock.set_x0(x)
    with torch.no_grad():
      self.set_solver_m2()
    z = self.odeblock(x)
    z = F.relu(z)
    cd = F.softmax(x, dim=-1)
    # Step 2: Compute a transition matrix: transP
    transP, sum_pipj = self.adgs1.compute_transP(cd, self.adgs1.edge_index)#cd
    # Step 3: gamma
    with torch.no_grad():
        from torch_geometric.utils import softmax, sort_edge_index, degree
        deg = degree(self.adgs1.edge_index[0])
        deg[deg==0] = 1
        cont_i = (sum_pipj) / torch.sqrt(deg) 

        gamma = args.beta + (1 - args.beta) * cont_i
        # gamma=F.softmax(gamma)


    #计算邻接节点中同类节点的比率---------------------------------------------------------------------------------------------
    data=self.data.cpu()
    gamma=gamma.cpu()
    from torch_geometric.utils import to_networkx
    import numpy as np
    from scipy.sparse import csr_matrix
    graph_nx=to_networkx(data)
    same_class_ratio_list=[]
    notsame_class_list=[]
    same_class_list=[]
    weight_matrix=csr_matrix((transP.cpu(), (data.edge_index[0], data.edge_index[1])),shape=(data.num_nodes,data.num_nodes))

    for node_idx in range(data.num_nodes):
      node=graph_nx.nodes[node_idx]
      neighbors=graph_nx.adj[node_idx]
      node_label = data.y[node_idx]
      same_class_count = 0
      total_neighbors=0#len(neighbors)
      for neighbors_idx in neighbors:
        neighbors_label=data.y[neighbors_idx]
        weight=weight_matrix[node_idx,neighbors_idx]#(1-gamma[node_idx])+gamma[node_idx]*weight_matrix[node_idx,neighbors_idx]
        if torch.equal(neighbors_label,node_label) :
          same_class_count += weight
          total_neighbors += weight
          same_class_list.append(weight)
        else:
          total_neighbors += weight
          notsame_class_list.append(weight)


      same_class_ratio =same_class_count/total_neighbors
      same_class_ratio_list.append(same_class_ratio)
      # same_class_ratio_list=same_class_ratio_list.cpu()
    print("mean of same_class_list",np.mean(same_class_list))
    print("mean of notsame_class_list",np.mean(notsame_class_list))
    print("mean of same_class_ratio_list",np.mean(same_class_ratio_list))
    print("min of same_class_ratio_list",np.min(same_class_ratio_list))
    print("max of same_class_ratio_list",np.max(same_class_ratio_list))
    print("standard of same_class_ratio_list",np.std(same_class_ratio_list))
    print("variance of same_class_ratio_list",np.var(same_class_ratio_list))
    print("median of same_class_ratio_list",np.median(same_class_ratio_list))
    #----------------------------------------------------------------------------------------------------------------------------


  def forward(self, x, pos_encoding=None):
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
      p = self.mp(p)
      x = torch.cat([x, p], dim=1)
    else:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)
      # x = F.dropout(x, self.opt['dropout'], training=self.training)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # x, ent, debug_tensor = self.adgs1(x,  False)
    # x, ent, debug_tensor = self.adgs(x,  False)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    self.odeblock.set_x0(x)

    with torch.no_grad():
      self.set_solver_m2()

    if self.training  and self.odeblock.nreg > 0:
      z, self.reg_states  = self.odeblock(x)
    else:
      # z=x
      z = self.odeblock(x)
      
    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    # z=F.leaky_relu(z,0.05)
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)
    

    #plus CAD-------------------------------------------------------
    # for i in range(4):
    #   z, ent, debug_tensor = self.adgs1(z, self.odeblock.odefunc.edge_index, self.data.train_mask, False)
    # z, ent, debug_tensor = self.adgs2(z,   False)
    # z, ent, debug_tensor = self.adgs(z,  False)
    # z = F.dropout(z,self.opt['dropout'] , training=self.training)#

    z, ent, debug_tensor = self.adgs1(z,  False)



    #plus CAD-------------------------------------------------------

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)#

    # Decode each node embedding to get node label.
    z = self.m2(z) 
    # z, ent, debug_tensor = self.adgs1(z,  False)
    

    return z

  def forward_encoder(self, x, pos_encoding):
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      x = self.mx(x)
      p = self.mp(pos_encoding)
      x = torch.cat([x, p], dim=1)
    else:
      x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = x + self.m11(F.relu(x))
      x = x + self.m12(F.relu(x))

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    return x

  def forward_ODE(self, x, pos_encoding):
    x = self.forward_encoder(x, pos_encoding)

    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    return z


def main(opt):
  dataset = get_dataset(opt, '../data', False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model, data = GNNEarly(opt, dataset, device).to(device), dataset.data.to(device)
  print(opt)
  # todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_val_acc = test_acc = best_epoch = 0
  best_val_acc_int = best_test_acc_int = best_epoch_int = 0
  for epoch in range(1, opt['epoch']):
    start_time = time.time()
    loss = train(model, optimizer, data)
    train_acc, val_acc, tmp_test_acc = test(model, data)
    val_acc_int = model.odeblock.test_integrator.solver.best_val
    tmp_test_acc_int = model.odeblock.test_integrator.solver.best_test
    # store best stuff inside integrator forward pass
    if val_acc_int > best_val_acc_int:
      best_val_acc_int = val_acc_int
      test_acc_int = tmp_test_acc_int
      best_epoch_int = epoch
    # store best stuff at the end of integrator forward pass
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      test_acc = tmp_test_acc
      best_epoch = epoch
    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(
      log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, tmp_test_acc))
    log = 'Performance inside integrator Val: {:.4f}, Test: {:.4f}'
    print(log.format(val_acc_int, tmp_test_acc_int))
    # print(
    # log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))
  print('best in integrator val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc_int,
                                                                                                test_acc_int,
                                                                                                best_epoch_int))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
  # ODE args
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=1, help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=int, default=100,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument(
    "--adjoint_method", type=str, default="adaptive_heun",
    help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument('--adjoint', dest='adjoint', action='store_true', help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--adjoint_step_size', type=float, default=1, help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')
  # SDE args
  parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
  parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
  parser.add_argument('--adaptive', dest='adaptive', action='store_true', help='use adaptive step sizes')
  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument("--max_nfe", type=int, default=1000, help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true', help="multiply attention scores by edge weights before softmax")
  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  # rewiring args
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001, help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')

  args = parser.parse_args()

  opt = vars(args)

  main(opt)
