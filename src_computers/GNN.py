import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function

# plus CAD--------------------------------------------------------------------------------------------------------------------------------------
import argparse
import torch
import torch.nn.functional as F
import time
from torch_geometric.utils import add_self_loops

from AdaCAD_cora import AdaCAD
# import sys
# sys.path.append(r"/home/hhj/file/graph-neural-pde-main/src")
# from AdaCAD_cora_for_grand_cad import AdaCAD


#cora citeseer
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
# parser.add_argument('--K', type=int, default=3)#13
# parser.add_argument('--beta', type=float, default=0.15)
# parser.add_argument('--entropy_regularization', type=float, default=0.5)
# args = parser.parse_args()


#Pubmed
parser = argparse.ArgumentParser()
parser.add_argument('--is_debug', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--dataset', type=str, default='Pubmed')
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
parser.add_argument('--beta', type=float, default=0.75)#0.85
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
# plus CAD--------------------------------------------------------------------------------------------------------------------------------------

# Define the GNN model.
class GNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

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

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)
    
    z, ent, debug_tensor = self.adgs1(z,  False)
    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z
