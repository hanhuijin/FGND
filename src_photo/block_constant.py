from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val,get_rw_adj_cad
from utils import get_edge_weight#plus



class ConstantODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1])):
    super(ConstantODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)

    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    if opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                   fill_value=opt['self_loop_weight'],
                                                                   num_nodes=data.num_nodes,
                                                                   dtype=data.x.dtype)
      # self.edge_weight=edge_weight
      # self.m2=torch.nn.Parameter(torch.ones(edge_index.shape[1]).cuda(device='cuda:1')*0.01, requires_grad= True)
      # m2=torch.nn.Parameter(torch.ones(edge_index.shape[1]).cuda(device='cuda:1'), requires_grad= True)
      # edge_index, edge_weight = get_rw_adj_cad(data.x, edge_index)
      # self.tau = torch.nn.Parameter(torch.tensor(5.0))
      # edge_index, edge_weight=get_edge_weight(data,norm_dim=1, fill_value=opt['self_loop_weight'], dtype=data.x.dtype, tau=[5, ] )#plus
    else:
      edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                           fill_value=opt['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=data.x.dtype)
    
    data.edge_index=edge_index.to(device)
    data.edge_weight=edge_weight.to(device)
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)#plus
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    self.device=device
    
  def forward(self, x):
    t = self.t.type_as(x)

    # self.odefunc.edge_weight = self.edge_weight.to(self.device)+self.m2
    # self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_weight

    integrator = self.train_integrator if self.training else self.test_integrator
    
    reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    state = (x,) + reg_states if self.training and self.nreg > 0 else x

    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options=dict(step_size = self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple( st[1] for st in state_dt[1:] )
      return z, reg_states
    else: 
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
