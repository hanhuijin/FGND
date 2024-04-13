import torch
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import softmax, sort_edge_index, degree
from torch_scatter import scatter_add
import math
from torch_sparse import SparseTensor

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class AdaCAD(MessagePassing):
    def __init__(self, K, beta, dropout=0.5, edge_index = None, inchannel=7, **kwargs ):
        super(AdaCAD, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.beta = beta
        self.dropout = dropout

        # plus,添加一层神经网络去预测节点的类别，对于下面的聚合更有意义
        self.m1=torch.nn.Linear(80,inchannel)

        # self.m2 = torch.nn.Linear(edge_index.shape[1],edge_index.shape[1])
        self.m2=torch.nn.Parameter(torch.ones(edge_index.shape[1]).cuda(device='cuda:1')*0.1, requires_grad= True)# torch.nn.Linear(edge_amount,edge_amount)
        # self.m2=torch.nn.Parameter(torch.rand(edge_index.shape[1]).cuda(), requires_grad= True)
        # self.m2_bias = torch.nn.Parameter(torch.zeros(edge_index.shape[1]).cuda(), requires_grad= True)
        self.edge_index=edge_index
        # self.scale_list=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.0/self.K]).cuda(device='cuda:1'), requires_grad= True) for i in range(K)])
        # self.lstm=torch.nn.LSTM(input_size=80, hidden_size=80, num_layers=2, batch_first=True).cuda(device='cuda:1')



    def reset_parameters(self):#plus
        self.m1.reset_parameters()
        self.m2=torch.nn.Parameter(torch.ones(self.edge_index.shape[1]).cuda(device='cuda:1')*0.1, requires_grad= True)# torch.nn.Linear(edge_amount,edge_amount)
        # self.scale_list=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.0/self.K]).cuda(device='cuda:1'), requires_grad= True) for i in range(self.K)])
        # self.lstm=torch.nn.LSTM(input_size=80, hidden_size=7, num_layers=2, batch_first=True).cuda(device='cuda:1')


    def forward(self, x,   is_debug=False):
        # plus,添加一层神经网络去预测节点的类别，对于下面的聚合更有意义
        # x=self.m1(x)

        # Step 1: Class Distribution & Entropy Regularization
        cd = F.softmax(x, dim=-1)
        EPS = 1e-15
        entropy = -(cd * torch.log(cd + EPS)).sum(dim=-1)

        # Step 2: Compute a transition matrix: transP
        transP, sum_pipj = self.compute_transP(cd, self.edge_index)

        # Step 3: gamma
        with torch.no_grad():
            deg = degree(self.edge_index[0])
            deg[deg==0] = 1
            cont_i = sum_pipj / deg

            gamma = self.beta + (1 - self.beta) * cont_i
        gamma=F.softmax(gamma)

        # Step 4: Aggregate features
        x = F.dropout(x, p=self.dropout, training=self.training)
        H = x
        x_list=[H]


        for k in range(self.K):
            x = self.propagate(self.edge_index, x=x, transP=transP)
            x_list.append(x)
        # x_concate=torch.stack(x_list, dim=0)

        # x, (h_n,c_n)=self.lstm(x_concate)

        # # x=x_list[0]*self.scale_list[0]
        # # for k in range(1,self.K):
        # #     x+=x_list[k]*self.scale_list[k]
        # x=x[0]

        x = (1 - gamma.unsqueeze(dim=-1)) * H + gamma.unsqueeze(dim=-1) * x#[-1]
       

        if is_debug:
            debug_tensor = []
            with torch.no_grad():
                debug_tensor.append(sort_edge_index(self.edge_index, transP))
                debug_tensor.append(cd)
                debug_tensor.append(sum_pipj)
                debug_tensor.append(gamma)
        else:
            debug_tensor = None

        return x, entropy, debug_tensor

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
        
        #plus,添加一层神经网络去学习结点间的关系
        # m2=torch.ones(pipj.shape[0],requires_grad=True)
        # m2.to(device='cuda')
        # # m2=torch.nn.Linear(pipj.shape[1], pipj.shape[1])
        # m2=torch.nn.Parameter(m2)
        # m2= torch.nn.Parameter(torch.ones(pipj.shape[0]).cuda(), requires_grad= True)
        
        pipj=self.m2*pipj
        # pipj=self.m2(pipj)
        # transP = torch.nn.functional.normalize(pipj,dim=0)
        with torch.no_grad():
            sum_pipj = scatter_add(pipj, row)
        transP = softmax(pipj, row, num_nodes=cd.size(0))
        

        

        return transP, sum_pipj

    def message(self, x_j, transP):
        return x_j * transP.view(-1, 1)

    def __repr__(self):
        return '{}(K = {}, beta={})'.format(self.__class__.__name__, self.K, self.beta)







