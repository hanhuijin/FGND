import torch
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import softmax, sort_edge_index, degree
from torch_scatter import scatter_add
import math
from torch_sparse import SparseTensor

class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.fc1 = torch.nn.Linear(in_features=64*3*20, out_features=80)
        # self.fc2 = torch.nn.Linear(in_features=256, out_features=80)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, 64*3*20)
        x = self.fc1(x)
        # x = torch.nn.functional.relu(x)
        # x = self.fc2(x)
        return x

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
        self.m2=torch.nn.Parameter(torch.ones(edge_index.shape[1]).cuda(device='cuda:0')*0.01, requires_grad= True)# torch.nn.Linear(edge_amount,edge_amount)
        # self.m2=torch.nn.Parameter(torch.rand(edge_index.shape[1]).cuda(), requires_grad= True)
        # self.m2_bias = torch.nn.Parameter(torch.zeros(edge_index.shape[1]).cuda(), requires_grad= True)
        self.edge_index=edge_index
        self.scale_list=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.0/(self.K)]).cuda(device='cuda:0'), requires_grad= True) for i in range(K)])
        self.lstm=torch.nn.LSTM(input_size=80, hidden_size=80, num_layers=2, batch_first=True).cuda(device='cuda:0')
        self.transformer=torch.nn.Transformer(d_model=80,nhead=8, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=80,dropout=0.1,activation='relu').cuda(device='cuda:0')
        self.cnn=MyCNN().cuda(device='cuda:0')
        self.m_cat=torch.nn.Linear(160,80).cuda(device='cuda:0')



    def reset_parameters(self):#plus
        self.m1.reset_parameters()
        self.m2=torch.nn.Parameter(torch.ones(self.edge_index.shape[1]).cuda(device='cuda:0')*0.01, requires_grad= True)# torch.nn.Linear(edge_amount,edge_amount)
        self.scale_list=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.0/(self.K)]).cuda(device='cuda:0'), requires_grad= True) for i in range(self.K)])
        self.lstm=torch.nn.LSTM(input_size=80, hidden_size=80, num_layers=2, batch_first=True).cuda(device='cuda:0')
        self.transformer=torch.nn.Transformer(d_model=80,nhead=8, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=80,dropout=0.1,activation='relu').cuda(device='cuda:0')
        self.cnn=MyCNN().cuda(device='cuda:0')
        self.m_cat=torch.nn.Linear(160,80).cuda(device='cuda:0')
        # self.bite_of_conti=torch.nn.Parameter(torch.ones(1),requires_grad=True).cuda(device='cuda:0')


    def forward(self, x,   is_debug=False):
        # plus,添加一层神经网络去预测节点的类别，对于下面的聚合更有意义
        # x=self.m1(x)

        # Step 1: Class Distribution & Entropy Regularization
        cd = F.softmax(x, dim=-1)
        EPS = 1e-15
        entropy = -(cd * torch.log(cd + EPS)).sum(dim=-1)

        # Step 2: Compute a transition matrix: transP
        transP, sum_pipj = self.compute_transP(cd, self.edge_index)#cd

        # Step 3: gamma
        with torch.no_grad():
            deg = degree(self.edge_index[0])
            deg[deg==0] = 1
            cont_i = (sum_pipj) / torch.sqrt(deg) 

            gamma = self.beta + (1 - self.beta) * cont_i
        # gamma=F.softmax(gamma)

        # Step 4: Aggregate features
        x = F.dropout(x, p=self.dropout, training=self.training)
        H = x
        x_list=[H]
        # pro_x=0

        # x = self.propagate(self.edge_index, x=x, transP=transP)

        for k in range(self.K):
        #     # x=pro_x+x
            x = self.propagate(self.edge_index, x=x, transP=transP)
        #     x_list.append(x)
        #     # pro_x=x


        # x_concate=torch.stack(x_list, dim=0)
        
        # x=self.cnn(x_concate.transpose(0,1))

        # x_list=self.transformer(x_concate,x_concate)
        # x_list, (h_n,c_n)=self.lstm(x_concate)

        # x=x_list[0]*self.scale_list[0]
        # for k in range(1,self.K):
        #     x+=x_list[k]*self.scale_list[k]

        



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
        # pipj = (p_i * p_j).sum(dim=-1)  # [E, 1]
        pipj=F.cosine_similarity(p_i,p_j,dim=-1)

        
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







