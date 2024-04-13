import pickle as pkl
import sys
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch
from torch_geometric.datasets import  Amazon

from normalization import fetch_normalization, row_normalize

datadir = "data"

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def load_citation(dataset_str="cora", normalization="AugNormAdj", porting_to_torch=True,data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    if dataset_str == "Computers" or dataset_str == "Photo":            #重写部分
        num_development = 1500
        num_per_class = 20
        data = Amazon(data_path, dataset_str)
        rnd_state = np.random.RandomState(1)
        num_nodes = data.y.shape[0]
        print(num_nodes)
        development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
        test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

        train_idx = []
        rnd_state = np.random.RandomState(1)
        for c in range(data.num_classes):
            class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

        val_idx = [i for i in development_idx if i not in train_idx]



        # adj = data.edge_index      ??咋获取啊
        #
        # labels = data.y
        features = data.x
        # features = sp.vstack(features).tolil()

        # labels = sp.vstack(data.y)
        labels = data.y

        edge_index = data[0].edge_index

        import torch
        from scipy.sparse import coo_matrix

        def edge_index_to_adj(edge_index, num_nodes=None):
            # edge_index: [2, E] 的张量
            if num_nodes is None:
                num_nodes = data[0].num_nodes  # int(torch.max(edge_index)) + 1

            # 转换为COO格式的稀疏矩阵
            indices = torch.stack((edge_index[0], edge_index[1])).t()
            values = torch.ones(indices.shape[0])
            shape = (num_nodes, num_nodes)

            # 创建一个稀疏张量
            adj = torch.sparse_coo_tensor(indices.t(), values, torch.Size(shape))

            return adj

        # edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 1, 2]])
        adj = edge_index_to_adj(data[0].edge_index, data[0].num_nodes)

        # 将稀疏张量转换为密集的张量
        adj = adj.to_dense()

        # 使用scipy.sparse.coo_matrix将numpy数组转换为COO格式的稀疏矩阵
        adj = coo_matrix(adj.numpy())

        # edge_index_np = edge_index.numpy()
        # weights = np.ones(edge_index.shape[1])
        # n_nodes = edge_index.max().item() + 1
        # adj = coo_matrix((weights, edge_index_np), shape=(n_nodes, n_nodes)).tocsr()
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        degree = np.sum(adj, axis=1)

        learning_type = "transductive"
        # porting to pytorch
        if porting_to_torch:
            features = torch.FloatTensor(features).float()
            labels = torch.LongTensor(labels)
            # labels = torch.max(labels, dim=1)[1]
            adj = sparse_mx_to_torch_sparse_tensor(adj).float()
            train_idx = torch.LongTensor(train_idx)
            val_idx = torch.LongTensor(val_idx)
            test_idx= torch.LongTensor(test_idx)
            degree = torch.LongTensor(degree)
        learning_type = "transductive"
        return adj, features, labels, train_idx, val_idx, test_idx, degree, learning_type







    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        G = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(G)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # degree = np.asarray(G.degree)
        degree = np.sum(adj, axis=1)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        if task_type == "full":
            print("Load full supervised task.")
            #supervised setting
            """
            idx_test = test_idx_range.tolist()
            idx_train = range(len(ally)- 500)
            idx_val = range(len(ally) - 500, len(ally))
            """
            idx_test = test_idx_range.tolist()
            num_nodes = ally.shape[0]
            print(num_nodes)
            print(np.max(ally))
            y_forindex=np.argmax(ally,axis=1)
            def set_train_val_test_split(
                num_development: int = 1500,
                num_per_class: int = 20, seed=1) :

                rnd_state = np.random.RandomState(seed)
                num_nodes = ally.shape[0]
                development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
                test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

                train_idx = []
                rnd_state = np.random.RandomState(seed)
                for c in range(ally.shape[1]):
                    class_idx = development_idx[np.where(y_forindex[development_idx] == c)[0]]
                    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))#True

                val_idx = [i for i in development_idx if i not in train_idx]
                return train_idx,val_idx,test_idx
            idx_train,idx_test,idx_val=set_train_val_test_split()
            # idx_trian = []
            # rnd_state = np.random.RandomState(42)
            # y_forindex=ally.index(1)
            # for c in range(ally.shape[1]):
            #     idx_class = num_nodes[np.where(y_forindex[num_nodes - 1] == c)[0]]
            #     idx_train.extend(rnd_state.choice(idx_class, 20, replace=False))
            # idx_val = [i for i in num_nodes if i not in idx_trian]
        elif task_type == "semi":
            print("Load semi-supervised task.")
            #semi-supervised setting
            idx_test = test_idx_range.tolist()
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+500)
        else:
            raise ValueError("Task type: %s is not supported. Available option: full and semi.")

        adj, features = preprocess_citation(adj, features, normalization)
        features = np.array(features.todense())
        labels = np.argmax(labels, axis=1)
        # porting to pytorch
        if porting_to_torch:
            features = torch.FloatTensor(features).float()
            labels = torch.LongTensor(labels)
            # labels = torch.max(labels, dim=1)[1]
            adj = sparse_mx_to_torch_sparse_tensor(adj).float()
            idx_train = torch.LongTensor(idx_train)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            degree = torch.LongTensor(degree)
        learning_type = "transductive"
        return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

def sgc_precompute(features, adj, degree):
    #t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = 0 #perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir=datadir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir +"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def load_reddit_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    degree = np.sum(train_adj, axis=1)

    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    train_features = torch.index_select(features, 0, torch.LongTensor(train_index))
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "inductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type



    
def data_loader(dataset, data_path=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type = "full"):
    if dataset == "reddit":
        return load_reddit_data(normalization, porting_to_torch, data_path)
    else:
        (adj,
         features,
         labels,
         idx_train,
         idx_val,
         idx_test,
         degree,
         learning_type) = load_citation(dataset, normalization, porting_to_torch, data_path, task_type)
        # (adj,
        #  features,
        #  labels,
        #  idx_train,
        #  idx_val,
        #  idx_test,
        #  degree,
        #  learning_type) = Amazon(data_path, dataset)
        train_adj = adj
        train_features = features
        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type

