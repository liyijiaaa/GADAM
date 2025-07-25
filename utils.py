import dgl
import torch
import torch.nn.functional as F
import random
import os
import dgl.function as fn
from dgl.data.utils import load_graphs
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score

import scipy.sparse as sp
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def t_v_t_split(train_ratio, val_ratio, num_nodes):
    node_idx = np.arange(num_nodes)
    train_num = int(train_ratio * num_nodes)
    val_num = int(val_ratio * num_nodes)

    selected_idx = np.random.choice(node_idx, train_num+val_num, replace=False)

    train_mask = torch.zeros(num_nodes).bool()
    val_mask = torch.zeros(num_nodes).bool()

    train_mask[selected_idx[:train_num]] = True
    val_mask[selected_idx[train_num:]] = True
    test_mask = torch.logical_and(~train_mask, ~val_mask)

    print(torch.sum(train_mask))
    print(torch.sum(val_mask))
    print(torch.sum(test_mask))
    return train_mask, val_mask, test_mask


def idx_sample(idxes):
    num_idx = len(idxes)
    random_add = torch.randint(low=1, high=num_idx,size=(1, ), device='cpu')
    idx = torch.arange(0, num_idx)

    shuffled_idx = torch.remainder(idx+random_add, num_idx)

    return shuffled_idx

def row_normalization(feats):
    return F.normalize(feats, p=2, dim=1)


def load_data(dataname, path='./raw_dataset/Flickr'):
    data = sio.loadmat(f'{path}/{dataname}.mat')

    adj = data['Network'].toarray()
    feats = torch.FloatTensor(data['Attributes'].toarray())
    label = torch.LongTensor(data['Label'].reshape(-1))

    graph = dgl.from_scipy(coo_matrix(adj)).remove_self_loop()
    graph.ndata['feat'] = feats
    graph.ndata['label'] = label

    return graph


def my_load_data(dataname, path='./data/'):
    data_dir = path+dataname+'.bin'
    graph = load_graphs(data_dir)

    return graph[0][0]


def pyg_to_dgl(pyg_graph):
    # Extract the PyG graph components
    edge_index = pyg_graph.edge_index
    edge_attr = pyg_graph.edge_attr
    num_nodes = pyg_graph.num_nodes
    node_attr = pyg_graph.x
    labels = pyg_graph.y
    # Create a DGL graph
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    dgl_graph.ndata['feat'] = node_attr
    dgl_graph.ndata['label'] = labels
    # Set edge attributes if they exist
    if edge_attr is not None:
        dgl_graph.edata['edge_attr'] = torch.tensor(edge_attr)

    return dgl_graph

#自适应邻居采样

def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def adaptive_sampler(num_node, eigen_adj, hop1_adj, hop2_adj, knn_adj, p=None, total_sample_size=20):

    data_list = []
    for id in range(num_node):
        s_ppr = eigen_adj[id]
        s_ppr[id] = -1000.0
        top_neighbor_index = s_ppr.argsort()[-total_sample_size:]  # padding

        s_ppr = eigen_adj[id]
        s_ppr[id] = 0
        s_ppr = np.maximum(s_ppr, 0)
        if p is not None:
            s_hop1 = hop1_adj[id]
            s_hop2 = hop2_adj[id]
            s_knn = knn_adj[id]
            s_hop1[id] = 0
            s_hop2[id] = 0
            s_knn[id] = 0
            s_hop1 = np.maximum(s_hop1, 0)
            s_hop2 = np.maximum(s_hop2, 0)
            s_knn = np.maximum(s_knn, 0)
            s = p[0] * s_ppr / (s_ppr.sum() + 1e-5) + p[1] * s_hop1 / (s_hop1.sum() + 1e-5) + p[2] * s_hop2 / (
                        s_hop2.sum() + 1e-5) + p[3] * s_knn / (s_knn.sum() + 1e-5)

        sampled_num = np.minimum(total_sample_size, (s > 0).sum())
        if sampled_num > 0:
            sampled_index = np.random.choice(a=np.arange(num_node), size=sampled_num, replace=False, p=s / s.sum())
        else:
            sampled_index = np.array([], dtype=int)

        sampled_ids = torch.cat([torch.tensor(sampled_index, dtype=int),
                                 torch.tensor(top_neighbor_index[:(total_sample_size - sampled_num)], dtype=int)])
        data_list.append(sampled_ids)

    return data_list


def get_reward(device, p, ppr_adj, hop1_adj, hop2_adj, knn_adj, num_nodes, sampled_nodes, cost_mat):
    r = [[], [], [], []]

    reward = np.zeros(4)
    with torch.no_grad():
        for id in range(num_nodes):
            sampled_ids = sampled_nodes[id]

            sampled_cost = cost_mat[sampled_ids]
            center_cost = cost_mat[id].unsqueeze(0)
            score_diff = F.softmax(1 / (torch.abs(sampled_cost - center_cost) + 1e-5), dim=0).to(device)

            s_ppr = ppr_adj[id]
            s_hop1 = hop1_adj[id]
            s_hop2 = hop2_adj[id]
            s_knn = knn_adj[id]
            s_ppr[id], s_hop1[id], s_hop2[id], s_knn[id] = 0, 0, 0, 0
            s_ppr = torch.tensor(np.maximum(s_ppr, 0)).to(device)
            s_ppr = s_ppr / (s_ppr.sum() + 1e-5)
            s_hop1 = torch.tensor(np.maximum(s_hop1, 0)).to(device)
            s_hop1 = s_hop1 / (s_hop1.sum() + 1e-5)
            s_hop2 = torch.tensor(np.maximum(s_hop2, 0)).to(device)
            s_hop2 = s_hop2 / (s_hop2.sum() + 1e-5)
            s_knn = torch.tensor(np.maximum(s_knn, 0)).to(device)
            s_knn = s_knn / (s_knn.sum() + 1e-5)
            phi = p[0] * s_ppr + p[1] * s_hop1 + p[2] * s_hop2 + p[3] * s_knn + 1e-5

            r[0].append(p[0] * s_ppr[sampled_ids] * score_diff / phi[sampled_ids])
            r[1].append(p[1] * s_hop1[sampled_ids] * score_diff / phi[sampled_ids])
            r[2].append(p[2] * s_hop2[sampled_ids] * score_diff / phi[sampled_ids])
            r[3].append(p[3] * s_knn[sampled_ids] * score_diff / phi[sampled_ids])

        reward[0] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[0]])).cpu().numpy()
        reward[1] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[1]])).cpu().numpy()
        reward[2] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[2]])).cpu().numpy()
        reward[3] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[3]])).cpu().numpy()
    return reward