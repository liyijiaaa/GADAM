import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import math
from utils import idx_sample, row_normalization

#添加

import numpy as np



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, activation) -> None:
        super().__init__()
        # self.encoder = nn.ModuleList([
        #     nn.Linear(in_dim, hid_dim),
        #     nn.Dropout(p=dropout),
        #     activation,
        #     nn.Linear(hid_dim, out_dim),
        #     nn.Dropout(p=dropout)
        # ])
        self.encoder = nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            activation,
        ])


    def forward(self, features):
        h = features
        for layer in self.encoder:
            h = layer(h)
        h = F.normalize(h, p=2, dim=1)  # row normalize
        return h
    

class GCN(nn.Module):
    def __init__(
        self, g, in_dim, hid_dim, activation, dropout
    ):
        super(GCN, self).__init__()
        self.g = g
        self.gcn = GraphConv(in_dim, hid_dim, activation=activation)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = self.gcn(self.g, features)
        return self.dropout(h)


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            return graph.ndata['neigh']


class Discriminator(nn.Module):
    def __init__(self, hid_dim) -> None:
        super().__init__()
    
    def forward(self, features, centers):
        # tmp = torch.matmul(features, self.weight)
        # res = torch.sum(tmp * centers, dim=1)
        # return torch.sigmoid(res) 
        return torch.sum(features * centers, dim=1)


class Encoder(nn.Module):
    def __init__(self, graph, in_dim,  out_dim, activation):
        super().__init__()
        self.encoder = MLP(in_dim, out_dim, activation)
        self.encoder2 = GCN(graph, in_dim, out_dim, activation, dropout=0.)
        self.meanAgg = MeanAggregator()
        self.g = graph
        
    def forward(self, h):
        h = self.encoder(h)
        mean_h = self.meanAgg(self.g ,h) #邻居聚合得到子图表示

        return h, mean_h


class LocalModel(nn.Module):
    # LIM module
    def __init__(self, graph, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = Encoder(graph, in_dim, out_dim, activation)
        self.g = graph
        self.discriminator = Discriminator(out_dim)
        self.loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()
    
    def forward(self, h):
        h, mean_h = self.encoder(h)
        
        # positive
        pos = self.discriminator(h, mean_h)
        # negtive
        idx = torch.arange(0, h.shape[0])
        neg_idx = idx_sample(idx)
        neg_neigh_h = mean_h[neg_idx]
        neg = self.discriminator(h, neg_neigh_h)
        
        self.g.ndata['pos'] = pos
        self.g.ndata['neg'] = neg

        l1 = self.loss(pos, torch.ones_like(pos))
        l2 = self.loss(neg, torch.zeros_like(neg))

        return l1 + l2, l1, l2



class GlobalModel(nn.Module):
    def __init__(self, graph, in_dim, out_dim, activation, nor_idx, ano_idx, center):
        super().__init__()
        self.g = graph
        self.discriminator = Discriminator(out_dim)
        self.beta = 0.9
        self.neigh_weight = 1. 
        self.loss = nn.BCEWithLogitsLoss()
        self.nor_idx = nor_idx
        self.ano_idx = ano_idx
        self.center = center # high confidence normal center
        self.encoder = Encoder(graph, in_dim, out_dim, activation)
        self.pre_attn = self.pre_attention()

    def pre_attention(self):
        # calculate pre-attn
        msg_func = lambda edges:{'abs_diff': torch.abs(edges.src['pos'] - edges.dst['pos'])}
        red_func = lambda nodes:{'pos_diff': torch.mean(nodes.mailbox['abs_diff'], dim=1)}
        self.g.update_all(msg_func, red_func)

        #pos = self.g.ndata['pos']
        pos = self.g.ndata['pos'].detach()
        pos.requires_grad = False

        pos_diff = self.g.ndata['pos_diff'].detach()

        diff_mean = pos_diff[self.nor_idx].mean()
        diff_std = torch.sqrt(pos_diff[self.nor_idx].var())

        normalized_pos = (pos_diff - diff_mean) / diff_std
        
        attn = 1-torch.sigmoid(normalized_pos)

        return attn.unsqueeze(1)

    def post_attention(self, h, mean_h):
        # calculate post-attn
        simi = self.discriminator(h, mean_h)
        return simi.unsqueeze(1)


    def msg_pass(self, h, mean_h, attn):
        # h+attn*mean_h
        nei = attn * self.neigh_weight
        h = nei*mean_h + (1-nei)*h
        return h

    def forward(self, feats, epoch):
        h, mean_h = self.encoder(feats)

        post_attn = self.post_attention(h, mean_h)
        beta = math.pow(self.beta, epoch)
        if beta < 0.1:
            beta = 0.
        attn = beta*self.pre_attn + (1-beta)*post_attn

        h = self.msg_pass(h, mean_h, attn)

        scores = self.discriminator(h, self.center)
        
        pos_center_simi = scores[self.nor_idx]
        neg_center_simi = scores[self.ano_idx]
        
        pos_center_loss = self.loss(pos_center_simi, torch.ones_like(pos_center_simi, dtype=torch.float32))
        neg_center_loss = self.loss(neg_center_simi, torch.zeros_like(neg_center_simi, dtype=torch.float32))

        center_loss = pos_center_loss + neg_center_loss

        return center_loss, scores



#修改图更新的节点嵌入

class GPR_ATT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, device):
        super(GPR_ATT, self).__init__()

        self.device = device
        self.inlinear = nn.Linear(in_channels, hidden_channels)
        self.outlinear = nn.Linear(hidden_channels, out_channels)

        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        torch.nn.init.xavier_uniform_(self.outlinear.weight)

        self.gnn = GPR_sparse(hidden_channels, num_layers, dropout, dropout_adj)
        self.extractor = ExtractorMLP(hidden_channels)

    def forward(self, x, g):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, g, edge_attn=True)
            return self.outlinear(h_gnn)

    def gen_node_emb(self, x, g):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, g, edge_attn=True)
            h_gnn = self.extractor.feature_extractor(h_gnn)
            return h_gnn



class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, activation='relu', dropout=0.2):
        super(ExtractorMLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.cos = nn.CosineSimilarity(dim=1)
        self._init_weight(self.feature_extractor)

    @staticmethod
    def _init_weight(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, emb, edge_index):
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        attn_logits = self.cos(self.feature_extractor(f1), self.feature_extractor(f2))
        return attn_logits


class GCNConv_dgl_attn(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl_attn, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        g.ndata['h'] = self.linear(x)
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
        return g.ndata['h']

class GPR_sparse(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, dropout_adj):
        super(GPR_sparse, self).__init__()

        self.layers = nn.ModuleList([GCNConv_dgl_attn(hidden_channels, hidden_channels) for _ in range(num_layers)])
        # GPR temprature initialize
        alpha = 0.1
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(torch.from_numpy(temp))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, g=None, edge_attn=False):
        # if edge_attn:
        #     g.edata['w'] = g.edata['w'] * g.edata['attn']
        # g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        hidden = x * self.temp[0]
        for i, conv in enumerate(self.layers):
            x = conv(x, g)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden += x * self.temp[i + 1]
        return hidden
