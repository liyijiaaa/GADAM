import torch
import torch.nn as nn
import dgl
from dgl.data import register_data_args
import argparse, time
from model import *
from utils import *
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score
from pytorch_memlab import LineProfiler, profile
from sklearn.preprocessing import MinMaxScaler


def train_local(net, graph, feats, opt, args, memorybank_nor,memorybank_abnor,init=True):

    memo = {}
    labels = graph.ndata['label']
    num_nodes=  graph.num_nodes()

    device = args.gpu
    if device >= 0:
        torch.cuda.set_device(device)
        net = net.to(device)
        labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    if init:
        net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []

    #修改,设置异常分数存储
    train_ano_score = torch.zeros((args.local_epochs, num_nodes), dtype=torch.float)


    # 修改
    for epoch in range(args.local_epochs): #local_epochs:100

        net.train()
        if epoch >= 3:
            t0 = time.time()
        opt.zero_grad()
        loss, l1, l2 = net(feats)
        loss.backward()
        opt.step()
        # 记录当前epoch的异常分数
        pos = graph.ndata['pos']
        train_ano_score[epoch] = -pos.detach().view(-1)

        if epoch > 0:
            #动态添加正太池
            _, train_list_temp = train_ano_score[epoch - 1].topk(
                int((epoch / args.local_epochs) ** 2 * num_nodes), dim=0,
                largest=False, sorted=True)
            train_list_temp = train_list_temp.cpu().numpy()
            train_list_temp = train_list_temp.tolist()
            memorybank_nor.append(train_list_temp)

            #动态添加异常池——数量设置的一样
            _, train_list_atemp = train_ano_score[epoch - 1].topk(
                int( (epoch / args.local_epochs) ** 2 * num_nodes), dim=0,
                largest=True, sorted=True)
            train_list_atemp = train_list_atemp.cpu().numpy()
            train_list_atemp = train_list_atemp.tolist()
            memorybank_abnor.append(train_list_atemp)

        if epoch == (args.local_epochs-1):
            # 归一化处理
            train_ano_score = train_ano_score.cpu().detach().numpy()
            scaler = MinMaxScaler()
            train_ano_score = scaler.fit_transform(train_ano_score.T).T
            train_ano_score = torch.DoubleTensor(train_ano_score).cuda()

            #克隆一份
            train_ano_scoreclone = train_ano_score.clone()

            # 计算每个节点在多个epoch中正太池的平均异常得分
            for idx in range(len(memorybank_nor)):
                train_ano_score[idx, memorybank_nor[idx]] = 0
            train_ano_score_nonzero = torch.count_nonzero(train_ano_score, dim=0)
            train_ano_score = torch.sum(train_ano_score, dim=0)
            train_ano_score = train_ano_score / train_ano_score_nonzero
            _, train_list = train_ano_score.topk(int(0.30 * num_nodes), dim=0, largest=False, sorted=True)

            train_list = train_list.cpu().numpy()
            train_list = train_list.tolist()
            nor_idx = train_list

            # 计算每个节点在多个epoch中异常池的平均异常得分
            for idx in range(len(memorybank_abnor)):
                train_ano_scoreclone[idx, memorybank_abnor[idx]] = 0
            abnormal_non_zero_count = torch.count_nonzero(train_ano_scoreclone, dim=0)
            train_ano_scoreclone = torch.sum(train_ano_scoreclone, dim=0)
            train_ano_scoreclone = train_ano_scoreclone / abnormal_non_zero_count
            _, abnormal_indices = train_ano_scoreclone.topk(int(0.05 * num_nodes), dim=0, largest=True, sorted=True)
            abnor_idx = abnormal_indices.cpu().numpy().tolist()


        if epoch >= 3:
            dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(net.state_dict(), 'best_local_model.pkl')
        
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | l1 {:.4f} | l2 {:.4f}"
              .format(epoch+1, np.mean(dur), loss.item(), l1.item(), l2.item()))


    #循环结束后加载最优模型
    memo['graph'] = graph
    net.load_state_dict(torch.load('best_local_model.pkl'))
    h, mean_h = net.encoder(feats)
    h, mean_h = h.detach(), mean_h.detach()
    memo['h'] = h
    memo['mean_h'] = mean_h
    torch.save(memo, 'memo.pth')

    return nor_idx,abnor_idx

def load_info_from_local(local_net,nor_idx,abnor_idx,device):
    if device >= 0:
        torch.cuda.set_device(device)
        local_net = local_net.to(device)

    memo = torch.load('memo.pth')
    local_net.load_state_dict(torch.load('best_local_model.pkl'))
    graph = memo['graph']
    feats = graph.ndata['feat']

    # 将列表转换为张量
    nor_idx = torch.tensor(nor_idx, dtype=torch.long)
    abnor_idx = torch.tensor(abnor_idx, dtype=torch.long)

    # 计算正常节点的中心
    h, _ = local_net.encoder(feats)
    center = h[nor_idx].mean(dim=0).detach()


    if device >= 0:
        memo = {k: v.to(device) for k, v in memo.items()}
        nor_idx = nor_idx.cuda()
        abnor_idx = abnor_idx.cuda()
        center = center.cuda()

    return memo, nor_idx, abnor_idx, center

def gen_edge_attn(emb, edge_index):
        cos = nn.CosineSimilarity(dim=1)
        # 源节点和目标节点的索引
        col, row = edge_index
        #得到两个特征
        f1, f2 = emb[col], emb[row]
        attn_logits = cos(f1, f2)
        return attn_logits

def gen_dgl_graph(index1, index2, edge_w=None, ndata1=None, ndata2=None,ndata3=None):
    g = dgl.graph((index1, index2))
    if ndata1 is not None:
        g.ndata['feat'] = ndata1

    if ndata2 is not None:
        g.ndata['pos'] = ndata2

    if ndata3 is not None:
        g.ndata['label'] = ndata3


    return g

 #全局修改，更新图结构
def update_graph(graph, h):
    # 得到新的隐藏节点的边
    new_edges = top_k_graph_based_on_edge_attn(h, k=5, device=args.gpu)

    # 计算homey矩阵——绝对值
    edge_attn = torch.abs(gen_edge_attn(h, graph.edges()))

    # 过滤边
    threshold = np.percentile(edge_attn.cpu().numpy(), 20)
    filtered_edge = (graph.edges()[0][edge_attn > threshold], graph.edges()[1][edge_attn > threshold])
    new_g = gen_dgl_graph(torch.cat((filtered_edge[0], new_edges[0])),
                          torch.cat((filtered_edge[1], new_edges[1])),
                          ndata1=graph.ndata['feat'],
                          ndata2 = graph.ndata['pos'],
                          ndata3 = graph.ndata['label']).to('cpu')

    new_g = dgl.to_simple(new_g)

    # 关键修改：将 DGL SparseMatrix 转换为 PyTorch 稀疏张量
    adj_sparse = new_g.adj_external(scipy_fmt='coo')
    adj_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([torch.tensor(adj_sparse.row), torch.tensor(adj_sparse.col)]),
        values=torch.tensor(adj_sparse.data),
        size=adj_sparse.shape
    )

    Adj = normalize1(adj_tensor, 'sym', 1) #对称
    new_g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[1], Adj.values(), graph.ndata['feat'].to('cpu'), graph.ndata['pos'].to('cpu'), graph.ndata['label'].to('cpu'))
    new_g = new_g.to(args.gpu)
    return new_g



def train_global(global_net, opt, graph, args):
    epochs = args.global_epochs

    labels = graph.ndata['label'].cpu().numpy()
    num_nodes=  graph.num_nodes()
    device = args.gpu
    feats = graph.ndata['feat']
    pos = graph.ndata['pos']

    if device >= 0:
        torch.cuda.set_device(device)
        global_net = global_net.to(device)
        # labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    init = True
    if init:
        global_net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []

    pred_labels = np.zeros_like(labels)

    for epoch in range(epochs):
        global_net.train()


        if epoch >= 3:
            t0 = time.time()

        opt.zero_grad()
        loss, scores = global_net(feats, epoch)
        loss.backward()
        opt.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(global_net.state_dict(), 'best_global_model.pkl')

        mix_score = -(scores + pos)
        mix_score = mix_score.detach().cpu().numpy()

        mix_auc = roc_auc_score(labels, mix_score)
        
        sorted_idx = np.argsort(mix_score)
        k = int(sum(labels))
        topk_idx = sorted_idx[-k:]
        pred_labels[topk_idx] = 1

        recall_k = recall_score(np.ones(k), labels[topk_idx])
        ap = average_precision_score(labels, mix_score)

        # print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | auc {:.4f} | mix_auc {:.4f}"
        #       .format(epoch+1, np.mean(dur), loss.item(), auc, mix_auc))
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | mix_auc {:.4f} | recall@k {:.4f} | ap {:.4f}"
            .format(epoch+1, np.mean(dur), loss.item(), mix_auc, recall_k, ap))
    
    return mix_auc, recall_k, ap

def main(args):
    seed_everything(args.seed)

    graph = my_load_data(args.data)
    # graph = graph.add_self_loop() test encoder=GCN
    feats = graph.ndata['feat']

    #修改,添加了一个正太池和异常池
    memorybank_nor = []
    memorybank_abnor=[]

    if args.gpu >= 0:
        graph = graph.to(args.gpu)

    in_feats = feats.shape[1]

    #初始化局部分数网络模型
    local_net = LocalModel(graph,
                     in_feats,
                     args.out_dim,
                     nn.PReLU(),)
    #初始化局部分数优化器
    local_opt = torch.optim.Adam(local_net.parameters(), 
                                 lr=args.local_lr, 
                                 weight_decay=args.weight_decay)
    t1 = time.time()

    #修改,将正态池异常池传递给训练函数
    nor_idx, abnor_idx = train_local(local_net, graph, feats, local_opt, args, memorybank_nor,memorybank_abnor)
    

    memo, nor_idx, ano_idx, center = load_info_from_local(local_net,nor_idx,abnor_idx, args.gpu)

    t2 = time.time()
    graph = memo['graph']

    #更新图
    h = memo['h']
    graph = update_graph(graph, h)

    global_net = GlobalModel(graph, 
                             in_feats, 
                             args.out_dim, 
                             nn.PReLU(), 
                             nor_idx, 
                             ano_idx, 
                             center)
    opt = torch.optim.Adam(global_net.parameters(), 
                                 lr=args.global_lr, 
                                 weight_decay=args.weight_decay)
    t3 = time.time()
    
    mix_auc, recall_k, ap = train_global(global_net, opt, graph, args)
    t4 = time.time()

    t_all = t2+t4-t1-t3
    print('mean_t:{:.4f}'.format(t_all / (args.local_epochs + args.global_epochs)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    register_data_args(parser)
    parser.add_argument("--data", type=str, default="Cora",
                        help="dataset")
    parser.add_argument("--seed", type=int, default=717,
                        help="random seed")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--local-lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--global-lr", type=float, default=5e-4,
                        help="learning rate")
    parser.add_argument("--local-epochs", type=int, default=100,
                        help="number of training local model epochs")
    parser.add_argument("--global-epochs", type=int, default=50,
                        help="number of training global model epochs")
    parser.add_argument("--out-dim", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--train-ratio", type=float, default=0.05,
                        help="train ratio")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=True)

    args = parser.parse_args()
    print(args)
    main(args)
    # multi_run(args)

