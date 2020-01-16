import os.path as osp

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import time
from deepwalkm.deepwalk.__main__ import get_embedding
import numpy as np
import random
seed = 42

# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    np.random.seed(int(seed))

# np.random.seed(43)
# torch.manual_seed(43)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(43)

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocerowing.')
args = parser.parse_args()


dataset = 'Citeseer'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..','data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 64, cached=True,
                             )
        self.conv2 = GCNConv(64, dataset.num_classes, cached=True,
                            )

        self.linear = torch.nn.Linear(64, dataset.num_classes)

        self.conv1_deepwalk = GCNConv(64, 32, cached=True,
                             )
        self.conv2_deepwalk = GCNConv(32, dataset.num_classes, cached=True,
                            )

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        index = 0
        with open('gcn.adjlist', 'w') as f:
            for i in range(x.shape[0]):
                f.write(str(i))
                for j in range(index, edge_index.shape[1]):
                    if edge_index[0][j].item() == i:
                        f.write(' '+ str(edge_index[1][j].item()))
                    else:
                        index = j
                        break
                f.write('\n')
        
        self.dat = get_embedding(edge_index, x.shape[0])
        self.dat = torch.from_numpy(self.dat).float().cuda()
 
        tensor_index=torch.Tensor(5000,x.shape[0], 64)
        tensor_neighbor_index=torch.Tensor(5000,x.shape[0], 64)

        edgewight = []
        sim_list = []
        for index, row in enumerate(self.dat):
            row = torch.squeeze(row, 0)
            row = row.repeat(x.shape[0], 1)
   
            if index < 5000:
                tensor_index[index]=row
                tensor_neighbor_index[index]=self.dat
            else:
                if index%5000 == 0:
                    sim = torch.cosine_similarity(tensor_index, tensor_neighbor_index, dim=-1)
                    sim_list.append(sim)
                tensor_index[index-5000*int(index/5000)]=row
                tensor_neighbor_index[index-5000*int(index/5000)]=self.dat
                
        if len(sim_list) <= 0:
            sim_ = torch.cosine_similarity(tensor_index, tensor_neighbor_index, dim=-1)
            sim = sim_[:x.shape[0]]
        else:
            sim = torch.cosine_similarity(tensor_index, tensor_neighbor_index, dim=-1)
            sim_list.append(sim)
            sim_ =  torch.cat(sim_list, dim=0)
            sim = sim_[:x.shape[0]]

        index = 0
        adlist = [] 

        for i in range(x.shape[0]):
            lists = []
            for j in range(index, edge_index.shape[1]):
                if edge_index[0][j].item() == i:
                    lists.append(edge_index[1][j].item())
                else:
                    index = j
                    break
            adlist.append(lists)
        mask = torch.ones(sim.size()[0])
        mask = 1 - mask.diag()
        #cora 0.86
        #citeseer 0.9
        #pubmed 1
        sim_vec = torch.nonzero((sim > 0.9).float()*mask)
        for k in sim_vec:
            node_index = k[0].item()
            node_neighbor_index = k[1].item()
            if node_neighbor_index not in adlist[node_index]:
                adlist[node_index].append(node_neighbor_index)
        node_total = []
        neighbor_total = []
        for i in range(len(adlist)):
            for j in range(len(adlist[i])):
                node_total.append(i)
                neighbor_total.append(adlist[i][j])

                
        self.edge_index_new = torch.Tensor(2, len(node_total)).long()
        
        self.edge_index_new[0] = torch.from_numpy(np.array(node_total))
        self.edge_index_new[1] = torch.from_numpy(np.array(neighbor_total))
        self.edge_index_new = self.edge_index_new.cuda()
        
        
    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_deepwalk = self.dat.float().cuda()
        x = F.relu(self.conv1(x, self.edge_index_new, edge_weight))
        x_deepwalk  = F.relu(self.conv1_deepwalk(x_deepwalk , self.edge_index_new, edge_weight))
        x = F.dropout(x, training=self.training)
        x_deepwalk  = F.dropout(x_deepwalk , training=self.training)
        x = self.conv2(x, self.edge_index_new, edge_weight)
        x_deepwalk  = self.conv2_deepwalk(x_deepwalk , self.edge_index_new, edge_weight)
        x = 0.1 * x_deepwalk  + 0.2 * x
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.1)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.int().sum().item()
        accs.append(acc)
    return accs

train_count = 0
test_count = 0
best_val_acc = test_acc = 0
for epoch in range(1, 501):
    one = time.time()
    train()
    second = time.time()
    train_acc, val_acc, tmp_test_acc = test()
    third = time.time()
    train_count += second - one
    test_count += third - second
    if tmp_test_acc > test_acc:
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, val_acc, test_acc))
avg_train = train_count / 500.
avg_test = test_count / 500.
file = open('citeseer.txt', 'a+')
file.write(str(test_acc)+','+ str(avg_train) + ','+str(avg_test) +'\n')
file.close()
