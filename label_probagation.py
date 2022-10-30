import torch
from torch_geometric.data import Data
from gtn_utils import GTN

import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer, GATConv, DenseGCNConv, GCNConv, GraphConv
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


from sklearn.metrics import roc_auc_score
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")

def load_data():

    raw_data = np.load("./Test001.npz")

    print(raw_data['edge_list'].shape)
    edges_list = raw_data['edge_list']
    print(edges_list[:5])

    print(raw_data['position'].shape)
    position = raw_data['position']
    print(position[0])

    labels = raw_data['labels']
    is_train = raw_data['is_train']
    fts = raw_data['fts']

    return edges_list, fts, position, labels, is_train


def label_probagation_GTN(edges_list, fts, position, labels, is_train):
    print(edges_list.shape, fts.shape, position.shape, labels.shape)

    N = len(labels)
    all_edges = set()

    for i in range(len(edges_list)):
        u = edges_list[i][0]
        v = edges_list[i][1]

        if (u != v) and ((u, v) not in all_edges):
            all_edges.add((u, v))

    input_edges = list(all_edges)
    print('Got input edges = ', len(input_edges))
    edge_index = np.array([[a[0] for a in all_edges],
                           [a[1] for a in all_edges]])
    print(edge_index.shape)

    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)
    print(edge_index.shape)

    node_features = fts
    classified_idx = is_train == 1
    unclassified_idx = is_train != 1

    # In[31]:

    node_features = torch.tensor(node_features, dtype=torch.double)

    # In[32]:

    data_train = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                      y=torch.tensor(labels, dtype=torch.double))

    # In[33]:

    y_train = labels[classified_idx]
    print(y_train.shape)

    # spliting train set and validation set
    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(node_features[classified_idx], y_train,
                                                                                np.arange(N)[classified_idx],
                                                                                test_size=0.15, random_state=2022)

    # In[34]:

    data_train.y[classified_idx].sum()

    # In[35]:

    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # In[37]:
    data_train = data_train.to(device)

    num_nodes = len(labels)
    A = []

    edge_tmp = torch.tensor(edge_index, dtype=torch.long).contiguous()
    value_tmp = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

    edge_tmp = edge_tmp.to(device)
    value_tmp = value_tmp.to(device)

    A.append((edge_tmp, value_tmp))
    edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.DoubleTensor)
    A.append((edge_tmp, value_tmp))

    # In[98]:

    gtn = GTN(num_edge=2,
              num_channels=2,
              w_in=2048,
              w_out=1024,
              num_class=1,
              num_layers=2,
              num_nodes=len(labels),
              args=None)

    gtn = gtn.to(device)
    gtn.double()

    # In[100]:

    data_train = data_train.to(device)
    optimizer = torch.optim.Adam(gtn.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.BCELoss()

    gtn.train()
    best_auc = -1
    best_out = None

    for epoch in range(10):
        gtn.zero_grad()
        out = gtn(A, data_train.x)

        # data_train.y.unsqueeze(1)
        out = out.reshape((data_train.x.shape[0]))
        # print(out[:10])
        loss = criterion(out[train_idx], data_train.y[train_idx])

        loss.backward()
        optimizer.step()
        auc = roc_auc_score(data_train.y.detach().cpu().numpy()[train_idx],
                            out.detach().cpu().numpy()[train_idx])  # [train_idx]

        print("epoch: {} - loss: {} - roc: {}".format(epoch, loss.item(), auc))
        if auc > best_auc:
            best_out = out.detach().cpu().numpy()
            best_auc = auc

    return best_out


class GCN(torch.nn.Module):
    def __init__(self, n_fts=2048):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(n_fts, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(64, 64)
        self.conv4 = GCNConv(128, 1)

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv4(x, edge_index)

        return F.sigmoid(x)


def label_probagation_GCN(edges_list, fts, position, labels, is_train):
    N = len(labels)
    all_edges = set()

    for i in range(len(edges_list)):
        u = edges_list[i][0]
        v = edges_list[i][1]

        if (u != v) and ((u, v) not in all_edges):
            all_edges.add((u, v))

    input_edges = list(all_edges)
    print('Got input edges = ', len(input_edges))
    edge_index = np.array([[a[0] for a in all_edges],
                           [a[1] for a in all_edges]])
    print(edge_index.shape)

    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)
    print(edge_index.shape)

    node_features = fts
    classified_idx = is_train == 1
    unclassified_idx = is_train != 1

    # In[31]:

    node_features = torch.tensor(node_features, dtype=torch.double)

    # In[32]:

    data_train = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                      y=torch.tensor(labels, dtype=torch.double))

    y_train = labels[classified_idx]
    print(y_train.shape)

    # spliting train set and validation set
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid, train_idx, valid_idx = \
        train_test_split(node_features[classified_idx], y_train, np.arange(N)[classified_idx], test_size=0.15,
                         random_state=2022)

    # In[33]:

    y_train = labels[classified_idx]
    print(y_train.shape)

    # spliting train set and validation set
    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(node_features[classified_idx], y_train,
                                                                                np.arange(N)[classified_idx],
                                                                                test_size=0.15, random_state=2022)

    # In[34]:

    data_train.y[classified_idx].sum()

    # In[35]:

    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # In[37]:
    data_train = data_train.to(device)

    num_nodes = len(labels)
    A = []

    edge_tmp = torch.tensor(edge_index, dtype=torch.long).contiguous()
    value_tmp = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

    edge_tmp = edge_tmp.to(device)
    value_tmp = value_tmp.to(device)

    A.append((edge_tmp, value_tmp))
    edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.DoubleTensor)
    A.append((edge_tmp, value_tmp))

    # In[98]:

    model = GCN(n_fts=len(fts[0])).to(device)
    model.double()
    # In[100]:

    data_train = data_train.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.BCELoss()

    best_auc = -1
    best_out = None

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(data_train)
        # data_train.y.unsqueeze(1)
        out = out.reshape((data_train.x.shape[0]))
        loss = criterion(out[train_idx], data_train.y[train_idx])
        auc = roc_auc_score(data_train.y.detach().cpu().numpy()[train_idx],
                            out.detach().cpu().numpy()[train_idx])  # [train_idx]
        loss.backward()
        optimizer.step()
        print("epoch: {} - loss: {} - roc: {}".format(epoch, loss.item(), auc))
        if auc > best_auc:
            best_out = out.detach().cpu().numpy()
            best_auc = auc

    return best_out


if __name__ == "__main__":
    edges_list, fts, position, labels, is_train = load_data()
    gtn_out = label_probagation_GTN(edges_list, fts, position, labels, is_train)

    print(gtn_out)

    gcn_out = label_probagation_GCN(edges_list, fts, position, labels, is_train)

    print(gcn_out)


# In[24]:



