import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix,average_precision_score

import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
from numpy.random import default_rng


def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, cur_epoch):
    train_nodes = getattr(dataCenter, ds+'_train')
    test_nodes = getattr(dataCenter, ds+'_test')
    val_nodes = getattr(dataCenter, ds+'_val')
    labels = getattr(dataCenter, ds+'_labels')
    

    models = [graphSage, classification]

    params = []
    
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    embs = graphSage(val_nodes)
    np.save('embs.npy', embs)
    logists = classification(embs)
    _, predicts = torch.max(logists, 1)
    labels_val = labels[val_nodes]
    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data)

    vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
    print("Validation F1:", vali_f1)

    if vali_f1 > max_vali_f1:
        max_vali_f1 = vali_f1
        embs = graphSage(test_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
        print("Test F1:", test_f1)

        for param in params:
            param.requires_grad = True

        torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))

    for param in params:
        param.requires_grad = True

    return max_vali_f1

def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800):
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
   	# train classification, detached from the current graph
   	#classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds+'_train')
    labels = getattr(dataCenter, ds+'_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            embs_batch = features[nodes_batch]

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))


            loss.backward()
			
            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()

        max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification, max_vali_f1

def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
    test_nodes = getattr(dataCenter, ds+'_test')
    val_nodes = getattr(dataCenter, ds+'_val')
    train_nodes = getattr(dataCenter, ds+'_train')
    labels = getattr(dataCenter, ds+'_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.SGD(params, lr=0.7)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)
        
        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]
        
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        embs_batch = graphSage(nodes_batch)

        if learn_method == 'sup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net
        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
		
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

    return graphSage, classification

###############################################


class node_mlp(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """
    def __init__(self, input, layers= [16, 16], normalize = False, dropout_rate = 0):
        """
        :param input: the feture size of input matrix; Number of the columns
        :param normalize: either use the normalizer layer or not
        :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
        """
        super(node_mlp, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(input, layers[0])])

        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i],layers[i+1]))

        self.norm_layers = None
        if normalize:
            self.norm_layers =  torch.nn.ModuleList([torch.nn.BatchNorm1d(c) for c in [input]+layers])
        self.dropout = torch.nn.Dropout(dropout_rate)
        # self.reset_parameters()

    def forward(self, in_tensor, activation = torch.tanh):
        h = in_tensor
        for i in range(len(self.layers)):
            if self.norm_layers!=None:
                if len(h.shape)==2:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h= h.reshape(-1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h=h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)
            h = activation(h)
        return h

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj, feature):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0
    assert adj.diagonal().sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    index = list(range(train_edges.shape[0]))
    np.random.shuffle(index)
    train_edges_true = train_edges[index[0:num_val]]

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue

        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])
    # print(test_edges_false)
    # print(val_edges_false)
    # print(test_edges)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    ignore_edges_inx = [list(np.array(val_edges_false)[:,0]),list(np.array(val_edges_false)[:,1])]
    ignore_edges_inx[0].extend(val_edges[:,0])
    ignore_edges_inx[1].extend(val_edges[:,1])
    import copy

    val_edge_idx = copy.deepcopy(ignore_edges_inx)
    ignore_edges_inx[0].extend(test_edges[:, 0])
    ignore_edges_inx[1].extend(test_edges[:, 1])
    ignore_edges_inx[0].extend(np.array(test_edges_false)[:, 0])
    ignore_edges_inx[1].extend(np.array(test_edges_false)[:, 1])

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, list(train_edges_true), train_edges_false,ignore_edges_inx, val_edge_idx


# objective Function
def optimizer_VAE(pred, labels, std_z, mean_z, num_nodes, pos_wight, norm):
    val_poterior_cost = 0
    posterior_cost = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_wight)


    z_kl = (-0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))
    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1])
    return z_kl, posterior_cost, acc, val_poterior_cost


def roc_auc_estimator(pos_edges, negative_edges, reconstructed_adj, origianl_agjacency):
    prediction = []
    true_label = []
    for edge in pos_edges:
        prediction.append(reconstructed_adj[edge[0],edge[1]])
        true_label.append(origianl_agjacency[edge[0], edge[1]])

    for edge in negative_edges:
        prediction.append(reconstructed_adj[edge[0], edge[1]])
        true_label.append(origianl_agjacency[edge[0], edge[1]])

    pred = [1 if x>.5 else 0 for x in prediction]
    auc = roc_auc_score(y_score= prediction, y_true= true_label)
    acc = accuracy_score(y_pred= pred, y_true= true_label, normalize= True)
    ap=average_precision_score(y_score= prediction, y_true= true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
    return auc , acc,ap, cof_mtx

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape




def make_test_train(adj, feat):
    num_test = int(np.floor(feat.shape[0] / 10.))
    num_val = int(np.floor(feat.shape[0] / 20.))
    rng = default_rng()
    numbers = rng.choice(feat.shape[0], feat.shape[0], replace=False)
    test_nodes = numbers[:num_test]
    val_nodes = numbers[-num_val:]
    train_nodes = numbers[num_val+ num_test:]
    
    
    feat_test = np.zeros((feat.shape[0], feat.shape[1]))
    feat_val = np.zeros((feat.shape[0], feat.shape[1]))
    feat_train = np.zeros((feat.shape[0], feat.shape[1]))
    for i in test_nodes:
        feat_test[i] = feat[i]
    for i in val_nodes:
        feat_val[i] = feat[i]
    for i in train_nodes:
        feat_train[i] = feat[i]
    

    #create adj
    adj_test = np.zeros((adj.shape[0], adj.shape[1]))
    adj_val = np.zeros((adj.shape[0], adj.shape[1]))
    adj_train = np.zeros((adj.shape[0], adj.shape[1]))
    # train_edges = []
    # val_edges = []
    # test_edges = []
    for i in range(len(adj)):
        if i % 1000 == 0:
            print("i is: ", i)
        for j in range(len(adj[0])):
            if adj[i][j] ==1:
                if i in train_nodes and j in train_nodes:
                    adj_train[i][j] = 1
                    # train_edges.append([i,j])
                elif i in val_nodes and j in val_nodes:
                    adj_val[i][j] = 1
                    # val_edges.append([i,j])
                elif i in test_nodes and j in test_nodes:
                    adj_test[i][j] = 1
                    # test_edges.append([i,j])
                    
    # test_edges = np.array(test_edges)
    # train_edges = np.array(train_edges)
    # val_edges = np.array(val_edges)
    
    
    # index = list(range(train_edges.shape[0]))
    # np.random.shuffle(index)
    # train_edges_true = train_edges[index[0:num_val]]
            
    # test_edges_false = []
    # while len(test_edges_false) < len(test_nodes):
    #     i = np.random.randint(0, adj.shape[0])
    #     j = np.random.randint(0, adj.shape[0])
    #     if i == j:
    #         continue
    #     if adj[i][j] == 1:        
    #         continue
    #     if test_edges_false:
    #         if [i,j] in test_edges_false:
    #             continue
    #     test_edges_false.append([i, j])
        
        
    
    # val_edges_false = []
    # while len(val_edges_false) < len(val_nodes):
    #     i = np.random.randint(0, adj.shape[0])
    #     j = np.random.randint(0, adj.shape[0])
    #     if i == j:
    #         continue
    #     if adj[i][j] == 1:        
    #         continue
    #     if [i,j] in test_edges_false:
    #             continue
    #     if val_edges_false:
    #         if [i,j] in val_edges_false:
    #             continue
    #     val_edges_false.append([i, j])
        
        
    # train_edges_false = []
    # while len(train_edges_false) < len(train_nodes):
    #     i = np.random.randint(0, adj.shape[0])
    #     j = np.random.randint(0, adj.shape[0])
    #     if i == j:
    #         continue
    #     if adj[i][j] == 1:        
    #         continue
    #     if [i,j] in val_edges_false:
    #             continue
    #     if [i,j] in test_edges_false:
    #             continue
    #     if train_edges_false:
    #         if [i,j] in train_edges_false:
    #             continue
    #     train_edges_false.append([i, j])
        
    
    # ignore_edges_inx = [list(np.array(val_edges_false)[:,0]),list(np.array(val_edges_false)[:,1])]
    # ignore_edges_inx[0].extend(val_edges[:,0])
    # ignore_edges_inx[1].extend(val_edges[:,1])
    # import copy

    # val_edge_idx = copy.deepcopy(ignore_edges_inx)
    # ignore_edges_inx[0].extend(test_edges[:, 0])
    # ignore_edges_inx[1].extend(test_edges[:, 1])
    # ignore_edges_inx[0].extend(np.array(test_edges_false)[:, 0])
    # ignore_edges_inx[1].extend(np.array(test_edges_false)[:, 1])
    
    

    return adj_train , adj_val, adj_test, feat_train, feat_val, feat_test
        
    
    
    




