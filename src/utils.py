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
from scipy.sparse import lil_matrix


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
    #np.save('embs.npy', embs)
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

        # torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))

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
        super().__init__()
        #super(node_mlp, self).__init__()
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

import copy
def make_test_train_gpu(adj, feat, split = []):
    if len(split) == 0:
        num_test = int(np.floor(feat.shape[0] / 3.))
        num_val = int(np.floor(feat.shape[0] / 6.))
        rng = default_rng()
        numbers = rng.choice(feat.shape[0], feat.shape[0], replace=False)
        test_nodes = numbers[:num_test]
        val_nodes = numbers[-num_val:]
        train_nodes = numbers[num_val+ num_test:]
    else:
        train_nodes = split[0]
        val_nodes = split[1]
        test_nodes = split[2]
    
    feat_train = np.zeros((feat.shape[0], feat.shape[1]))
    feat_val = []
    feat_test = []
    for i in train_nodes:
        feat_train[i] = feat[i].cpu().data.numpy()
    

    #create adj
    adj_test = []
    adj_val = []
    adj_train = np.zeros((adj.shape[0], adj.shape[1]))
    
    adj_train[train_nodes, :] = adj[train_nodes, :]
    adj_train[:, train_nodes] = adj[:, train_nodes]
    
    #adj_train[:, val_nodes] = 0

    return adj_train , adj_val, adj_test, feat_train, feat_val, feat_test

def make_test_train(adj, feat, split = []):
    if len(split) == 0:
        num_test = int(np.floor(feat.shape[0] / 3.))
        num_val = int(np.floor(feat.shape[0] / 6.))
        rng = default_rng()
        numbers = rng.choice(feat.shape[0], feat.shape[0], replace=False)
        test_nodes = numbers[:num_test]
        val_nodes = numbers[-num_val:]
        train_nodes = numbers[num_val+ num_test:]
    else:
        train_nodes = split[0]
        val_nodes = split[1]
        test_nodes = split[2]
    
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
        
####################### NIPS UTILS ##########################



def test_(number_of_samples, model ,graph_size,max_size, path_to_save_g, device, remove_self=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, max_size, model.latent_dim]))
            z = torch.randn_like(z)
            start_time = time.time()
            if type(model.decode) == GRAPHITdecoder:
                pass
                # adj_logit = model.decode(z.float(), features)
            elif type(model.decode) == RNNDecoder:
                adj_logit = model.decode(z.to(device).float(), [g_size])
            elif type(model.decode) in (FCdecoder, FC_InnerDOTdecoder):
                g_size = max_size
                z = torch.tensor(numpy.random.normal(size=[1, max_size, model.latent_dim]))
                z = torch.randn_like(z)
                adj_logit = model.decode(z.to(device).float())
            else:
                adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            sample_graph = sample_graph[:g_size,:g_size]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_matrix(sample_graph)
            # generated_graph_list.append(G)
            f_name = path_to_save_g+ str(g_size)+ str(j) + dataset
            # plot and save the generated graph
            plotter.plotG(G, "generated" + dataset, file_name=f_name)
            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)
            plotter.plotG(G, "generated" + dataset, file_name=f_name+"_ConnectedComponnents")
    return generated_graph_list

            # save to pickle file



def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, num_nodes, alpha, reconstructed_adj_logit, pos_wight, norm,node_num, ignore_indexes=None ):
    if ignore_indexes ==None:
        loss = norm*torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), targert_adj.float(),pos_weight=pos_wight)
    else:
        loss = norm*torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), targert_adj.float(),pos_weight=pos_wight,
                                                                   reduction='none')
        loss[0][ignore_indexes[1], ignore_indexes[0]] = 0
        loss = loss.mean()
    norm =    mean.shape[0] * mean.shape[1] * mean.shape[2]
    kl = (1/norm)* -0.5 * torch.sum(1+2*log_std - mean.pow(2)-torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum()/float(reconstructed_adj.shape[0]*reconstructed_adj.shape[1]*reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []
    for i in range(len(target_kernel_val)):
        l = torch.nn.MSELoss()
        step_loss = l(reconstructed_kernel_val[i].float(), target_kernel_val[i].float())
        each_kernel_loss.append(step_loss.cpu().detach().numpy()*alpha[i])
        kernel_diff += l(reconstructed_kernel_val[i].float(), target_kernel_val[i].float())*alpha[i]
    each_kernel_loss.append(loss.cpu().detach().numpy()*alpha[-2])
    each_kernel_loss.append(kl.cpu().detach().numpy()*alpha[-1])
    kernel_diff += loss*alpha[-2]
    kernel_diff += kl * alpha[-1]
    return kl , loss, acc, kernel_diff, each_kernel_loss

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

    


class Datasets():
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_adjs,self_for_none, list_Xs, padding =True, Max_num = None):
        """
        :param list_adjs: a list of adjacency in sparse format
        :param list_Xs: a list of node feature matrix
        """
        'Initialization'
        self.paading = padding
        self.list_Xs = list_Xs
        self.list_adjs = list_adjs
        self.toatl_num_of_edges = 0
        self.max_num_nodes = 0
        for i, adj in enumerate(list_adjs):
            list_adjs[i] =  adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            list_adjs[i] += sp.eye(list_adjs[i].shape[0])
            if self.max_num_nodes < adj.shape[0]:
                self.max_num_nodes = adj.shape[0]
            self.toatl_num_of_edges += adj.sum().item()
            # if list_Xs!=None:
            #     self.list_adjs[i], list_Xs[i] = self.permute(list_adjs[i], list_Xs[i])
            # else:
            #     self.list_adjs[i], _ = self.permute(list_adjs[i], None)
        if Max_num!=None:
            self.max_num_nodes = Max_num
        self.processed_Xs = []
        self.processed_adjs = []
        self.num_of_edges = []
        for i in range(self.__len__()):
            a,x,n = self.process(i,self_for_none)
            self.processed_Xs.append(x)
            self.processed_adjs.append(a)
            self.num_of_edges.append(n)
        if list_Xs!=None:
            self.feature_size = list_Xs[0].shape[1]
        else:
            self.feature_size = self.max_num_nodes

  def get(self, shuffle= True):
      indexces = list(range(self.__len__()))
      random.shuffle()
      return [self.processed_adjs[i] for i in indexces], [self.processed_Xs[i] for i in indexces]

  def get__(self,from_, to_, self_for_none):
      adj_s = []
      x_s = []
      num_nodes = []
      # padded_to = max([self.list_adjs[i].shape[1] for i in range(from_, to_)])
      # padded_to = 225
      for i in range(from_, to_):
          adj, x, num_node = self.process(i, self_for_none)#, padded_to)
          adj_s.append(adj)
          x_s.append(x)
          num_nodes.append(num_node)
      return adj_s, x_s, num_nodes


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_adjs)

  def process(self,index,self_for_none, padded_to=None,):

      num_nodes = self.list_adjs[index].shape[0]
      if self.paading == True:
          max_num_nodes = self.max_num_nodes if padded_to==None else padded_to
      else:
          max_num_nodes = num_nodes
      adj_padded = lil_matrix((max_num_nodes,max_num_nodes))  # make the size equal to maximum graph
      adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index][:, :]
      adj_padded -= sp.dia_matrix((adj_padded.diagonal()[np.newaxis, :], [0]), shape=adj_padded.shape)
      if self_for_none:
        adj_padded += sp.eye(max_num_nodes)
      else:
        adj_padded[:num_nodes, :num_nodes] += sp.eye(num_nodes)
      # adj_padded+= sp.eye(max_num_nodes)




      if self.list_Xs == None:
          # if the feature is not exist we use identical matrix
          X = np.identity( max_num_nodes)

      else:
          #ToDo: deal with data with diffrent number of nodes
          X = self.list_Xs[index].toarray()

      # adj_padded, X = self.permute(adj_padded, X)

      # Converting sparse matrix to sparse tensor
      coo = adj_padded.tocoo()
      values = coo.data
      indices = np.vstack((coo.row, coo.col))
      i = torch.LongTensor(indices)
      v = torch.FloatTensor(values)
      shape = coo.shape
      adj_padded = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
      X = torch.tensor(X, dtype=torch.float32)

      return adj_padded.reshape(1,*adj_padded.shape), X.reshape(1, *X.shape), num_nodes

  def permute(self, list_adj, X):
            p = list(range(list_adj.shape[0]))
            np.random.shuffle(p)
            # for i in range(list_adj.shape[0]):
            #     list_adj[:, i] = list_adj[p, i]
            #     X[:, i] = X[p, i]
            # for i in range(list_adj.shape[0]):
            #     list_adj[i, :] = list_adj[i, p]
            #     X[i, :] = X[i, p]
            list_adj[:, :] = list_adj[p, :]
            list_adj[:, :] = list_adj[:, p]
            if X !=None:
                X[:, :] = X[p, :]
                X[:, :] = X[:, p]
            return list_adj , X

  def shuffle(self):
      indx = list(range(len(self.list_adjs)))
      np.random.shuffle(indx)
      if  self.list_Xs !=None:
        self.list_Xs=[self.list_Xs[i] for i in indx]
      self.list_adjs=[self.list_adjs[i] for i in indx]
  def __getitem__(self, index):
        'Generates one sample of data'
        # return self.processed_adjs[index], self.processed_Xs[index],torch.tensor(self.list_adjs[index].todense(), dtype=torch.float32)
        return self.processed_adjs[index], self.processed_Xs[index]




