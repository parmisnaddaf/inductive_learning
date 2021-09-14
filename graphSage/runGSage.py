# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:06:27 2021

@author: amber
"""
import sys
import os
import argparse

import numpy as np
from scipy.sparse import lil_matrix
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl
import timeit

from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv

from src.dataCenter import *
from src.graphSage.models import *
from src.classification import *


#%% GraphSage Model
def train_graphSage(dataCenter, features, args, config, device):
    ds = args.dataSet
    
    # define GraphSage  GraphSage_XEmbed
    graphSage = GraphSage(config['setting.num_layers'], features.size(1), 
                          config['setting.hidden_emb_size'], 
                          features, getattr(dataCenter, ds+'_adj_train'), 
                          device, gcn=args.gcn, agg_func=args.agg_func)
    graphSage.to(device)
    
    
    classification = Classification(config['setting.hidden_emb_size'], 
                                    len(set(getattr(dataCenter, ds+'_labels'))))
    classification.to(device)
    
    unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_train'), 
                                         getattr(dataCenter, ds+'_train'), device)

    if args.learn_method == 'sup':
        print('GraphSage with Supervised Learning')
    elif args.learn_method == 'plus_unsup':
        print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
    else:
        print('GraphSage with Net Unsupervised Learning')
    
    print('\nBegin to train the GraphSAGE:')
    loss = []
    for epoch in range(args.epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)       
        graphSage, classification, avg_loss = apply_model(dataCenter, ds, graphSage, 
                                                classification, unsupervised_loss, 
                                                args.b_sz, 
                                                args.unsup_loss, device, 
                                                args.learn_method, args.lrate)
        loss.append(avg_loss)
        print(graphSage, classification)

        if (epoch == args.epochs-1 or (epoch+1) % 100 == 0) and args.learn_method == 'unsup':
            classification, args.max_vali_f1 = train_classification(dataCenter, 
                                                        graphSage, 
                                                        classification, ds, device, 
                                                        args.max_vali_f1, 
                                                        args.name, epochs = 2)

        if args.learn_method != 'unsup':
            args.max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, 
                                                  device, args.max_vali_f1, 
                                                  args.name, epoch)
            
    return graphSage, classification, loss


''' This function really trains the GraphSAGE'''
def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, 
                b_sz, unsup_loss, device, learn_method, lrate = 0.1):
    test_nodes = getattr(dataCenter, ds+'_test')
    val_nodes = getattr(dataCenter, ds+'_val')
    train_nodes = getattr(dataCenter, ds+'_train')
    labels = getattr(dataCenter, ds+'_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 20
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

    optimizer = torch.optim.SGD(params, lr=lrate)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    avg_loss = 0
    for index in range(batches):
        nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
        print('batch size = ', len(nodes_batch))
        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        print('node size after extending = ', nodes_batch.shape,', Num_negative = ', num_neg)
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
            # unsupervised learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net
        avg_loss += loss.item()
        
        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, 
                                    loss.item(), len(visited_nodes), len(train_nodes)))
		
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
    avg_loss = avg_loss/batches
    print('average loss = ', avg_loss)

    return graphSage, classification, avg_loss

def get_gnn_embeddings(gnn_model, ds, nodes = []):
    print('Loading embeddings from trained GraphSAGE model.')
    #features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
    #nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    if nodes == []:
        nodes = np.arange(gnn_model.get_datasetSize()).tolist()
    
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



def train_classification(dataCenter, graphSage, classification, ds, device, 
                         max_vali_f1, name, epochs=800):
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
   	# train classification, detached from the current graph
   	#classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds+'_train')
    labels = getattr(dataCenter, ds+'_labels')
    features = get_gnn_embeddings(graphSage, ds)
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

def train_classification_individually(features, labels, learning_rate = 0.01, epochs = 200, b_sz = 64):

    # Hyper-parameters 
    input_size = features.shape[1]
    
    num_classes = len(np.unique(labels, return_counts=False))
    y = torch.Tensor(labels).type(torch.LongTensor) 
    x = torch.Tensor(features)
    train_nodes = torch.Tensor(np.arange(len(labels))).type(torch.LongTensor) 

    print('Training Classification ...')
    classification = Classification(input_size, num_classes)
    
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        
        visited_nodes = set()
        avg_loss = 0
        for index in range(batches):
            nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
            visited_nodes |= set(nodes_batch)
            
            labels_batch = y[nodes_batch]
            embs_batch = x[nodes_batch]

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            avg_loss += loss.item()
            '''print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(
                                epoch+1, epochs, index, batches, loss.item(), 
                                len(visited_nodes), len(train_nodes)))'''

            loss.backward()
            
            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
        print('Epoch [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(
                                epoch+1, epochs, avg_loss/batches, 
                                len(visited_nodes), len(train_nodes)))
    
    labels_pred = classification.predict(x)
    result = get_metrices(labels, labels_pred)
    return classification, result


def update_complete_adj(model, dataCenter, ds, device):
    """
    Update dataCenter: train Adjacency matrix -> complete Adjacency matrix
    """
    print('Updating training adjacency matrix with the complete adjacency matrix.')
    adjacency_matrix = getattr(dataCenter, ds+'_adj_matrix')
    
    adj_lists = defaultdict(set)
    for row, col in zip(adjacency_matrix.nonzero()[0], adjacency_matrix.nonzero()[1]):
        adj_lists[row].add(col)
        
    setattr(dataCenter, ds + '_adj_train', adj_lists)
    model.adj_lists = adj_lists
    
def update_train_adj(model, dataCenter, ds, device):
    """
    Update dataCenter: complete Adjacency matrix -> train Adjacency matrix
    """
    print('Updating the complete adjacency matrix with the training adjacency matrix.')
    adjacency_matrix = getattr(dataCenter, ds+'_adj_matrix')
    
    adj_lists = defaultdict(set)
    train_set = getattr(dataCenter, ds + '_train')
    for row, col in zip(adjacency_matrix.nonzero()[0],adjacency_matrix.nonzero()[1]):
        if row in train_set and col in train_set:
            adj_lists[row].add(col)
            
    setattr(dataCenter, ds + '_adj_train', adj_lists)
    model.adj_lists = adj_lists
    