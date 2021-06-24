#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:38:07 2021

@author: parmis
"""


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
    train_edges = []
    val_edges = []
    test_edges = []
    for i in range(len(adj)):
        for j in range(len(adj[0])):
            if adj[i][j] ==1:
                if i in train_nodes and j in train_nodes:
                    adj_train[i][j] = 1
                    train_edges.append([i,j])
                elif i in val_nodes and j in val_nodes:
                    adj_val[i][j] = 1
                    val_edges.append([i,j])
                elif i in test_nodes and j in test_nodes:
                    adj_test[i][j] = 1
                    test_edges.append([i,j])
                    
    test_edges = np.array(test_edges)
    train_edges = np.array(train_edges)
    val_edges = np.array(val_edges)
    
    
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
    
    

    return 
        
    
    
    
    
    

from numpy.random import default_rng
import numpy as np

feat = np.random.rand(1000,100)
adj = np.random.randint(2, size=(1000, 1000))

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, train_true, train_false, ignore_edges_inx, val_edge_idx = make_test_train(
        adj, feat)
