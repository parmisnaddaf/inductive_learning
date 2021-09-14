# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:17:19 2021

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

from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv

from src.dataCenter import *
from src.utils import *
from src.graphSage.models import *
import src.graphSage.runGSage as runGSage
import src.plotter as plotter
import src.graph_statistics as GS
import src.compare_gsageKdd_helper as helper
from src import classification


#%%  arg setup
parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')


parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MAX')
parser.add_argument('--b_sz', type=int, default=128)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='unsup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--lrate', type=float, default=0.1)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='src/experiments.conf')

args_graphsage = parser.parse_args()
print("")
print("graphSage SETING: "+str(args_graphsage))

pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])
if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cpu")
# print('DEVICE:', device)


#%% load config

random.seed(args_graphsage.seed)
np.random.seed(args_graphsage.seed)
torch.manual_seed(args_graphsage.seed)
torch.cuda.manual_seed_all(args_graphsage.seed)

# load config file
config = pyhocon.ConfigFactory.parse_file(args_graphsage.config)

#%% load data
ds = args_graphsage.dataSet
if ds == 'cora'or ds == 'IMDB'or ds == 'ACM':
    dataCenter_sage = DataCenter(config)
    dataCenter_sage.load_dataSet(ds, "graphSage")
    features_sage = torch.FloatTensor(getattr(dataCenter_sage, ds+'_feats')).to(device)
    
elif ds == 'DBLP':
    dataCenter_kdd = DataCenter(config)
    dataCenter_kdd.load_dataSet(ds, "KDD")
    features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats')).to(device)

    dataCenter_sage = datasetConvert(dataCenter_kdd, ds)
    features_sage = features_kdd


#%% train graphSAGE 

# train graphsage
from models import *
graphSage, classification_sage, loss = runGSage.train_graphSage(dataCenter_sage, 
                                        features_sage, args_graphsage,
                                        config, device)


#%% get embedding
embedding_type = 'replace'

trainId = getattr(dataCenter_sage, ds + '_train')
testValId = [i for i in range(len(features_sage)) if i not in trainId]
labels = getattr(dataCenter_sage, ds + '_labels')

# get training embedding
Z_sage_tr0 = runGSage.get_gnn_embeddings(graphSage, ds)[trainId].cpu().data

# load the complete adjacency matrix
runGSage.update_complete_adj(graphSage, dataCenter_sage, ds, device)

#%% get embedding of all nodes
if embedding_type == 'replace':
  features_new = getattr(dataCenter_sage, ds + '_feats')
  
  feat_ztr0 = np.zeros((len(trainId), len(features_new[0])))
  feat_ztr0[:, :len(Z_sage_tr0[0])] = Z_sage_tr0
  features_new[trainId, :] = feat_ztr0

  graphSage.reset_features(torch.FloatTensor(features_new).to(device))

Z_sage = runGSage.get_gnn_embeddings(graphSage, ds).cpu().data
Z_sage_tr = Z_sage[trainId]
z_sage_te = Z_sage[testValId]

print((Z_sage_tr != Z_sage_tr0).sum(axis = 0))
print(((Z_sage_tr != Z_sage_tr0).sum(axis = 1)!= 0).sum(), 'out of ',len(Z_sage_tr))

#%% train Classifier based on training embedding (Z_tr^0)

# train classification/prediction model - NN
if args_graphsage.learn_method == 'unsup':
    res_train_sage, classifier_sage = classification.NN_all(Z_sage_tr0, labels[trainId])
    #res_train_sage, classifier_sage = classification.logistic_regression_all(Z_sage_tr0, labels[trainId])
elif args_graphsage.learn_method == 'sup':
    classifier_sage = classification_sage.to('cpu')
    
    labels_pred_ztr0 = classifier_sage.predict(torch.Tensor(Z_sage_tr0))
    res_train_sage = classification.get_metrices(labels[trainId], labels_pred_ztr0.detach().numpy())


#%% evaluate on whole dataset

# ********************** TRAIN SET
print('\n# ****************** TRAIN SET ******************')
print('#  GraphSAGE')
print(res_train_sage[-1])

labels_pred_sage = classifier_sage.predict(torch.Tensor(Z_sage))

# ********************** TRAIN SET
print('\n# ****************** TRAIN SET ******************')
print('#  GraphSAGE')
helper.print_eval(labels[trainId], labels_pred_sage[trainId])

# ********************** TEST SET
print('\n# ****************** TEST SET ******************')
print('#  GraphSAGE')
helper.print_eval(labels[testValId], labels_pred_sage[testValId])

#%% train classifier based on Z_tr

# train on Z_tr, then try again
if args_graphsage.learn_method == 'unsup':
    print('Unsupervised learning ...')
    res_train_sage, classifier_sage = classification.NN_all(Z_sage_tr, labels[trainId])
    #res_train_sage, classifier_sage = classification.logistic_regression_all(Z_sage_tr,labels[trainId])
elif args_graphsage.learn_method == 'sup':
    classifier_sage, results = runGSage.train_classification_individually(Z_sage_tr, 
                                                    labels[trainId], learning_rate = 0.1, 
                                                    epochs = 300, b_sz = 64)


#%% ********************** TRAIN SET

print('\n# ****************** TRAIN SET ******************')
print('#  GraphSAGE')
helper.print_eval(labels[trainId], classifier_sage.predict(torch.Tensor(Z_sage_tr0)))

labels_pred_sage = classifier_sage.predict(torch.Tensor(Z_sage))

# ********************** TRAIN SET
print('\n# ****************** TRAIN SET ******************')
print('#  GraphSAGE')
helper.print_eval(labels[trainId], labels_pred_sage[trainId])

# ********************** TEST SET
print('\n# ****************** TEST SET ******************')
print('#  GraphSAGE')
helper.print_eval(labels[testValId], labels_pred_sage[testValId])



