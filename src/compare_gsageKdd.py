# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:17:19 2021

@author: amber
"""
import sys
import os
import argparse

import numpy as np
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl

from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv

from dataCenter import *
from utils import *
from models import *
import plotter as plotter
import graph_statistics as GS
import compare_gsageKdd_helper as helper
import classification


#%%  arg setup
parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')


parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--dataSet', type=str, default='ACM')
parser.add_argument('--agg_func', type=str, default='MAX')
parser.add_argument('--b_sz', type=int, default=400)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='unsup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='experiments.conf')

args_graphsage = parser.parse_args()
print("")
print("graphSage SETING: "+str(args_graphsage))
##################################################################


parser = argparse.ArgumentParser(description='Inductive KDD')


parser.add_argument('-e', dest="epoch_number", default=5, help="Number of Epochs")
parser.add_argument('--model', type=str, default='KDD')
parser.add_argument('--dataSet', type=str, default='ACM')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('-num_node', dest="num_node", default=-1, type=str,
                    help="the size of subgraph which is sampled; -1 means use the whule graph")
parser.add_argument('--config', type=str, default='/Users/parmis/Desktop/parmis-thesis/related-work/codes/graphSAGE-pytorch-master/src/experiments.conf')
parser.add_argument('-decoder_type', dest="decoder_type", default="multi_inner_product",
                    help="the decoder type, Either SBM or InnerDot  or TransE or MapedInnerProduct_SBM or multi_inner_product and TransX or SBM_REL")
parser.add_argument('-encoder_type', dest="encoder_type", default="Multi_GCN",
                    help="the encoder type, Either ,mixture_of_GCNs, mixture_of_GatedGCNs , Multi_GCN or Edge_GCN ")
parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-NofRels', dest="num_of_relations", default=2,
                    help="Number of latent or known relation; number of deltas in SBM")
parser.add_argument('-NofCom', dest="num_of_comunities", default=128,
                    help="Number of comunites, tor latent space dimention; len(z)")
parser.add_argument('-BN', dest="batch_norm", default=True,
                    help="either use batch norm at decoder; only apply in multi relational decoders")
parser.add_argument('-DR', dest="DropOut_rate", default=.3, help="drop out rate")
parser.add_argument('-encoder_layers', dest="encoder_layers", default="64", type=str,
                    help="a list in which each element determine the size of gcn; Note: the last layer size is determine with -NofCom")
parser.add_argument('-lr', dest="lr", default=0.005, help="model learning rate")
parser.add_argument('-NSR', dest="negative_sampling_rate", default=1,
                    help="the rate of negative samples which should be used in each epoch; by default negative sampling wont use")
parser.add_argument('-v', dest="Vis_step", default=50, help="model learning rate")
parser.add_argument('-modelpath', dest="mpath", default="VGAE_FrameWork_MODEL", type=str,
                    help="The pass to save the learned model")
parser.add_argument('-Split', dest="split_the_data_to_train_test", default=True,
                    help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-s', dest="save_embeddings_to_file", default=True, help="save the latent vector of nodes")

args_kdd = parser.parse_args()

print("")
print("KDD SETING: "+str(args_kdd))



###########################################################################
parser = argparse.ArgumentParser(description='Inductive nips')

parser.add_argument('-e', dest="epoch_number" , default=5, help="Number of Epochs")
parser.add_argument('-dataset', dest="dataset", default="ACM", help="possible choices are:  grid, community, citeseer, lobster, DD")#citeceer: ego
parser.add_argument('-v', dest="Vis_step", default=100, help="model learning rate")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-lr', dest="lr", default=0.001, help="model learning rate") # for RNN decoder use 0.0001
parser.add_argument('-NSR', dest="negative_sampling_rate", default=1, help="the rate of negative samples which shold be used in each epoch; by default negative sampling wont use")
parser.add_argument('-NofCom', dest="num_of_comunities", default=16, help="Number of comunites")
parser.add_argument('-s', dest="save_embeddings_to_file", default=False, help="save the latent vector of nodes")
parser.add_argument('-graph_save_path', dest="graph_save_path", default="develope/", help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature" , default=True, help="either use features or identity matrix")
parser.add_argument('-Split', dest="split_the_data_to_train_test" , default=True, help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-PATH', dest="PATH" , default="model", help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder" , default="FC_InnerDOTdecoder", help="the decoder type,SBMdecoder, FC_InnerDOTdecoder, GRAPHdecoder,FCdecoder,InnerDOTdecoder")
parser.add_argument('-batchSize', dest="batchSize" , default=100, help="the size of each batch")
parser.add_argument('-device', dest="device" , default="cuda:0", help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model" , default="kernel", help="kipf or kernel")
parser.add_argument('-UseGPU', dest="UseGPU" , default=True, help="either use GPU or not if availabel")
parser.add_argument('-task', dest="task" , default="linkPrediction", help="nodeClassification, graphGeneration")
parser.add_argument('-autoencoder', dest="autoencoder" , default=True, help="nodeClassification, graphGeneration")
parser.add_argument('-appendX', dest="appendX" , default=False, help="doese append x to Z for nodeclassification")
args_nips = parser.parse_args()

print("")
print("NIPS SETING: "+str(args_nips))

pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])

if torch.cuda.is_available():
	device_id = torch.cuda.current_device()
	print('using device', device_id, torch.cuda.get_device_name(device_id))
else:
    device_id = 'CPU'

device = torch.device(device_id)
print('DEVICE:', device)


#%% load config

random.seed(args_graphsage.seed)
np.random.seed(args_graphsage.seed)
torch.manual_seed(args_graphsage.seed)
torch.cuda.manual_seed_all(args_graphsage.seed)

# load config file
config = pyhocon.ConfigFactory.parse_file(args_graphsage.config)

#%% load data
ds = args_graphsage.dataSet
if ds == 'cora':
    dataCenter_sage = DataCenter(config)
    dataCenter_sage.load_dataSet(ds, "graphSage")
    features_sage = torch.FloatTensor(getattr(dataCenter_sage, ds+'_feats')).to(device)
    
    dataCenter_kdd = DataCenter(config)
    dataCenter_kdd.load_dataSet(ds, "KDD")
    features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats')).to(device)
elif ds == 'IMDB' or ds == 'ACM'or ds == 'DBLP':
    dataCenter_kdd = DataCenter(config)
    dataCenter_kdd.load_dataSet(ds, "KDD")
    features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats')).to(device)

    dataCenter_sage = datasetConvert(dataCenter_kdd, ds)
    features_sage = features_kdd


#%% train graphSAGE and KDD model

# print(features_sage)

# zeros = np.zeros((features_kdd.shape[0], 128),  dtype= np.float32)
# x = np.concatenate((features_kdd, zeros), axis=1).astype(np.float32)
# features_kdd = torch.from_numpy(x)

# train graphsage
from models import *
graphSage, classification_sage = helper.train_graphSage(dataCenter_sage, 
                                        features_sage,args_graphsage,
                                        config, device)

#%%  train inductive_kdd
inductive_kdd = helper.train_kddModel(dataCenter_kdd, features_kdd, 
                                      args_kdd, device)




#%%  train inductive_nips
inductive_nips = helper.train_nipsModel(dataCenter_kdd, features_kdd, 
                                      args_nips, device)



#%% get embedding of GraphSAGE


embedding_sage = get_gnn_embeddings(graphSage, dataCenter_sage, ds)

#%% get embedding of KDD

# GET TRAIN EMBEDDINGS
features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats'))
trainId = getattr(dataCenter_kdd, ds + '_train')
labels = getattr(dataCenter_kdd, ds + '_labels')
adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))
adj_list_train = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))[trainId]
adj_list_train = adj_list_train[:, trainId]
graph_dgl = dgl.from_scipy(adj_list_train)
graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
std_z, m_z, z, reconstructed_adj = inductive_kdd(graph_dgl, features_kdd[trainId])
embedding_kdd_train = z.detach().numpy()


# features_kdd = features_kdd.cpu().detach().numpy()
# for i , idd in enumerate(trainId):
#     features_kdd[idd][-128:] = embedding_kdd_train[i]
# features_kdd = torch.from_numpy(features_kdd)


features_kdd = features_kdd.cpu().detach().numpy()
for i , idd in enumerate(trainId):
    features_kdd[idd] = np.pad(embedding_kdd_train[i], (0, features_kdd.shape[1] - embedding_kdd_train.shape[1]), 'constant', constant_values=(0,0))
features_kdd = torch.from_numpy(features_kdd)


# GET ALL EMBEDDINGS
adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))
graph_dgl = dgl.from_scipy(adj_list)
graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
std_z, m_z, z, reconstructed_adj = inductive_kdd(graph_dgl, features_kdd)
embedding_kdd = z.detach().numpy()





#%% get embedding of NIPS

self_for_none = False
full_list = getattr(dataCenter_kdd, ds+'_adj_lists')
trainId = getattr(dataCenter_kdd, ds + '_train')
features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats'))


# GET TRAIN EMBEDDINGS

full_list_train = full_list[trainId]
full_list_train = full_list_train[:, trainId]
num_zeros = len(full_list) - len(full_list_train)
mask_zeros_rows = np.zeros((num_zeros, len(full_list_train)))
full_list_train = np.concatenate((full_list_train, mask_zeros_rows), axis=0)
mask_zeros_cols = np.zeros((len(full_list_train), num_zeros))
full_list_train = np.concatenate((full_list_train, mask_zeros_cols), axis=1)


features_kdd_train = features_kdd[trainId]
zeros_rows = np.zeros((num_zeros, features_kdd.shape[1]))
features_kdd_train = np.concatenate((features_kdd_train, zeros_rows), axis=0)

list_graphs_full_train = Datasets([full_list_train], self_for_none, [sparse.csr_matrix(features_kdd_train)])
org_adj,x_s, node_num = list_graphs_full_train.get__(0, len(list_graphs_full_train.list_adjs), self_for_none)
org_adj = torch.cat(org_adj).to(device)
x_s = torch.cat(x_s)
reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val,reconstructed_adj_logit = inductive_nips(org_adj.to(device), x_s.to(device), node_num)
embedding_nips_train_full = np.concatenate((prior_samples[0].cpu().detach().numpy(),x_s.detach().numpy()[0]),axis=1)

embedding_nips_train = prior_samples[0].cpu().detach().numpy()


# GET ALL EMBEDDINGS

# features_kdd = features_kdd.cpu().detach().numpy()
# for i , idd in enumerate(trainId):
#     features_kdd[idd][-16:] = embedding_nips_train[i]
# features_kdd = torch.from_numpy(features_kdd)


features_kdd = features_kdd.cpu().detach().numpy()
for i , idd in enumerate(trainId):
    features_kdd[idd] = np.pad(embedding_nips_train[i], (0, features_kdd.shape[1] - embedding_nips_train.shape[1]), 'constant', constant_values=(0,0))
features_kdd = torch.from_numpy(features_kdd)


list_graphs_full = Datasets([full_list], self_for_none, [sparse.csr_matrix(features_kdd)])
org_adj,x_s, node_num = list_graphs_full.get__(0, len(list_graphs_full.list_adjs), self_for_none)
org_adj = torch.cat(org_adj).to(device)
x_s = torch.cat(x_s)
reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val,reconstructed_adj_logit = inductive_nips(org_adj.to(device), x_s.to(device), node_num)
embedding_nips = np.concatenate((prior_samples[0].cpu().detach().numpy(),x_s.detach().numpy()[0]),axis=1)


#%% train classification/prediction model - NN
trainId = getattr(dataCenter_kdd, ds + '_train')
labels = getattr(dataCenter_kdd, ds + '_labels')

res_train_sage, classifier_sage = classification.NN_all(embedding_sage[trainId, :].cpu().detach().numpy(), 
                                                             labels[trainId])

res_train_kdd, classifier_kdd = classification.NN_all(embedding_kdd_train, 
                                                           labels[trainId])

res_train_nips, classifier_nips = classification.NN_all(embedding_nips_train_full[trainId, :], 
                                                           labels[trainId])

#%% evaluate on whole dataset


# ********************** TRAIN SET
print('\n# ****************** TRAIN SET Z_tr0 ******************')
print('#  GraphSAGE')
print(res_train_sage[-1])
print('#  KDD Model')
print(res_train_kdd[-1])
print('#  NIPS Model')
print(res_train_nips[-1])






labels_pred_sage = classifier_sage.predict(torch.Tensor(embedding_sage.cpu().detach().numpy()))
labels_pred_kdd = classifier_kdd.predict(torch.Tensor(embedding_kdd))
labels_pred_nips = classifier_nips.predict(torch.Tensor(embedding_nips))

#************************ TRAIN SET
print('\n# ****************** TRAIN SET Z_tr ******************')
print('#  GraphSAGE')
helper.print_eval(labels[trainId], labels_pred_sage[trainId])
print('#  KDD Model')
helper.print_eval(labels[trainId], labels_pred_kdd[trainId])
print('#  NIPS Model')
helper.print_eval(labels[trainId], labels_pred_nips[trainId])




# ********************** TEST SET
print('\n# ****************** TEST SET ******************')
testId = [i for i in range(len(labels)) if i not in trainId]
print('#  GraphSAGE')
helper.print_eval(labels[testId], labels_pred_sage[testId])
print('#  KDD Model')
helper.print_eval(labels[testId], labels_pred_kdd[testId])
print('#  NIPS Model')
helper.print_eval(labels[testId], labels_pred_nips[testId])


# ********************** WHOLE SET
print('\n# ****************** WHOLE SET ******************')
print('#  GraphSAGE')
helper.print_eval(labels, labels_pred_sage)
print('#  KDD Model')
helper.print_eval(labels, labels_pred_kdd)
print('#  NIPS Model')
helper.print_eval(labels, labels_pred_nips)

