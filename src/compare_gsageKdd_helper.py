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

from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv

from dataCenter import *
from utils import *
from models import *
import timeit
#import src.plotter as plotter
#import src.graph_statistics as GS

import classification


#%% GraphSage Model
def train_graphSage(dataCenter, features, args, config, device):
    ds = args.dataSet
    graphSage = GraphSage(config['setting.num_layers'], features.size(1), 
                          config['setting.hidden_emb_size'], 
                          features, getattr(dataCenter, ds+'_adj_lists'), 
                          device, gcn=args.gcn, agg_func=args.agg_func)
    graphSage.to(device)
    
    num_labels = len(set(getattr(dataCenter, ds+'_labels')))
    classification = Classification(config['setting.hidden_emb_size'], num_labels)
    classification.to(device)
    
    unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), 
                                         getattr(dataCenter, ds+'_train'), device)
    if args.learn_method == 'sup':
        print('GraphSage with Supervised Learning')
    elif args.learn_method == 'plus_unsup':
        print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
    else:
        print('GraphSage with Net Unsupervised Learning')
    
    for epoch in range(args.epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        
        graphSage, classification = apply_model(dataCenter, ds, graphSage, 
                                                classification, unsupervised_loss, 
                                                args.b_sz, 
                                                args.unsup_loss, device, 
                                                args.learn_method)
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
            
    return graphSage, classification
   

#%% KDD model
def train_kddModel(dataCenter, features, args, device):
    decoder = args.decoder_type
    encoder = args.encoder_type
    num_of_relations = args.num_of_relations  # diffrent type of relation
    num_of_comunities = args.num_of_comunities  # number of comunities
    batch_norm = args.batch_norm
    DropOut_rate = args.DropOut_rate
    encoder_layers = [int(x) for x in args.encoder_layers.split()]
    split_the_data_to_train_test = args.split_the_data_to_train_test
    epoch_number = args.epoch_number
    negative_sampling_rate = args.negative_sampling_rate
    visulizer_step = args.Vis_step
    PATH = args.mpath
    subgraph_size = args.num_node
    use_feature = args.use_feature
    lr = args.lr
    save_embeddings_to_file = args.save_embeddings_to_file
    
    synthesis_graphs = {"grid", "community", "lobster", "ego"}
    
    ds = args.dataSet
    if ds in synthesis_graphs:
        synthetic = True
    else: 
        synthetic = False
    
    original_adj_full= torch.FloatTensor(getattr(dataCenter, ds+'_adj_lists')).to(device)

    node_label_full= torch.FloatTensor(getattr(dataCenter, ds+'_labels')).to(device)
    
    # if edge labels exist
    edge_labels = None
    if ds == 'IMDB' or ds == 'ACM' or ds == 'DBLP':
        edge_labels= torch.FloatTensor(getattr(dataCenter, ds+'_edge_labels')).to(device)
    circles = None

    
    # shuffling the data, and selecting a subset of it
    if subgraph_size == -1:
        subgraph_size = original_adj_full.shape[0]
    elemnt = min(original_adj_full.shape[0], subgraph_size)
    indexes = list(range(original_adj_full.shape[0]))
    np.random.shuffle(indexes)
    indexes = indexes[:elemnt]
    original_adj = original_adj_full[indexes, :]
    original_adj = original_adj[:, indexes]
    features = features[indexes]
    if synthetic != True:
        if node_label_full != None:
            node_label = [node_label_full[i] for i in indexes]
        if edge_labels != None:
            edge_labels = edge_labels[indexes, :]
            edge_labels = edge_labels[:, indexes]
        if circles != None:
            shuffles_cir = {}
            for ego_node, circule_lists in circles.items():
                shuffles_cir[indexes.index(ego_node)] = [[indexes.index(x) for x in circule_list] for circule_list in
                                                         circule_lists]
            circles = shuffles_cir
    # Check for Encoder and redirect to appropriate function
    if encoder == "Multi_GCN":
        encoder_model = multi_layer_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "mixture_of_GCNs":
        encoder_model = mixture_of_GCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                        latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)
    elif encoder == "mixture_of_GatedGCNs":
        encoder_model = mixture_of_GatedGCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                             latent_dim=num_of_comunities, layers=encoder_layers, dropOutRate=DropOut_rate)
    elif encoder == "Edge_GCN":
        haveedge = True
        encoder_model = edge_enabled_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)
    # asakhuja End
    else:
        raise Exception("Sorry, this Encoder is not Impemented; check the input args")
    
    # Check for Decoder and redirect to appropriate function
    if decoder == "SBM":
        decoder_model = SBM_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "multi_inner_product":
        decoder_model = MapedInnerProductDecoder([32, 32], num_of_relations, num_of_comunities, batch_norm, DropOut_rate)
    
    elif decoder == "MapedInnerProduct_SBM":
        decoder_model = MapedInnerProduct_SBM([32, 32], num_of_relations, num_of_comunities, batch_norm, DropOut_rate)
    
    elif decoder == "TransE":
        decoder_model = TransE_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "TransX":
        decoder_model = TransX_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "SBM_REL":
        haveedge = True
        decoder_model = edge_enabeled_SBM_decoder(num_of_comunities, num_of_relations)
    
    # asakhuja - Start Added Inner Dot product decoder
    elif decoder == "InnerDot":
        decoder_model = InnerProductDecoder()
    # asakhuja End
    else:
        raise Exception("Sorry, this Decoder is not Impemented; check the input args")
    if use_feature == False:
        features = torch.eye(features.shape[0])
        features = sp.csr_matrix(features)
    
    if split_the_data_to_train_test == True:
        trainId = getattr(dataCenter, ds + '_train')
        validId = getattr(dataCenter, ds + '_val')
        testId = getattr(dataCenter, ds + '_test')
        adj_train , adj_val, adj_test, feat_train, feat_val, feat_test= make_test_train_gpu(
                        original_adj.cpu().detach().numpy(), features,
                        [trainId, validId, testId])
        print('Finish spliting dataset to train and test. ')
    
    
    #pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])
    
    adj_train = sp.csr_matrix(adj_train)
    
    #graph_dgl = dgl.DGLGraph()
    #graph_dgl.from_scipy_sparse_matrix(adj_train)
    graph_dgl = dgl.from_scipy(adj_train)

    # origianl_graph_statistics = GS.compute_graph_statistics(np.array(adj_train.todense()) + np.identity(adj_train.shape[0]))
    
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops
    
    num_nodes = graph_dgl.number_of_dst_nodes()
    adj_train = torch.tensor(adj_train.todense())  # use sparse man
    
    if (type(feat_train) == np.ndarray):
        feat_train = torch.tensor(feat_train, dtype=torch.float32)
    else:
        feat_train = feat_train
        
    model = GVAE_FrameWork(num_of_comunities,
                           num_of_relations, encoder=encoder_model,
                           decoder=decoder_model)  # parameter namimng, it should be dimentionality of distriburion
    
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    pos_wight = torch.true_divide((adj_train.shape[0] ** 2 - torch.sum(adj_train)), torch.sum(
        adj_train))  # addrressing imbalance data problem: ratio between positve to negative instance
    norm = torch.true_divide(adj_train.shape[0] * adj_train.shape[0],
                             ((adj_train.shape[0] * adj_train.shape[0] - torch.sum(adj_train)) * 2))
    
    best_recorded_validation = None
    best_epoch = 0
    for epoch in range(epoch_number):
        print(epoch)
        model.train()
        # forward propagation by using all nodes
        std_z, m_z, z, reconstructed_adj = model(graph_dgl, feat_train)
        # compute loss and accuracy
        z_kl, reconstruction_loss, acc, val_recons_loss = optimizer_VAE(reconstructed_adj,
                                                                       adj_train + sp.eye(adj_train.shape[0]).todense(),
                                                                       std_z, m_z, num_nodes, pos_wight, norm)
        loss = reconstruction_loss + z_kl
    
        reconstructed_adj = torch.sigmoid(reconstructed_adj).detach().numpy()
        model.eval()
    
        model.train()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # print some metrics
        print("Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
            epoch + 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc))
    
    # save the log plot on the current directory
    #pltr.save_plot("VGAE_Framework_log_plot")
    model.eval()
    
    return model





def train_nipsModel(dataCenter, features, args, device):
    PATH = args.PATH # the dir to save the with the best performance on validation data
    visulizer_step = args.Vis_step
    device = device
    redraw = args.redraw
    task = args.task
    epoch_number = args.epoch_number
    autoencoder = args.autoencoder
    lr = args.lr
    negative_sampling_rate = args.negative_sampling_rate
    hidden_1 = 128  # ?????????????? naming
    decoder_type = args.decoder
    hidden_2 =  args.num_of_comunities # number of comunities;
    ds = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
    mini_batch_size = args.batchSize
    use_gpu=args.UseGPU
    
    appendX = args.appendX
    use_feature = args.use_feature
    save_embeddings_to_file = args.save_embeddings_to_file
    graph_save_path = args.graph_save_path
    split_the_data_to_train_test = args.split_the_data_to_train_test
    
    kernl_type = []
    
    if args.model == "kernel_tri":
        kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "tri", "square"]
        alpha = [1, 1, 1, 1, 1, 1e-06, 1e-06, 0, 0, .001, .001 ]
        step_num = 5
    # alpha= [1, 1, 1, 1, 1e-06, 1e-06, 0, 0,.001]
    if args.model == "kernel":
        kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist"]
        alpha = [1,1, 1, 1, 1, 1e-06, 1e-06,.001,.001*20]#GRID
        alpha= [10,10, 10, 10, 10, 1e-08*.5, 1e-08*.5,.001,.001] #cora#
        alpha=[24, 24, 24, 24, 24, 5e-09, 5e-09, 0.001, 0.001] #IMDB
        alpha = [10, 10, 10, 10, 10, 5e-09, 5e-09, 0.001, 0.001]#DBLP
        step_num = 5
    if args.model == "kipf":
        alpha= [ .001,.001]
        step_num = 0
    
    if autoencoder==True:
        alpha[-1]=0
    print("kernl_type:"+str(kernl_type))
    print("alpha: "+str(alpha) +" num_step:"+str(step_num))
    
    bin_center = torch.tensor([[x / 10000] for x in range(0, 1000, 1)])
    bin_width = torch.tensor([[9000] for x in range(0, 1000, 1)])# with is propertion to revese of this value;

    # setting the plots legend
    functions= ["Accuracy", "loss", "AUC"]
    functions.extend(["Kernel"+str(i) for i in range(step_num)])
    functions.extend(kernl_type[1:])
    functions.append("Binary_Cross_Entropy")
    functions.append("KL-D")
    
    
    synthesis_graphs = {"grid","small_grid", "community", "lobster", "ego"}
    dataset = ds
    if dataset in synthesis_graphs:
        split_the_data_to_train_test = False
            
    ignore_indexes=None
    node_label = None
    # load the data
    
    original_adj_full= torch.FloatTensor(getattr(dataCenter, ds+'_adj_lists')).to(device)
    node_label = torch.FloatTensor(getattr(dataCenter, ds+'_labels')).to(device)
    
    list_adj = [original_adj_full]
    list_x = [features]
    
    # import input_data
    # if task=="nodeClassification":
    #     list_adj, list_x, node_label, _, _ = input_data.load_data(dataset)
    #     list_adj = [list_adj]
    #     list_x = [list_x]
    # else:
    #     list_adj, list_x, node_label, _, _ = input_data.load_data(dataset)
    #     list_adj = [list_adj]
    #     list_x = [list_x]
    
    
    
    if len(list_adj) == 1 and task=="linkPrediction":
        trainId = getattr(dataCenter, ds + '_train')
        validId = getattr(dataCenter, ds + '_val')
        testId = getattr(dataCenter, ds + '_test')
        adj_train , adj_val, adj_test, feat_train, feat_val, feat_test= make_test_train_gpu(
                        original_adj_full.cpu().detach().numpy(), features,
                        [trainId, validId, testId])
        print('Finish spliting dataset to train and test. ')
        
    
    list_adj = [adj_train]
    print("")
    self_for_none = False
    
    if (decoder_type)in  ("FCdecoder"):#,"FC_InnerDOTdecoder"
        self_for_none = True
    
    if len(list_adj)==1:
        test_list_adj=list_adj.copy()
        list_graphs = Datasets(list_adj, self_for_none, [sparse.csr_matrix(feat_train)])
    
    else:
        list_adj, test_list_adj = data_split(list_adj)
        list_graphs = Datasets(list_adj, self_for_none, None)
    
    
    degree_center = torch.tensor([[x] for x in range(0, list_graphs.max_num_nodes, 1)])
    degree_width = torch.tensor([[.1] for x in range(0, list_graphs.max_num_nodes,1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
    # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
    
    kernel_model = kernel(kernel_type = kernl_type, step_num = step_num,
                bin_width= bin_width, bin_center=bin_center, degree_bin_center=degree_center, degree_bin_width=degree_width)
    # 225#
    in_feature_dim = list_graphs.feature_size # ToDo: consider none Synthasis data
    print("Feature dim is: ", in_feature_dim)
    
    if decoder_type=="SBMdecoder":
        decoder = SBMdecoder_(hidden_2)
    elif decoder_type=="FCdecoder":
        decoder= FCdecoder(list_graphs.max_num_nodes*hidden_2,list_graphs.max_num_nodes**2)
    elif decoder_type == "InnerDOTdecoder":
        decoder = InnerDOTdecoder()
    elif decoder_type == "FC_InnerDOTdecoder":
        decoder = FC_InnerDOTdecoder(list_graphs.max_num_nodes * hidden_2, list_graphs.max_num_nodes *hidden_2, laten_size = hidden_2)
    elif decoder_type=="GRAPHITdecoder":
        decoder = GRAPHITdecoder(hidden_2,25)
    elif decoder_type=="GRAPHdecoder":
        decoder = GRAPHdecoder(hidden_2)
    elif decoder_type=="GRAPHdecoder2":
        decoder = GRAPHdecoder(hidden_2,type="nn",)
    elif decoder_type=="RNNDecoder":
        decoder = RNNDecoder(hidden_2)
    
    model = kernelGVAE(in_feature_dim, hidden_1,  hidden_2,  kernel_model,decoder, device=device, autoencoder=autoencoder) # parameter namimng, it should be dimentionality of distriburion
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    num_nodes = list_graphs.max_num_nodes
    print("NUM NODES IS:  ", num_nodes)
    #ToDo Check the effect of norm and pos weight
    
    
    start = timeit.default_timer()
    # Parameters
    step =0
    for epoch in range(epoch_number):
        # list_graphs.shuffle()
        batch = 0
        for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
            from_ = iter
            to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+1)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)
            org_adj,x_s, node_num = list_graphs.get__(from_, to_, self_for_none)
            if decoder_type == "InnerDOTdecoder": #
                node_num = len(node_num)*[list_graphs.max_num_nodes]
            org_adj = torch.cat(org_adj).to(device)
            x_s = torch.cat(x_s)
            pos_wight = torch.true_divide(sum([x**2 for x in node_num])-org_adj.sum(),org_adj.sum())
            model.train()
            target_kelrnel_val = kernel_model(org_adj, node_num)
            reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val,reconstructed_adj_logit = model(org_adj.to(device), x_s.to(device), node_num)
            kl_loss, reconstruction_loss, acc, kernel_cost,each_kernel_loss = OptimizerVAE(reconstructed_adj, generated_kernel_val, org_adj, target_kelrnel_val, post_log_std, post_mean, num_nodes, alpha,reconstructed_adj_logit, pos_wight, 2,node_num, ignore_indexes)
    
    
            loss = kernel_cost
    
            step+=1
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
            optimizer.step()
    
        
            k_loss_str=""
            for indx,l in enumerate(each_kernel_loss):
                k_loss_str+=functions[indx+3]+":"
                k_loss_str+=str(l)+".   "
    
            print("Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
                epoch + 1,batch,  loss.item(), reconstruction_loss.item(), kl_loss.item(), acc),k_loss_str)
    
    
            batch+=1
    stop = timeit.default_timer()
    print("trainning time:", str(stop-start))
    # torch.save(model, PATH)
        
    return model


    
def print_eval(labels_gtruth, labels_pred, verbose = False):
    labels_test, labels_pred, accuracy, micro_recall, macro_recall, \
        micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, \
        report = classification.get_metrices(labels_gtruth, labels_pred)
    
    if verbose:
        print("******************************************")
        print("Accuracy:{}".format(accuracy),
            "Macro_AvgPrecision:{}".format(macro_precision), 
            "Micro_AvgPrecision:{}".format(micro_precision),
            "Macro_AvgRecall:{}".format(macro_recall), 
            "Micro_AvgRecall:{}".format(micro_recall),
            "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
            "confusion matrix:{}".format(conf_matrix))
    print(report)
