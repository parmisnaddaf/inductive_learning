#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:17:00 2021

@author: parmis
"""
import numpy as np
from scipy.sparse import lil_matrix
import pickle
import argparse
import random
import torch
import plotter
import pyhocon
import dgl
from dgl.nn.pytorch import GraphConv as GraphConv
from dataCenter import *
from utils import *
from models import *
from scipy import sparse
import graph_statistics as GS
import torch.nn.functional as F
import classification



parser = argparse.ArgumentParser(description='Inductive Interface')

parser.add_argument('--model', type=str, default='KDD')
parser.add_argument('--dataSet', type=str, default='IMDB')
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
parser.add_argument('-NofCom', dest="num_of_comunities", default=64,
                    help="Number of comunites, tor latent space dimention; len(z)")
parser.add_argument('-BN', dest="batch_norm", default=True,
                    help="either use batch norm at decoder; only apply in multi relational decoders")
parser.add_argument('-DR', dest="DropOut_rate", default=.3, help="drop out rate")
parser.add_argument('-encoder_layers', dest="encoder_layers", default="64", type=str,
                    help="a list in which each element determine the size of gcn; Note: the last layer size is determine with -NofCom")
parser.add_argument('-lr', dest="lr", default=0.001, help="model learning rate")
parser.add_argument('-e', dest="epoch_number", default=100, help="Number of Epochs")
parser.add_argument('-NSR', dest="negative_sampling_rate", default=1,
                    help="the rate of negative samples which should be used in each epoch; by default negative sampling wont use")
parser.add_argument('-v', dest="Vis_step", default=50, help="model learning rate")
parser.add_argument('-modelpath', dest="mpath", default="VGAE_FrameWork_MODEL", type=str,
                    help="The pass to save the learned model")
parser.add_argument('-Split', dest="split_the_data_to_train_test", default=True,
                    help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-s', dest="save_embeddings_to_file", default=True, help="save the latent vector of nodes")

args = parser.parse_args()
pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cpu")


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)
    
    # load data
    ds = args.dataSet
    model_name = args.model
    dataCenter = DataCenter(config)
    dataCenter.load_dataSet(ds, model_name)
    features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')).to(device)
    
    if model_name == "KDD":
        
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
        
        if ds in synthesis_graphs:
            synthetic = True
        else: 
            synthetic = False
        
              

        original_adj_full= torch.FloatTensor(getattr(dataCenter, ds+'_adj_lists')).to(device)

        node_label_full= torch.FloatTensor(getattr(dataCenter, ds+'_labels')).to(device)
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
            
            adj_train , adj_val, adj_test, feat_train, feat_val, feat_test= make_test_train(original_adj.cpu().detach().numpy(), features)
        
        
        pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])
        
        adj_train = sp.csr_matrix(adj_train)
        
        graph_dgl = dgl.DGLGraph()

        graph_dgl.from_scipy_sparse_matrix(adj_train)
        

        origianl_graph_statistics = GS.compute_graph_statistics(np.array(adj_train.todense()) + np.identity(adj_train.shape[0]))
        
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
        
        optimizer = torch.optim.Adam(model.parameters(), lr)
        
        pos_wight = torch.true_divide((adj_train.shape[0] ** 2 - torch.sum(adj_train)), torch.sum(
            adj_train))  # addrressing imbalance data problem: ratio between positve to negative instance
        norm = torch.true_divide(adj_train.shape[0] * adj_train.shape[0],
                                 ((adj_train.shape[0] * adj_train.shape[0] - torch.sum(adj_train)) * 2))
        
        best_recorded_validation = None
        best_epoch = 0
        for epoch in range(epoch_number):
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
        
            # Ploting the recinstructed Graph
            if epoch % visulizer_step == 0:
                pltr.redraw()
        
            if epoch % visulizer_step == 0 and ds in synthesis_graphs:
                reconstructed_adj[reconstructed_adj >= 0.5] = 1
                reconstructed_adj[reconstructed_adj < 0.5] = 0
                G = nx.from_numpy_matrix(reconstructed_adj)
                plotter.plotG(G, "generated" + ds)
        
            # print some metrics
            print("Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
                epoch + 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc))
        
        # save the log plot on the current directory
        pltr.save_plot("VGAE_Framework_log_plot")
        model.eval()
        
        # #Loading the best model
        # if dataset not in synthesis_graphs and split_the_data_to_train_test == True:
        #     model = torch.load(PATH)
        
        # print("the best Elbow on validation is " + str(best_recorded_validation) + " at epoch " + str(best_epoch))
        
        # # Link Prediction Task
        # print("=====================================")
        # print("Result on Link Prediction Task")
        # _, post_mean, z, re_adj = model(graph_dgl, features)
        # re_adj = torch.sigmoid(re_adj)
        # # save the node embedding
        # print("=====================================")
        # if (save_embeddings_to_file):  # save the embedding on the current directory   
        #     np.savetxt(ds + "_embeddings", post_mean.detach().numpy())

    
        # Node Classification Task
        if node_label != None:
            # DBLP Node Label Fix 1.0
            if min(node_label) != 0:
                for i in range(len(node_label)):
                    node_label[i] -= 1


        adj_train = sp.csr_matrix(adj_train)
        train_nodes = np.where(adj_train.toarray().any(axis=1))[0]
        z_train = z[train_nodes]
        label_train = node_label_full[train_nodes]
        print("=====================================")
        print("Result on Node Classification Task")
        print("results for NN:")
        (labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report), NN = classification.NN(
            z_train.detach().numpy(), label_train)
        print("Accuracy:{}".format(accuracy),
              "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
              "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
              "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
              "confusion matrix:{}".format(conf_matrix))
        print(report)
        print("******************************************")
        print("results for KNN:")
        (labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report) , KNN = classification.knn(
            z_train.detach().numpy(), label_train)
        print("Accuracy:{}".format(accuracy),
              "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
              "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
              "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
              "confusion matrix:{}".format(conf_matrix))
        print(report)
        print("******************************************")
        print("results for logistic regression:")
        (labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report) , LR = classification.logistiic_regression(
            z_train.detach().numpy(), label_train)
        print("Accuracy:{}".format(accuracy),
              "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
              "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
              "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
              "confusion matrix:{}".format(conf_matrix))
        print(report)
        


    
    
        # Node Classification Task
        if node_label_full != None:
            # DBLP Node Label Fix 1.0
            if min(node_label_full) != 0:
                for i in range(len(node_label_full)):
                    node_label_full[i] -= 1
                        
            
        print("RESULTS ON FULL DS")
            
        graph_dgl = dgl.DGLGraph(sparse.csr_matrix(original_adj_full.cpu().detach().numpy()))
        # asakhuja Changing Following line coz of depracted function - Starts
        graph_dgl.from_scipy_sparse_matrix(sparse.csr_matrix(original_adj_full.cpu().detach().numpy()))
        # dgl.from_scipy(adj_train)
        # asakhuja Changing Following line coz of depracted function - Ends
        origianl_graph_statistics = GS.compute_graph_statistics(np.array(sparse.csr_matrix(original_adj_full).todense()) + np.identity(original_adj_full.shape[0]))
        
        graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops
        std_z, m_z, z, reconstructed_adj = model(graph_dgl, features)

        labels_pred = LR.predict(z.detach().numpy())
        # max returns (value ,index)
        labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = classification.get_metrices(node_label_full, labels_pred)
        print("******************************************")
        print("results for logistic regression:")
        print("Accuracy:{}".format(accuracy),
            "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
            "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
            "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
            "confusion matrix:{}".format(conf_matrix))
        print(report)
        
        labels_pred = KNN.predict(z.detach().numpy())
        # max returns (value ,index)
        labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = classification.get_metrices(node_label_full, labels_pred)
        print("******************************************")
        print("results for KNN:")
        print("Accuracy:{}".format(accuracy),
            "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
            "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
            "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
            "confusion matrix:{}".format(conf_matrix))
        print(report)
         
    
 
    
