import sys
import os

from collections import defaultdict
from scipy.sparse import lil_matrix
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy import sparse

import copy

class DataCenter(object):
    """docstring for DataCenter"""
    def __init__(self, config):
        #super(DataCenter, self).__init__()
        super().__init__()
        self.config = config
        self.test_split = 0.3
        self.val_split = 0.2
        
        
    def _load_Cora(self, model_name):
        dataSet='cora'
        
        cora_content_file = self.config['file_path.cora_content']
        cora_cite_file = self.config['file_path.cora_cite']
        
        feat_data = []
        labels = [] # label sequence of node
        node_map = {} # map node to Node_ID
        label_map = {} # map label to Label_ID
        with open(cora_content_file) as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                feat_data.append([float(x) for x in info[1:-1]])
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels.append(label_map[info[-1]])                
        feat_data = np.asarray(feat_data)
        labels = np.asarray(labels, dtype=np.int64)
        
        # get adjacency matrix                
        adjacency_matrix = lil_matrix((len(feat_data), len(feat_data)))
        with open(cora_cite_file) as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                assert len(info) == 2
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adjacency_matrix[paper1, paper2] = 1
                adjacency_matrix[paper2, paper1] = 1
                
        assert len(feat_data) == len(labels) == adjacency_matrix.shape[0]
        
        # split dataset to train, val, test
        test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0],
                                                            self.test_split, self.val_split)
            
        if model_name == 'graphSage':
            # get train adj in dictionary type
            adj_lists = defaultdict(set)
            train_set = set(train_indexs)
            for row, col in zip(adjacency_matrix.nonzero()[0],
                                    adjacency_matrix.nonzero()[1]):
                if row in train_set and col in train_set:
                    adj_lists[row].add(col)
            setattr(self, dataSet + '_adj_train', adj_lists)


        setattr(self, dataSet+'_test', test_indexs)
        setattr(self, dataSet+'_val', val_indexs)
        setattr(self, dataSet+'_train', train_indexs)

        setattr(self, dataSet+'_feats', feat_data)
        setattr(self, dataSet+'_labels', labels)
        setattr(self, dataSet+'_adj_matrix', adjacency_matrix.toarray())
        
    def _load_Imdb(self, model_name):
        dataSet = "IMDB"
        
        obj = []                
        adj_file_name = self.config['file_path.imdb_edges']
        
        with open(adj_file_name, 'rb') as f:
            obj.append(pkl.load(f))
    
        # merging diffrent edge type into a single adj matrix
        adjacency_matrix = lil_matrix(obj[0][0].shape)
        for matrix in obj[0]:
            adjacency_matrix +=matrix
    
        matrix = obj[0]
        edge_labels = matrix[0] + matrix[1]
        edge_labels += (matrix[2] + matrix[3])*2
    
        node_label= []
        in_1 = matrix[0].indices.min()
        in_2 = matrix[0].indices.max()+1
        in_3 = matrix[2].indices.max()+1
        node_label.extend([0 for i in range(in_1)])
        node_label.extend([1 for i in range(in_1,in_2)])
        node_label.extend([2 for i in range(in_2, in_3)])
    
        obj = []                
        feat_file_name = self.config['file_path.imdb_feats']
        with open(feat_file_name, 'rb') as f:
            obj.append(pkl.load(f))
        feature = sp.csr_matrix(obj[0])
        
        index = -1
        test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0],
                                                        self.test_split, self.val_split)
        
        if model_name == 'graphSage':
            # get train adj in dictionary type
            adj_lists = defaultdict(set)
            train_set = set(train_indexs)
            for row, col in zip(adjacency_matrix.nonzero()[0],
                                    adjacency_matrix.nonzero()[1]):
                if row in train_set and col in train_set:
                    adj_lists[row].add(col)
            setattr(self, dataSet + '_adj_train', adj_lists)
              
        setattr(self, dataSet+'_test', test_indexs)
        setattr(self, dataSet+'_val', val_indexs)
        setattr(self, dataSet+'_train', train_indexs)

        setattr(self, dataSet+'_feats', feature[:index].toarray())
        setattr(self, dataSet+'_labels', np.array(node_label[:index]))
        setattr(self, dataSet+'_adj_matrix', adjacency_matrix[:index,:index].toarray())
        setattr(self, dataSet+'_edge_labels', edge_labels[:index].toarray())
        
    
    def _load_Acm(self, model_name):
        dataSet = 'ACM'
        
        obj = []
        adj_file_name = self.config['file_path.acm_edges']
        with open(adj_file_name, 'rb') as f:
                obj.append(pkl.load(f))
                
        adjacency_matrix = sp.csr_matrix(obj[0][0].shape)
        for matrix in obj:
            nnz = matrix[0].nonzero() # indices of nonzero values
            for i, j in zip(nnz[0], nnz[1]):
                adjacency_matrix[i,j] = 1
                adjacency_matrix[j,i] = 1
            #adj +=matrix[0]
        
        # to fix the bug on running GraphSAGE
        """
        adjacency_matrix = adjacency_matrix.toarray()
        for i in range(len(adjacency_matrix)):
            if sum(adjacency_matrix[i, :]) == 0:
                idx = np.random.randint(0, len(adjacency_matrix))
                adjacency_matrix[i,idx] = 1
                adjacency_matrix[idx,i] = 1
        """
    
        edge_labels = matrix[0] + matrix[1]
        edge_labels += (matrix[2] + matrix[3])*2
    
        node_label= []
        in_1 = matrix[0].indices.min()
        in_2 = matrix[0].indices.max()+1
        in_3 = matrix[2].indices.max()+1
        node_label.extend([0 for i in range(in_1)])
        node_label.extend([1 for i in range(in_1,in_2)])
        node_label.extend([2 for i in range(in_2, in_3)])
    
        obj = []
        feat_file_name = self.config['file_path.acm_feats']
        with open(feat_file_name, 'rb') as f:
            obj.append(pkl.load(f))
        feature = sp.csr_matrix(obj[0])
    
        index = -1
        test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0],
                                                        self.test_split, self.val_split)
        
        if model_name == 'graphSage':
            # get train adj in dictionary type
            adj_lists = defaultdict(set)
            train_set = set(train_indexs)
            for row, col in zip(adjacency_matrix.nonzero()[0],
                                    adjacency_matrix.nonzero()[1]):
                if row in train_set and col in train_set:
                    adj_lists[row].add(col)
            setattr(self, dataSet + '_adj_train', adj_lists)
            
        setattr(self, dataSet+'_test', test_indexs)
        setattr(self, dataSet+'_val', val_indexs)
        setattr(self, dataSet+'_train', train_indexs)
        
        setattr(self, dataSet+'_feats', feature[:index].toarray())
        setattr(self, dataSet+'_labels',np.array(node_label[:index]))
        setattr(self, dataSet+'_adj_matrix', adjacency_matrix[:index,:index].toarray())
        setattr(self, dataSet+'_edge_labels', edge_labels[:index,:index].toarray())
        
    def _load_Dblp(self, model_name):
        dataSet = 'DBLP'
        obj = []
    
        adj_file_name = self.config['file_path.dblp_edges']
        with open(adj_file_name, 'rb') as f:
                obj.append(pkl.load(f))
    
        # merging diffrent edge type into a single adj matrix
        adjacency_matrix = sp.csr_matrix(obj[0][0].shape)
        for matrix in obj[0]:
            adjacency_matrix +=matrix
    
        matrix = obj[0]
        edge_labels = matrix[0] + matrix[1]
        edge_labels += (matrix[2] + matrix[3])*2
    
        node_label= []
        in_1 = matrix[0].nonzero()[0].min()
        in_2 = matrix[0].nonzero()[0].max()+1
        in_3 = matrix[3].nonzero()[0].max()+1
        matrix[0].nonzero()
        node_label.extend([0 for i in range(in_1)])
        node_label.extend([1 for i in range(in_1,in_2)])
        node_label.extend([2 for i in range(in_2, in_3)])
    
    
        obj = []
        feat_file_name = self.config['file_path.dblp_feats']
        with open(feat_file_name, 'rb') as f:
            obj.append(pkl.load(f))
        feature = sp.csr_matrix(obj[0])
        
        
        index = -1
        test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0],
                                                            self.test_split, self.val_split)
              
        if model_name == 'graphSage':
            # get train adj in dictionary type
            adj_lists = defaultdict(set)
            train_set = set(train_indexs)
            for row, col in zip(adjacency_matrix.nonzero()[0],
                                    adjacency_matrix.nonzero()[1]):
                if row in train_set and col in train_set:
                    adj_lists[row].add(col)
            setattr(self, dataSet + '_adj_train', adj_lists)
            
        setattr(self, dataSet+'_test', test_indexs)
        setattr(self, dataSet+'_val', val_indexs)
        setattr(self, dataSet+'_train', train_indexs)

        setattr(self, dataSet+'_feats', feature[:index].toarray())
        setattr(self, dataSet+'_labels', np.array(node_label[:index]))
        setattr(self, dataSet+'_adj_matrix', adjacency_matrix[:index,:index].toarray())
        setattr(self, dataSet+'_edge_labels', edge_labels[:index].toarray())

        
        
    def load_dataSet(self, dataSet='cora', model_name= 'KDD'):
        if model_name == 'graphSage':
            if dataSet == 'cora':
                self._load_Cora(model_name)
            elif dataSet == 'IMDB':
                self._load_Imdb(model_name)
            elif dataSet == 'ACM':
                self._load_Acm(model_name)
            elif dataSet == 'DBLP':
                self._load_Dblp(model_name)
                
            elif dataSet == 'pubmed':
                pubmed_content_file = self.config['file_path.pubmed_paper']
                pubmed_cite_file = self.config['file_path.pubmed_cites']
    
                feat_data = []
                labels = [] # label sequence of node
                node_map = {} # map node to Node_ID
                with open(pubmed_content_file) as fp:
                    fp.readline()
                    feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
                    for i, line in enumerate(fp):
                        info = line.split("\t")
                        node_map[info[0]] = i
                        labels.append(int(info[1].split("=")[1])-1)
                        tmp_list = np.zeros(len(feat_map)-2)
                        for word_info in info[2:-1]:
                            word_info = word_info.split("=")
                            tmp_list[feat_map[word_info[0]]] = float(word_info[1])
                        feat_data.append(tmp_list)
                
                feat_data = np.asarray(feat_data)
                labels = np.asarray(labels, dtype=np.int64)
                
                adj_lists = defaultdict(set)
                with open(pubmed_cite_file) as fp:
                    fp.readline()
                    fp.readline()
                    for line in fp:
                        info = line.strip().split("\t")
                        paper1 = node_map[info[1].split(":")[1]]
                        paper2 = node_map[info[-1].split(":")[1]]
                        adj_lists[paper1].add(paper2)
                        adj_lists[paper2].add(paper1)
                
                assert len(feat_data) == len(labels) == len(adj_lists)
                test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
    
                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
    
                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
        elif model_name == "KDD":
            if dataSet == 'cora':
                cora_content_file = self.config['file_path.cora_content']
                cora_cite_file = self.config['file_path.cora_cite']
                
                with open(cora_content_file) as f:
                    content = f.readlines()
                content = [x.strip() for x in content]
                id_list = []
                for x in content:
                    x = x.split()
                    id_list.append(int(x[0]))
                id_list = list(set(id_list))
                old_to_new_dict = {}
                for idd in id_list:
                        old_to_new_dict[idd] = len(old_to_new_dict.keys())
                        
                with open(cora_cite_file) as f:
                    content = f.readlines()
                content = [x.strip() for x in content] 
                edge_list = []
                for x in content:
                    x = x.split()
                    edge_list.append([old_to_new_dict[int(x[0])] , old_to_new_dict[int(x[1])]])
                    
                all_nodes = set()            
                for pair in edge_list:
                    all_nodes.add(pair[0])
                    all_nodes.add(pair[1])
                                
                adjancy_matrix = lil_matrix((len(all_nodes), len(all_nodes)))

                for pair in edge_list:
                    adjancy_matrix[pair[0],pair[1]] = 1
                    
                feat_data = []
                labels = [] # label sequence of node
                node_map = {} # map node to Node_ID
                label_map = {} # map label to Label_ID
                with open(cora_content_file) as fp:
                    for i,line in enumerate(fp):
                        info = line.strip().split()
                        feat_data.append([float(x) for x in info[1:-1]])
                        node_map[info[0]] = i
                        if not info[-1] in label_map:
                            label_map[info[-1]] = len(label_map)
                        labels.append(label_map[info[-1]])
                feat_data = np.asarray(feat_data)
                labels = np.asarray(labels, dtype=np.int64)
                                  
                test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
    
                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
    
                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adjancy_matrix.toarray())
                
            if dataSet == "IMDB":
                obj = []
                
                adj_file_name = self.config['file_path.imdb_edges']
                
                with open(adj_file_name, 'rb') as f:
                    obj.append(pkl.load(f))
            
                # merging diffrent edge type into a single adj matrix
                adj = lil_matrix(obj[0][0].shape)
                for matrix in obj[0]:
                    adj +=matrix
            
                matrix = obj[0]
                edge_labels = matrix[0] + matrix[1]
                edge_labels += (matrix[2] + matrix[3])*2
            
                node_label= []
                in_1 = matrix[0].indices.min()
                in_2 = matrix[0].indices.max()+1
                in_3 = matrix[2].indices.max()+1
                node_label.extend([0 for i in range(in_1)])
                node_label.extend([1 for i in range(in_1,in_2)])
                node_label.extend([2 for i in range(in_2, in_3)])
            
                obj = []                
                feat_file_name = self.config['file_path.imdb_feats']
                with open(feat_file_name, 'rb') as f:
                    obj.append(pkl.load(f))
                feature = sp.csr_matrix(obj[0])
                
                index = 9000
                test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0])
                                
                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
    
                setattr(self, dataSet+'_feats', feature[:index].toarray())
                setattr(self, dataSet+'_labels', np.array(node_label[:index]))
                setattr(self, dataSet+'_adj_lists', adj[:index,:index].toarray())
                setattr(self, dataSet+'_edge_labels', edge_labels[:index].toarray())

            if dataSet == "ACM":
                obj = []
                adj_file_name = self.config['file_path.acm_edges']
                with open(adj_file_name, 'rb') as f:
                        obj.append(pkl.load(f))
                        
                adj = sp.csr_matrix(obj[0][0].shape)
                for matrix in obj:
                    nnz = matrix[0].nonzero() # indices of nonzero values
                    for i, j in zip(nnz[0], nnz[1]):
                        adj[i,j] = 1
                        adj[j,i] = 1
                    #adj +=matrix[0]
                
                # to fix the bug on running GraphSAGE
                adj = adj.toarray()
                for i in range(len(adj)):
                    if sum(adj[i, :]) == 0:
                        idx = np.random.randint(0, len(adj))
                        adj[i,idx] = 1
                        adj[idx,i] = 1
            
                edge_labels = matrix[0] + matrix[1]
                edge_labels += (matrix[2] + matrix[3])*2
            
                node_label= []
                in_1 = matrix[0].indices.min()
                in_2 = matrix[0].indices.max()+1
                in_3 = matrix[2].indices.max()+1
                node_label.extend([0 for i in range(in_1)])
                node_label.extend([1 for i in range(in_1,in_2)])
                node_label.extend([2 for i in range(in_2, in_3)])
            
            
                obj = []
                feat_file_name = self.config['file_path.acm_feats']
                with open(feat_file_name, 'rb') as f:
                    obj.append(pkl.load(f))
                feature = sp.csr_matrix(obj[0])
            
            
                index = -1
                test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0])
                
                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                
                setattr(self, dataSet+'_feats', feature[:index].toarray())
                setattr(self, dataSet+'_labels',np.array(node_label[:index]))
                setattr(self, dataSet+'_adj_lists', adj[:index,:index])
                setattr(self, dataSet+'_edge_labels', edge_labels[:index,:index].toarray())
            
            elif dataSet == "DBLP":

                obj = []

                adj_file_name = "/Users/parmis/Desktop/indd/inductive_learning/DBLP/edges.pkl"
            
            
                with open(adj_file_name, 'rb') as f:
                        obj.append(pkl.load(f))
            
                # merging diffrent edge type into a single adj matrix
                adj = sp.csr_matrix(obj[0][0].shape)
                for matrix in obj[0]:
                    adj +=matrix
            
                matrix = obj[0]
                edge_labels = matrix[0] + matrix[1]
                edge_labels += (matrix[2] + matrix[3])*2
            
                node_label= []
                in_1 = matrix[0].nonzero()[0].min()
                in_2 = matrix[0].nonzero()[0].max()+1
                in_3 = matrix[3].nonzero()[0].max()+1
                matrix[0].nonzero()
                node_label.extend([0 for i in range(in_1)])
                node_label.extend([1 for i in range(in_1,in_2)])
                node_label.extend([2 for i in range(in_2, in_3)])
            
            
                obj = []
                with open("/Users/parmis/Desktop/indd/inductive_learning/DBLP/node_features.pkl", 'rb') as f:
                    obj.append(pkl.load(f))
                feature = sp.csr_matrix(obj[0])
                
                
                index = -1000
                test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0])
                                
                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
    
                setattr(self, dataSet+'_feats', feature[:index].toarray())
                setattr(self, dataSet+'_labels', np.array(node_label[:index]))
                setattr(self, dataSet+'_adj_lists', adj[:index,:index].toarray())
                setattr(self, dataSet+'_edge_labels', edge_labels[:index].toarray())
            

    def _split_data(self, num_nodes, test_split = 0.2, val_split = 0.2):
        rand_indices = np.random.permutation(num_nodes)
        
        test_size = int(num_nodes * test_split)
        val_size = int(num_nodes * val_split)
        train_size = num_nodes - (test_size + val_size)
        print(num_nodes, train_size, val_size, test_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size+val_size)]
        train_indexs = rand_indices[(test_size+val_size):]
        
        return test_indexs, val_indexs, train_indexs


"""
convert KDD dataset to GraphSAGE one
"""
def datasetConvert(dataCenter_kdd, ds):
    if ds == 'IMDB' or ds == 'ACM' or ds == 'DBLP':
        dataCenter_sage = copy.deepcopy(dataCenter_kdd)
        
        adj_lists = defaultdict(set)
        adj_kdd = getattr(dataCenter_kdd, ds + '_adj_lists')
        for row in range(len(adj_kdd)):
            for col in range(len(adj_kdd[0])):
                if adj_kdd[row][col] == 1:
                    adj_lists[row].add(col)
        print(adj_lists)
                
        setattr(dataCenter_sage, ds+'_adj_lists', adj_lists)
    return dataCenter_sage

        
       
        
