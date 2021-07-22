import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
from utils import *
from dgl.nn.pytorch import GraphConv as GraphConv

global haveedge
haveedge = False

class Classification(nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        #self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
        self.layer = nn.Sequential(
                                nn.Linear(emb_size, num_classes)      
                                #nn.ReLU()
                            )
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        logists = torch.log_softmax(self.layer(embeds), 1)
        return logists


class UnsupervisedLoss(object):
    """docstring for UnsupervisedLoss"""
    def __init__(self, adj_lists, train_nodes, device):
        super(UnsupervisedLoss, self).__init__()
        self.Q = 10
        self.N_WALKS = 6
        self.WALK_LEN = 1
        self.N_WALK_LEN = 5
        self.MARGIN = 3
        self.adj_lists = adj_lists
        self.train_nodes = train_nodes
        self.device = device

        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []

    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            #print(neg_score)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            #print(pos_score)

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
                
        loss = torch.mean(torch.cat(nodes_score, 0))
        
        return loss

    def get_loss_margin(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

            nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score-pos_score+self.MARGIN).view(1,-1))
            # nodes_score.append((-pos_score - neg_score).view(1,-1))

        loss = torch.mean(torch.cat(nodes_score, 0),0)

        # loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))
        
        return loss


    def extend_nodes(self, nodes, num_neg=6):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        self.node_negtive_pairs = {}

        self.target_nodes = nodes
        self.get_positive_nodes(nodes)
        # print(self.positive_pairs)
        self.get_negtive_nodes(nodes, num_neg)
        # print(self.negtive_pairs)
        self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
        assert set(self.target_nodes) < set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_positive_nodes(self, nodes):
        return self._run_random_walks(nodes)

    def get_negtive_nodes(self, nodes, num_neg):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train_nodes) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    def _run_random_walks(self, nodes):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                continue
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    neighs = self.adj_lists[int(curr_node)]
                    next_node = random.choice(list(neighs))
                    # self co-occurrences are useless
                    if next_node != node and next_node in self.train_nodes:
                        self.positive_pairs.append((node,next_node))
                        cur_pairs.append((node,next_node))
                    curr_node = next_node

            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs
        

class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, input_size, out_size, gcn=False): 
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size


        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined

class GraphSage(nn.Module):
    """docstring for GraphSage"""
    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func

        self.raw_features = raw_features
        self.adj_lists = adj_lists

        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer'+str(index),SageLayer(layer_size, out_size, gcn=self.gcn)) 

    def forward(self, nodes_batch):
        """
        erates embeddings for a batch of nodes.
        es_batch    -- batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index-1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                        aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        
        # TODO is this adj the whole adj or only the training set?
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
            
        
        samp_neighs = [samp_neigh | set([nodes[i]]) 
                       for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
        # self.dc.logger.info('2')
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # self.dc.logger.info('3')
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # self.dc.logger.info('4')

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            # print(mask)
            indexs = [x.nonzero() for x in mask==1]
            aggregate_feats = []
            # self.dc.logger.info('5')
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)


        
        return aggregate_feats
    
    def printModel(self):
        print('*********** GraphSAGE Info ************') 
        print('Input size: ', self.input_size)
        print('Output size: ', self.out_size)
        print('Num layers: ', self.num_layers)
        print('Aggregation: ', self.agg_func)
        
    
###########################################################################3 
    

class GVAE_FrameWork(torch.nn.Module):
    def __init__(self, latent_dim, numb_of_rel, encoder, decoder, mlp_decoder=False, layesrs=None):
        """
        :param latent_dim: the dimention of each embedded node; |z| or len(z)
        :param numb_of_rel:
        :param decoder:
        :param encoder:
        :param mlp_decoder: either apply an multi layer perceptorn on each decoeded embedings
        """
        super(GVAE_FrameWork, self).__init__()
        # self.relation_type_param = torch.nn.ParameterList(torch.nn.Parameter(torch.Tensor(2*latent_space_dim)) for x in range(latent_space_dim))
        self.numb_of_rel = numb_of_rel
        self.decoder = decoder
        self.encoder = encoder

        if mlp_decoder:
            self.embedding_level_mlp = node_mlp(input=latent_dim, layers=layesrs)

        self.dropout = torch.nn.Dropout(0)
        self.reset_parameters()

        # self.mlp_decoder = torch.nn.ModuleList([edge_mlp(2*latent_space_dim,[16,8,1]) for i in range(self.numb_of_rel)])

    def forward(self, adj, x):
        # asakhuja - Start Calling edge generator function if flag of edge embedding is set

        if (haveedge):
            # x = self.dropout(x)
            z, m_z, std_z, edge_emb = self.inference(adj, x)
            if decoder == "SBM_REL":
                generated_adj = self.generator_edge(z, edge_emb)
            else:
                generated_adj = self.generator(z)

        else:
            z, m_z, std_z = self.inference(adj, x)
            z = self.dropout(z)
            generated_adj = self.generator(z)
        return std_z, m_z, z, generated_adj

    # asakhuja - End
    def reset_parameters(self):
        pass

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)

    # inference model
    def inference(self, adj, x):
        # asakhuja - Start Calling edge encoder function if flag of edge embedding is set
        if (haveedge):
            z, m_q_z, std_q_z, edge_emb = self.encoder(adj, x)
            return z, m_q_z, std_q_z, edge_emb
        else:
            z, m_q_z, std_q_z = self.encoder(adj, x)
            return z, m_q_z, std_q_z
        # asakhuja - End

    # generation model
    # asakhuja - Start Added edge generator
    def generator_edge(self, z, edge_emb):
        # apply chain of mlp on nede embedings
        # z = self.embedding_level_mlp(z)

        gen_adj = []
        if (haveedge):
            adj = self.decoder(z, edge_emb)
        else:
            adj = self.decoder(z)
        return adj

    def generator(self, z):
        # apply chain of mlp on nede embedings
        # z = self.embedding_level_mlp(z)

        gen_adj = []
        adj = self.decoder(z)
        return adj
        # asakhuja - End

    
class multi_layer_GCN(torch.nn.Module):
    def __init__(self, in_feature, latent_dim=32, layers=[64]):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param latent_dim: the dimention of each embedded node; |z| or len(z)
        :param layers: a list in which each element determine the siae of corresponding GCNN Layer.
        """
        super(multi_layer_GCN, self).__init__()
        layers = [in_feature] + layers
        if len(layers) < 1: raise Exception("sorry, you need at least two layer")
        self.ConvLayers = torch.nn.ModuleList(
            GraphConv(layers[i], layers[i + 1], activation=None, bias=False, weight=True) for i in
            range(len(layers) - 1))

        self.q_z_mean = GraphConv(layers[-1], latent_dim, activation=None, bias=False, weight=True)

        self.q_z_std = GraphConv(layers[-1], latent_dim, activation=None, bias=False, weight=True)

    def forward(self, adj, x):
        dropout = torch.nn.Dropout(0)
        for conv_layer in self.ConvLayers:
            x = torch.tanh(conv_layer(adj, x))
            x = dropout(x)

        m_q_z = self.q_z_mean(adj, x)
        std_q_z = torch.relu(self.q_z_std(adj, x)) + .0001

        z = self.reparameterize(m_q_z, std_q_z)
        return z, m_q_z, std_q_z,

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)


from utils import *
class MapedInnerProductDecoder(torch.nn.Module):
    """Decoder for using inner product of multiple transformed Z"""

    def __init__(self, layers, num_of_relations, in_size, normalize, DropOut_rate):
        #
        super(MapedInnerProductDecoder, self).__init__()
        self.models = torch.nn.ModuleList(
            node_mlp(in_size, layers, normalize, DropOut_rate) for i in range(num_of_relations))

    def forward(self, z):
        A = []
        for trans_model in self.models:
            tr_z = trans_model(z)
            layer_i = torch.mm(tr_z, tr_z.t())
            A.append(layer_i)
        return torch.sum(torch.stack(A), 0)

    def get_edges_features(self, z):
        gen_adj = []
        for trans_model in self.models:
            tr_z = trans_model(z)
            layer_i = torch.mm(tr_z, tr_z.t())
            # gen_adj.append((h) * self.p_of_relation(z, i))
            gen_adj.append(layer_i)
            # gen_adj.append(self.mlp_decoder[i](self.to_3D(z)))
        return torch.stack(gen_adj)
    
############################# NIPS MODELS ########################################
class kernelGVAE(torch.nn.Module):
    def __init__(self, in_feature_dim, hidden1,  latent_size, ker, decoder, device, encoder_fcc_dim = [128] , autoencoder=False):
        super(kernelGVAE, self).__init__()
        self.first_conv_layer = GraphConvNN(in_feature_dim, hidden1)
        self.second_conv_layer = GraphConvNN(hidden1, hidden1)
        self.stochastic_mean_layer = GraphConvNN(hidden1, latent_size)
        self.stochastic_log_std_layer = GraphConvNN(hidden1, latent_size)
        self.kernel = ker #TODO: bin and width whould be determined if kernel is his

        # self.reset_parameters()
        self.Drop = torch.nn.Dropout(0)
        self.Drop = torch.nn.Dropout(0)
        self.latent_dim = latent_size
        self.mlp = None
        self.decode = decoder
        self.autoencoder = autoencoder
        self.device = device

        if None !=encoder_fcc_dim:

            self.fnn =node_mlp(hidden1, encoder_fcc_dim)
            self.stochastic_mean_layer = node_mlp(encoder_fcc_dim[-1], [latent_size])
            self.stochastic_log_std_layer = node_mlp(encoder_fcc_dim[-1], [latent_size])

    def forward(self, graph, features, num_node, ):
        """

        :param graph: normalized adjacency matrix of graph
        :param features: normalized node feature matrix
        :return:
        """
        samples, mean, log_std = self.encode( graph, features,self.autoencoder)
        reconstructed_adj_logit = self.decode(samples)
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)
        kernel_value = self.kernel(reconstructed_adj,num_node)

        mask = torch.zeros(graph.shape)

        # removing the effect of none existing nodes
        for i in range(graph.shape[0]):
            reconstructed_adj_logit[i, :, num_node[i]:] = -100
            reconstructed_adj_logit[i, num_node[i]:, :] = -100
            mask[i, :num_node[i], :num_node[i]] = 1
            mean[i,num_node[i]:, :]=  0
            log_std[i,num_node[i]:, :]=  0

        reconstructed_adj = reconstructed_adj * mask.to(self.device)
        # reconstructed_adj_logit  = reconstructed_adj_logit + mask_logit
        return reconstructed_adj, samples, mean, log_std, kernel_value, reconstructed_adj_logit

    def encode(self, graph, features, autoencoder):
        h = self.first_conv_layer(graph, features)
        h = self.Drop(h)
        h= torch.tanh(h)
        h = self.second_conv_layer(graph, h)
        h = torch.tanh(h)
        if type(self.stochastic_mean_layer) ==GraphConvNN:
            mean = self.stochastic_mean_layer(graph, h)
            log_std = self.stochastic_log_std_layer(graph, h)
        else:
            h = self.fnn(h)
            mean = self.stochastic_mean_layer(h,activation = lambda x:x)
            log_std = self.stochastic_log_std_layer(h,activation = lambda x:x)

        if autoencoder==False:
            sample = self.reparameterize(mean, log_std, node_num)
        else:
            sample = mean*1
        return sample, mean, log_std


    def reparameterize(self, mean, log_std, node_num):
        # std = torch.exp(log_std)
        # eps = torch.randn_like(std)
        # return eps.mul(std).add(mean)

        var = torch.exp(log_std).pow(2)
        eps = torch.randn_like(var)
        sample = eps.mul(var).add(mean)

        for i, node_size in enumerate(node_num):
            sample[i][node_size:,:]=0
        return sample


class kernel(torch.nn.Module):
    """
     this class return a list of kernel ordered by keywords in kernel_type
    """
    def __init__(self, **ker):
        
        """
        :param ker:
        kernel_type; a list of string which determine needed kernels
        """
        super(kernel, self).__init__()
        self.kernel_type = ker.get("kernel_type")
        kernel_set = set(self.kernel_type)
        
        
        self.device = ker.get("device")

        if "in_degree_dist" in kernel_set or "out_degree_dist" in kernel_set:
            self.degree_hist = Histogram( self.device, ker.get("degree_bin_width").to(self.device), ker.get("degree_bin_center").to(self.device))

        if "RPF" in kernel_set:
            self.num_of_steps = ker.get("step_num")
            self.hist = Histogram(self.device, ker.get("bin_width"), ker.get("bin_center") )

        if "trans_matrix" in kernel_set:
            self.num_of_steps = ker.get("step_num")



    def forward(self,adj, num_nodes):
        vec = self.kernel_function(adj, num_nodes)
        # return self.hist(vec)
        return vec

    def kernel_function(self, adj, num_nodes): # TODO: another var for keeping the number of moments
        # ToDo: here we assumed the matrix is symetrix(undirected) which might not
        vec = []  # feature vector
        for kernel in self.kernel_type:
            if "in_degree_dist" == kernel:
                degree_hit = []
                for i in range(adj.shape[0]):
                    degree = adj[i,:num_nodes[i],:num_nodes[i]].sum(1).view(1, num_nodes[i])
                    degree_hit.append(self.degree_hist(degree.to(self.device)))
                vec.append(torch.cat(degree_hit))
            if "out_degree_dist" == kernel:
                degree_hit = []
                for i in range(adj.shape[0]):
                    degree = adj[i, :num_nodes[i], :num_nodes[i]].sum(0).view(1, num_nodes[i])
                    degree_hit.append(self.degree_hist(degree))
                vec.append(torch.cat(degree_hit))
            if "RPF" == kernel:
                raise("should be changed") #ToDo: need to be fixed
                tr_p = self.S_step_trasition_probablity( self.device, adj, num_nodes, self.num_of_steps)
                for i in range(len(tr_p)):
                    vec.append(self.hist(torch.diag(tr_p[i])))

            if "trans_matrix" == kernel:
                vec.extend(self.S_step_trasition_probablity(self.device, adj, num_nodes, self.num_of_steps))
                # vec = torch.cat(vec,1)

            if "tri" == kernel:  # compare the nodes degree in the given order
                tri, square = self.tri_square_count(adj)
                vec.append(tri), vec.append(square)

        return vec

    def tri_square_count(self, adj):
        ind = torch.eye(adj[0].shape[0]).to(device)
        adj = adj - ind
        two__ = torch.matmul(adj, adj)
        tri_ = torch.matmul(two__, adj)
        squares = torch.matmul(two__, two__)
        return (torch.diagonal(tri_, dim1=1, dim2=2), torch.diagonal(squares, dim1=1, dim2=2))

    def S_step_trasition_probablity(self, device, adj, num_node, s=4):
        """
         this method take an adjacency matrix and return its j<s adjacency matrix, sorted, in a list
        :param s: maximum step
        :param Adj: adjacency matrixy of the grap
        :return: a list in whcih the ith elemnt is the i srep transition probablity
        """
        mask = torch.zeros(adj.shape).to(device)
        for i in range(adj.shape[0]):
            mask[i,:num_node[i],:num_node[i]] = 1

        p1 = adj.to(device)
        p1 = p1 * mask
        # ind = torch.eye(adj[0].shape[0])
        # p1 = p1 - ind
        TP_list = []
        p1 = p1*(p1.sum(2).float().clamp(min=1) ** -1).view(adj.shape[0],adj.shape[1], 1)

        # p1[p1!=p1] = 0
        # p1 = p1 * mask

        if s>0:
            # TP_list.append(torch.matmul(p1,p1))
            TP_list.append( p1)
        for i in range(s-1):
            TP_list.append(torch.matmul(p1, TP_list[-1] ))
        return TP_list



class Histogram(torch.nn.Module):
    def __init__(self, device, bin_width = None, bin_centers = None):
        
        super(Histogram, self).__init__()
        self.bin_width = bin_width.to(device)
        self.bin_center = bin_centers.to(device)
        if self.bin_width == None:
            self.prism()
        else:
            self.bin_num = self.bin_width.shape[0]

    def forward(self, vec):
        score_vec = vec.view(vec.shape[0],1, vec.shape[1], ) - self.bin_center
        # score_vec = vec-self.bin_center
        score_vec = 1-torch.abs(score_vec)*self.bin_width
        score_vec = torch.relu(score_vec)
        return score_vec.sum(2)

    def prism(self):
        pass
    
    
class FC_InnerDOTdecoder(torch.nn.Module):
    def __init__(self,input,output,laten_size ,layer=[256,1024,256]):
        super(FC_InnerDOTdecoder,self).__init__()
        self.lamda = torch.nn.Parameter(torch.Tensor(laten_size, laten_size))
        layer = [input]+layer + [output]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layer[i],layer[i+1]) for i in range(len(layer)-1)])
        self.reset_parameters()
    # def forward(self,Z):
    #     shape = Z.shape
    #     z = Z.reshape(shape[0],-1)
    #     for i in range(len(self.layers)):
    #         z  = self.layers[i](z)
    #         z = torch.tanh(z)
    #     # Z = torch.sigmoid(Z)
    # return z.reshape(shape[0], shape[-2], shape[-2])
    def forward(self, in_tensor, activation=torch.nn.ReLU()):
        h = in_tensor.reshape(in_tensor.shape[0],-1)
        for i in range(len(self.layers)):
            # if self.norm_layers != None:
            #     if len(h.shape) == 2:
            #         h = self.norm_layers[i](h)
            #     else:
            #         shape = h.shape
            #         h = h.reshape(-1, h.shape[-1])
            #         h = self.norm_layers[i](h)
            #         h = h.reshape(shape)
            # h = self.dropout(h)
            h = self.layers[i](h)
            if ((i!=len(self.layers))):
              h = activation(h)
        h = h.reshape(in_tensor.shape[0], in_tensor.shape[1],-1)
        return torch.matmul(torch.matmul(h,self.lamda), h.permute(0, 2, 1))

    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)
class InnerDOTdecoder(torch.nn.Module):
    def __init__(self):
        super(InnerDOTdecoder,self).__init__()
    # def forward(self,Z):
    #     shape = Z.shape
    #     z = Z.reshape(shape[0],-1)
    #     for i in range(len(self.layers)):
    #         z  = self.layers[i](z)
    #         z = torch.tanh(z)
    #     # Z = torch.sigmoid(Z)
    # return z.reshape(shape[0], shape[-2], shape[-2])
    def forward(self, h):
       return torch.matmul(h, h.permute(0, 2, 1))




class GraphConvNN(torch.nn.Module):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=False,
                 activation=None):
        super(GraphConvNN, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise ('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The adg of graph. It should include self loop
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """


        if self._norm == 'both':
            degs = graph.sum(-2).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,)
            norm = torch.reshape(norm, shp)
            feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise ('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = torch.matmul(feat, weight)
            # graph.srcdata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            #                  fn.sum(msg='m', out='h'))
            rst = torch.matmul(graph, feat)
        else:
            # aggregate first then mult W
            # graph.srcdata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            #                  fn.sum(msg='m', out='h'))
            # rst = graph.dstdata['h']
            rst = torch.matmul(graph, feat)
            if weight is not None:
                rst = torch.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.sum(-1).float().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)



         



         