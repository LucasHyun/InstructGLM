# STEP 1: Add these imports at the top with other imports (around line 10)
from fileinput import lineno
from platform import node
import re
from urllib.parse import urldefrag 
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
import copy
import torch.nn.functional as F  # Add this
from collections import defaultdict  # Add this

import torch.nn.functional as F
from torch.nn import CosineSimilarity

_do_abl = False

_kval = 5

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

    
 # STEP 2: Replace the existing __init__ method with this enhanced version
class PubMed_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks    
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        # Initialize SNS components
        self.node_pairs = []
        self.similarity_scores = []
        
        # Initialize PinSAGE components
        self.random_walk_length = 3
        self.num_random_walks = 10
        self.neighbor_samples = defaultdict(list)
        
        print('Data sources: ', split.split(','))
        self.mode = mode
        self.prefix_1='Perform Link Prediction for the node: Node represents academic paper with a specific topic, link represents a citation between the two papers. Pay attention to the multi-hop link relationship between the nodes. '
       
        self.prefix_2='Classify the article according to its topic into one of the following categories: [ experimental, second, first ]. Node represents academic paper with a specific topic, link represents a citation between the two papers. Pay attention to the multi-hop link relationship between the nodes. '
        self.label_map=load_pickle(os.path.join('PubMed','final_pub_label_map.pkl'))  #1
        self.re_id=load_pickle(os.path.join('PubMed','Llama_final_pub_re_id.pkl'))  #2   
        self.llama_embed=load_pickle(os.path.join('PubMed','Llama_embeds.pkl'))  #3
        self.l_max=self.args.max_text_length
        self.real_feature=load_pickle(os.path.join('PubMed','Llama_final_norm_pub_real_feature.pkl')) #4
        self.node_feature_BERT=load_pickle(os.path.join('PubMed','final_pubmed_node_feature_BERT.pkl')) #4.5
        self.train_L1=load_pickle(os.path.join('PubMed','final_pub_L1.pkl'))  #5
        self.transductive=load_pickle(os.path.join('PubMed','final_pub_transductive.pkl'))  #6 
        #self.transductive=load_pickle(os.path.join('PubMed','small_pub_transductive.pkl'))  #6 
        self.classification=load_pickle(os.path.join('PubMed','final_pub_classification.pkl'))  #7
        #self.classification=load_pickle(os.path.join('PubMed','small_pub_classification.pkl'))  #7
        self.node_feature=load_pickle(os.path.join('PubMed','full_final_pubmed_node_feature.pkl'))  #8

        self.cosSim = CosineSimilarity(dim=0)
        
        LA=[]
        LAA=list(set(self.label_map.values()))
        for laa in tqdm(range(len(LAA))):
            LA.append(LAA[laa])
        assert len(LA)==3 
        self.LA=LA

        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        if self.mode=='train':
            self.compute_datum_info_train()    
        else:
            self.compute_datum_info_val()
            
        if self.mode=='val':
            self.len_transductive=len(self.transductive)   

    # self._simdeg_dic_calc()
    def _simdeg_dic_calc(self, point_orig, point_list):
        #return self._sim_dic_calc(point_orig, point_list)
        return {nodeid : self._sim_calc(point_orig,nodeid) * len(self.train_L1[nodeid]) for nodeid in point_list}
    
    def _sim_dic_calc(self, point_orig, point_list):
        return {nodeid : self._sim_calc(point_orig,nodeid) for nodeid in point_list}
    
    def _sim_calc(self,point1, point2):
        #return 1.0
        return 1.0 + self.cosSim(
            self.node_feature_BERT[point1],
            self.node_feature_BERT[point2]
        )
    
    # STEP 3: Add these new methods right after __init__ and before compute_datum_info_train
    def initialize_sns(self):
        """Initialize Supervised Node Similarity components"""
        for node in range(len(self.node_feature)):
            # Only process nodes that have connections
            if node in self.train_L1 and self.train_L1[node]:
                positive_pairs = [(node, pos) for pos in self.train_L1[node]]
                
                # Get valid negative nodes (excluding self and connected nodes)
                valid_neg_nodes = [n for n in range(len(self.node_feature)) 
                                 if n != node and n not in self.train_L1[node]]
                
                # Only sample if we have valid negative nodes
                if valid_neg_nodes:
                    num_neg_samples = min(len(positive_pairs), len(valid_neg_nodes))
                    negative_pairs = [(node, neg) for neg in 
                                    random.sample(valid_neg_nodes, num_neg_samples)]
                    
                    self.node_pairs.extend(positive_pairs)
                    self.node_pairs.extend(negative_pairs)
                    
                    # Compute similarity scores (1 for positive pairs, 0 for negative)
                    self.similarity_scores.extend([1] * len(positive_pairs))
                    self.similarity_scores.extend([0] * len(negative_pairs))
        
        if not self.node_pairs:
            print("Warning: No valid node pairs generated for SNS")

    def initialize_pinsage(self):
        """Initialize PinSAGE components"""
        for node in range(len(self.node_feature)):
            for _ in range(self.num_random_walks):
                walk = self.random_walk(node)
                self.neighbor_samples[node].extend(walk)
            
            # Keep unique neighbors
            self.neighbor_samples[node] = list(set(self.neighbor_samples[node]))

    def random_walk(self, start_node):
        """Perform random walk starting from given node"""
        walk = [start_node]
        current_node = start_node
        
        for _ in range(self.random_walk_length):
            neighbors = self.train_L1[current_node]
            if not neighbors:
                break
            current_node = random.choice(neighbors)
            walk.append(current_node)
            
        return walk

    def get_sns_loss(self, node_embeddings):
        """Compute SNS loss for node embeddings"""
        # Check if we have valid pairs
        if not self.node_pairs or len(self.node_pairs) == 0:
            return torch.tensor(0.0)  # Return zero loss if no pairs
            
        try:
            loss = 0
            valid_pairs = 0
            
            for (node1, node2), similarity in zip(self.node_pairs, self.similarity_scores):
                # Ensure nodes are within bounds
                if (node1 < len(node_embeddings) and node2 < len(node_embeddings)):
                    # Compute cosine similarity
                    emb1 = node_embeddings[node1]
                    emb2 = node_embeddings[node2]
                    cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                    
                    # Binary cross entropy loss
                    loss += F.binary_cross_entropy_with_logits(cos_sim, 
                                                           torch.tensor(similarity, dtype=torch.float))
                    valid_pairs += 1
            
            # Return average loss only if we have valid pairs
            return loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0)
            
        except Exception as e:
            print(f"Error in SNS loss calculation: {str(e)}")
            return torch.tensor(0.0)

    def aggregate_neighbors(self, node, node_embeddings):
        """PinSAGE neighbor aggregation"""
        neighbors = self.neighbor_samples[node]
        if not neighbors:
            return node_embeddings[node]
        
        # Get neighbor embeddings
        neighbor_embeds = node_embeddings[neighbors]
        
        # Compute attention scores
        attention = torch.matmul(node_embeddings[node].unsqueeze(0), 
                               neighbor_embeds.transpose(0, 1))
        attention = F.softmax(attention, dim=1)
        
        # Weighted aggregation
        aggregated = torch.matmul(attention, neighbor_embeds)
        return aggregated.squeeze(0)
        
    def compute_datum_info_train(self):
        curr = 0
        for key in list(self.task_list.keys()):     
            if key == 'link':
                for tems in self.task_list[key]: 
                    if '1-1-1-1' in tems:
                        self.total_length += 19717 * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-1'))
                        curr = self.total_length
                    elif '1-1-2-1' in tems:  
                        self.total_length += 19717 * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-2'))
                        curr = self.total_length
                    elif '1-1-3-1' in tems:  
                        self.total_length += 19717 * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-3'))  
                        curr = self.total_length
                    elif '1-7-7-3' in tems:  
                        self.total_length += 19717 * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-7'))  
                        curr = self.total_length

            elif key == 'classification': 
                for tems in self.task_list[key]:
                    if '2-1-1-2' in tems:

                        self.total_length += len(self.classification) * 1
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'2-1','transductive')) 
                        curr = self.total_length
                    elif '2-1-2-2' in tems:

                        self.total_length += len(self.classification) * 2
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 2,'2-2','transductive'))
                        curr = self.total_length
                    elif '2-1-3-2' in tems:

                        self.total_length += len(self.classification) * 2
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 2,'2-3','transductive'))
                        curr = self.total_length
                    elif '6-6-6-6' in tems:

                        self.total_length += len(self.classification) * 1
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'5-6','transductive'))
                        curr = self.total_length
            elif key == 'intermediate':
                pass
            else:
                raise NotImplementedError

    def compute_datum_info_val(self):
        curr = 0
        for key in list(self.task_list.keys()):     
            if key == 'link':
                pass
            elif key == 'classification':
                for tems in self.task_list[key]:
                    if '2-3-1-2' in tems:
                        self.total_length += len(self.transductive) * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,tems[i % 1],'transductive'))  
                        curr = self.total_length
                    elif '2-3-2-4' in tems:

                        self.total_length += len(self.transductive) * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,tems[i % 1],'transductive'))
                        curr = self.total_length
                    elif '2-3-3-4' in tems:

                        self.total_length += len(self.transductive) * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,tems[i % 1],'transductive'))
                        curr = self.total_length
                    elif '6-6-6-6' in tems:

                        self.total_length += len(self.transductive) * 2
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,tems[i % 2],'transductive'))
                        curr = self.total_length
            elif key == 'intermediate':
                pass
            else:
                raise NotImplementedError
    
            
    def __len__(self):
        return self.total_length

    # STEP 4: Update beginning of __getitem__ where it currently starts
    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args
        
        # Initialize SNS and PinSAGE components if not done yet
        if not hasattr(self, 'node_pairs'):
            self.initialize_sns()
        if not hasattr(self, 'neighbor_samples'):
            self.initialize_pinsage()
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        
        

        if self.mode=='train':
            if len(datum_info_idx) == 5:
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]

                task_template_range = datum_info_idx[3]

                if task_template_range=='2-1':
                    t_set=['2-1-1-2','2-3-1-2']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='2-2':
                    t_set=['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4']
                    #t_set=['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4','2-3-2-1','2-3-2-2','2-3-2-3','2-3-2-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='2-3':
                    t_set=['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']
                    #t_set=['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4','2-3-3-1','2-3-3-2','2-3-3-3','2-3-3-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='5-6':
                    t_set=['6-6-6-6','6-6-6-7']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                
                
                if task_name=='classification':
                    cate=datum_info_idx[4]
                else:
                    which_idx=datum_info_idx[4]
                    flip=0
            elif len(datum_info_idx)==4:
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]

                task_template_range = datum_info_idx[3]
                if task_template_range=='1-1':
                    t_set=['1-3-1-1','1-1-1-1','1-3-2-3','1-3-3-3','1-1-2-3','1-1-3-3']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='1-2':
                    t_set=['1-3-2-1','1-3-2-3','1-1-2-1','1-1-2-3']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='1-3':
                    t_set=['1-3-3-1','1-3-3-3','1-1-3-1','1-1-3-3']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='1-7':
                    t_set=['1-7-7-3','1-7-7-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]

            else:
                raise NotImplementedError
        elif self.mode=='val': 
            if len(datum_info_idx) == 5:
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]
                task_template = self.all_tasks[task_name][datum_info_idx[3]]
                if task_name=='classification':
                    cate=datum_info_idx[4]
                else:
                    which_idx=datum_info_idx[4]
                    flip=0
            else:
                raise NotImplementedError


        if task_name == 'link':
            if self.mode=='train': 
                link_datum=[datum_idx]  # Central Node
            elif self.mode=='val':
                pass

            if task_template['id'] == '1-1-1-1':   
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                        link_datum.append(point)
                        node_list=''   
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        
                        # Added Part
                        similarity_dic = self._sim_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                    
                        while go_on and count < len(self.train_L1[link_datum[0]]) and ((not _do_abl) or count < _kval**1):  
                            temp_text=source_text 

                            # select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                            
                            # Added + Changed Part
                            selectID = sorted_sim[count][0]
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[selectID])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  
                        node_list=''    
                        count=0

                        negative=random.randint(0,19716)
                        while negative in self.train_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        
                        # Added Part
                        similarity_dic = self._sim_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                        
                        while go_on and count < len(self.train_L1[link_datum[0]]) and ((not _do_abl) or count < _kval**1):  
                            temp_text=source_text   

                            # select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                            
                            # Added + Changed Part
                            selectID = sorted_sim[count][0]
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[selectID])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', '<extra_id_0>')
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass

            elif task_template['id'] == '1-1-1-2':   
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                    link_datum.append(point)
                    
                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    # already_idx=[]
                    
                    # Added Part
                    similarity_dic = self._sim_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                    sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                
                    while go_on and count < len(self.train_L1[link_datum[0]]) and ((not _do_abl) or count < _kval**1):  
                        temp_text=source_text   

                        # select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                        # idx=int(np.random.choice(select,1,replace=False)[0])
                        # already_idx.append(idx)
                        # node_list=node_list+'<extra_id_0>, '
                        # real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                        
                        # Added + Changed Part
                        selectID = sorted_sim[count][0]
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[selectID])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])
                    target_text=[self.re_id[link_datum[1]],1]

                elif self.mode=='val':   
                    pass





            elif task_template['id'] == '1-7-7-4':   
                link_abs=self.node_feature[link_datum[0]][1] 
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                    link_datum.append(point)
                    
                    source_text=task_template['source'].format('<extra_id_0>', link_abs,'<extra_id_0>')
                    real_id.append(self.re_id[link_datum[0]])
                    target_text=[self.re_id[link_datum[1]],1]

                elif self.mode=='val':   
                    pass

            elif task_template['id'] == '1-7-7-3': 
                link_abs=self.node_feature[link_datum[0]][1] 
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    rand_prob = random.random()

                    if rand_prob > 0.5:
                        point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                        link_datum.append(point)
                        source_text =task_template['source'].format('<extra_id_0>', link_abs,'<extra_id_0>', '<extra_id_0>')
                        
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  
                        negative=random.randint(0,19716)
                        while negative in self.train_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =task_template['source'].format('<extra_id_0>', link_abs,'<extra_id_0>', '<extra_id_0>')
                        
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass
            

            elif task_template['id'] == '1-1-2-1':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    temp_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                                temp_L2.append(ttt)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L2[random.randint(0, len(train_L2) - 1)]
                        link_datum.extend(points)
                        node_list=''   
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                real_id.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # real_id.append(self.re_id[train_L2[idx][1]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:   
                        node_list=''    
                        count=0
                        negative=random.randint(0,19716)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                real_id.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # real_id.append(self.re_id[train_L2[idx][1]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>','<extra_id_0>')
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass



            elif task_template['id'] == '1-1-2-2':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                    
                    while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L1_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops not in already_idx
                                select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                
                                if len(select_L1_hops) > 0:
                                    go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                                    select_L1 = None
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                        
                        if select_L1 is None:
                            go_on = False
                        else:
                            for pair in select_L1_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                            selectPair = select_L1_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[selectPair[1]])
                            
                            # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])

                    target_text = [self.re_id[link_datum[2]],1]

                elif self.mode=='val':   
                    pass


            elif task_template['id'] == '1-1-2-3':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''   
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                middle_list=middle_list+'<extra_id_0>, '
                                id_1.append(self.re_id[selectPair[1]])
                                id_2.append(self.re_id[selectPair[0]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # id_1.append(self.re_id[train_L2[idx][1]])
                                # middle_list=middle_list+'<extra_id_0>, '
                                # id_2.append(self.re_id[train_L2[idx][0]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>','<extra_id_0>','<extra_id_0>')
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        target_text = task_template['target'].format('yes')

                    else:  
                        temp_L2=self.train_L1[link_datum[1]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,19716)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                middle_list=middle_list+'<extra_id_0>, '
                                id_1.append(self.re_id[selectPair[1]])
                                id_2.append(self.re_id[selectPair[0]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)

                                # node_list=node_list+'<extra_id_0>, '
                                # id_1.append(self.re_id[train_L2[idx][1]])

                                # middle_list=middle_list+'<extra_id_0>, '
                                # id_2.append(self.re_id[train_L2[idx][0]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass

            elif task_template['id'] == '1-1-2-4':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>','<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                    
                    while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2): 
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L1_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops not in already_idx
                                select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                
                                if len(select_L1_hops) > 0:
                                    go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                                    select_L1 = None
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                        
                        if select_L1 is None:
                            go_on = False
                        else:
                            for pair in select_L1_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                            selectPair = select_L1_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'<extra_id_0>, '
                            middle_list=middle_list+'<extra_id_0>, '
                            id_1.append(self.re_id[selectPair[1]])
                            id_2.append(self.re_id[selectPair[0]])
                            
                            # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)

                            # node_list=node_list+'<extra_id_0>, '
                            # id_1.append(self.re_id[train_L2[idx][1]])

                            # middle_list=middle_list+'<extra_id_0>, '
                            # id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>','<extra_id_0>')
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        id_1.pop(-1)
                        id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[link_datum[0]])
                    real_id.append(self.re_id[link_datum[1]])

                    target_text =[self.re_id[link_datum[2]],1]

                elif self.mode=='val':   
                    pass

            elif task_template['id'] == '1-1-3-1':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    temp_L3=[]
                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0] and el!=ele[0]:
                                train_L3.append(ele+[el])
                                temp_L3.append(el)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L3[random.randint(0, len(train_L3) - 1)]
                        link_datum.extend(points)
                        node_list=''   
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                real_id.append(self.re_id[selectPair[2]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # real_id.append(self.re_id[train_L3[idx][2]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else: 

                        node_list=''    
                        count=0
                        negative=random.randint(0,19716)                                               
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3): 
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                real_id.append(self.re_id[selectPair[2]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # real_id.append(self.re_id[train_L3[idx][2]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])

                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass

            elif task_template['id'] == '1-1-3-2':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0] and el!=ele[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                    L2_set = list(set([i[1] for i in train_L2]))
                    simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                    
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                    train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                    
                    while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L2 = None
                        select_L1_hops = None
                        select_L2_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops in train_L2_undepleted
                                select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                
                                # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                if len(select_L1_hops) > 0:
                                    go_on_L2 = True
                                    
                                    while go_on_L2:
                                        toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                        if len(toselectfrom) > 0:
                                            # select random node from select_L1_hops
                                            weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                            select_L2_ID = np.random.choice(
                                                range(len(toselectfrom)),
                                                p = weights_L2 / np.sum(weights_L2)
                                            )
                                            select_L2 = toselectfrom[select_L2_ID]
                                            
                                            # check number of L3 hops not in already_idx
                                            select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                            
                                            if len(select_L2_hops) > 0:
                                                go_on_L2 = False
                                            else:
                                                train_L2_undepleted.remove(select_L2)
                                                select_L2 = None
                                        else:
                                            go_on_L2 = False
                                            
                                    if select_L2 is None:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                    else:
                                        go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                        
                        if select_L2 is None:
                            go_on = False
                        else:
                            for pair in select_L2_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                            selectPair = select_L2_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[selectPair[2]])
                            
                            # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])

                    target_text = [self.re_id[link_datum[3]],1]


                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-1-3-3':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0] and el!=ele[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        
                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                
                                node_list=node_list+'<extra_id_0>, '
                                middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                                id_1.append(self.re_id[selectPair[2]])
                                id_2.append(self.re_id[selectPair[0]])
                                id_2.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)

                                # node_list=node_list+'<extra_id_0>, '
                                # id_1.append(self.re_id[train_L3[idx][2]])
                                # middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                                # id_2.append(self.re_id[train_L3[idx][0]])
                                # id_2.append(self.re_id[train_L3[idx][1]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])

                        target_text = task_template['target'].format('yes')

                    else: 
                        temp_L3=self.train_L1[link_datum[2]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,19716)
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                
                                node_list=node_list+'<extra_id_0>, '
                                middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                                id_1.append(self.re_id[selectPair[2]])
                                id_2.append(self.re_id[selectPair[0]])
                                id_2.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)

                                # node_list=node_list+'<extra_id_0>, '
                                # id_1.append(self.re_id[train_L3[idx][2]])
                                # middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                                # id_2.append(self.re_id[train_L3[idx][0]])
                                # id_2.append(self.re_id[train_L3[idx][1]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass
            elif task_template['id'] == '1-1-3-4': 
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0] and el!=ele[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                    L2_set = list(set([i[1] for i in train_L2]))
                    simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                    
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                    train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                    
                    while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L2 = None
                        select_L1_hops = None
                        select_L2_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops in train_L2_undepleted
                                select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                
                                # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                if len(select_L1_hops) > 0:
                                    go_on_L2 = True
                                    
                                    while go_on_L2:
                                        toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                        if len(toselectfrom) > 0:
                                            # select random node from select_L1_hops
                                            weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                            select_L2_ID = np.random.choice(
                                                range(len(toselectfrom)),
                                                p = weights_L2 / np.sum(weights_L2)
                                            )
                                            select_L2 = toselectfrom[select_L2_ID]
                                            
                                            # check number of L3 hops not in already_idx
                                            select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                            
                                            if len(select_L2_hops) > 0:
                                                go_on_L2 = False
                                            else:
                                                train_L2_undepleted.remove(select_L2)
                                                select_L2 = None
                                        else:
                                            go_on_L2 = False
                                            
                                    if select_L2 is None:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                    else:
                                        go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                        
                        if select_L2 is None:
                            go_on = False
                        else:
                            for pair in select_L2_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                            selectPair = select_L2_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            
                            node_list=node_list+'<extra_id_0>, '
                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_1.append(self.re_id[selectPair[2]])
                            id_2.append(self.re_id[selectPair[0]])
                            id_2.append(self.re_id[selectPair[1]])
                            
                            # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # id_1.append(self.re_id[train_L3[idx][2]])
                            # middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            # id_2.append(self.re_id[train_L3[idx][0]])
                            # id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        id_1.pop(-1)
                        id_2.pop(-1)
                        id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[link_datum[0]])
                    real_id.append(self.re_id[link_datum[1]])
                    real_id.append(self.re_id[link_datum[2]])

                    target_text =[self.re_id[link_datum[3]],1]

                elif self.mode=='val':  
                    pass


            elif task_template['id'] == '1-3-1-1':    
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    rand_prob = random.random()

                    if rand_prob > 0.5:
                        point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                        link_datum.append(point)
                        node_list=''    
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[1]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        # already_idx=[]
                        
                        # Added Part
                        similarity_dic = self._sim_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                        
                        while go_on and count < len(self.train_L1[link_datum[0]]) and ((not _do_abl) or count < _kval**1):  
                            temp_text=source_text  

                            # select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[link_datum[0]][idx]][0])
                            # real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                            
                            # Added + Changed Part
                            selectID = sorted_sim[count][0]
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectID][0])
                            real_id.append(self.re_id[selectID])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[1]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            

                            count+=1 
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  
                        node_list=''    
                        count=0

                        negative=random.randint(0,19716)
                        while negative in self.train_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        # already_idx=[]
                        
                        # Added Part
                        similarity_dic = self._sim_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                        
                        while go_on and count < len(self.train_L1[link_datum[0]]) and ((not _do_abl) or count < _kval**1):  
                            temp_text=source_text   

                            # select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[link_datum[0]][idx]][0])
                            # real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                            
                            # Added + Changed Part
                            selectID = sorted_sim[count][0]
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectID][0])
                            real_id.append(self.re_id[selectID])
                            
                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass

            elif task_template['id'] == '1-3-1-2':  
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                    link_datum.append(point)
                    
                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    # already_idx=[]
                    
                    # Added Part
                    similarity_dic = self._sim_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                    sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                    
                    while go_on and count < len(self.train_L1[link_datum[0]]) and ((not _do_abl) or count < _kval**1):  
                        temp_text=source_text   

                        # select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                        # idx=int(np.random.choice(select,1,replace=False)[0])
                        # already_idx.append(idx)
                        # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[link_datum[0]][idx]][0])
                        # real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])

                        # Added + Changed Part
                        selectID = sorted_sim[count][0]
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectID][0])
                        real_id.append(self.re_id[selectID])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])
                    target_text=[self.re_id[link_datum[1]],1]

                elif self.mode=='val':   
                    pass



            elif task_template['id'] == '1-3-2-1':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    temp_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                                temp_L2.append(ttt)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L2[random.randint(0, len(train_L2) - 1)]
                        link_datum.extend(points)
                        node_list=''    
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                                real_id.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                                # real_id.append(self.re_id[train_L2[idx][1]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else: 
                        node_list=''    
                        count=0
                        negative=random.randint(0,19716)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                                real_id.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                                # real_id.append(self.re_id[train_L2[idx][1]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass



            elif task_template['id'] == '1-3-2-2':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                    
                    while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L1_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops not in already_idx
                                select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                
                                if len(select_L1_hops) > 0:
                                    go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                                    select_L1 = None
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                        
                        if select_L1 is None:
                            go_on = False
                        else:
                            for pair in select_L1_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                            selectPair = select_L1_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                            real_id.append(self.re_id[selectPair[1]])
                            
                            # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            # real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])

                    target_text = [self.re_id[link_datum[2]],1]

                elif self.mode=='val':   
                    pass


            elif task_template['id'] == '1-3-2-3':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                                middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0])
                                id_1.append(self.re_id[selectPair[1]])
                                id_2.append(self.re_id[selectPair[0]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)

                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                                # id_1.append(self.re_id[train_L2[idx][1]])

                                # middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                                # id_2.append(self.re_id[train_L2[idx][0]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])

                        target_text = task_template['target'].format('yes')

                    else:  
                        temp_L2=self.train_L1[link_datum[1]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,19716)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                                middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0])
                                id_1.append(self.re_id[selectPair[1]])
                                id_2.append(self.re_id[selectPair[0]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                                # id_1.append(self.re_id[train_L2[idx][1]])
                                # middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                                # id_2.append(self.re_id[train_L2[idx][0]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-2-4':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0],self.train_L1[link_datum[0]])
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                    
                    while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L1_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops not in already_idx
                                select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                
                                if len(select_L1_hops) > 0:
                                    go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                                    select_L1 = None
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                        
                        if select_L1 is None:
                            go_on = False
                        else:
                            for pair in select_L1_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                            selectPair = select_L1_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0])
                            id_1.append(self.re_id[selectPair[1]])
                            id_2.append(self.re_id[selectPair[0]])
                            
                            # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)

                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            # id_1.append(self.re_id[train_L2[idx][1]])

                            # middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            # id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        id_1.pop(-1)
                        id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[link_datum[0]])
                    real_id.append(self.re_id[link_datum[1]])

                    target_text =[self.re_id[link_datum[2]],1]

                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-3-1':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    temp_L3=[]
                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0] and el!=ele[0]:
                                train_L3.append(ele+[el])
                                temp_L3.append(el)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L3[random.randint(0, len(train_L3) - 1)]
                        link_datum.extend(points)
                        node_list=''   
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3): 
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                                real_id.append(self.re_id[selectPair[2]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                                # real_id.append(self.re_id[train_L3[idx][2]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else: 

                        node_list=''    
                        count=0
                        negative=random.randint(0,19716)                                               
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                                real_id.append(self.re_id[selectPair[2]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                                # real_id.append(self.re_id[train_L3[idx][2]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])

                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass

            elif task_template['id'] == '1-3-3-2':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0] and el!=ele[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                    L2_set = list(set([i[1] for i in train_L2]))
                    simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                    
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                    train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                    
                    while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L2 = None
                        select_L1_hops = None
                        select_L2_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops in train_L2_undepleted
                                select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                
                                # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                if len(select_L1_hops) > 0:
                                    go_on_L2 = True
                                    
                                    while go_on_L2:
                                        toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                        if len(toselectfrom) > 0:
                                            # select random node from select_L1_hops
                                            weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                            select_L2_ID = np.random.choice(
                                                range(len(toselectfrom)),
                                                p = weights_L2 / np.sum(weights_L2)
                                            )
                                            select_L2 = toselectfrom[select_L2_ID]
                                            
                                            # check number of L3 hops not in already_idx
                                            select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                            
                                            if len(select_L2_hops) > 0:
                                                go_on_L2 = False
                                            else:
                                                train_L2_undepleted.remove(select_L2)
                                                select_L2 = None
                                        else:
                                            go_on_L2 = False
                                            
                                    if select_L2 is None:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                    else:
                                        go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                        
                        if select_L2 is None:
                            go_on = False
                        else:
                            for pair in select_L2_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                            selectPair = select_L2_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                            real_id.append(self.re_id[selectPair[2]])
                            
                            # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            # real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])

                    target_text = [self.re_id[link_datum[3]],1]


                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-3-3':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0] and el!=ele[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        
                        node_list=''   
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                                middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0],self.node_feature[selectPair[1]][0])
                                id_1.append(self.re_id[selectPair[2]])
                                id_2.append(self.re_id[selectPair[0]])
                                id_2.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)

                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                                # id_1.append(self.re_id[train_L3[idx][2]])
                                # middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                                # id_2.append(self.re_id[train_L3[idx][0]])
                                # id_2.append(self.re_id[train_L3[idx][1]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])

                        target_text = task_template['target'].format('yes')

                    else:  
                        temp_L3=self.train_L1[link_datum[2]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,19716)
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,19716)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                                middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0],self.node_feature[selectPair[1]][0])
                                id_1.append(self.re_id[selectPair[2]])
                                id_2.append(self.re_id[selectPair[0]])
                                id_2.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)

                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                                # id_1.append(self.re_id[train_L3[idx][2]])
                                # middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                                # id_2.append(self.re_id[train_L3[idx][0]])
                                # id_2.append(self.re_id[train_L3[idx][1]])

                                source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-3-4': 
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]  
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0] and el!=ele[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(link_datum[0], self.train_L1[link_datum[0]])
                    L2_set = list(set([i[1] for i in train_L2]))
                    simdeg_L2_dict = self._simdeg_dic_calc(link_datum[0], L2_set)
                    
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[link_datum[0]]]
                    train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                    
                    while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L2 = None
                        select_L1_hops = None
                        select_L2_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops in train_L2_undepleted
                                select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                
                                # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                if len(select_L1_hops) > 0:
                                    go_on_L2 = True
                                    
                                    while go_on_L2:
                                        toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                        if len(toselectfrom) > 0:
                                            # select random node from select_L1_hops
                                            weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                            select_L2_ID = np.random.choice(
                                                range(len(toselectfrom)),
                                                p = weights_L2 / np.sum(weights_L2)
                                            )
                                            select_L2 = toselectfrom[select_L2_ID]
                                            
                                            # check number of L3 hops not in already_idx
                                            select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                            
                                            if len(select_L2_hops) > 0:
                                                go_on_L2 = False
                                            else:
                                                train_L2_undepleted.remove(select_L2)
                                                select_L2 = None
                                        else:
                                            go_on_L2 = False
                                            
                                    if select_L2 is None:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                    else:
                                        go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                        
                        if select_L2 is None:
                            go_on = False
                        else:
                            for pair in select_L2_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(link_datum[0], pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                            selectPair = select_L2_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0],self.node_feature[selectPair[1]][0])
                            id_1.append(self.re_id[selectPair[2]])
                            id_2.append(self.re_id[selectPair[0]])
                            id_2.append(self.re_id[selectPair[1]])
                            
                            # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            # id_1.append(self.re_id[train_L3[idx][2]])
                            # middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            # id_2.append(self.re_id[train_L3[idx][0]])
                            # id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        id_1.pop(-1)
                        id_2.pop(-1)
                        id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[link_datum[0]])
                    real_id.append(self.re_id[link_datum[1]])
                    real_id.append(self.re_id[link_datum[2]])

                    target_text =[self.re_id[link_datum[3]],1]

                elif self.mode=='val':   
                    pass



        
        elif task_name == 'classification':
            if self.mode=='train':   
                point=self.classification[datum_idx]  # Central Node
            elif self.mode=='val':
                if cate=='inductive':
                    pass
                elif cate=='transductive':
                    point=self.transductive[datum_idx]

            label=self.label_map[point]
            
            negative=str(np.random.choice(list(set(self.LA).difference({label})),1,replace=False)[0])

            tit=self.node_feature[point][0]

            if task_template['id'] == '5-5-5-5': 
                abs=self.node_feature[point][1] 
                rand_prob=random.random()
                if rand_prob>0.5:
                    source_text =task_template['source'].format('<extra_id_0>', abs, '<extra_id_0>', label)
                    real_id=[self.re_id[point],self.re_id[point]]
                    target_text = task_template['target'].format('yes')
                else:
                    source_text =task_template['source'].format('<extra_id_0>', abs, '<extra_id_0>', negative)
                    real_id=[self.re_id[point],self.re_id[point]]
                    target_text = task_template['target'].format('no')

            elif task_template['id']=='6-6-6-6':
                abss=self.node_feature[point][1] 
                while len(self.tokenizer.encode(tit))>100:
                    tit=self.tokenizer.decode(self.tokenizer.encode(tit)[:99],skip_special_tokens=True)
                while len(self.tokenizer.encode(abss))>1024:
                    abss=self.tokenizer.decode(self.tokenizer.encode(abss)[:1023],skip_special_tokens=True)
                source_text =task_template['source'].format('<extra_id_0>',tit, abss, '<extra_id_0>',tit)
                real_id=[self.re_id[point],self.re_id[point]]   
                target_text = task_template['target'].format(label)

            elif task_template['id']=='6-6-6-7':
                abss=self.node_feature[point][1] 
                while len(self.tokenizer.encode(tit))>100:
                    tit=self.tokenizer.decode(self.tokenizer.encode(tit)[:99],skip_special_tokens=True)
                while len(self.tokenizer.encode(abss))>1024:
                    abss=self.tokenizer.decode(self.tokenizer.encode(abss)[:1023],skip_special_tokens=True)
                source_text =task_template['source'].format('<extra_id_0>',tit, abss, '<extra_id_0>',tit)
                real_id=[self.re_id[point],self.re_id[point]]   
                target_text = task_template['target'].format(label)


            elif task_template['id'] == '2-1-1-1':
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        # already_idx=[]
                        
                        # Added Part
                        similarity_dic = self._sim_dic_calc(point,self.train_L1[point])
                        sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                        
                        while go_on and count < len(self.train_L1[point]) and ((not _do_abl) or count < _kval**1):  
                            temp_text=source_text   

                            # select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # real_id.append(self.re_id[self.train_L1[point][idx]])

                            # Added + Changed Part
                            selectID = sorted_sim[count][0]
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[selectID])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])    
                        target_text = task_template['target'].format('yes')

                    else:   
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        # already_idx=[]
                        
                        # Added Part
                        similarity_dic = self._sim_dic_calc(point,self.train_L1[point])
                        sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                        
                        while go_on and count < len(self.train_L1[point]) and ((not _do_abl) or count < _kval**1): 
                            temp_text=source_text   

                            # select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # real_id.append(self.re_id[self.train_L1[point][idx]])
                            
                            # Added + Changed Part
                            selectID = sorted_sim[count][0]
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[selectID])
                            
                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')

                elif self.mode=='val':   
                    pass


            elif task_template['id'] == '2-1-1-2': #원래 2-1-1-2임
                if self.mode!=None: 
                    
                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    
                    # Added Part
                    similarity_dic = self._sim_dic_calc(point,self.train_L1[point])
                    sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                    
                    while go_on and count < len(self.train_L1[point]) and ((not _do_abl) or count < _kval**1):  
                        temp_text=source_text   

                        # select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        # idx=int(np.random.choice(select,1,replace=False)[0])
                        # already_idx.append(idx)
                        # node_list=node_list+'<extra_id_0>, '
                        # real_id.append(self.re_id[self.train_L1[point][idx]])

                        # Added + Changed Part
                        selectID = sorted_sim[count][0]
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[selectID])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   
                    pass
            
            elif task_template['id'] == '2-1-2-1': #원래 2-1-2-1임
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text  
                            
                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                real_id.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # real_id.append(self.re_id[train_L2[idx][1]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')
                        
                        # print(f"train_L2: {train_L2}")
                        # print(f"no. train_L2: {len(train_L2)}")
                        # print(f"real_id: {real_id}")
                        # print(f"no. real_id: {len(real_id)}")
                        # print(f"token len: {len(self.tokenizer.tokenize(source_text))}")
                        # print(f"source_text: {source_text}")

                    else:  
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                real_id.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # real_id.append(self.re_id[train_L2[idx][1]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)


                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
                
            elif task_template['id'] == '2-1-2-2':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                    
                    while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                        temp_text=source_text   
                        
                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L1_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops not in already_idx
                                select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                
                                if len(select_L1_hops) > 0:
                                    go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                                    select_L1 = None
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                        
                        if select_L1 is None:
                            go_on = False
                        else:
                            for pair in select_L1_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                            selectPair = select_L1_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[selectPair[1]])
                            
                            # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': 
                    pass
                
            elif task_template['id'] == '2-1-2-3':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                middle_list=middle_list+'<extra_id_0>, '
                                id_1.append(self.re_id[selectPair[1]])
                                id_2.append(self.re_id[selectPair[0]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # id_1.append(self.re_id[train_L2[idx][1]])

                                # middle_list=middle_list+'<extra_id_0>, '
                                # id_2.append(self.re_id[train_L2[idx][0]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', label)
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2): 
                            temp_text=source_text  
                            
                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                middle_list=middle_list+'<extra_id_0>, '
                                id_1.append(self.re_id[selectPair[1]])
                                id_2.append(self.re_id[selectPair[0]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # id_1.append(self.re_id[train_L2[idx][1]])
                                # middle_list=middle_list+'<extra_id_0>, '
                                # id_2.append(self.re_id[train_L2[idx][0]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        real_id=real_id+id_1+id_2

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-1-2-4':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                    
                    while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                        temp_text=source_text 
                        
                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L1_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops not in already_idx
                                select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                
                                if len(select_L1_hops) > 0:
                                    go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                                    select_L1 = None
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                        
                        if select_L1 is None:
                            go_on = False
                        else:
                            for pair in select_L1_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                            selectPair = select_L1_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'<extra_id_0>, '
                            middle_list=middle_list+'<extra_id_0>, '
                            id_1.append(self.re_id[selectPair[1]])
                            id_2.append(self.re_id[selectPair[0]])
                            
                            # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # id_1.append(self.re_id[train_L2[idx][1]])
                            # middle_list=middle_list+'<extra_id_0>, '
                            # id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>')
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)

                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  
                    pass




            elif task_template['id'] == '2-1-3-1':    
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=point and el!=ele[0]:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''   
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   
                            
                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                real_id.append(self.re_id[selectPair[2]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # real_id.append(self.re_id[train_L3[idx][2]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', label)
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else: 
                        real_id=[self.re_id[point]]

                        
                        node_list=''    
                        count=0
                        

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'<extra_id_0>, '
                                real_id.append(self.re_id[selectPair[2]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # real_id.append(self.re_id[train_L3[idx][2]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                                
                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass
                
            elif task_template['id'] == '2-1-3-2': # 원래 2-1-3-2임
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                            
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=point and el!=ele[0]:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]

                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                    L2_set = list(set([i[1] for i in train_L2]))
                    simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                    
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                    train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                    
                    while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L2 = None
                        select_L1_hops = None
                        select_L2_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops in train_L2_undepleted
                                select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                
                                # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                if len(select_L1_hops) > 0:
                                    go_on_L2 = True
                                    
                                    while go_on_L2:
                                        toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                        if len(toselectfrom) > 0:
                                            # select random node from select_L1_hops
                                            weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                            select_L2_ID = np.random.choice(
                                                range(len(toselectfrom)),
                                                p = weights_L2 / np.sum(weights_L2)
                                            )
                                            select_L2 = toselectfrom[select_L2_ID]
                                            
                                            # check number of L3 hops not in already_idx
                                            select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                            
                                            if len(select_L2_hops) > 0:
                                                go_on_L2 = False
                                            else:
                                                train_L2_undepleted.remove(select_L2)
                                                select_L2 = None
                                        else:
                                            go_on_L2 = False
                                            
                                    if select_L2 is None:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                    else:
                                        go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                        
                        if select_L2 is None:
                            go_on = False
                        else:
                            for pair in select_L2_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                            selectPair = select_L2_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[selectPair[2]])
                            
                            # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'<extra_id_0>, '
                            # real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)
                    
                    #print(f"train_L3: {train_L3}")
                    #print(f"no. train_L2: {len(train_L3)}")
                    #print(f"real_id: {real_id}")
                    #print(f"no. real_id: {len(real_id)}")
                    #print(f"token len: {len(self.tokenizer.tokenize(source_text))}")
                    #print(f"source_text: {source_text}")


                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '2-1-3-3':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=point and el!=ele[0]:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>',label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   
                            
                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                
                                node_list=node_list+'<extra_id_0>, '
                                middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                                id_1.append(self.re_id[selectPair[2]])
                                id_2.append(self.re_id[selectPair[0]])
                                id_2.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'<extra_id_0>, '
                                # id_1.append(self.re_id[train_L3[idx][2]])

                                # middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                                # id_2.append(self.re_id[train_L3[idx][0]])
                                # id_2.append(self.re_id[train_L3[idx][1]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>',label)
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                        
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>',negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                
                                node_list=node_list+'<extra_id_0>, '
                                middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                                id_1.append(self.re_id[selectPair[2]])
                                id_2.append(self.re_id[selectPair[0]])
                                id_2.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)

                                # node_list=node_list+'<extra_id_0>, '
                                # id_1.append(self.re_id[train_L3[idx][2]])

                                # middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                                # id_2.append(self.re_id[train_L3[idx][0]])
                                # id_2.append(self.re_id[train_L3[idx][1]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass
            elif task_template['id'] == '2-1-3-4':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]  
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]: 
                            if el!=point and el!=ele[0]:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                    L2_set = list(set([i[1] for i in train_L2]))
                    simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                    
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                    train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                    
                    while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3): 
                        temp_text=source_text   

                        
                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L2 = None
                        select_L1_hops = None
                        select_L2_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops in train_L2_undepleted
                                select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                
                                # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                if len(select_L1_hops) > 0:
                                    go_on_L2 = True
                                    
                                    while go_on_L2:
                                        toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                        if len(toselectfrom) > 0:
                                            # select random node from select_L1_hops
                                            weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                            select_L2_ID = np.random.choice(
                                                range(len(toselectfrom)),
                                                p = weights_L2 / np.sum(weights_L2)
                                            )
                                            select_L2 = toselectfrom[select_L2_ID]
                                            
                                            # check number of L3 hops not in already_idx
                                            select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                            
                                            if len(select_L2_hops) > 0:
                                                go_on_L2 = False
                                            else:
                                                train_L2_undepleted.remove(select_L2)
                                                select_L2 = None
                                        else:
                                            go_on_L2 = False
                                            
                                    if select_L2 is None:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                    else:
                                        go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                        
                        if select_L2 is None:
                            go_on = False
                        else:
                            for pair in select_L2_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                            selectPair = select_L2_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            
                            node_list=node_list+'<extra_id_0>, '
                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_1.append(self.re_id[selectPair[2]])
                            id_2.append(self.re_id[selectPair[0]])
                            id_2.append(self.re_id[selectPair[1]])
                            
                            # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)

                            # node_list=node_list+'<extra_id_0>, '
                            # id_1.append(self.re_id[train_L3[idx][2]])
                            # middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            # id_2.append(self.re_id[train_L3[idx][0]])
                            # id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>')
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   
                    pass
            


            elif task_template['id'] == '2-3-1-1':    
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        # already_idx=[]
                        
                        # Added Part
                        similarity_dic = self._sim_dic_calc(point,self.train_L1[point])
                        sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                        
                        
                        while go_on and count < len(self.train_L1[point]) and ((not _do_abl) or count < _kval**1):  
                            temp_text=source_text   

                            # select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                            # real_id.append(self.re_id[self.train_L1[point][idx]])

                            # Added + Changed Part
                            selectID = sorted_sim[count][0]
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectID][0])
                            real_id.append(self.re_id[selectID])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])     
                        target_text = task_template['target'].format('yes')

                    else:     
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        # already_idx=[]
                        
                        # Added Part
                        similarity_dic = self._sim_dic_calc(point,self.train_L1[point])
                        sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                        
                        while go_on and count < len(self.train_L1[point]) and ((not _do_abl) or count < _kval**1):  
                            temp_text=source_text   

                            # select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                            # real_id.append(self.re_id[self.train_L1[point][idx]])

                            # Added + Changed Part
                            selectID = sorted_sim[count][0]
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectID][0])
                            real_id.append(self.re_id[selectID])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')

                elif self.mode=='val':  
                    pass


            elif task_template['id'] == '2-3-1-2':
                if self.mode!=None: 
                    
                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    # already_idx=[]
                    
                    # Added Part
                    similarity_dic = self._sim_dic_calc(point,self.train_L1[point])
                    sorted_sim = sorted(similarity_dic.items(), key = lambda x: (x[1], x[0]), reverse = True)
                    
                    while go_on and count < len(self.train_L1[point]) and ((not _do_abl) or count < _kval**1):  
                        temp_text=source_text   

                        # select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        # idx=int(np.random.choice(select,1,replace=False)[0])
                        # already_idx.append(idx)
                        # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                        # real_id.append(self.re_id[self.train_L1[point][idx]])
                        
                        # Added + Changed Part
                        selectID = sorted_sim[count][0]
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectID][0])
                        real_id.append(self.re_id[selectID])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit)
                        
                        # print("select: " + str(select))
                        # print("idx: " +  str(idx))
                        # print("node_list: " + str(node_list))
                        # print("source_text: " + source_text)
                        # exit()

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)

                elif self.mode=='val': 
                    pass
            
            elif task_template['id'] == '2-3-2-1':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2): 
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                                real_id.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                                # real_id.append(self.re_id[train_L2[idx][1]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                                real_id.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                                # real_id.append(self.re_id[train_L2[idx][1]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-2':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                    
                    while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L1_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops not in already_idx
                                select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                
                                if len(select_L1_hops) > 0:
                                    go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                                    select_L1 = None
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                        
                        if select_L1 is None:
                            go_on = False
                        else:
                            for pair in select_L1_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                            selectPair = select_L1_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                            real_id.append(self.re_id[selectPair[1]])
                            
                            # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            # real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit)
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-3':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''   
                        middle_list=''
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text  

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                                middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0])
                                id_1.append(self.re_id[selectPair[1]])
                                id_2.append(self.re_id[selectPair[0]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                                # id_1.append(self.re_id[train_L2[idx][1]])

                                # middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                                # id_2.append(self.re_id[train_L2[idx][0]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit, label)
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        
                        while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L1_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops not in already_idx
                                    select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                    
                                    if len(select_L1_hops) > 0:
                                        go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                            
                            if select_L1 is None:
                                go_on = False
                            else:
                                for pair in select_L1_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                                selectPair = select_L1_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                                middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0])
                                id_1.append(self.re_id[selectPair[1]])
                                id_2.append(self.re_id[selectPair[0]])
                                
                                # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                                # id_1.append(self.re_id[train_L2[idx][1]])

                                # middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                                # id_2.append(self.re_id[train_L2[idx][0]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        real_id=real_id+id_1+id_2

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-4':
                # Added part
                simdeg_L1_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(point,self.train_L1[point])
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                    
                    while go_on and count < len(train_L2) and ((not _do_abl) or count < _kval**2):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L1_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops not in already_idx
                                select_L1_hops = [hop for hop in train_L2 if hop[0] == select_L1 and hop not in already_idx]
                                
                                if len(select_L1_hops) > 0:
                                    go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                                    select_L1 = None
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L1_undepleted left, select_L1 will be None
                        
                        if select_L1 is None:
                            go_on = False
                        else:
                            for pair in select_L1_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L1_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L1_hops)), p=weights)
                            selectPair = select_L1_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[1]][0])
                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0])
                            id_1.append(self.re_id[selectPair[1]])
                            id_2.append(self.re_id[selectPair[0]])
                            
                            # select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            # id_1.append(self.re_id[train_L2[idx][1]])

                            # middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            # id_2.append(self.re_id[train_L2[idx][0]])


                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit)
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)

                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  
                    pass




            elif task_template['id'] == '2-3-3-1':    
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])
                    
                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=point and el!=ele[0]:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''   
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text  

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                                real_id.append(self.re_id[selectPair[2]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                                # real_id.append(self.re_id[train_L3[idx][2]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit, label)
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        

                        node_list=''    
                        count=0
                        

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                                real_id.append(self.re_id[selectPair[2]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                                # real_id.append(self.re_id[train_L3[idx][2]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass
                
            elif task_template['id'] == '2-3-3-2':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])
                    
                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=point and el!=ele[0]:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]

                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                    L2_set = list(set([i[1] for i in train_L2]))
                    simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                    
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                    train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                    
                    while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L2 = None
                        select_L1_hops = None
                        select_L2_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops in train_L2_undepleted
                                select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                
                                # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                if len(select_L1_hops) > 0:
                                    go_on_L2 = True
                                    
                                    while go_on_L2:
                                        toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                        if len(toselectfrom) > 0:
                                            # select random node from select_L1_hops
                                            weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                            select_L2_ID = np.random.choice(
                                                range(len(toselectfrom)),
                                                p = weights_L2 / np.sum(weights_L2)
                                            )
                                            select_L2 = toselectfrom[select_L2_ID]
                                            
                                            # check number of L3 hops not in already_idx
                                            select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                            
                                            if len(select_L2_hops) > 0:
                                                go_on_L2 = False
                                            else:
                                                train_L2_undepleted.remove(select_L2)
                                                select_L2 = None
                                        else:
                                            go_on_L2 = False
                                            
                                    if select_L2 is None:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                    else:
                                        go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                        
                        if select_L2 is None:
                            go_on = False
                        else:
                            for pair in select_L2_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                            selectPair = select_L2_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                            real_id.append(self.re_id[selectPair[2]])
                            
                            # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)
                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            # real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)


                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '2-3-3-3':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]  
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=point and el!=ele[0]:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit,label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text  

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                                middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0],self.node_feature[selectPair[1]][0])
                                id_1.append(self.re_id[selectPair[2]])
                                id_2.append(self.re_id[selectPair[0]])
                                id_2.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)
                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                                # id_1.append(self.re_id[train_L3[idx][2]])

                                # middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                                # id_2.append(self.re_id[train_L3[idx][0]])
                                # id_2.append(self.re_id[train_L3[idx][1]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit,label)
                                

                                count+=1   
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                        
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit,negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        
                        #추가한 부분
                        simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                        L2_set = list(set([i[1] for i in train_L2]))
                        simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                        
                        train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                        train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                        
                        while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                            temp_text=source_text   

                            #추가한 부분
                            go_on_L1 = True
                            select_L1 = None
                            select_L2 = None
                            select_L1_hops = None
                            select_L2_hops = None
                            while go_on_L1:
                                if len(train_L1_undepleted) > 0:
                                    # select random node from train_L1_undepleted
                                    weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                    select_L1 = np.random.choice(
                                        train_L1_undepleted,
                                        p = weights/np.sum(weights)
                                    )
                                    
                                    # check number of L2 hops in train_L2_undepleted
                                    select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                    
                                    # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                    if len(select_L1_hops) > 0:
                                        go_on_L2 = True
                                        
                                        while go_on_L2:
                                            toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                            if len(toselectfrom) > 0:
                                                # select random node from select_L1_hops
                                                weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                                select_L2_ID = np.random.choice(
                                                    range(len(toselectfrom)),
                                                    p = weights_L2 / np.sum(weights_L2)
                                                )
                                                select_L2 = toselectfrom[select_L2_ID]
                                                
                                                # check number of L3 hops not in already_idx
                                                select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                                
                                                if len(select_L2_hops) > 0:
                                                    go_on_L2 = False
                                                else:
                                                    train_L2_undepleted.remove(select_L2)
                                                    select_L2 = None
                                            else:
                                                go_on_L2 = False
                                                
                                        if select_L2 is None:
                                            train_L1_undepleted.remove(select_L1)
                                            select_L1 = None
                                        else:
                                            go_on_L1 = False
                                    else:
                                        train_L1_undepleted.remove(select_L1)
                                else:
                                    go_on_L1 = False
                            # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                            
                            if select_L2 is None:
                                go_on = False
                            else:
                                for pair in select_L2_hops:
                                    if pair[-1] not in sim_dict:
                                        sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                                simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                                weights = simlist/np.sum(simlist)
                                selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                                selectPair = select_L2_hops[selectPairIdx]
                                
                                already_idx.append(selectPair)
                                
                                node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                                middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0],self.node_feature[selectPair[1]][0])
                                id_1.append(self.re_id[selectPair[2]])
                                id_2.append(self.re_id[selectPair[0]])
                                id_2.append(self.re_id[selectPair[1]])
                                
                                # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                                # idx=int(np.random.choice(select,1,replace=False)[0])
                                # already_idx.append(idx)

                                # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                                # id_1.append(self.re_id[train_L3[idx][2]])

                                # middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                                # id_2.append(self.re_id[train_L3[idx][0]])
                                # id_2.append(self.re_id[train_L3[idx][1]])

                                source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                                

                                count+=1  
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass
                
            elif task_template['id'] == '2-3-3-4':
                # Added part
                simdeg_L1_dict = {}
                simdeg_L2_dict = {}
                sim_dict = {}
                
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    for ele in train_L2:
                        for el in self.train_L1[ele[1]]:
                            if el!=point and el!=ele[0]:
                                train_L3.append(ele+[el])
                        

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    
                    #추가한 부분
                    simdeg_L1_dict = self._simdeg_dic_calc(point, self.train_L1[point])
                    L2_set = list(set([i[1] for i in train_L2]))
                    simdeg_L2_dict = self._simdeg_dic_calc(point, L2_set)
                    
                    train_L1_undepleted=[nodeid for nodeid in self.train_L1[point]]
                    train_L2_undepleted=[nodeidpair for nodeidpair in train_L2]
                    
                    while go_on and count < len(train_L3) and ((not _do_abl) or count < _kval**3):  
                        temp_text=source_text   

                        #추가한 부분
                        go_on_L1 = True
                        select_L1 = None
                        select_L2 = None
                        select_L1_hops = None
                        select_L2_hops = None
                        while go_on_L1:
                            if len(train_L1_undepleted) > 0:
                                # select random node from train_L1_undepleted
                                weights = [simdeg_L1_dict[nodeid] for nodeid in train_L1_undepleted]
                                select_L1 = np.random.choice(
                                    train_L1_undepleted,
                                    p = weights/np.sum(weights)
                                )
                                
                                # check number of L2 hops in train_L2_undepleted
                                select_L1_hops = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                
                                # 만약 L1의 neighbor 중에서 선택할게 남아있다면.
                                if len(select_L1_hops) > 0:
                                    go_on_L2 = True
                                    
                                    while go_on_L2:
                                        toselectfrom = [hop for hop in train_L2_undepleted if hop[0] == select_L1]
                                        if len(toselectfrom) > 0:
                                            # select random node from select_L1_hops
                                            weights_L2 = [simdeg_L2_dict[pair[1]] for pair in toselectfrom]
                                            select_L2_ID = np.random.choice(
                                                range(len(toselectfrom)),
                                                p = weights_L2 / np.sum(weights_L2)
                                            )
                                            select_L2 = toselectfrom[select_L2_ID]
                                            
                                            # check number of L3 hops not in already_idx
                                            select_L2_hops = [hop for hop in train_L3 if hop[0] == select_L1 and hop[1] == select_L2[1] and hop not in already_idx]
                                            
                                            if len(select_L2_hops) > 0:
                                                go_on_L2 = False
                                            else:
                                                train_L2_undepleted.remove(select_L2)
                                                select_L2 = None
                                        else:
                                            go_on_L2 = False
                                            
                                    if select_L2 is None:
                                        train_L1_undepleted.remove(select_L1)
                                        select_L1 = None
                                    else:
                                        go_on_L1 = False
                                else:
                                    train_L1_undepleted.remove(select_L1)
                            else:
                                go_on_L1 = False
                        # At this point, if there is no train_L2_undepleted left, select_L2 will be None
                        
                        if select_L2 is None:
                            go_on = False
                        else:
                            for pair in select_L2_hops:
                                if pair[-1] not in sim_dict:
                                    sim_dict[pair[-1]] = self._sim_calc(point, pair[-1])
                            simlist = [sim_dict[pair[-1]] for pair in select_L2_hops]
                            weights = simlist/np.sum(simlist)
                            selectPairIdx = np.random.choice(range(len(select_L2_hops)), p=weights)
                            selectPair = select_L2_hops[selectPairIdx]
                            
                            already_idx.append(selectPair)
                            
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[selectPair[2]][0])
                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[selectPair[0]][0],self.node_feature[selectPair[1]][0])
                            id_1.append(self.re_id[selectPair[2]])
                            id_2.append(self.re_id[selectPair[0]])
                            id_2.append(self.re_id[selectPair[1]])
                            
                            # select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            # idx=int(np.random.choice(select,1,replace=False)[0])
                            # already_idx.append(idx)

                            # node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            # id_1.append(self.re_id[train_L3[idx][2]])
                            # middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            # id_2.append(self.re_id[train_L3[idx][0]])
                            # id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit)
                                

                            count+=1  

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  
                    pass

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
            

        input_ids = self.tokenizer.encode(source_text)
        extra_num=0
        for idi in range(len(input_ids)):
            idid=input_ids[idi]
            if idid==32000:
                input_ids[idi]=real_id[extra_num]
                extra_num+=1
        if extra_num!=len(real_id):
            print(task_template['id'])
            print(source_text)
            print(extra_num,len(real_id))
        assert extra_num==len(real_id)

                
        if task_template['id'].startswith('1') and (task_template['id'].endswith('2') or task_template['id'].endswith('4')):
            target_text=[1]+target_text[:-1]
            target_ids=target_text   
        else:
            target_ids = self.tokenizer.encode(target_text)

        out_dict['input_ids'] = input_ids
        out_dict['input_length'] = len(input_ids)
        out_dict['target_ids'] = target_ids
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight
        out_dict['temp_id'] = task_template['id']

        if self.mode=='val':
            out_dict['cate']='None' if task_template['task']!='classification' else cate

        return out_dict
    
    # STEP 5: Update beginning of collate_fn where it currently starts
    def collate_fn(self, batch):   
        batch_entry = {}
        
        # Add SNS and PinSAGE computations for training mode
        if self.mode == 'train':
            with torch.no_grad():
                node_embeds = self.llama_embed.clone().detach()
            sns_loss = self.get_sns_loss(node_embeds)
            batch_entry['sns_loss'] = sns_loss
            
            # Add neighbor aggregations for nodes in batch
            neighbor_aggs = []
            for i, entry in enumerate(batch):
                if 'task' in entry and entry['task'] in ['link', 'classification']:
                    datum_info = self.datum_info[i]  # Use batch index directly
                    if len(datum_info) >= 3:
                        node_idx = datum_info[2]
                        agg = self.aggregate_neighbors(node_idx, node_embeds)
                        neighbor_aggs.append(agg)
            
            if neighbor_aggs:
                batch_entry['neighbor_aggregations'] = torch.stack(neighbor_aggs)

        B = len(batch)

        args = self.args

        if self.mode=='train':
            S_W_L = max(entry['input_length']+entry['target_length']+1 for entry in batch)   
            target_ids = torch.ones(B, S_W_L, dtype=torch.long) * (-100)   
        else:
            S_W_L = max(entry['input_length'] for entry in batch)  
            target_ids=None

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        target_text = []
        temp_ids=[]
        cate=[]

        for i, entry in enumerate(batch):
            if self.mode=='train':
                input_ids[i, -(entry['input_length']+entry['target_length']+1):] = torch.LongTensor(entry['input_ids']+entry['target_ids']+[2])
                target_ids[i, -(entry['target_length']):] = torch.LongTensor(entry['target_ids'][1:]+[2])
            else:
                input_ids[i, -(entry['input_length']):] = torch.LongTensor(entry['input_ids'])

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

            if 'temp_id' in entry:
                temp_ids.append(entry['temp_id'])

            if 'cate' in entry:
                cate.append(entry['cate'])
            
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id).to(dtype=input_ids.dtype, device=input_ids.device)

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['attn_mask']= attn_mask

        batch_entry['loss_weights'] = loss_weights
        batch_entry['temp_ids'] = temp_ids 
        if len(cate)!=0:
            batch_entry['cate'] = cate

        return batch_entry     
