import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd
from tqdm import tqdm

import pickle

from transformers import BertTokenizer, BertModel

import nltk

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

#BERT GENERATION
# Set a random seed
random_seed = 11451419198109315 
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Download and load NLTK punkt tokenizer
nltk.download('punkt_tab')
punkt = nltk.data.load('tokenizers/punkt/english.pickle')

# Load pickle file
raw = load_pickle('final_pubmed_node_feature.pkl')

final_tensor = None

for i in tqdm(range(len(raw))):
    
    text = ''
    
    abstract = punkt.tokenize(raw[i][1])
    
    token_sentences = []
    token_sentences.append(tokenizer.tokenize(raw[i][0])) # title
    for sentence in abstract:
        token_sentences.append(tokenizer.tokenize(sentence))
    
    # Pad sentences
    for j in range(len(token_sentences)):
        if j == 0:
            token_sentences[j] = ['[CLS]'] + token_sentences[j] + ['[SEP]']
        else:
            token_sentences[j] = token_sentences[j] + ['[SEP]']
        
    # Merge sentences
    tokens = []
    for sentence in token_sentences:
        tokens = tokens + sentence
    
    # Adjust size to 512 tokens
    if (len(tokens) > 512):
        tokens = tokens[:512]
    elif (len(tokens) < 512):
        tokens = tokens + ['[PAD]' for _ in range(512 - len(tokens))]

    # Create input tensors
    sent_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor(sent_ids).unsqueeze(0) 
    
    attn_mask = [ 1 if token != '[PAD]' else 0 for token in tokens]
    attn_mask = torch.tensor(attn_mask).unsqueeze(0)
    
    seg_ids = [0 for _ in range(len(tokens))] 
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
    with torch.no_grad():
        output = model(token_ids, attention_mask=attn_mask,token_type_ids=seg_ids)
        pooler_output = output[1]
    
    if final_tensor is None:
        final_tensor = pooler_output.detach().clone()
    else:
        final_tensor = torch.cat((final_tensor,pooler_output),dim=0)
        
save_pickle(final_tensor, 'final_pubmed_node_feature_BERT.pkl')

print(final_tensor)
print(final_tensor.shape)
    