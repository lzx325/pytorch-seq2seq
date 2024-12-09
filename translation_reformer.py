import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

from reformer_pytorch import LSHAttention,ReformerLM, Autopadder

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))

if False: 
    n_hashes=16
    attn = LSHAttention(
        bucket_size = 64,
        n_hashes = n_hashes,
        causal = True,
        return_attn = True
    )

    qk = torch.randn(10, 1024, 128)
    v = torch.randn(10, 1024, 128)

    out, attn, buckets = attn(qk, v) # (10, 1024, 128)
    print(attn.shape)
    buckets=buckets.reshape(10,n_hashes,1024)

    batch_idx=0

    for seq_idx in range(1024):
        attended_idx_set=attn[batch_idx,seq_idx,:].cpu().numpy().nonzero()[0]
        bucket_idx_set=set()
        for hash_idx in range(n_hashes):
            bucket_idx=buckets[batch_idx,hash_idx,seq_idx].item()
            same_bucket=(buckets[batch_idx,hash_idx,:]==bucket_idx).cpu().numpy().nonzero()[0]
            bucket_idx_set.update(same_bucket)
            bucket_idx_set.update(same_bucket+1)
            bucket_idx_set.update(same_bucket-1)
            # attn contains the unsorted attention weights, provided return_attn is set to True (costly otherwise)
            # buckets will contain the bucket number (post-argmax) of each token of each batch 
        print(len(attended_idx_set),len(bucket_idx_set),len(set(attended_idx_set).intersection(bucket_idx_set))) 

if False:
    bucket_size=64
    n_hashes=16
    seq_len=1024
    batch_size=10
    dim=128
    attn = LSHAttention(
        bucket_size = bucket_size,
        n_hashes = n_hashes,
        causal = True,
        return_attn = True
    )

    qk = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)

    out, attn, buckets = attn(qk, v) # (10, 1024, 128)
    buckets=buckets.reshape(batch_size,n_hashes,seq_len)

    for batch_idx in range(batch_size):
        diag=torch.diag(attn[batch_idx,:,:]).cpu().numpy()
        print(np.histogram(diag))

if False:
    CONTEXT_LEN = 512
    SEQ_LEN = 8192

    model = ReformerLM(
        num_tokens= 20000,
        dim = 1024,
        depth = 1,
        max_seq_len = SEQ_LEN,
        ff_chunks = 8,
        causal = True
    )

    c = torch.randn(1, CONTEXT_LEN, 1024)
    x = torch.randint(0, 20000, (1, SEQ_LEN)).long()

    i_mask = torch.ones(1, SEQ_LEN).bool()
    c_mask = torch.ones(1, CONTEXT_LEN).bool()

    y = model(x, keys = c, input_mask = i_mask, context_mask = c_mask)

    print(x.shape)
    print(c.shape)
    print(y.shape)

if False:
    from reformer_pytorch import Reformer, Recorder

    model = Reformer(
        dim = 512,
        depth = 12,
        heads = 8,
        lsh_dropout = 0.1,
        causal = True
    ).cuda()

    # model = Recorder(model)

    x = torch.randn(1, 8192, 512).cuda()
    y = model(x)

    print(model.recordings[0]) # a list of attention weights and buckets for the first forward pass

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 2,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    ff_dropout = 0.1,
    post_attn_dropout = 0.1,
    layer_dropout = 0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    emb_dim = 128,        # embedding factorization for further memory savings
    dim_head = 64,        # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    attn_chunks = 8,      # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
    full_attn_thres = 1024, # use full attention if context length is less than set value
    reverse_thres = 1024,   # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value
    use_scale_norm = False,  # use scale norm from 'Transformers without tears' paper
    use_rezero = False,      # remove normalization and use rezero from 'ReZero is All You Need'
    one_value_head = False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
    weight_tie = False,           # tie parameters of each layer for no memory per additional depth
    weight_tie_embedding = False, # use token embedding for projection of output, some papers report better results
    n_local_attn_heads = 2,       # many papers suggest mixing local attention heads aids specialization and improves on certain tasks
    pkm_layers = (4,7),           # specify layers to use product key memory. paper shows 1 or 2 modules near the middle of the transformer is best
    pkm_num_keys = 128,           # defaults to 128, but can be increased to 256 or 512 as memory allows
    use_full_attn = False    # only turn on this flag to override and turn on full attention for all sequence lengths. for comparison with LSH to show that it is working
).cuda()
model=Autopadder(model)
x = torch.randint(0, 20000, (1, 20000)).long().cuda()
y = model(x) # (1, 8192, 20000)
import ipdb; ipdb.set_trace()