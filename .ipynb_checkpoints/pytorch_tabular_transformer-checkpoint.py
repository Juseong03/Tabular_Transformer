import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn, einsum

from einops import rearrange

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
    
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            
            layers.append(linear)

            if is_last:
                continue

            if ind < (len(dims_pairs) -1) : 
                batch_norm = nn.BatchNorm1d(dim_out)
                layers.append(batch_norm)
                
            act = default(act, nn.ReLU())
            layers.append(act)
            
            dropout = nn.Dropout(0.1)
            layers.append(dropout)
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        x = (self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = (self.norm2(forward + x))
        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, dim_head, dropout, forward_expansion):
        super().__init__()
        self.embed_size = dim

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size = dim,
                    heads = dim_head,
                    dropout = dropout,
                    forward_expansion = forward_expansion,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x, x, x)
        return out
    
class EmbNN(torch.nn.Module) :
    def __init__(self, dim, Emb_hidden, dis_emb = 20) :
        super().__init__()
        self.dis_emb = dis_emb
        self.embedding = nn.Embedding(self.dis_emb, dim)
        
        self.fc1 = nn.Linear(1, Emb_hidden, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(Emb_hidden)
        self.fc2 = nn.Linear(Emb_hidden, self.dis_emb, bias=False)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x) :
        x = self.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        x = self.softmax(x)        
        output_emb = torch.matmul(x, self.embedding.weight)
        
        return output_emb
    
class NN(torch.nn.Module):
    def __init__(
        self, 
        cat_idxs, 
        con_idxs, 
        cat_dim, 
        emb_dim, 
        output_size,
        Emb_hidden = 32,
        dis_emb = 20,
        depth = 6,
        dim_head = 8,
        forward_expansion = 2,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2
    ):
        super(NN, self).__init__()
        
        self.cat_idxs = cat_idxs
        self.con_idxs = con_idxs
        
        self.num_continuous = len(con_idxs)
        
        # Categorical Embeddings
        self.num_categories = len(cat_dim)
        self.num_unique_categories = sum(cat_dim)
        self.special_tokens = 2
        
        self.total_cat_size = self.num_unique_categories + self.special_tokens
        self.categories_offset = F.pad(torch.tensor(list(cat_dim)), (1, 0), value = 2)
        self.categories_offset = self.categories_offset.cumsum(dim = -1)[:-1]
        self.categories_offset = self.categories_offset.to(DEVICE)

            
        self.cat_embed = nn.Embedding(self.total_cat_size, emb_dim).to(DEVICE)
        
        # Continuous Embeddings
        self.embeddings = []
            
        self.embnn = nn.ModuleList([])
        for _ in range(self.num_continuous) :
            self.embnn.append(
                EmbNN(
                    dim = emb_dim,
                    Emb_hidden = Emb_hidden,
                    dis_emb = dis_emb
                    
                ).to(DEVICE)
            )
            
        # transformer
        self.transformer = Transformer(
            dim = emb_dim,
            depth = depth,
            dim_head = dim_head,
            forward_expansion = forward_expansion,
            dropout = 0.1,
        ).to(DEVICE)
        print('Transformer : ', self.transformer)
        
        self.fc_input_size = (len(cat_idxs) + len(con_idxs)) * emb_dim
        print('MLP input size : ', self.fc_input_size)
        l = self.fc_input_size // 8
        
        hidden_dimensions = list(map(lambda t: int(l * t), mlp_hidden_mults))
        all_dimensions = [self.fc_input_size, *hidden_dimensions, output_size]
        print('MLP dimensions : ', all_dimensions)
        self.mlp = MLP(all_dimensions, act = mlp_act).to(DEVICE)

    def forward(self, x):
        # categorical 
        cat_x = x[:,self.cat_idxs]
        # convert with offset
        cat_input = self.categories_offset + cat_x
        cat_input = cat_input.type(torch.long)
        cat_input = self.cat_embed(cat_input)
        
        
        # Continuous
        con_x = x[:,self.con_idxs]
        
        # discretization with embedding
        con_emb = []    
        for i in range(self.num_continuous) :
            con_x_T = con_x.transpose(1,0)
            con_tmp = con_x_T[i]
            con_tmp = con_tmp.reshape(len(con_tmp),1)
            con_tmp = con_tmp.type(torch.float)
            
            con_emb.append(self.embnn[i](con_tmp))
        
        con_input = torch.stack(con_emb, dim = 1)
                    
        # concat embeddings (con, cat)
        input_values = torch.cat([cat_input, con_input], dim = 1)
        
        # Transformer
        output = self.transformer(input_values)
        
        # MLP
        output = output.flatten(1)
        output = self.mlp(output)
        
        return output