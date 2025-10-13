import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim,emb_dim)
        self.ff1 = nn.Linear(emb_dim, 4*emb_dim)
        self.ff2 = nn.Linear(4*emb_dim,emb_dim)
        self.n_heads = n_heads
        self.head_dim = (emb_dim//n_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.gelu = nn.GELU()
    def forward(self, token_embeddings, attn_masks=None):
        batch_size, max_len, emb_dim = token_embeddings.shape
        Q = self.W_q(token_embeddings)
        K = self.W_k(token_embeddings)
        V = self.W_v(token_embeddings)

        Q = Q.view(-1,max_len,self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(-1,max_len,self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(-1,max_len,self.n_heads, self.head_dim).transpose(1,2)

        attn_score = torch.matmul(Q,K.transpose(-2,-1))/(self.head_dim**0.5)
        if  attn_masks is not None:
            mask = attn_masks.unsqueeze(1).unsqueeze(2)
            attn_score = attn_score.masked_fill(mask==0,float('-inf'))
        attn_score = torch.softmax(attn_score,dim=-1)
        new_embeddings = attn_score@V
        new_embeddings = new_embeddings.transpose(1,2).contiguous().view(-1,max_len,emb_dim)
        new_embeddings = self.W_o(new_embeddings)
        new_embeddings = self.norm1(token_embeddings + new_embeddings)
        old_embeddings = new_embeddings
        new_embeddings = self.gelu(self.ff1(new_embeddings))
        new_embeddings = self.ff2(new_embeddings)
        new_embeddings = self.norm2(new_embeddings+old_embeddings)

        return new_embeddings