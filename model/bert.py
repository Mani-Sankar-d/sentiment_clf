from blocks.encoder import Encoder
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, n_heads, n_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(max_len, embed_dim)
        self.dense_layer = nn.Linear(embed_dim,n_classes)
        self.dropout = nn.Dropout(0.1)
        self.encoder1 = Encoder(embed_dim,n_heads)
        self.encoder2 = Encoder(embed_dim,n_heads)
        self.encoder3 = Encoder(embed_dim,n_heads)
        self.encoder4 = Encoder(embed_dim,n_heads)
        
        self.encoder5 = Encoder(embed_dim,n_heads)
        self.encoder6 = Encoder(embed_dim,n_heads)
        self.encoder7 = Encoder(embed_dim,n_heads)
        self.encoder8 = Encoder(embed_dim,n_heads)

        
        self.encoder9 = Encoder(embed_dim,n_heads)
        self.encoder10 = Encoder(embed_dim,n_heads)
        self.encoder11 = Encoder(embed_dim,n_heads)
        self.encoder12 = Encoder(embed_dim,n_heads)

        self.encoder_stack = nn.ModuleList([Encoder(embed_dim, n_heads) for _ in range(12)])

    def forward(self, token_ids, attn_masks):
        batch_size ,max_len = token_ids.shape
        token_embeddings = self.embeddings(token_ids)
        pos_embeddings = self.pos_embeddings(torch.arange(0,max_len,device=token_ids.device).unsqueeze(0).expand(batch_size,-1))
        final_embeddings = token_embeddings+pos_embeddings
        for enc in self.encoder_stack:
            final_embeddings = enc(final_embeddings,attn_masks = attn_masks)
        cls_token = final_embeddings[:,0,:]
        final_embeddings = self.dropout(cls_token)
        res = self.dense_layer(cls_token)
        return res