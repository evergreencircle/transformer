# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

embed_dim = 64
num_heads = 8
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

# %%
query = torch.rand(6, 32, embed_dim)  # (sequence_length, batch_size, embed_dim)
key = torch.rand(10, 32, embed_dim)
value = torch.rand(10, 32, embed_dim)

attn_output, attn_output_weights = multihead_attn(query, key, value)
# %%
attn_output.shape
# %% md
#
# %%
key_padding_mask = torch.zeros(32, 10, dtype=torch.bool)
key_padding_mask[:, 5:] = 1  # Mask out positions after the 5th token

attn_output, attn_output_weights = multihead_attn(
    query, key, value, key_padding_mask=key_padding_mask
)
attn_output, attn_output_weights

# %%
key_padding_mask = torch.zeros(32, 10, dtype=torch.bool)
key_padding_mask[:, 5:] = 1  # Mask out positions after the 5th token


# %%
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# Instantiate the layer
embed_dim = 512
num_heads = 8
layer = TransformerEncoderLayer(embed_dim, num_heads)
dummy_input = torch.rand(10, 32, embed_dim)

# Forward pass through the layer
output = layer(dummy_input)
print(output.shape)
# %%

# %%
