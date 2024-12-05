# %% md
# ### Self Attention
# %%
import torch
import torch.nn.functional as F

sentence = "Life is short, eat dessert first"

dc = {s: i for i, s in enumerate(sorted(sentence.replace(",", "").split()))}
print(dc)
# %%
sentence_int = torch.tensor([dc[s] for s in sentence.replace(",", "").split()])
print(sentence_int)
# %%
torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)
# %%
torch.manual_seed(123)

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 24, 24, 28

W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))
# %%
x_2 = embedded_sentence[1]
query_2 = W_query.matmul(x_2)
key_2 = W_key.matmul(x_2)
value_2 = W_value.matmul(x_2)

print(query_2.shape)
print(key_2.shape)
print(value_2.shape)
# %%
keys = W_key.matmul(embedded_sentence.T).T
values = W_value.matmul(embedded_sentence.T).T

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# %%
omega_24 = query_2.dot(keys[4])
print(omega_24)

# %%

attention_weights_2 = F.softmax(omega_24 / d_k**0.5, dim=0)
print(attention_weights_2)
# %%
context_vector_2 = attention_weights_2.matmul(values)

print(context_vector_2.shape)
print(context_vector_2)

# %%
h = 3
multihead_W_query = torch.nn.Parameter(torch.rand(h, d_q, d))
multihead_W_key = torch.nn.Parameter(torch.rand(h, d_k, d))
multihead_W_value = torch.nn.Parameter(torch.rand(h, d_v, d))
# %%
multihead_query_2 = multihead_W_query.matmul(x_2)
print(multihead_query_2.shape)
# %%
multihead_key_2 = multihead_W_key.matmul(x_2)
multihead_value_2 = multihead_W_value.matmul(x_2)
# %%
stacked_inputs = embedded_sentence.T.repeat(3, 1, 1)
print(stacked_inputs.shape)
# %%
multihead_keys = torch.bmm(multihead_W_key, stacked_inputs)
multihead_values = torch.bmm(multihead_W_value, stacked_inputs)
print("multihead_keys.shape:", multihead_keys.shape)
print("multihead_values.shape:", multihead_values.shape)
# %% md
# ### Cross Attention
# %%
embedded_sentence_2 = torch.rand(8, 16)  # 2nd input sequence

keys = W_key.matmul(embedded_sentence_2.T).T
values = W_value.matmul(embedded_sentence_2.T).T

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# %%
