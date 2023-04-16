import torch
from torch.nn import functional as F
import torch.nn as nn

emb_entrada = "Ali es el "
emb_salida =  "es el mejor"




B, T, C = 4, 8 , 32
x = torch.randn(B,T,C)

head_size = 3
key = nn.Linear(C,head_size,bias=False)
query = nn.Linear(C,head_size,bias=False)
value = nn.Linear(C,head_size,bias=False)

k = key(x)
q = query(x)
v = value(x)

wei = q @ k.transpose(1,2)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

scaled_attention = wei @ v

print(wei)


print("ENDED")
