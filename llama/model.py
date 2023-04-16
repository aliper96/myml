# La clase RMSNorm es una implementación de una capa de normalización en PyTorch
# que puede ser utilizada en redes neuronales para mejorar la estabilidad y el rendimiento del entrenamiento.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

from llama.tokenizer import Tokenizer




@dataclass
class ModelArgs:
    dim: int = 128  #orignal 512
    n_layers: int = 1
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    dropout: float = 0.2
    max_iters : int = 3000
    eval_interval : int = 500
    learning_rate : float= 3e-4
    max_batch_size: int = 32
    max_seq_len: int = 2048
    batch_size = 128  # how many independent sequences will we process in parallel?
    block_size = 128  # what is the maximum context length for predictions?
    eval_iters = 200


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer(model_path="weights/tokenizer/tokenizer.model")
modelArgs = ModelArgs()
modelArgs.vocab_size = tokenizer.n_words

class RMSNorm(torch.nn.Module):
    # El constructor de la clase acepta dos argumentos:
    # - dim: la dimensión de la entrada
    # - eps (opcional): un pequeño valor para evitar la división por cero en la normalización
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm es una función interna que acepta un tensor de entrada x y normaliza
    # utilizando la raíz cuadrada inversa de la media de los cuadrados de los elementos
    # de x a lo largo de la última dimensión.
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # forward es el método principal que se llama cuando se procesa la entrada a través de esta capa.
    # Normaliza el tensor de entrada y multiplica por el parámetro weight.
    def forward(self, x):
        # output = self._norm(x.float()).type_as(x)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# precompute_freqs_cis calcula las frecuencias y la función coseno para la entrada de la función apply_rotary_emb.
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

# reshape_for_broadcast ajusta la forma del tensor freqs_cis para permitir la transmisión
# de la operación en la función apply_rotary_emb.
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# apply_rotary_emb es una función que aplica la técnica de rotary embeddings
# a los tensores de entrada xq y xk utilizando las frecuencias y la función coseno calculadas previamente.
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



"""
La transformación de rotary embedding es una técnica utilizada en modelos de procesamiento del lenguaje natural, especialmente en modelos de atención como los transformadores. La idea principal es aplicar una modulación periódica a las representaciones de entrada antes de calcular la atención, lo que permite capturar información relativa de posición en la secuencia de entrada.

En resumen, el rotary embedding realiza lo siguiente:

Calcula las frecuencias y la función coseno basada en los parámetros proporcionados. En el ejemplo anterior, utilizamos precompute_freqs_cis para calcular estas frecuencias y la función coseno.

Reajusta la forma del tensor de frecuencias y la función coseno para que se pueda transmitir con los tensores de entrada. En el ejemplo anterior, utilizamos reshape_for_broadcast para hacer esto.

Aplica una modulación periódica a las representaciones de entrada utilizando las frecuencias y la función coseno. En el ejemplo anterior, utilizamos apply_rotary_emb para aplicar la modulación a los tensores de entrada xq y xk.

La transformación de rotary embedding ayuda a mejorar la capacidad del modelo de atención para capturar información relativa de posición en la secuencia de entrada sin depender de las incrustaciones de posición absoluta. Esto es especialmente útil en tareas donde la posición relativa de las palabras en la secuencia es más importante que su posición absoluta.
"""


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.key = nn.Linear(args.dim, args.dim // args.n_heads, bias=False)
        self.query = nn.Linear(args.dim, args.dim // args.n_heads, bias=False)
        self.value = nn.Linear(args.dim, args.dim // args.n_heads, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(args.dim, args.dim)))

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, freqs_cis: torch.Tensor):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)

        # q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.heads = nn.ModuleList([Head(args) for _ in range(args.n_heads)])
        self.proj = nn.Linear(args.n_heads * (args.dim // args.n_heads), args.dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, freqs_cis: torch.Tensor):
        out = torch.cat([h(x, freqs_cis) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.dim, 4 * args.dim),
            nn.ReLU(),
            nn.Linear(4 * args.dim, args.dim),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output.float()


class GPTLanguageModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.transformer = Transformer(params)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor):
        logits = self.transformer(tokens, start_pos=0)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, context: torch.Tensor, max_new_tokens: int):
        output = context
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.transformer(output)
                probabilities = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.argmax(probabilities, dim=-1).unsqueeze(1)
            output = torch.cat((output, next_token), dim=1)
        return output


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(modelArgs.eval_iters)
        for k in range(modelArgs.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


with open('data/coran2.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Train and test splits
data = torch.tensor(tokenizer.encode(text,True,True), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - modelArgs.block_size, (modelArgs.batch_size,))
    x = torch.stack([data[i:i+modelArgs.block_size] for i in ix])
    y = torch.stack([data[i+1:i+modelArgs.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def main():
    # Asegúrate de especificar la ruta correcta al modelo de tokenizador

    model = GPTLanguageModel(modelArgs)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=modelArgs.learning_rate)
    print("Before the training loop")
    for iter in range(modelArgs.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % modelArgs.eval_interval == 0 or iter == modelArgs.max_iters - 1:
            losses = estimate_loss(model)  # Asegúrate de definir esta función
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch("train")  # Asegúrate de definir esta función

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "gpt_language_model.pth")

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = m.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(generated_tokens))

if __name__ == "__main__":
    main()