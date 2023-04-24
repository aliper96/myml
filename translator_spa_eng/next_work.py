import torch
import torch.nn as nn
import re
import torch.nn.functional as F
import copy
import spacy
import numpy as np
import string
import random
import math
import time
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

SEED = 1234
BATCH_SIZE = 256
BLOCK_SIZE = 100  # what is the maximum context length for predictions?
N_EPOCHS = 100
d_model = 128
n_heads = 8
N = 1
pin_memory = True
num_workers = 8
num_batches_per_epoch_train = 1000  # or any other value you want to set
num_batches_per_epoch_valid = 200  # or any other value you want to set for validation

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_file = "coran2.txt"

def dataLoader():
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

nlp_en = spacy.load('en_core_web_sm')

def tokenizer_en(sentence):
    sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
    sentence = re.sub(r"[ ]+", " ", sentence)
    sentence = re.sub(r"\!+", "!", sentence)
    sentence = re.sub(r"\,+", ",", sentence)
    sentence = re.sub(r"\?+", "?", sentence)
    sentence = sentence.lower()
    return [tok.text for tok in nlp_en.tokenizer(sentence) if tok.text != " "]


class TextEncoderDecoder:
    def __init__(self, text):
        self.text = text
        self.vocab = self.build_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def build_vocab(self):
        tokens = set()
        for sentence in self.text.split('\n'):
            for token in tokenizer_en(sentence):
                tokens.add(token)
        return ['<pad>', '<sos>', '<eos>','<',  '>', '[', ']', '&', '|', '=', '+', '-', '*', '/', '^', '%', '$', '#', '@'] + sorted(tokens)

    def encode(self, sentence):
        return [self.token_to_id[token] for token in tokenizer_en(sentence)]

    def decode(self, token_ids):
        return ' '.join(self.id_to_token[token_id] for token_id in token_ids)

text = dataLoader()
text_encoder_decoder = TextEncoderDecoder(text)



# data loading
def get_batch():
    data = dataLoader()
    tokenized_data = [text_encoder_decoder.encode(sentence) for sentence in data.split('\n') if sentence.strip()]

    ix = torch.randint(len(tokenized_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.zeros(BATCH_SIZE, BLOCK_SIZE, dtype=torch.long)
    y = torch.zeros(BATCH_SIZE, BLOCK_SIZE, dtype=torch.long)

    for i, index in enumerate(ix):
        tokens = tokenized_data[index]
        tokens = tokens[:BLOCK_SIZE]  # Truncate tokens if longer than BLOCK_SIZE
        x[i, :len(tokens)] = torch.tensor(tokens)
        y[i, :len(tokens) - 1] = torch.tensor(tokens[1:])
        y[i, len(tokens) - 1] = torch.tensor(text_encoder_decoder.token_to_id['<eos>'])

    x, y = x.to(device), y.to(device)
    return x, y




def train_iterator(num_batches_per_epoch):
    for _ in range(num_batches_per_epoch):
        yield get_batch()

def valid_iterator(num_batches_per_epoch):
    for _ in range(num_batches_per_epoch):
        yield get_batch()




# In the create_mask function, replace word2idx_spa with token_to_id and idx2word_spa with id_to_token
def create_mask(src_input, trg_input):
    # pad = token_to_id["[PAD]"]
    src_mask = (src_input ).unsqueeze(1)
    trg_mask = (trg_input).unsqueeze(1)
    seq_len = trg_input.size(1)
    nopeak_mask = np.tril(np.ones((1, seq_len, seq_len)), k=0).astype('uint8')
    nopeak_mask = torch.from_numpy(nopeak_mask) != 0
    trg_input_device = trg_input.device
    trg_mask = trg_mask.to(trg_input_device)
    nopeak_mask = nopeak_mask.to(trg_input_device)
    trg_mask = trg_mask & nopeak_mask
    return src_mask.cuda(), trg_mask.cuda()



class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=BLOCK_SIZE):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # se calcula las posiciones pares
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                # se calcula las posiciones impares
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Aqui incrementamos el valor del embedding para que la matriz de posiciones no domine
        x = x * math.sqrt(self.d_model)
        # sumamos para obtener el vector de entrada final tal como se comenta en el post
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        # Aqui definimos las matrices WQ, WK y WV explicadas en el post
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # realizamos operaciones para obtener las dimensiones adecuadas

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculamos la atención
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatemos y obtenemos la salida
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output, scores


def get_clones(Module, N):
    return nn.ModuleList([copy.deepcopy(Module) for i in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        xx,_ = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(xx)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size,d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = nn.LayerNorm(d_model)
    def forward(self,src,mask):
        src = src.cuda()
        x = self.embed(src)
        x =self.pe (x)
        for i in range(self.N):
            x  = self.layers[i](x,mask)
        x = self.norm(x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        xx , _ = self.attn_1(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(xx)
        x2 = self.norm_2(x)
        xx2,_ = self.attn_2(x2, e_outputs, e_outputs,src_mask)
        x = x + self.dropout_2(xx2)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size,d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        trg = trg.cuda()
        x = self.embed(trg)
        x =self.pe (x)
        for i in range(self.N):
            x  = self.layers[i](x,e_outputs,src_mask,trg_mask)
        x = self.norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # definimos el tamaño por defecto a 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = x.cuda()
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
        self.apply(self._init_weights)

    # esto para inicializar los pesos
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output



def train(model, optimizer, criterion, iterator,num_batches):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        # print(i)
        src_input = batch[0]  # size (batch_size, seq_len)
        trg = batch[1]  # size (batch_size, seq_len)

        trg_input = trg[:, :-1]
        ys = trg[:, 1:].contiguous().view(-1).cuda()

        # create masks
        src_mask, trg_mask = create_mask(src_input, trg_input)
        preds = model(src_input, trg_input, src_mask, trg_mask)
        # apply backpropagation
        optimizer.zero_grad()
        loss = criterion(preds.view(-1, preds.size(-1)), ys)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / num_batches


def evaluate(model, criterion, iterator,num_batches):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_input = batch[0]  # size (batch_size, seq_len)
            trg = batch[1]  # size (batch_size, seq_len)

            trg_input = trg[:, :-1]
            ys = trg[:, 1:].contiguous().view(-1).cuda()
            src_mask, trg_mask = create_mask(src_input, trg_input)
            preds = model(src_input, trg_input, src_mask, trg_mask)
            loss = criterion(preds.view(-1, preds.size(-1)), ys)
            epoch_loss += loss.item()

    return epoch_loss / num_batches


# %%time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

train_iter = train_iterator(num_batches_per_epoch_train)
valid_iter = valid_iterator(num_batches_per_epoch_valid)
vocab_size = len(text_encoder_decoder.vocab)

if __name__ == "__main__":
    # Configure model parameters



    model = Transformer(vocab_size, vocab_size, d_model, N, n_heads)
    model.cuda()

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    optimizer = torch.optim.AdamW(model.parameters())

    criterion = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1} / {N_EPOCHS}")

        start_time = time.time()

        train_loss = train(model, optimizer, criterion, train_iter, num_batches_per_epoch_train)
        valid_loss = evaluate(model, criterion, valid_iter, num_batches_per_epoch_valid)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')






