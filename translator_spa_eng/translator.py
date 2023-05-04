import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dill as pickle
import copy
import spacy
import numpy as np
import string
import random
import math
import time

SEED = 1234
BATCH_SIZE = 256
pin_memory = True
num_workers = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_file = "spa-eng/spa.txt"

def dataLoader():
    """
    Loads the data from the given text file and returns a list of tuples containing
    Spanish and English text pairs.
    """
    with open(text_file, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]
    text_pairs = []
    for line in lines:
        eng, spa = line.split("\t")
        text_pairs.append((spa, eng))

    return text_pairs

nlp_en = spacy.load('en_core_web_sm')

def tokenizer_en(sentence):
    """
    Tokenizes the input English sentence and returns a list of tokens after
    applying basic preprocessing such as removing special characters, multiple spaces, etc.
    """
    sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
    sentence = re.sub(r"[ ]+", " ", sentence)
    sentence = re.sub(r"\!+", "!", sentence)
    sentence = re.sub(r"\,+", ",", sentence)
    sentence = re.sub(r"\?+", "?", sentence)
    sentence = sentence.lower()
    return [tok.text for tok in nlp_en.tokenizer(sentence) if tok.text != " "]

nlp_es = spacy.load('es_core_news_sm')

def tokenizer_es(sentence):
    """
    Tokenizes the input Spanish sentence and returns a list of tokens after
    applying basic preprocessing such as removing special characters, multiple spaces, etc.
    """
    sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
    sentence = re.sub(r"[ ]+", " ", sentence)
    sentence = re.sub(r"\!+", "!", sentence)
    sentence = re.sub(r"\,+", ",", sentence)
    sentence = re.sub(r"\?+", "?", sentence)
    sentence = sentence.lower()
    return [tok.text for tok in nlp_es.tokenizer(sentence) if tok.text != " "]

def ws2seq(s, word2idx_spa):
    """
    Converts a list of Spanish words into a list of corresponding indices using the word2idx_spa dictionary.
    """
    return [word2idx_spa[i] for i in s if i in word2idx_spa.keys()]

def seq2ws(s, idx2word_spa):
    """
    Converts a list of Spanish word indices into a list of corresponding words using the idx2word_spa dictionary.
    """
    return [idx2word_spa[i] for i in s if idx2word_spa[i]]

def we2seq(s, word2idx_eng):
    """
    Converts a list of English words into a list of corresponding indices using the word2idx_eng dictionary.
    Adds start of sentence (SOS) and end of sentence (EOS) tokens to the sequence.
    """
    return [word2idx_eng['[SOS]']] + [word2idx_eng[i] for i in s if i in word2idx_eng.keys()] + [word2idx_eng['[EOS]']]

def seq2we(s, idx2word_eng):
    """
    Converts a list of English word indices into a list of corresponding words using the idx2word_eng dictionary.
    """
    return [idx2word_eng[i] for i in s]



class TextLoader(torch.utils.data.Dataset):
    def __init__(self, word2idx_spa,word2idx_eng,path= "//spa-eng/spa.txt"):
        self.x, self.y = [], []
        with open(text_file,encoding="utf-8") as f:
            lines = f.read().split("\n")[:-1]
        for line in lines:
            eng, spa = line.split("\t")
            eng_l = tokenizer_en(eng)
            spa_l = tokenizer_en(spa)
            self.x.append(ws2seq(spa_l,word2idx_spa))
            self.y.append(we2seq(eng_l, word2idx_eng))
    def __getitem__(self, index):
        return (torch.LongTensor(self.x[index]), torch.LongTensor(self.y[index]))

    def __len__(self):
        return len(self.x)


class TextCollate():
    def __call__(self, batch):
        max_x_len = max([i[0].size(0) for i in batch])
        x_padded = torch.LongTensor( len(batch), max_x_len)
        x_padded.zero_()

        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor( len(batch), max_y_len)
        y_padded.zero_()

        for i in range(len(batch)):
            x = batch[i][0]
            x_padded[i, :x.size(0)] = x
            y = batch[i][1]
            y_padded[i,:y.size(0)] = y

        return x_padded, y_padded


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
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

def create_mask(src_input, trg_input):
    # mascara de entrada para evitar el padding
    pad = word2idx_spa["[PAD]"]
    src_mask = (src_input != pad).unsqueeze(1)

    # mascara de salida
    trg_mask = (trg_input != pad).unsqueeze(1)

    seq_len = trg_input.size(1)
    nopeak_mask = np.tril(np.ones((1, seq_len, seq_len)), k=0).astype('uint8')
    nopeak_mask = torch.from_numpy(nopeak_mask) != 0
    trg_mask = trg_mask & nopeak_mask

    return src_mask.cuda(), trg_mask.cuda()

def train(model, optimizer, criterion, iterator):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src_input = batch[0]  # tamaño (batch_size, seq_len)
        trg = batch[1]  # tamaño (batch_size, seq_len)

        trg_input = trg[:, :-1]
        ys = trg[:, 1:].contiguous().view(-1).cuda()

        # creamos las mascaras
        src_mask, trg_mask = create_mask(src_input, trg_input)
        preds = model(src_input, trg_input, src_mask, trg_mask)
        # aplicamos el backpropagation
        optimizer.zero_grad()
        loss = criterion(preds.view(-1, preds.size(-1)), ys)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, criterion, iterator):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_input = batch[0]  # tamaño (batch_size, seq_len)
            trg = batch[1]  # tamaño (batch_size, seq_len)

            trg_input = trg[:, :-1]
            ys = trg[:, 1:].contiguous().view(-1).cuda()
            src_mask, trg_mask = create_mask(src_input, trg_input)
            preds = model(src_input, trg_input, src_mask, trg_mask)
            loss = criterion(preds.view(-1, preds.size(-1)), ys)
            epoch_loss += loss.item()

            if i % 32 == 0:
                for i in [5, 12, 15]:
                    out = F.softmax(preds[i], dim=-1)
                    val, ix = out.data.topk(1)
                    print("Oración en Español: ", seq2ws(src_input[i].tolist(), idx2word_spa))
                    print("Oración Real en Ingles: ", seq2we(trg_input[i].tolist(), idx2word_eng))
                    print("Oración Predicha en Ingles: ", seq2we([ix1[0] for ix1 in ix.tolist()], idx2word_eng))
                    print("\n")

    return epoch_loss / len(iterator)





# %%time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    # Configuramos los parametros del modelo
    N_EPOCHS = 100
    d_model = 256
    n_heads = 8
    N = 1
    pin_memory = True
    num_workers = 2

    word2idx_spa = {}
    word2idx_eng = {}
    idx2word_eng = {}
    idx2word_spa = {}

    strip_chars = string.punctuation + "¿"
    j = 1
    z = 3
    word2idx_eng["[PAD]"] = 0
    word2idx_eng["[SOS]"] = 1
    word2idx_eng["[EOS]"] = 2
    word2idx_spa["[PAD]"] = 0


    text_pairs = dataLoader()

    for i, (spa, eng) in enumerate(text_pairs):
        for w1 in tokenizer_es(spa):
            if w1 not in word2idx_spa:
                word2idx_spa[w1] = j
                j += 1
        for w in tokenizer_en(eng):
            if w not in word2idx_eng:
                word2idx_eng[w] = z
                z += 1


    dataset = TextLoader(word2idx_spa,word2idx_eng)

    train_len = int(len(dataset) * 0.9)
    trainset, valset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    collate_fn = TextCollate()

    train_loader = torch.utils.data.DataLoader(trainset, num_workers=num_workers, shuffle=True,
                              batch_size=BATCH_SIZE, pin_memory=pin_memory,
                              drop_last=True, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(valset, num_workers=num_workers, shuffle=False,
                            batch_size=BATCH_SIZE, pin_memory=pin_memory,
                            drop_last=False, collate_fn=collate_fn)

    idx2word_eng = {v: k for k, v in word2idx_eng.items()}
    idx2word_spa = {v: k for k, v in word2idx_spa.items()}

    src_vocab_size = len(word2idx_spa) + 1
    trg_vocab_size = len(word2idx_eng) + 1
    print(src_vocab_size, trg_vocab_size, d_model, N, n_heads)
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, N, n_heads)
    model.cuda()

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    optimizer = torch.optim.AdamW(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx_spa["[PAD]"])


    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch+1:02}')

        start_time = time.time()

        train_loss = train(model, optimizer, criterion, train_loader)
        valid_loss = evaluate(model, criterion, val_loader)

        epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), 's2e_model.pt') # si quieres guardar el modelo

        print(f'Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val   Loss: {valid_loss:.3f}')
    torch.save(model.state_dict(), 's2e_model.pt')

    print(best_valid_loss)