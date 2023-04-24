import torch
import time
import numpy as np
import string
from translator import dataLoader, TextLoader, TextCollate
from translator import tokenizer_es, tokenizer_en
from translator import Transformer
from translator import train, evaluate, epoch_time
from translator import BATCH_SIZE,  seq2we
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_attention_heatmap(scores, src_words, trg_words, layer_idx, head_idx, file_name='attention_heatmap.png'):
    fig, ax = plt.subplots()
    attn_scores = scores[layer_idx][0, head_idx]
    cax = ax.matshow(attn_scores, cmap='viridis')

    ax.set_xticks(np.arange(len(src_words)))
    ax.set_yticks(np.arange(len(trg_words)))
    ax.set_xticklabels(src_words)
    ax.set_yticklabels(trg_words)

    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel(f"Layer {layer_idx + 1}, Head {head_idx + 1}")
    plt.colorbar(cax)
    plt.savefig(file_name)
    plt.close(fig)


def get_attention_scores(model, e_outputs, trg_input):
    trg_mask = (trg_input != word2idx_eng["[PAD]"]).unsqueeze(-2)

    # Initialize the decoder input
    x = model.decoder.embed(trg_input)
    x = model.decoder.pe(x)

    # Iterate through the decoder layers and collect attention scores
    attention_scores = []
    for layer in model.decoder.layers:
        x, attn_scores = layer.attn_1(x, x, x, trg_mask)
        attention_scores.append(attn_scores.detach().cpu().numpy())

    return attention_scores

# Other imports and configurations
def translate(model, src, max_len=80, custom_string=False):
    model.eval()

    if custom_string == True:
        src = tokenizer_es(src)
        src = (torch.LongTensor([[word2idx_spa[tok] for tok in src]])).cuda()
    src_mask = (src != word2idx_spa["[PAD]"]).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([word2idx_eng['[SOS]']])
    attention_scores = []
    for i in range(1, max_len):

        trg_mask = np.triu(np.ones((1, i, i))).astype('uint8')
        trg_mask = (torch.from_numpy(trg_mask) == 0).cuda()

        trg_input = outputs[:i].unsqueeze(0)
        out = model.out(model.decoder(trg_input, e_outputs, src_mask, trg_mask))
        attention_scores = get_attention_scores(model, e_outputs, trg_input)
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == word2idx_eng['[EOS]']:
            break

    # Visualize the attention scores
    src_words = [idx2word_spa[idx.item()] for idx in src[0]]
    trg_words = [idx2word_eng[idx] for idx in outputs[:i].tolist()]


    layer_idx = 1  # Choose the layer index
    head_idx = 8  # Choose the head index
    for layer in range(layer_idx):
        for head in range(head_idx):
            plot_attention_heatmap(attention_scores, src_words, trg_words, layer, head, f"plots/my_plot{layer}_{head}.png")

    return ' '.join(seq2we(outputs[:i].tolist(), idx2word_eng))


if __name__ == "__main__":
    # Configuramos los parametros del modelo
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
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, N, n_heads)
    model.cuda()
    # Load the trained model
    model.load_state_dict(torch.load('s2e_model.pt'))

    # Test the translate function
    test_sentence = "Voy a ser uno de los mejores programadores de la historia"
    translation = translate(model, test_sentence, custom_string=True)
    print(f"Original: {test_sentence}")
    print(f"Translated: {translation}")
