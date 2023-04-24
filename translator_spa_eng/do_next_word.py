# Load the best model and make predictions
import torch
import time
import numpy as np
import string
from next_work import *
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

def greedy_decode(model, src, src_mask, max_len=BLOCK_SIZE):
    memory = model.encoder(src, src_mask)
    ys = torch.ones(1, 1).fill_(text_encoder_decoder.encode('<sos>')[0]).type_as(src.data)
    for _ in range(max_len-1):
        trg_mask = (ys != 0).unsqueeze(-2)
        out = model.decoder(ys, memory, src_mask, trg_mask)
        prob = F.softmax(out[:, -1], dim=-1)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, next_word.unsqueeze(0).unsqueeze(0)], dim=1)
        if next_word.item() == text_encoder_decoder.encode('<eos>')[0]:
            break
    return ys

def get_attention_scores(model, e_outputs, trg_input):
    trg_mask = (trg_input != 0).unsqueeze(-2)

    # Initialize the decoder input
    x = model.decoder.embed(trg_input)
    x = model.decoder.pe(x)

    # Iterate through the decoder layers and collect attention scores
    attention_scores = []
    for layer in model.decoder.layers:
        x, attn_scores = layer.attn_1(x, x, x, trg_mask)
        attention_scores.append(attn_scores.detach().cpu().numpy())

    return attention_scores


if __name__ == "__main__":


    model = Transformer(vocab_size, vocab_size, d_model, N, n_heads)
    model.cuda()

    # Load the trained model
    model.load_state_dict(torch.load('tut6-model.pt'))

    # Test the translate function
    test_sentence = "Allah is the best"
    input = text_encoder_decoder.encode(test_sentence)
    input = torch.tensor(input, dtype=torch.long).unsqueeze(0).cuda()
    src_mask, trg_mask = create_mask(input, input)
    output = greedy_decode(model, input, src_mask)

    next_pred = model(input, input, src_mask, trg_mask)

    print(f"Original: {test_sentence}")
    # Get token IDs and convert to list of integers
    output = output.squeeze(0).cpu().numpy().tolist()
    translated_sentence = text_encoder_decoder.decode(output[1:])

    print(f"Translated: {translated_sentence}")

    #lets convert next_pred to numpy array