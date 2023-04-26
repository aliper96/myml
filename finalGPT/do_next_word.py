import torch

from next_word_decoder import GPT
from train import vocab_size , encode, decode
import numpy as np

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = vocab_size
model_config.block_size = 100
model = GPT(model_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


    model.cuda()

    # Load the trained model
    # model.load_state_dict(torch.load('trained_model.pth'))
    model.load_state_dict(torch.load('trained_model_finetuned.pth'))

    # Test the translate function
    encoded_phrase = torch.tensor(encode("Who sent down the Quran?"), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(model.generate(encoded_phrase, max_new_tokens=500,do_sample=True,top_k=2)[0].tolist()))

    encoded_phrase = torch.tensor(encode("Who sent down the Quran?"), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(model.generate(encoded_phrase, max_new_tokens=500,do_sample=False)[0].tolist()))

    encoded_phrase = torch.tensor(encode("To whom was the Quran revealed?"), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(model.generate(encoded_phrase, max_new_tokens=500,do_sample=True)[0].tolist()))

    encoded_phrase = torch.tensor(encode("To whom was the Quran revealed?"), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(model.generate(encoded_phrase, max_new_tokens=500,do_sample=False)[0].tolist()))

    encoded_phrase = torch.tensor(encode("Through whom was the revelation of the Quran sent?"), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(model.generate(encoded_phrase, max_new_tokens=500)[0].tolist()))
    #lets convert next_pred to numpy array