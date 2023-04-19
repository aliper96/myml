import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch
from torchvision import transforms
from torchvision.datasets import CocoCaptions, CocoDetection
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



# hyperparameters
batch_size = 1 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 128
n_head = 1
n_layer = 1
dropout = 0.2


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, padding=1),
            *[Residual(num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)],
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1),
            *[Residual(num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)],
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens // 2, int(num_hiddens // 3), kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(int(num_hiddens // 3), int(num_hiddens // 4), kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(int(num_hiddens // 4), out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class VQVAE(nn.Module):
    def __init__(self, codebook_size, codebook_dim, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim))

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.vector_quantize(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q, indices

    def vector_quantize(self, z_e):
        z_e_flatten = z_e.view(z_e.size(0), z_e.size(1), -1)  # Shape: [batch_size, num_hiddens, H*W]
        z_e_flatten = z_e_flatten.permute(0, 2, 1)  # Shape: [batch_size, H*W, num_hiddens]

        codebook = self.codebook.unsqueeze(0)  # Shape: [1, codebook_size, codebook_dim]
        distances = torch.cdist(z_e_flatten, codebook.squeeze(0))  # Shape: [batch_size, H*W, codebook_size]

        indices = distances.argmin(dim=-1)  # Shape: [batch_size, H*W]
        z_q = self.codebook[indices].view(z_e.size())  # Shape: [batch_size, num_hiddens, H, W]
        return z_q, indices


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(4146, 4146)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DALLE(nn.Module):
    def __init__(self, vqvae, transformer, text_encoder):
        super().__init__()
        self.vqvae = vqvae
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.adjust_dim = None

    def forward(self, images, text):
        _, z_e, z_q, indices = self.vqvae(images)

        # Flatten and permute z_q
        z_q_flat = z_q.view(z_q.size(0), z_q.size(1), -1).permute(0, 2, 1)  # Shape: [batch_size, H*W, num_hiddens]

        # Encode text input
        text_embeddings = self.text_encoder(text)

        # Create the adjust_dim layer if it doesn't exist or is on the wrong device
        if self.adjust_dim is None or self.adjust_dim.weight.device != text_embeddings.device:
            self.adjust_dim = nn.Linear(text_embeddings.size(-1), z_q_flat.size(-1)).to(text_embeddings.device)

        # Adjust text embeddings dimensions
        text_embeddings_adjusted = self.adjust_dim(text_embeddings)

        # Concatenate image and text embeddings
        embeddings = torch.cat([z_q_flat, text_embeddings_adjusted], dim=1)

        # Pass concatenated embeddings through the transformer
        transformer_output = self.transformer(embeddings)

        # Use the transformer output to predict the next token in the image sequence
        prediction = transformer_output[:, -1]

        # Decode the predicted token back to image space
        generated_images = self.vqvae.decoder(prediction.unsqueeze(-1).unsqueeze(-1))

        return generated_images




class CocoCaptionsDataset(Dataset):
    def __init__(self, root, annFile, transform=None, tokenizer=None):
        self.coco_captions = CocoCaptions(root=root, annFile=annFile, transform=transform)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.coco_captions)

    def __getitem__(self, idx):
        image, captions = self.coco_captions[idx]

        # Tokenize the first caption as an example
        text = self.tokenizer(captions[0], return_tensors='pt', padding='max_length', truncation=True, max_length=50)
        input_ids = text['input_ids'].squeeze()

        return image, input_ids


def train_dalle(dalle, dataloader, device, epochs):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dalle.parameters(), lr=learning_rate)

    # Move the model to the appropriate device (CPU or GPU)
    dalle.to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            # Get the input images and text
            images, text = data
            images, text = images.to(device), text.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, compute the loss, and perform backpropagation
            generated_images = dalle(images, text)
            loss = criterion(generated_images, images)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                running_loss = 0.0

        torch.save(dalle.state_dict(), "dalle.pth")

    print("Finished training")
def load_data():
    data_path = "coco_data"
    img_folder = "train2017"
    anno_json = "annotations/captions_train2017.json"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)


    coco_dataset = CocoCaptionsDataset(root=f"{data_path}/{img_folder}",
                                       annFile=f"{data_path}/{anno_json}",
                                       transform=transform,
                                       tokenizer=tokenizer)

    dataloader = torch.utils.data.DataLoader(coco_dataset,
                                             batch_size=2,
                                             shuffle=True,
                                             num_workers=4)
    return  dataloader, vocab_size


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Define the VQ-VAE model
    encoder = Encoder(3, 128, 2, 32)
    decoder = Decoder(128, 128, 2, 32, 3)
    vqvae = VQVAE(512, 128, encoder, decoder)

    # Define the Transformer model
    transformer = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
    dataloader, vocab_size = load_data()
    text_encoder = nn.Embedding(vocab_size, n_embd)
    dalle_model = DALLE(vqvae, transformer, text_encoder)
    train_dalle(dalle_model, dataloader, device, epochs=10)

if __name__ == '__main__':
    main()