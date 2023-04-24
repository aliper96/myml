from vqvae import VectorQuantizedVAE
from transformer import Head,MultiHeadAttention,FeedFoward,Block
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch
from torchvision import transforms
from transformers import AutoTokenizer
from utils import CocoCaptionsDataset
from config import n_embd, n_head, n_layer, lr,latent_vector_size

#we will create a VQVAE model with the transformer architecture from coco_data import CocoCaptionsDataset


#we will create a VQVAE model with the transformer architecture
#now we will create a VQVAE model with the transformer architecture

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


class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, device):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, 256)  # Change output dimension to 256
        self.linear_reshape = nn.Linear(256 * 50, 32 * 32)  # Add another linear layer to reshape the output

    def forward(self, input_text):
        x = self.embedding(input_text)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        text_embeddings = self.linear(x)
        text_embeddings = text_embeddings.view( text_embeddings.shape[0], -1)  # Flatten the embeddings
        text_embeddings = self.linear_reshape(text_embeddings)  # Reshape the embeddings
        text_embeddings = text_embeddings.view(text_embeddings.shape[0], 32, 32)  # Reshape output tensor

        return text_embeddings




class TextToImageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, device):
        super().__init__()
        self.transformer = CustomTransformer(vocab_size, n_embd, n_head, n_layer, device)
        self.vqvae = VectorQuantizedVAE(input_dim=3, dim=n_embd, K=latent_vector_size)

    def forward(self, input_text, input_image=None):
        text_embeddings = self.transformer(input_text)

        if input_image is not None:
            latents = self.vqvae.encode(input_image)
            output_image = self.vqvae.decode(torch.cat((latents, text_embeddings),dim =-1))
        else:
            # Generate a random latent code for the bottom code
            random_bottom_code = torch.randint(
                high=self.vqvae.codeBook.embedding.num_embeddings,
                size=(text_embeddings.shape[0], self.vqvae.encoder[-1].out_channels, self.vqvae.encoder[-1].kernel_size[0], self.vqvae.encoder[-1].kernel_size[1]),
                device=input_text.device
            )
            output_image = self.vqvae.decode(torch.cat((random_bottom_code, text_embeddings.unsqueeze(1)), dim=-1))

        return output_image








def train(model, dataloader, optimizer, epochs, device):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, (images, captions) in enumerate(dataloader):
            images, captions = images.to(device), captions.to(device)

            optimizer.zero_grad()
            output = model(captions, images)

            # You may need to modify this loss function based on your specific requirements
            loss = F.mse_loss(output, images)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Step {i}: Loss = {loss.item()}")




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, vocab_size = load_data()

    model = TextToImageModel(vocab_size, n_embd, n_head, n_layer, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(model, dataloader, optimizer, epochs=10, device=device)
