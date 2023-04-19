import random
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from  model import *

def generate_image_from_text(dalle, tokenizer, text, device):
    dalle.eval()  # Set the model to evaluation mode
    random_image = torch.randn(1, 3, 128, 128).to(device)  # Create a random image as a starting point

    tokens = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
    input_ids = tokens['input_ids'].to(device)

    with torch.no_grad():  # Disable gradient calculation
        generated_image = dalle(random_image, input_ids)

    return to_pil_image(generated_image[0].cpu())  # Convert the tensor to a PIL image and return it

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved DALLE model
encoder = Encoder(3, 128, 2, 32)
decoder = Decoder(128, 128, 2, 32, 3)
vqvae = VQVAE(512, 128, encoder, decoder)

# Define the Transformer model
transformer = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
dataloader, vocab_size = load_data()
text_encoder = nn.Embedding(vocab_size, n_embd)
dalle_model = DALLE(vqvae, transformer, text_encoder)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)

dalle_weights_path = "dalle.pth"
dalle_model.load_state_dict(torch.load(dalle_weights_path, map_location=device))

# Test the model with a text input
input_text = "A red apple on a table."
result_image = generate_image_from_text(dalle_model, tokenizer, input_text, device)
result_image.save("output_image.png")
result_image.show()
