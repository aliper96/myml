from  gpt import  GPTLanguageModel, encode, decode
import  torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Instantiate the model and load the saved state
loaded_model = GPTLanguageModel()
loaded_model.load_state_dict(torch.load("gpt_language_model.pth"))
loaded_model = loaded_model.to(device)
loaded_model.eval()  # Set the model to evaluation mode


context = torch.zeros((1, 1), dtype=torch.long, device=device)
encoded_phrase = torch.tensor(encode("Who is Allah"), dtype=torch.long, device=device).unsqueeze(0)
print(decode(loaded_model.generate(encoded_phrase, max_new_tokens=500)[0].tolist()))