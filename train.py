import os
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from src.Transformer import Transformer
from src.cli_interface import CLI_Interface

load_dotenv()

# --- CONFIGURATION ---
# Set this to True to start fine-tuning, or False for pre-training from scratch
FINETUNE = True 
# ---------------------

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device} | Mode: {'Fine-tuning' if FINETUNE else 'Pre-training'}")

# Hyperparameters (Adjusted for mode)
block_size = 256
batch_size = 16 if FINETUNE else 64
learning_rate = 1e-5 if FINETUNE else 3e-4
train_iterations = 500 if FINETUNE else 5000
eval_iterations = 100
temperature = 1.0
number_embeddings = 384
num_heads = 6
dropout_rate = 0.2
n_layers = 6

# Path setup
shakespeare_path = os.getenv("SHAKESPEARE_PATH")
fine_tuning_path = "fine_tuning_data.txt"

# Always load the original Shakespeare to ensure consistent vocabulary
with open(shakespeare_path, "r") as file:
    shakespeare_text = file.read()
chars = sorted(list(set(shakespeare_text)))
vocab_size = len(chars)

# Load the data for the current mode
data_path = fine_tuning_path if FINETUNE else shakespeare_path
with open(data_path, "r") as file:
    text = file.read()

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
def train(model):
    # If fine-tuning, load the pre-trained weights first
    if FINETUNE:
        pre_trained_path = "transformer_model.pth"
        if os.path.exists(pre_trained_path):
            print(f"Loading pre-trained weights from {pre_trained_path}...")
            model.load_state_dict(torch.load(pre_trained_path, map_location=device))
        else:
            print("Warning: pre-trained weights not found! Fine-tuning from scratch (unlikely to work well).")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Encode data
    data = torch.tensor(model.encode(text), dtype=torch.long).to(device)
    
    print(f"Starting {'fine-tuning' if FINETUNE else 'training'} for {train_iterations} steps...")
    model.train()
    
    for step in range(train_iterations):
        xb, yb = get_batch(data)
        
        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = yb.view(B*T)
        
        loss = F.cross_entropy(logits, targets)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: loss {loss.item():.4f}")

    # Save to different files based on mode
    save_path = "fine_tuned_transformer.pth" if FINETUNE else "transformer_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")
def main():
    # Initialize model with original chars to maintain consistency
    model = Transformer(
        chars=chars, 
        temperature=temperature, 
        num_embeddings=number_embeddings, 
        block_size=block_size, 
        num_heads=num_heads, 
        dropout_rate=dropout_rate
    )

    model.load_state_dict(torch.load('fine_tuned_transformer.pth', map_location=device))
 


    

    print("\nGenerating text:")
    #model.generate()
    cli = CLI_Interface(model)
    cli.interact()


if __name__ == "__main__":
    main()
