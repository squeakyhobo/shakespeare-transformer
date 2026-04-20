import torch.nn as nn
import torch
import torch.nn.functional as F
from time import sleep

class Head(nn.Module):
    def __init__(self, temperature, num_embed, block_size, head_size, dropout_rate):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        
        # register_buffer ensures tril moves with the model to GPU/MPS
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.temperature = temperature 
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) / (self.head_size**0.5)
        # tril is now correctly on the same device as x
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v 
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size, num_embeddings, block_size, temperature, dropout_rate):
       super().__init__()
       self.heads = nn.ModuleList([
           Head(temperature, num_embeddings, block_size, head_size, dropout_rate) 
           for _ in range(num_heads)
       ])
       self.projection = nn.Linear(num_heads * head_size, num_embeddings)
       self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out 
            
class Block(nn.Module):
    def __init__(self, temperature, num_embeddings, block_size, num_heads, head_size, dropout_rate):
        super().__init__() 
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)
        self.heads = MultiHeadedAttention(num_heads, head_size, num_embeddings, block_size, temperature, dropout_rate)
        self.ffwd = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Pre-Norm structure
        x = x + self.heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module): 
    def __init__(self, chars, temperature, num_embeddings, block_size, num_heads, dropout_rate):
        super().__init__()
        self.block_size = block_size
        self.temperature = temperature
        self.head_size = num_embeddings // num_heads

        self.vocab_embedding_table = nn.Embedding(len(chars), num_embeddings)
        self.position_embedding = nn.Embedding(block_size, num_embeddings)
        
        # Using 6 blocks as in Karpathy's final model
        self.blocks = nn.Sequential(*[
            Block(temperature, num_embeddings, block_size, num_heads, self.head_size, dropout_rate)
            for _ in range(6)
        ])
        
        self.ln_f = nn.LayerNorm(num_embeddings)
        self.lm_head = nn.Linear(num_embeddings, len(chars))
        self.dropout = nn.Dropout(dropout_rate)
        
        self.encoding_pairing = {char: i for i, char in enumerate(chars)}
        self.decoding_pairing = {i: char for i, char in enumerate(chars)}
        
    def forward(self, idx):
        B, T = idx.shape
        # Create positions on the same device as input idx
        positions = torch.arange(T, device=idx.device)

        pos_embed = self.position_embedding(positions)
        token_embed = self.vocab_embedding_table(idx) 
        
        x = self.dropout(token_embed + pos_embed)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self,prompt = None):
        self.eval()
        # Find the device the model is currently on
        device = next(self.parameters()).device
        if(prompt!=None):
            idx = torch.tensor(self.encode(prompt), dtype=torch.long, device=device).unsqueeze(0) #add extra dim for batch
        else:
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        with torch.no_grad():
            while True:
                idx_cropped = idx[:, -self.block_size:]
                logits = self(idx_cropped)
                logits = logits[:, -1, :] / self.temperature
                probs = F.softmax(logits, dim=-1)
                index = torch.multinomial(probs, num_samples=1)
                
                output = self.decoding_pairing.get(index.item())
                if output is None: # handle potential unknown characters or index out of bounds
                    break
                idx = torch.cat((idx, index), dim=1)
                print(output, end="", flush=True)
                sleep(0.05)

    def encode(self, string):
        return [self.encoding_pairing.get(char) for char in string]
