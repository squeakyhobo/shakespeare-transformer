import torch.nn as nn
import torch
import torch.nn.functional as F
from time import sleep




class Head(nn.Module):
    def __init__(self,temprature,num_embed,block_size,head_size):
        super().__init__()
        
        
        
        self.key = nn.Linear(num_embed,head_size,bias =False)
        self.query = nn.Linear(num_embed,head_size,bias= False)
        self.value = nn.Linear(num_embed,head_size,bias= False)
       
       
        self.tril = torch.tril(torch.ones(block_size, block_size))

        
        self.temprature =temprature 
        self.block_size = block_size
        self.head_size = head_size

        

    def forward(self,x):

        B,T,C =x.shape
        
        k = self.key(x) # (B,T,head_size) have waht each token is  
        q = self.query(x)# B,T,head_size) we have what each token is looking for 
        v = self.value(x)
        

        # I now want each token in their respetice block size to communicate to each otehr and find what they are looking for 
        wei = q @ k.transpose(-2, -1) / self.head_size**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        out = wei @ v # B,T,head_size
        
        
        return out
    
   
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size, num_embeddings, block_size, temperature):
       super().__init__()
       self.heads=  nn.ModuleList(Head(temprature=temperature,num_embed=num_embeddings,block_size=block_size,head_size=head_size) for _ in range(num_heads))
   


    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out 
            
            

class Transformer(nn.Module): 

    def __init__(self,chars,temprature,num_embeddings=64,block_size =8,num_heads =8,):
        super().__init__()


        self.block_size = block_size
        self.temprature =temprature
        self.head_size = num_embeddings//num_heads

        
        self.vocab_embedding_table = nn.Embedding(len(chars),num_embeddings)
        self.positon_embedding = nn.Embedding(block_size,num_embeddings)
        self.Heads = MultiHeadedAttention(num_heads,self.head_size,num_embeddings,block_size,temprature)
        self.lm_head =nn.Linear(num_embeddings,len(chars))
        
        
        self.encoding_paring ={}
        self.decoding_paring ={}

    

        for encoding,char in enumerate(chars):
            self.encoding_paring[char] = encoding
            self.decoding_paring[encoding] = char
        
    def forward(self,idx):
        _, T = idx.shape
        positions = torch.arange(T)

        pos_embed = self.positon_embedding(positions)
        token_embed = self.vocab_embedding_table(idx) 
        
        
        x = token_embed + pos_embed 

        x =self.Heads(x)

        logits = self.lm_head(x)

        return logits



  
    
    def generate(self):
        
        idx = torch.zeros((1,1),dtype=torch.long,device='cpu')
        

        while True:
            
            
            idx_cropped =idx[:, -self.block_size:]
            logits = self(idx_cropped)
            logits = logits[:, -1, :]/self.temprature
            probs = F.softmax(logits,dim =-1)
            
            index = torch.multinomial(probs, num_samples=1)
            
            
            output =self.decoding_paring.get(index.item())

            idx = torch.cat((idx,index),dim=1)
            print(output, end="", flush=True)
            sleep(0.05)

    def encode(self,string):
        values =[]
        for char in string:
            values.append(self.encoding_paring.get(char))
        return values

        
    
        