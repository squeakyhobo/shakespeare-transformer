import torch.nn as nn
import torch
import torch.nn.functional as F
from time import sleep




class Bigram(nn.Module):
    def __init__(self,vocab_size,chars,temprature,num_embed,block_size):
        super().__init__()
        
        self.vocab_embedding_table = nn.Embedding(vocab_size,num_embed)
        self.positon_embedding = nn.Embedding(block_size,num_embed)
        self.lm_head = nn.Linear(num_embed,vocab_size)

        self.encoding_paring ={}
        self.decoding_paring ={}
        self.temprature =temprature
        self.block_size = block_size

        for encoding,char in enumerate(chars):
            self.encoding_paring[char] = encoding
            self.decoding_paring[encoding] = char

    def forward(self,idx):

        B, T = idx.shape
        positions = torch.arange(T)

        pos_embed = self.positon_embedding(positions)
        token_embed = self.vocab_embedding_table(idx) 
        
        
        x = token_embed + pos_embed
        
        


        logits = self.lm_head(x)
        return logits
    
   
    
    def encode(self,string):
        values =[]
        for char in string:
            values.append(self.encoding_paring.get(char))
        return values


    def decode(self,encodings):
        decodings =[]
        for value in encodings:
            decodings.append(self.decoding_paring.get(value))
        return decodings
    
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
            
            
        
        
        
    
        