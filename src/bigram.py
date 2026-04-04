import torch.nn as nn
import torch
import torch.nn.functional as F
from time import sleep




class Bigram(nn.Module):
    def __init__(self,vocab_size,chars):
        super().__init__()
        self.vocab_embedding_table = nn.Embedding(vocab_size,vocab_size)
        self.encoding_paring ={}
        self.decoding_paring ={}
        self.temprature =1.2

        for encoding,char in enumerate(chars):
            self.encoding_paring[char] = encoding
            self.decoding_paring[encoding] = char

    def forward(self,idx):
        logits = self.vocab_embedding_table(idx) # raw data not normalised
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
            
            
            
            logits = self(idx)
            logits = logits[:, -1, :]/self.temprature
            probs = F.softmax(logits,dim =-1)
            
            index = torch.multinomial(probs, num_samples=1)
            
            
            output =self.decoding_paring.get(index.item())

            idx = torch.cat((idx,index),dim=1)
            print(output, end="", flush=True)
            sleep(0.05)
            
            
        
        
        
    
        