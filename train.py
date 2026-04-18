import os
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from src.Transformer import Transformer

load_dotenv()


#device = 'mps' if torch.mps.is_available() else 'cpu'
device = 'cpu'


block_size = 8
batch_size = 32
learning_rate = 10e-3
train_iterations = 1000
eval_iterations = 500
temprature =1.0
number_embeddings = 32






shakespeare_path = os.getenv("SHAKESPEARE_PATH")


with open(shakespeare_path,"r") as file:
    text = file.read() # gets tiny shakespeare text
chars = sorted(list(set(text))) # get characters in text
vocab_size = len(chars)


#encode and decode chars




def get_batch(data):
    # Generate random starting points in the text
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Stack the chunks into a 2D Tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # The targets are the same chunks, but shifted right by one
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y
# split date into training and val


        


   




def main():


    model = Transformer(chars=chars,temprature=temprature) # init model
    model.to(device)

    data = torch.tensor(model.encode(text), dtype=torch.long)  #encode whole dataset


    n = int(len(data)*0.9) # index to split data at 

    train_data =data[:n]
    val_data =data[n:]  

    train_data.to(device)
    val_data.to(device)

    optimiser = torch.optim.Adam(model.parameters(),learning_rate)

    for step in range(train_iterations):

        xb,yb = get_batch(train_data) # shape is (B,T) T =8 
        
        logits = model(xb) #hsape should be (BATCH_size,T)
        B, T, C = logits.shape
        logits = logits.view(B*T,C)
        targets =yb.view(B*T)
        loss = F.cross_entropy(logits,targets)
        #print(loss)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.generate()


        

        







    
    

main()



    
