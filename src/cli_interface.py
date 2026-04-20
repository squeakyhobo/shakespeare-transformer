class CLI_Interface():
    def __init__(self,model):
        self.model = model
    


    def interact(self):
        prompt  = input()
        if len(prompt) <= self.model.block_size:
            self.model.generate(prompt)  

