import pandas as pd

class Util:
    
    def __init__(self, pathdata:str):
        
        self.pathdata = pathdata
        self.dataset = self.get_dataset()
        
    def get_dataset(self):
        df = pd.read_csv(self.pathdata)
        return df
    
    def train_and_target(self):
        X , y = self.dataset.drop(["Strength"], axis=1) , self.dataset["Strength"]
        return X , y
        
        
        