import torch 
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer 

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def get_tokens(text):
    tokens = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt',          
        return_attention_mask=True   
    )
    return tokens

class IMDb_dataset(torch.utils.data.Dataset):
    def __init__(self,path):
        self.df = pd.read_csv(path)
        super().__init__()
    def __len__(self):
        return (len(self.df))
    def __getitem__(self, index):
        tokens = get_tokens(self.df.loc[index]["review"])
        label = torch.tensor(int(self.df.loc[index]["sentiment"]=="positive"),dtype=torch.long)
        return {
           "input_ids":tokens["input_ids"].squeeze(0),
           "attention_mask":tokens["attention_mask"].squeeze(0)},label