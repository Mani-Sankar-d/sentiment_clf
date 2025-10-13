import torch
import torch.nn as nn
from model.bert import Model
from dataset.IMDb import IMDb_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer 
from utils.checkpoint import save_ckpt, load_ckpt

path = "./sentiment_data/val.csv"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = IMDb_dataset(path=path)
loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)
criterion = nn.CrossEntropyLoss()
def validate(model,device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for text_info, label in loader:
            input_ids = text_info["input_ids"].to(device)
            label = label.to(device)
            attn_masks = text_info["attention_mask"].to(device)
            outputs = model(input_ids,attn_masks)
            loss = criterion(outputs,label)
            val_loss+=loss.item()
    print(f"Val_loss: {val_loss/len(loader):.4f}")
    return val_loss