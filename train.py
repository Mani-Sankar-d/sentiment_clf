import torch
import torch.nn as nn
from model.bert import Model
from dataset.IMDb import IMDb_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer 
from utils.checkpoint import save_ckpt, load_ckpt
from validate_ import validate


path = "./sentiment_data/train.csv"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = IMDb_dataset(path=path)
loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)
def train(epochs,start_epoch):
    for epoch in range(start_epoch,epochs):
        model.train()
        epoch_loss = 0
        for text_info, labels in loader:
            input_ids = text_info["input_ids"].to(device)
            attn_masks = text_info["attention_mask"].to(device)
            labels = labels.to(device)
            outputs = model(input_ids,attn_masks)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: loss {epoch_loss/len(loader):.4f}")
        val_loss = validate(model,device)
        save_ckpt(model=model,optimizer=optimizer,epoch=epoch,val_loss=val_loss,counter=counter,title="last_checkpoint.pth")
        if(val_loss<best_val_loss):
            save_ckpt(model=model,optimizer=optimizer,epoch=epoch,val_loss=val_loss,counter=counter,title="best_checkpoint.pth")

if __name__=="__main__":
    lr=2e-4
    model = Model(vocab_size=tokenizer.vocab_size,embed_dim=256,max_len=512,n_heads=8,n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.005)
    criterion = nn.CrossEntropyLoss()
    counter=0
    start_epoch=0
    best_val_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    model,optimizer,start_epoch,best_val_loss,counter = load_ckpt(model=model,optimizer=optimizer,path="last_checkpoint.pth",device=device)
    print(f"Resuming training from epoch {start_epoch}")
    train(100,start_epoch)