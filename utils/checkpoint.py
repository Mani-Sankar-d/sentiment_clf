import torch
import os
 
def save_ckpt(model,optimizer,epoch,val_loss,counter,title):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': val_loss,
            'counter': counter
        }, title)

def get_best_val_loss(path,device):
    if os.path.exists(path):
        best_checkpoint = torch.load(path,map_location=device)
        return best_checkpoint["best_val_loss"]
    return float('inf')

def load_ckpt(model,optimizer,path,device):
    if os.path.exists(path):
        checkpoint = torch.load(path,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = get_best_val_loss("best_checkpoint.pth",device)
        start_epoch = checkpoint["epoch"]+1
        counter = checkpoint["counter"]
        
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        counter = 0
        if(path=="last_checkpoint.pth"):
            print("Starting training from scratch.")
    return model,optimizer,start_epoch,best_val_loss,counter