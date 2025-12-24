import torch
import os

def save_checkpoint(model, optimizer, epoch, filename="checkpoints/checkpoint.pth"):
    print("=> Checkpoint kaydediliyor...")
    
    # Checkpoints klasörü yoksa oluştur
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename="checkpoints/checkpoint.pth"):
    print("=> Checkpoint yükleniyor...")
    if not os.path.isfile(filename):
        print(f"⚠️ Uyarı: '{filename}' bulunamadı, sıfırdan başlanıyor.")
        return model, optimizer, 0 # epoch 0 döndür
        
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Kaydedilen epoch'un bir fazlasından devam et
    epoch = checkpoint.get("epoch", 0) + 1 
    return model, optimizer, epoch