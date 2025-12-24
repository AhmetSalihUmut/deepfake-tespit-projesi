import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Kendi modüllerimiz
from dataset import DeepfakeDataset
from models.xception_rnn import XceptionRNN
from utils import save_checkpoint, load_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 1. Config
    CONFIG_PATH = 'config/base.yaml'
    print(f"📄 Config yükleniyor: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    
    if 'reproducibility' in config:
        set_seed(config['reproducibility']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Çalışma Ortamı: {device}")
    
    # 2. Veri Seti
    print("📦 Veri setleri hazırlanıyor...")
    full_dataset = DeepfakeDataset(config_path=CONFIG_PATH, split='train')
    
    # Train/Val Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
    
    # 3. Model
    print("🧠 Model (Standart Mod) kuruluyor...")
    model = XceptionRNN(config).to(device)
    
    # 4. Loss & Optimizer (Ağırlık Yok - Standart BCE)
    criterion = nn.BCEWithLogitsLoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    
    num_epochs = config['training']['epoch_count']
    start_epoch = 0
    
    print("✨ Eğitim başlıyor...")
    best_val_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # İstatistikler
            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        # Epoch Sonu
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        
        print(f"\n📊 Sonuçlar: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, 'checkpoints/best_model.pth')
            print("💾 En iyi model kaydedildi!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()