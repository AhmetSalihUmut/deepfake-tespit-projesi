import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import json
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Kendi modüllerimiz
from dataset import DeepfakeDataset
from models.xception_rnn import XceptionRNN

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def evaluate():
    # 1. AYARLAR
    CONFIG_PATH = 'config/base.yaml'
    MODEL_PATH = 'outputs/models/best_model_xception.pth' # Train.py'nin kaydettiği dosya
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Değerlendirme Ortamı: {device}")
    
    # Config yükle
    config = load_config(CONFIG_PATH)
    batch_size = config['training']['batch_size']

    # 2. TEST VERİ SETİ (Split='test' olmalı)
    print("📦 Test verisi hazırlanıyor...")
    test_dataset = DeepfakeDataset(config_path=CONFIG_PATH, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 3. MODELİ YÜKLE
    print(f"🧠 Model yükleniyor: {MODEL_PATH}")
    # Model mimarisini başlat (Eğitimdeki parametrelerin aynısı olmalı)
    model = XceptionRNN(num_classes=1, hidden_size=256, num_layers=2).to(device)
    
    # Eğitilmiş ağırlıkları yükle
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Ağırlıklar başarıyla yüklendi.")
    else:
        print(f"❌ HATA: Model dosyası bulunamadı! ({MODEL_PATH})")
        print("Önce train.py çalıştırıp modeli eğitmelisiniz.")
        return

    # 4. TEST DÖNGÜSÜ
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("🔍 Test seti taranıyor...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(images)
            probs = torch.sigmoid(outputs) # 0 ile 1 arasına sıkıştır
            
            # Tahminler (0.5 eşik değeri)
            preds = (probs > 0.5).float()
            
            # Listelere ekle (CPU'ya alarak)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. METRİKLERİ HESAPLA (Bilimsel Çıktılar)
    print("\n" + "="*30)
    print("📊 TEST SONUÇLARI")
    print("="*30)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0 # Sadece tek bir sınıf varsa AUC hesaplanamaz
    
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Accuracy (Doğruluk): {acc:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(f"AUC Score          : {auc:.4f}")
    print("\nConfusion Matrix (Karmaşıklık Matrisi):")
    print(cm)
    print("\nDetaylı Rapor:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    # 6. SONUÇLARI KAYDET
    results = {
        'accuracy': acc,
        'f1_score': f1,
        'auc_score': auc,
        'confusion_matrix': cm.tolist()
    }
    
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n💾 Sonuçlar kaydedildi: outputs/metrics/test_results.json")

if __name__ == "__main__":
    evaluate()