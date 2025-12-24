import torch
import torch.nn as nn
import timm

class XceptionRNN(nn.Module):
    def __init__(self, config):
        super(XceptionRNN, self).__init__()
        
        # --- ANAYASA UYUM MODU & AYARLAR ---
        # Config varsa oradan al, yoksa varsayılanları kullan
        self.hidden_size = config.get('hidden_size', 256) if config else 256
        self.num_layers = config.get('num_layers', 2) if config else 2
        self.num_classes = config.get('num_classes', 1) if config else 1
        dropout_rate = config.get('dropout', 0.5) if config else 0.5
        
        # 1. CNN (Xception) - Özellik Çıkarıcı
        self.cnn = timm.create_model('xception', pretrained=True)
        self.cnn.reset_classifier(0) # Sınıflandırma katmanını kaldır
        self.input_size = 2048       # Xception 2048 özellik verir

        # 2. RNN (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate if self.num_layers > 1 else 0
        )
        
        # 3. Son Sınıflandırıcı
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # --- HATA ÖNLEYİCİ KALKAN (BOYUT DÜZELTME) ---
        # Eğer app.py yanlışlıkla (Seq, C, H, W) gönderirse (4 boyut),
        # biz onu (1, Seq, C, H, W) yaparız (5 boyut).
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        # x shape beklentisi: (batch_size, seq_len, channels, height, width)
        try:
            batch_size, seq_len, c, h, w = x.size()
        except ValueError:
            # Olası bir durumda boyutları manuel yazdıralım ki loglarda görelim
            print(f"!!! HATA: Model girdi boyutu beklendiği gibi değil. Gelen: {x.shape}")
            raise

        # CNN için birleştir: (batch * seq, c, h, w)
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        # CNN'den geçir
        features = self.cnn.forward_features(c_in)
        
        # --- POOLING DÜZELTMESİ (100352 -> 2048) ---
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.flatten(1)
        
        # RNN için tekrar ayır: (batch, seq, 2048)
        r_in = features.view(batch_size, seq_len, -1)
        
        # LSTM'den geçir
        lstm_out, _ = self.lstm(r_in)
        
        # Sadece son zaman adımını al (Classification için)
        last_out = lstm_out[:, -1, :]
        
        # Sınıflandır
        out = self.fc(self.dropout(last_out))
        
        return out