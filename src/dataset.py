import torch
from torch.utils.data import Dataset
import lmdb
import json
import yaml
import numpy as np
from PIL import Image
import io
import os  # <-- os kütüphanesi
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    def __init__(self, config_path, split='train'):
        # 1. Config'i Yükle
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 2. Metadata'yı Yükle
        with open('data/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        # JSON yapısına göre veriyi al
        if 'videos' in self.metadata and split in self.metadata['videos']:
            self.samples = self.metadata['videos'][split]
        else:
            print(f"UYARI: '{split}' bulunamadı, varsayılan olarak 'train' deneniyor.")
            self.samples = self.metadata.get('videos', {}).get('train', [])
            
        self.split = split
        self.img_size = self.config['data']['image_size']
        self.seq_len = self.config['data']['frames_per_video']
        
        # --- AKILLI YOL BULUCU (OTOMATİK TESPİT) ---
        current_file_path = os.path.abspath(__file__) 
        src_dir = os.path.dirname(current_file_path)
        project_root = os.path.dirname(src_dir)
        
        # İki ihtimali de kontrol et:
        path_option_1 = os.path.join(project_root, 'data')         # Senin durumun bu
        path_option_2 = os.path.join(project_root, 'data', 'lmdb') # Standart durum
        
        if os.path.exists(os.path.join(path_option_1, 'data.mdb')):
            self.lmdb_path = path_option_1
            print(f"✅ DOĞRU YOL BULUNDU (Seçenek 1): {self.lmdb_path}")
        elif os.path.exists(os.path.join(path_option_2, 'data.mdb')):
            self.lmdb_path = path_option_2
            print(f"✅ DOĞRU YOL BULUNDU (Seçenek 2): {self.lmdb_path}")
        else:
            # Bulamazsa yine de 'data'yı dene ama uyarı ver
            print("⚠️ UYARI: data.mdb dosyası otomatik bulunamadı! 'data' klasörü varsayılıyor.")
            self.lmdb_path = path_option_1
            
        print(f"📂 KULLANILAN LMDB YOLU: {self.lmdb_path}")
        # -------------------------------------------

        self.env = None 

        # Augmentation Pipeline
        if split == 'train':
            aug_params = self.config['training']['augmentation_pipeline']
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=aug_params['RandomHorizontalFlip']),
                transforms.ColorJitter(
                    brightness=aug_params['ColorJitter']['brightness'],
                    contrast=aug_params['ColorJitter']['contrast'],
                    saturation=aug_params['ColorJitter']['saturation'],
                    hue=aug_params['ColorJitter']['hue']
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.env is None:
            # LMDB'yi aç
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        video_info = self.samples[idx]
        
        key_prefix = video_info['id'] 
        
        # Label Dönüşümü
        raw_label = video_info['label']
        if isinstance(raw_label, str):
            if raw_label.lower() == 'fake':
                label = 1.0
            else:
                label = 0.0
        else:
            label = float(raw_label)
        
        total_frames = video_info.get('num_frames', 30) 

        # Kare Seçimi
        frame_indices = np.linspace(0, total_frames - 1, self.seq_len, dtype=int)
        
        frames = []
        with self.env.begin(write=False) as txn:
            for frame_idx in frame_indices:
                full_key = f"{key_prefix}_{frame_idx:03d}".encode('ascii')
                byteflow = txn.get(full_key)
                
                if byteflow is None:
                    img = Image.new('RGB', (self.img_size, self.img_size))
                else:
                    buffer = io.BytesIO(byteflow)
                    img = Image.open(buffer).convert('RGB')
                
                img_tensor = self.transform(img)
                frames.append(img_tensor)

        frames_stack = torch.stack(frames)
        return frames_stack, torch.tensor(label, dtype=torch.float32)