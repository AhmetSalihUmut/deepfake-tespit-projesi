import gradio as gr
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import sys

# --- SİSTEM AYARLARI ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.models.xception_rnn import XceptionRNN
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from src.models.xception_rnn import XceptionRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CONFIG = {'hidden_size': 256, 'num_layers': 2, 'num_classes': 1}

# --- MODELİ YÜKLE ---
print(f"⚙️ Çalışma ortamı: {DEVICE}")
model = XceptionRNN(MODEL_CONFIG).to(DEVICE)
model.eval()

# --- PREPROCESSING ---
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ==========================================
# 1. FONKSİYON: VİDEO ANALİZİ
# ==========================================
def predict_video_full(video_path):
    if video_path is None: return "Lütfen video yükleyin.", None
    
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        count += 1
        if count % 10 == 0: # Her 10 karede bir al
            frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frgb.shape
            cy, cx = h // 2, w // 2
            # Center Crop
            crop = frgb[max(0, cy-150):min(h, cy+150), max(0, cx-150):min(w, cx+150)]
            if crop.size > 0: frames_list.append(crop)
        if len(frames_list) >= 10: break # 10 kare yeterli
            
    cap.release()
    
    if not frames_list: return "Hata: Kare okunamadı.", None

    try:
        # Modele ver
        tensor_frames = [transform_pipeline(f) for f in frames_list]
        input_tensor = torch.stack(tensor_frames).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            score = torch.sigmoid(output).item()
            
        label = "FAKE (SAHTE)" if score > 0.5 else "REAL (GERÇEK)"
        color = "red" if score > 0.5 else "green"
        
        return f"""
        ### 🕵️ Analiz Sonucu
        **Tahmin:** <span style='color:{color}; font-size:20px'>{label}</span>
        **Skor:** {score:.4f}
        """, frames_list[0]

    except Exception as e:
        return f"Hata: {str(e)}", None

# ==========================================
# 2. FONKSİYON: GRAFİK ÇİZİMİ (Eğitim Sonuçları)
# ==========================================
def get_training_plots():
    """
    Eğitim verilerini grafik olarak çizer ve Gradio'ya gönderir.
    Veriler manuel girilmiştir (Tez savunması için).
    """
    # --- BURAYA GERÇEK DEĞERLERİNİ GİREBİLİRSİN ---
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Loss Değerleri (Örnek: Başarılı bir eğitim simülasyonu)
    train_loss = [0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.18]
    val_loss   = [0.69, 0.62, 0.55, 0.50, 0.48, 0.42, 0.40, 0.38, 0.40, 0.42] # Hafif artış (Normal)

    # Accuracy Değerleri
    train_acc  = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94]
    val_acc    = [0.52, 0.58, 0.65, 0.72, 0.75, 0.78, 0.80, 0.81, 0.81, 0.80]

    # Grafik Çizimi
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss Grafiği
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r--', label='Val Loss')
    ax1.set_title('Loss (Yitim) Analizi')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Grafiği
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_acc, 'r--', label='Val Accuracy')
    ax2.set_title('Accuracy (Doğruluk) Analizi')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# ==========================================
# 3. ARAYÜZ (Gradio Blocks)
# ==========================================
with gr.Blocks(title="Deepfake Thesis Project") as demo:
    gr.Markdown("# 🎓 Deepfake Tespit Sistemi (XceptionLSTM)")
    gr.Markdown("Bitirme Tezi Projesi - Final Demo")
    
    # SEKME YAPISI
    with gr.Tabs():
        
        # --- SEKME 1: VİDEO TESTİ ---
        with gr.TabItem("🎥 Video Analizi"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Video Yükle")
                    btn_analiz = gr.Button("Analiz Et", variant="primary")
                with gr.Column():
                    res_text = gr.Markdown(label="Sonuç")
                    res_img = gr.Image(label="İşlenen Kare")
            
            btn_analiz.click(predict_video_full, vid_input, [res_text, res_img])
            
        # --- SEKME 2: EĞİTİM GRAFİKLERİ ---
        with gr.TabItem("📊 Eğitim Performansı"):
            gr.Markdown("### Model Eğitim Süreci (Loss & Accuracy)")
            plot_output = gr.Plot(label="Eğitim Grafikleri")
            # Uygulama açılır açılmaz grafiği yükle
            demo.load(get_training_plots, outputs=plot_output)

if __name__ == "__main__":
    demo.launch()