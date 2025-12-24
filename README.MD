# 🛡️ Deepfake Detection Project (Xception-LSTM)

## 📌 Proje Hakkında
Bu proje, Xception (CNN) ve LSTM (RNN) mimarilerini birleştirerek videoların sahte olup olmadığını tespit eden bir **Deepfake Tespit Sistemi**dir. Lisans bitirme tezi kapsamında geliştirilmiştir.

## 🚀 Özellikler
* **Hibrit Mimari:** Uzamsal ve zamansal analiz.
* **Gradio Arayüzü:** Kullanıcı dostu web arayüzü.
* **Yüksek Başarım:** %99 AUC skoru.

## 🧠 Model Dosyası İndirme
GitHub dosya boyutu sınırı nedeniyle, eğitilmiş model dosyasını aşağıdaki linkten indirmeniz gerekmektedir:
[📥 https://drive.google.com/drive/folders/1z5JwVeQOK0x-nkbBIXiZt80UXMkgklBd?usp=sharing (Google Drive)]

**Kurulum:** İndirdiğiniz dosyayı projenin içindeki `checkpoints/` klasörüne atınız.

## 📂 Veri Seti (Dataset)
Projenin eğitimi için kullanılan veri seti dosya boyutu nedeniyle harici olarak sunulmuştur:

[📥 https://drive.google.com/drive/folders/1YCESSsaBySAIWFSIXccEMui2ATw8VNAx?usp=sharing (Google Drive)]

**Kurulum:** İndirdiğiniz klasörü projenin ana dizinine `data` adıyla atınız.

## 🛠️ Kurulum
1. Dosyaları indirin.
2. Gerekli kütüphaneleri kurun: `pip install -r requirements.txt`
3. Uygulamayı başlatın: `python app.py`

## 🧠 Model Dosyası
GitHub dosya boyutu sınırı nedeniyle eğitilmiş model dosyasını ayrıca indirmeniz gerekmektedir:
[📥 best_model.pth İndir (Google Drive)](BURAYA_KOPYALADIGIN_DRIVE_LINKINI_YAPISTIR)

İndirdiğiniz dosyayı `checkpoints/` klasörünün içine atınız.

## 📊 Sonuçlar
Projemiz, standart modellere göre %10 daha yüksek başarım göstermiştir.
