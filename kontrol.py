import os

# Projenin ana dizinini bul
current_dir = os.getcwd()
print(f"📍 Şu anki çalışma dizini: {current_dir}")

# Hedeflenen LMDB yolu
target_path = os.path.join(current_dir, 'data', 'lmdb')
print(f"🎯 Hedeflenen yol: {target_path}")

# Yol var mı?
if os.path.exists(target_path):
    print("✅ Klasör bulundu!")
    # İçindeki dosyaları listele
    files = os.listdir(target_path)
    print(f"📂 Klasör içindekiler: {files}")
    
    if 'data.mdb' in files:
        print("🎉 data.mdb bulundu! Yol doğru.")
    else:
        print("⚠️ Klasör var ama içinde data.mdb YOK!")
else:
    print("❌ HATA: Bu yol bilgisayarında bulunamadı.")